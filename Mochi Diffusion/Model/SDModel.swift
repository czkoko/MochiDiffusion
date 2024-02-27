//
//  SDModel.swift
//  Mochi Diffusion
//
//  Created by Joshua Park on 2/12/23.
//

import CoreML
import Foundation
import os.log

private let logger = Logger()

struct SDModel: Identifiable {
    let url: URL
    let name: String
    let attention: SDModelAttentionType
    let controlNet: [String]
    let isXL: Bool
    var inputSize: CGSize?
    var controltype: ControlType?
    var allowsVariableSize: Bool

    var id: URL { url }

    init?(url: URL, name: String, controlNet: [SDControlNet]) {
        guard let attention = identifyAttentionType(url) else {
            return nil
        }

        let isXL = identifyIfXL(url)
        let size = identifyInputSize(url)
        let controltype = identifyControlNetType(url)

        self.url = url
        self.name = name
        self.attention = attention
        if let size = size {
            self.controlNet = controlNet.filter { $0.size == size && $0.attention == attention && $0.controltype == controltype ?? .all}.map { $0.name }
        } else {
            self.controlNet = []
        }
        self.isXL = isXL
        self.inputSize = size
        self.controltype = controltype
        self.allowsVariableSize = identifyAllowsVariableSize(url)!
    }
}

extension SDModel: Hashable {
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

private func identifyAttentionType(_ url: URL) -> SDModelAttentionType? {
    guard let metadataURL = unetMetadataURL(from: url) else {
        logger.warning("No model metadata found at '\(url)'")
        return nil
    }

    struct ModelMetadata: Decodable {
        let mlProgramOperationTypeHistogram: [String: Int]
    }

    do {
        let jsonData = try Data(contentsOf: metadataURL)
        let metadatas = try JSONDecoder().decode([ModelMetadata].self, from: jsonData)

        guard metadatas.count == 1 else {
            return nil
        }

        return metadatas[0].mlProgramOperationTypeHistogram["Ios16.einsum"] != nil ? .splitEinsum : .original
    } catch {
        logger.warning("Failed to parse model metadata at '\(metadataURL)': \(error)")
        return nil
    }
}

private func identifyIfXL(_ url: URL) -> Bool {
    guard let metadataURL = unetMetadataURL(from: url) else {
        logger.warning("No model metadata found at '\(url)'")
        return false
    }

    struct ModelMetadata: Decodable {
        let inputSchema: [[String: String]]
    }

    do {
        let jsonData = try Data(contentsOf: metadataURL)
        let metadatas = try JSONDecoder().decode([ModelMetadata].self, from: jsonData)

        guard metadatas.count == 1 else {
            return false
        }

        // XL models have 5 inputs total (added: time_ids and text_embeds)
        let inputNames = metadatas[0].inputSchema.compactMap { $0["name"] }
        return inputNames.contains("time_ids") && inputNames.contains("text_embeds")
    } catch {
        logger.warning("Failed to parse model metadata at '\(metadataURL)': \(error)")
        return false
    }
}

private func unetMetadataURL(from url: URL) -> URL? {
    let potentialMetadataURLs = [
        url.appending(components: "Unet.mlmodelc", "metadata.json"),
        url.appending(components: "UnetChunk1.mlmodelc", "metadata.json")
    ]

    return potentialMetadataURLs.first {
        FileManager.default.fileExists(atPath: $0.path(percentEncoded: false))
    }
}

private func identifyInputSize(_ url: URL) -> CGSize? {
    let encoderMetadataURL = url.appending(path: "VAEDecoder.mlmodelc").appending(path: "metadata.json")
    if let jsonData = try? Data(contentsOf: encoderMetadataURL),
        let jsonArray = try? JSONSerialization.jsonObject(with: jsonData) as? [[String: Any]],
        let jsonItem = jsonArray.first,
        let inputSchema = jsonItem["outputSchema"] as? [[String: Any]],
        let controlnetCond = inputSchema.first,
        let shapeString = controlnetCond["shape"] as? String {
        let shapeIntArray = shapeString.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
            .components(separatedBy: ", ")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let width = shapeIntArray[3]
        let height = shapeIntArray[2]
        return CGSize(width: width, height: height)
    } else {
        return nil
    }
}

private func identifyControlNetType(_ url: URL) -> ControlType? {
    let metadataURL = url.appending(path: "Unet.mlmodelc").appending(path: "metadata.json")

    guard let jsonData = try? Data(contentsOf: metadataURL) else {
        print("Error: Could not read data from \(metadataURL)")
        return nil
    }

    guard let jsonArray = (try? JSONSerialization.jsonObject(with: jsonData)) as? [[String: Any]] else {
        print("Error: Could not parse JSON data")
        return nil
    }

    guard let jsonItem = jsonArray.first else {
        print("Error: JSON array is empty")
        return nil
    }

    guard let inputSchema = jsonItem["inputSchema"] as? [[String: Any]] else {
        print("Error: Missing 'inputSchema' in JSON")
        return nil
    }

    if inputSchema.first(where: { ($0["name"] as? String) == "adapter_res_samples_00" }) != nil && inputSchema.first(where: { ($0["name"] as? String) == "down_block_res_samples_00" }) != nil {
        return .all
    }else if inputSchema.first(where: { ($0["name"] as? String) == "adapter_res_samples_00" }) != nil {
        return .T2IAdapter
    }else{
        return .ControlNet
    }
}

private func identifyAllowsVariableSize(_ url: URL) -> Bool? {
    let metadataURL = url.appending(path: "Unet.mlmodelc").appending(path: "metadata.json")

    guard let jsonData = try? Data(contentsOf: metadataURL) else {
        print("Error: Could not read data from \(metadataURL)")
        return nil
    }

    guard let jsonArray = (try? JSONSerialization.jsonObject(with: jsonData)) as? [[String: Any]] else {
        print("Error: Could not parse JSON data")
        return nil
    }

    guard let jsonItem = jsonArray.first else {
        print("Error: JSON array is empty")
        return nil
    }

    guard let inputSchema = jsonItem["inputSchema"] as? [[String: Any]] else {
        print("Error: Missing 'inputSchema' in JSON")
        return nil
    }

    if inputSchema.first(where: { ($0["hasShapeFlexibility"] as? String) == "1" }) != nil {
        return true
    }else{
        return false
    }
}

extension SDModel {
    public func resized(width: Int, height: Int) async -> SDModel?  {
        if let currentWidth = inputSize?.width,
           let currentHeight = inputSize?.height,
           width == Int(currentWidth) && height == Int(currentHeight) {
            return self
        }
        let modelSizeName = "\(name)_\(width)x\(height)"
        let newURL = FileManager.default.temporaryDirectory.appending(path: modelSizeName)
        if FileManager.default.fileExists(atPath: newURL.path(percentEncoded: false)) {
            return SDModel(url: newURL, name: name, controlNet: []) // TODO: variable size controlnet
        } else {
            do {
                try FileManager.default.copyItem(at: self.url, to: newURL)

                guard let decoderURL = Bundle.main.url(forResource: "decoder-coremldata", withExtension: "bin"),
                      let encoderURL = Bundle.main.url(forResource: "encoder-coremldata", withExtension: "bin") else {
                    return nil
                }
                try FileManager.default.removeItem(at: newURL.appending(components: "VAEEncoder.mlmodelc", "coremldata.bin"))
                try FileManager.default.copyItem(at: encoderURL, to: newURL.appending(components: "VAEEncoder.mlmodelc", "coremldata.bin"))
                try FileManager.default.removeItem(at: newURL.appending(components: "VAEDecoder.mlmodelc", "coremldata.bin"))
                try FileManager.default.copyItem(at: decoderURL, to: newURL.appending(components: "VAEDecoder.mlmodelc", "coremldata.bin"))

                let encoderMIL = newURL.appending(components: "VAEEncoder.mlmodelc", "model.mil")
                let decoderMIL = newURL.appending(components: "VAEDecoder.mlmodelc", "model.mil")

                if isXL {
                    try await vaeEnSDXL(vaeMIL: encoderMIL, height: ImageController.shared.height, width: ImageController.shared.width)
                    try await vaeDeSDXL(vaeMIL: decoderMIL, height: ImageController.shared.height, width: ImageController.shared.width)
                } else {
                    try await vaeEnSD(vaeMIL: encoderMIL, height: ImageController.shared.height, width: ImageController.shared.width)
                    try await vaeDeSD(vaeMIL: decoderMIL, height: ImageController.shared.height, width: ImageController.shared.width)
                }

                modifyMetadataInputSize(url: newURL, height: height, width: width)
                return SDModel(url: newURL, name: name, controlNet: []) // TODO: variable size controlnet
            } catch {
                print("Error resizing model \(error)")
                return nil
            }
        }
    }

    private func vaeDeSDXL(vaeMIL: URL, height: Int, width: Int) throws {
        var fileContent = try String(contentsOf: vaeMIL, encoding: .utf8)
        fileContent = fileContent.replacingOccurrences(of: "[1, 4, 128, 128]", with: "[1, 4, \(height / 8), \(width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 512, 128, 128]", with: "[1, 512, \(height / 8), \(width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 16, 128, 128]", with: "[1, 32, 16, \(height / 8), \(width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 16, 16384]", with: "[1, 32, 16, \(height / 8 * width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 512, 16384]", with: "[1, 512, \(height / 8 * width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 16384, 512]", with: "[1, \(height / 8 * width / 8), 512]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 16384, 1, 512]", with: "[1, \(height / 8 * width / 8), 1, 512]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 1, 16384, 512]", with: "[1, 1, \(height / 8 * width / 8), 512]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 1, 16384, 16384]", with: "[1, 1, \(height / 8 * width / 8), \(height / 8 * width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 512, 256, 256]", with: "[1, 512, \(height / 4), \(width / 4)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 16, 256, 256]", with: "[1, 32, 16, \(height / 4), \(width / 4)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 512, 512, 512]", with: "[1, 512, \(height / 2), \(width / 2)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 16, 512, 512]", with: "[1, 32, 16, \(height / 2), \(width / 2)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 256, 512, 512]", with: "[1, 256, \(height / 2), \(width / 2)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 8, 512, 512]", with: "[1, 32, 8, \(height / 2), \(width / 2)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 256, 1024, 1024]", with: "[1, 256, \(height), \(width)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 8, 1024, 1024]", with: "[1, 32, 8, \(height), \(width)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 128, 1024, 1024]", with: "[1, 128, \(height), \(width)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 4, 1024, 1024]", with: "[1, 32, 4, \(height), \(width)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 3, 1024, 1024]", with: "[1, 3, \(height), \(width)]")
        try fileContent.write(to: vaeMIL, atomically: false, encoding: .utf8)
    }

    private func vaeEnSDXL(vaeMIL: URL, height: Int, width: Int) throws {
        var fileContent = try String(contentsOf: vaeMIL, encoding: .utf8)
        fileContent = fileContent.replacingOccurrences(of: "[1, 8, 128, 128]", with: "[1, 8, \(height / 8), \(width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 1, 16384, 512]", with: "[1, 1, \(height / 8 * width / 8), 512]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 1, 16384, 16384]", with: "[1, 1, \(height / 8 * width / 8), \(height / 8 * width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 16384, 1, 512]", with: "[1, \(height / 8 * width / 8), 1, 512]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 16384, 512]", with: "[1, \(height / 8 * width / 8), 512]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 512, 16384]", with: "[1, 512, \(height / 8 * width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 16, 16384]", with: "[1, 32, 16, \(height / 8 * width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 16, 128, 128]", with: "[1, 32, 16, \(height / 8), \(width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 512, 128, 128]", with: "[1, 512, \(height / 8), \(width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 512, 257, 257]", with: "[1, 512, \(height / 4 + 1), \(width / 4 + 1)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 16, 256, 256]", with: "[1, 32, 16, \(height / 4), \(width / 4)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 512, 256, 256]", with: "[1, 512, \(height / 4), \(width / 4)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 8, 256, 256]", with: "[1, 32, 8, \(height / 4), \(width / 4)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 256, 256, 256]", with: "[1, 256, \(height / 4), \(width / 4)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 256, 513, 513]", with: "[1, 256, \(height / 2 + 1), \(width / 2 + 1)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 8, 512, 512]", with: "[1, 32, 8, \(height / 2), \(width / 2)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 256, 512, 512]", with: "[1, 256, \(height / 2), \(width / 2)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 4, 512, 512]", with: "[1, 32, 4, \(height / 2), \(width / 2)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 128, 512, 512]", with: "[1, 128, \(height / 2), \(width / 2)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 128, 1025, 1025]", with: "[1, 128, \(height + 1), \(width + 1)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 128, 1024, 1024]", with: "[1, 128, \(height), \(width)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 4, 1024, 1024]", with: "[1, 32, 4, \(height), \(width)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 3, 1024, 1024]", with: "[1, 3, \(height), \(width)]")
        try fileContent.write(to: vaeMIL, atomically: false, encoding: .utf8)
    }

    private func vaeDeSD(vaeMIL: URL, height: Int, width: Int) throws {
        var fileContent = try String(contentsOf: vaeMIL, encoding: .utf8)
        fileContent = fileContent.replacingOccurrences(of: "[1, 4, 64, 64]", with: "[1, 4, \(height / 8), \(width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 512, 64, 64]", with: "[1, 512, \(height / 8), \(width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 16, 64, 64]", with: "[1, 32, 16, \(height / 8), \(width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 16, 4096]", with: "[1, 32, 16, \(height / 8 * width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 512, 4096]", with: "[1, 512, \(height / 8 * width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 4096, 512]", with: "[1, \(height / 8 * width / 8), 512]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 4096, 1, 512]", with: "[1, \(height / 8 * width / 8), 1, 512]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 1, 4096, 512]", with: "[1, 1, \(height / 8 * width / 8), 512]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 1, 4096, 4096]", with: "[1, 1, \(height / 8 * width / 8), \(height / 8 * width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 512, 128, 128]", with: "[1, 512, \(height / 4), \(width / 4)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 16, 128, 128]", with: "[1, 32, 16, \(height / 4), \(width / 4)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 512, 256, 256]", with: "[1, 512, \(height / 2), \(width / 2)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 16, 256, 256]", with: "[1, 32, 16, \(height / 2), \(width / 2)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 256, 256, 256]", with: "[1, 256, \(height / 2), \(width / 2)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 8, 256, 256]", with: "[1, 32, 8, \(height / 2), \(width / 2)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 256, 512, 512]", with: "[1, 256, \(height), \(width)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 8, 512, 512]", with: "[1, 32, 8, \(height), \(width)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 128, 512, 512]", with: "[1, 128, \(height), \(width)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 4, 512, 512]", with: "[1, 32, 4, \(height), \(width)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 3, 512, 512]", with: "[1, 3, \(height), \(width)]")
        try fileContent.write(to: vaeMIL, atomically: false, encoding: .utf8)
    }

    private func vaeEnSD(vaeMIL: URL, height: Int, width: Int) throws {
        var fileContent = try String(contentsOf: vaeMIL, encoding: .utf8)
        fileContent = fileContent.replacingOccurrences(of: "[1, 8, 64, 64]", with: "[1, 8, \(height / 8), \(width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 1, 4096, 512]", with: "[1, 1, \(height / 8 * width / 8), 512]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 1, 4096, 4096]", with: "[1, 1, \(height / 8 * width / 8), \(height / 8 * width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 4096, 1, 512]", with: "[1, \(height / 8 * width / 8), 1, 512]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 4096, 512]", with: "[1, \(height / 8 * width / 8), 512]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 512, 4096]", with: "[1, 512, \(height / 8 * width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 16, 4096]", with: "[1, 32, 16, \(height / 8 * width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 16, 64, 64]", with: "[1, 32, 16, \(height / 8), \(width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 512, 64, 64]", with: "[1, 512, \(height / 8), \(width / 8)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 512, 129, 129]", with: "[1, 512, \(height / 4 + 1), \(width / 4 + 1)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 16, 128, 128]", with: "[1, 32, 16, \(height / 4), \(width / 4)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 512, 128, 128]", with: "[1, 512, \(height / 4), \(width / 4)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 8, 128, 128]", with: "[1, 32, 8, \(height / 4), \(width / 4)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 256, 128, 128]", with: "[1, 256, \(height / 4), \(width / 4)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 256, 257, 257]", with: "[1, 256, \(height / 2 + 1), \(width / 2 + 1)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 8, 256, 256]", with: "[1, 32, 8, \(height / 2), \(width / 2)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 256, 256, 256]", with: "[1, 256, \(height / 2), \(width / 2)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 4, 256, 256]", with: "[1, 32, 4, \(height / 2), \(width / 2)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 128, 256, 256]", with: "[1, 128, \(height / 2), \(width / 2)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 128, 513, 513]", with: "[1, 128, \(height + 1), \(width + 1)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 128, 512, 512]", with: "[1, 128, \(height), \(width)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 32, 4, 512, 512]", with: "[1, 32, 4, \(height), \(width)]")
        fileContent = fileContent.replacingOccurrences(of: "[1, 3, 512, 512]", with: "[1, 3, \(height), \(width)]")
        try fileContent.write(to: vaeMIL, atomically: false, encoding: .utf8)
    }

    private func modifyMetadataInputSize(url: URL, height: Int, width: Int) {
        let encoderMetadataURL = url.appending(components: "VAEEncoder.mlmodelc", "metadata.json")
        guard let jsonData = try? Data(contentsOf: encoderMetadataURL),
              var jsonArray = try? JSONSerialization.jsonObject(with: jsonData) as? [[String: Any]],
              var jsonItem = jsonArray.first,
              var inputSchema = jsonItem["inputSchema"] as? [[String: Any]],
              var controlnetCond = inputSchema.first,
              var shapeString = controlnetCond["shape"] as? String else {
                  return
        }

        var shapeIntArray = shapeString.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
            .components(separatedBy: ", ")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

        shapeIntArray[3] = width
        shapeIntArray[2] = height
        shapeString = "[\(shapeIntArray.map { String($0) }.joined(separator: ", "))]"

        controlnetCond["shape"] = shapeString
        inputSchema[0] = controlnetCond
        jsonItem["inputSchema"] = inputSchema
        jsonArray[0] = jsonItem

        if let updatedJsonData = try? JSONSerialization.data(withJSONObject: jsonArray, options: .prettyPrinted) {
            try? updatedJsonData.write(to: encoderMetadataURL)
            print("update metadata.")
        } else {
            print("Failed to update metadata.")
        }
    }
}
