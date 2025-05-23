import MLX
import Tokenizers
import Vapor
import mlx_embeddings

struct EmbeddingRequest: Content {
    let input: EmbeddingInput
    let model: String? 
    let encoding_format: String?
    let dimensions: Int? 
    let user: String?
    let batch_size: Int?
}

enum EmbeddingInput: Codable {
    case string(String)
    case array([String])

    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self = .string(str)
        } else if let arr = try? container.decode([String].self) {
            self = .array(arr)
        } else {
            throw DecodingError.typeMismatch(
                EmbeddingInput.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath, debugDescription: "Expected String or [String]"))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let str):
            try container.encode(str)
        case .array(let arr):
            try container.encode(arr)
        }
    }

    var values: [String] {
        switch self {
        case .string(let str): return [str]
        case .array(let arr): return arr
        }
    }
}

struct EmbeddingData: Content {
    var object: String = "embedding"
    let embedding: EmbeddingOutput
    let index: Int
}

enum EmbeddingOutput: Codable {
    case floats([Float])
    case base64(String)

    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let arr = try? container.decode([Float].self) {
            self = .floats(arr)
        } else if let str = try? container.decode(String.self) {
            self = .base64(str)
        } else {
            throw DecodingError.typeMismatch(
                EmbeddingOutput.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath, debugDescription: "Expected [Float] or String"))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .floats(let arr):
            try container.encode(arr)
        case .base64(let str):
            try container.encode(str)
        }
    }
}

struct UsageData: Content {
    let prompt_tokens: Int
    let total_tokens: Int
}

struct EmbeddingResponse: Content {
    var object: String = "list"
    let data: [EmbeddingData]
    let model: String 
    let usage: UsageData
}


func registerEmbeddingsRoute(_ app: Application, embeddingModelProvider: EmbeddingModelProvider) {

    app.post("v1", "embeddings") { req async throws -> EmbeddingResponse in
        let embeddingRequest = try req.content.decode(EmbeddingRequest.self)
        let logger = req.logger
        let embeddingReqId = "emb-\(UUID().uuidString)"

        logger.info("Received embedding request (ID: \(embeddingReqId)) for model: \(embeddingRequest.model ?? "Default")")

        let (modelContainer, loadedModelName) = try await embeddingModelProvider.getModel(
            requestedModelId: embeddingRequest.model
        )

        let texts = embeddingRequest.input.values
        guard !texts.isEmpty, texts.allSatisfy({ !$0.isEmpty }) else {
            logger.error("Embedding request (ID: \(embeddingReqId)) input is empty or contains empty strings.")
            throw Abort(.badRequest, reason: "Input text(s) cannot be empty.")
        }

        let encodingFormat = embeddingRequest.encoding_format ?? "float"
        let batchSize = embeddingRequest.batch_size ?? texts.count

        logger.debug("Processing \(texts.count) text(s) for embedding (ID: \(embeddingReqId)) with model \(loadedModelName). Batch size: \(batchSize), Format: \(encodingFormat)")

        return try await modelContainer.perform { model, tokenizer in
            var allData: [EmbeddingData] = []
            var promptTokens = 0
            var index = 0

            for batchStart in stride(from: 0, to: texts.count, by: batchSize) {
                let batchEnd = min(batchStart + batchSize, texts.count)
                let batchTexts = Array(texts[batchStart..<batchEnd])
                logger.trace("Processing batch \(batchStart/batchSize + 1) for \(embeddingReqId), indices \(batchStart)..<\(batchEnd)")

                let tokenized = batchTexts.map { tokenizer.encode(text: $0, addSpecialTokens: true) }
                let currentBatchTokens = tokenized.reduce(0) { $0 + $1.count }
                promptTokens += currentBatchTokens
                logger.trace("Batch tokens for \(embeddingReqId): \(currentBatchTokens)")

                let maxLength = tokenized.map { $0.count }.max() ?? 16
                let padId = tokenizer.eosTokenId ?? 0
                guard let padIdInt = padId as? Int else {
                     logger.error("Could not determine valid integer padding token ID for \(embeddingReqId).")
                     struct EmbeddingError: Error, AbortError {
                         var status: HTTPResponseStatus { .internalServerError }
                         var reason: String { "Failed to get padding token ID" }
                     }
                     throw EmbeddingError()
                }

                let paddedArrays = tokenized.map { elem in
                     MLXArray(elem + Array(repeating: padIdInt, count: maxLength - elem.count))
                }

                guard !paddedArrays.isEmpty else { continue }

                let padded = try MLX.stacked(paddedArrays)
                let attentionMask = padded .!= MLXArray(padIdInt)
                let tokenTypeIds = MLXArray.zeros(like: padded) 

                let output = try  model(
                    padded, positionIds: nil, tokenTypeIds: tokenTypeIds, attentionMask: attentionMask) 
                guard let embeddings = output.textEmbeds as? MLXArray else {
                     logger.error("Could not extract embeddings from model output for batch (ID: \(embeddingReqId)).")
                     struct EmbeddingError: Error, AbortError {
                         var status: HTTPResponseStatus { .internalServerError }
                         var reason: String { "Failed to get embeddings from model output." }
                     }
                     throw EmbeddingError()
                }


                switch encodingFormat {
                case "base64":
                    for i in 0..<embeddings.shape[0] {
                        let arr = embeddings[i].asArray(Float.self)
                        let data = arr.withUnsafeBufferPointer { Data(buffer: $0) }
                        let base64 = data.base64EncodedString()
                        allData.append(EmbeddingData(embedding: .base64(base64), index: index))
                        index += 1
                    }
                default: 
                    for i in 0..<embeddings.shape[0] {
                        let arr = embeddings[i].asArray(Float.self)
                        allData.append(EmbeddingData(embedding: .floats(arr), index: index))
                        index += 1
                    }
                }
                logger.trace("Finished processing batch \(batchStart/batchSize + 1) for \(embeddingReqId). Total embeddings so far: \(index)")
            }
            let usage = UsageData(prompt_tokens: promptTokens, total_tokens: promptTokens)
            logger.info("Embedding generation complete (ID: \(embeddingReqId)) for model \(loadedModelName). Total tokens: \(promptTokens)")

            return EmbeddingResponse(data: allData, model: loadedModelName, usage: usage)

        } 
    } 
} 
