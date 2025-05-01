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

actor EmbeddingsManager {
  private let modelContainer: mlx_embeddings.ModelContainer?
  private let modelId: String

  init(modelContainer: mlx_embeddings.ModelContainer?, modelId: String = "default_model") {
    self.modelId = modelId
    self.modelContainer = modelContainer
  }

  func embed(request: EmbeddingRequest) async throws -> EmbeddingResponse {
    let texts = request.input.values
    guard !texts.isEmpty, texts.allSatisfy({ !$0.isEmpty }) else {
      throw Abort(.badRequest)
    }
    let encodingFormat = request.encoding_format ?? "float"
    guard request.model == "default_model" || request.model == modelId else {
      throw Abort(
        .badRequest,
        reason:
          "Model '\(request.model!)' is not available or does not match expected model '\(modelId)'"
      )
    }
    let batchSize = request.batch_size ?? texts.count
    return await modelContainer!.perform { model, tokenizer in
      var allData: [EmbeddingData] = []
      var promptTokens = 0
      var index = 0
      for batchStart in stride(from: 0, to: texts.count, by: batchSize) {
        let batchTexts = Array(texts[batchStart..<min(batchStart + batchSize, texts.count)])
        let tokenized = batchTexts.map { tokenizer.encode(text: $0, addSpecialTokens: true) }
        let maxLength = tokenized.reduce(16) { max($0, $1.count) }
        let padId = tokenizer.eosTokenId ?? 0
        let padded = stacked(
          tokenized.map { elem in
            MLXArray(elem + Array(repeating: padId, count: maxLength - elem.count))
          })
        let attentionMask = padded .!= MLXArray(padId)
        let tokenTypeIds = MLXArray.zeros(like: padded)
        let output = model(
          padded, positionIds: nil, tokenTypeIds: tokenTypeIds, attentionMask: attentionMask)
        let embeddings = output.textEmbeds!
        promptTokens += tokenized.reduce(0) { $0 + $1.count }
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
      }
      let usage = UsageData(prompt_tokens: promptTokens, total_tokens: promptTokens)
      return EmbeddingResponse(data: allData, model: modelId, usage: usage)
    }
  }
}

func registerEmbeddingsRoute(_ app: Application, modelContainer: mlx_embeddings.ModelContainer?) {

  let manager = EmbeddingsManager(modelContainer: modelContainer)
  app.post("v1", "embeddings") { req async throws -> EmbeddingResponse in
    guard let modelContainer else {
      throw Abort(
        .serviceUnavailable,
        reason: "Embedding model not loaded. Start server with --embedding-model option.")
    }
    let embeddingRequest = try req.content.decode(EmbeddingRequest.self)
    return try await manager.embed(request: embeddingRequest)
  }
}
