import Vapor

struct CompletionRequest: Content {
  let model: String?
  let prompt: String
  let maxTokens: Int?
  let temperature: Float?
  let topP: Float?
  let n: Int?
  let stream: Bool?
  let logprobs: Int?
  let stop: [String]?
  let repetitionPenalty: Float?
  let repetitionContextSize: Int?

  enum CodingKeys: String, CodingKey {
    case model
    case prompt
    case maxTokens = "max_tokens"
    case temperature
    case topP = "top_p"
    case n
    case stream
    case logprobs
    case stop
    case repetitionPenalty = "repetition_penalty"
    case repetitionContextSize = "repetition_context_size"
  }
}

enum GenerationDefaults {
  static let model = "default_model"
  static let maxTokens = 128
  static let temperature: Float = 0.8
  static let topP: Float = 1.0
  static let stream = false
  static let repetitionPenalty: Float = 1.0
  static let repetitionContextSize = 20
  static let stopSequences: [String] = []
}
