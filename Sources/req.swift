import Vapor
import Foundation

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
        case model, prompt, temperature, n, stream, logprobs, stop
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case repetitionPenalty = "repetition_penalty"
        case repetitionContextSize = "repetition_context_size"
    }
}

struct ChatMessageRequestData: Content {
    let role: String
    let content: String?
}

struct ChatCompletionRequest: Content {
    let messages: [ChatMessageRequestData]
    let model: String?
    let maxTokens: Int?
    let temperature: Float?
    let topP: Float?
    let stream: Bool?
    let stop: [String]?
    let repetitionPenalty: Float?
    let repetitionContextSize: Int?

    enum CodingKeys: String, CodingKey {
        case messages, model, temperature, stream, stop
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case repetitionPenalty = "repetition_penalty"
        case repetitionContextSize = "repetition_context_size"
    }
}