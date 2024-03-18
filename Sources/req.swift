import Vapor

struct CompletionRequest: Content {
    let model: String?
    let prompt: String
    let maxTokens: Int
    let temperature: Float
    let topP: Float?
    let n: Int?
    let stream: Bool?
    let logprobs: Int?
    let stop: [String]?
    
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
        }
}
