import Vapor
import Foundation
import CoreImage

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

struct ContentFragment: Content {
    let type: String
    let text: String?
    let imageUrl: URL?
    let videoUrl: URL?
    
    enum CodingKeys: String, CodingKey {
        case type
        case text
        case imageUrl = "image_url"
        case videoUrl = "video_url"
    }
}

struct ChatMessageRequestData: Content {
    let role: String
    let content: ContentFragmentType
    
    enum ContentFragmentType: Codable {
        case text(String)
        case fragments([ContentFragment])
        case none
        
        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            
            if let string = try? container.decode(String.self) {
                self = .text(string)
            } else if let fragments = try? container.decode([ContentFragment].self) {
                self = .fragments(fragments)
            } else if container.decodeNil() {
                self = .none
            } else {
                throw DecodingError.dataCorruptedError(
                    in: container,
                    debugDescription: "Expected String, [ContentFragment], or nil"
                )
            }
        }
        
        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            
            switch self {
            case .text(let string):
                try container.encode(string)
            case .fragments(let fragments):
                try container.encode(fragments)
            case .none:
                try container.encodeNil()
            }
        }
        
        var asString: String? {
            switch self {
            case .text(let string):
                return string
            case .fragments(let fragments):
                let textFragments = fragments.compactMap { $0.type == "text" ? $0.text : nil }
                return textFragments.isEmpty ? nil : textFragments.joined()
            case .none:
                return nil
            }
        }
    }
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
    let resize: [CGFloat]?

    enum CodingKeys: String, CodingKey {
        case messages, model, temperature, stream, stop, resize
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case repetitionPenalty = "repetition_penalty"
        case repetitionContextSize = "repetition_context_size"
    }
}
