import Vapor
import Foundation

struct CompletionUsage: Content {
    let promptTokens: Int
    let completionTokens: Int
    let totalTokens: Int

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }
}

struct CompletionChoice: Content {
    let text: String
    let index: Int
    let logprobs: [String: Double]?
    let finishReason: String?

    init(text: String, index: Int = 0, logprobs: [String: Double]? = nil, finishReason: String?) {
        self.text = text
        self.index = index
        self.logprobs = logprobs
        self.finishReason = finishReason
    }

     enum CodingKeys: String, CodingKey {
        case text, index, logprobs
        case finishReason = "finish_reason"
    }
}

struct CompletionResponse: AsyncResponseEncodable, Content {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [CompletionChoice]
    let usage: CompletionUsage

    init(id: String = "cmpl-\(UUID().uuidString)", object: String = "text_completion", model: String, choices: [CompletionChoice], usage: CompletionUsage) {
        self.id = id
        self.object = object
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.choices = choices
        self.usage = usage
    }

     func encodeResponse(for request: Request) async throws -> Response {
        let response = Response(status: .ok) 
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        response.body = try .init(data: encoder.encode(self)) 
        response.headers.contentType = .json 
        return response
    }

     enum CodingKeys: String, CodingKey {
        case id, object, created, model, choices, usage
    }
}

struct CompletionChunkResponse: Content {
    let id: String
    let object: String = "text_completion"
    let created: Int
    let choices: [Choice]
    let model: String
    let systemFingerprint: String

    struct Choice: Content {
        let text: String
        let index: Int = 0
        var logprobs: String?
        var finishReason: String?

         enum CodingKeys: String, CodingKey {
            case text, index, logprobs
            case finishReason = "finish_reason"
        }
    }

    init(completionId: String, requestedModel: String, nextChunk: String, systemFingerprint: String = "fp_\(UUID().uuidString)") {
        self.id = completionId
        self.created = Int(Date().timeIntervalSince1970)
        self.choices = [Choice(text: nextChunk)]
        self.model = requestedModel
        self.systemFingerprint = systemFingerprint
    }

    enum CodingKeys: String, CodingKey {
        case id, object, created, model, choices
        case systemFingerprint = "system_fingerprint"
    }
}

struct ChatMessageResponseData: Content {
    let role: String
    let content: String?
    let refusal: String? = nil

    enum CodingKeys: String, CodingKey {
        case role, content, refusal
    }

    init(role: String, content: String?) {
        self.role = role
        self.content = content
    }
}

struct ChatCompletionDelta: Content {
    var role: String?
    var content: String?
}

struct ChatCompletionChoiceDelta: Content {
    let index: Int
    let delta: ChatCompletionDelta
    let logprobs: String? = nil
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case index, delta, logprobs
        case finishReason = "finish_reason"
    }
}

struct ChatCompletionChunkResponse: Content {
    let id: String
    let object: String = "chat.completion.chunk"
    let created: Int
    let model: String
    let systemFingerprint: String?
    let choices: [ChatCompletionChoiceDelta]

    init(id: String, created: Int = Int(Date().timeIntervalSince1970), model: String, systemFingerprint: String? = nil, choices: [ChatCompletionChoiceDelta]) {
        self.id = id
        self.created = created
        self.model = model
        self.systemFingerprint = systemFingerprint
        self.choices = choices
    }

    enum CodingKeys: String, CodingKey {
        case id, object, created, model, choices
        case systemFingerprint = "system_fingerprint"
    }
}

struct ChatCompletionChoice: Content {
    let index: Int
    let message: ChatMessageResponseData
    let logprobs: String? = nil
    let finishReason: String

    enum CodingKeys: String, CodingKey {
        case index, message, logprobs
        case finishReason = "finish_reason"
    }
}

struct ChatCompletionResponse: AsyncResponseEncodable, Content {
    let id: String
    let object: String = "chat.completion"
    let created: Int
    let model: String
    let choices: [ChatCompletionChoice]
    let usage: CompletionUsage
    let systemFingerprint: String? = nil
    var serviceTier: String? = "default"

    init(id: String = "chatcmpl-\(UUID().uuidString)", created: Int = Int(Date().timeIntervalSince1970), model: String, choices: [ChatCompletionChoice], usage: CompletionUsage, serviceTier: String? = "default") {
        self.id = id
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
        self.serviceTier = serviceTier
    }

    func encodeResponse(for request: Request) async throws -> Response {
        let response = Response(status: .ok) 
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        response.body = try .init(data: encoder.encode(self)) 
        response.headers.contentType = .json 
        return response
    }

    enum CodingKeys: String, CodingKey {
        case id, object, created, model, choices, usage
        case systemFingerprint = "system_fingerprint" 
        case serviceTier = "service_tier"
    }
}
