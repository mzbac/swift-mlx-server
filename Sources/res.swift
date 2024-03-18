import Vapor

struct CompletionResponse: Content {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [CompletionChoice]
    let usage: CompletionUsage
}

struct CompletionChoice: Content {
    let text: String
    let index: Int
    let logprobs: [String: Double]?
    let finishReason: String?
}

struct CompletionUsage: Content {
    let promptTokens: Int
    let completionTokens: Int
    let totalTokens: Int
}

struct CompletionChunkResponse: Content {
    let id: String
    var object: String = "text_completion"
    let created: Int
    let choices: [Choice]
    let model: String
    let systemFingerprint: String

    struct Choice: Content {
        let text: String
        var index: Int = 0
        var logprobs: String? = nil
        var finishReason: String? = nil
    }

    init(completionId: String, requestedModel: String, nextChunk: String) {
        self.id = completionId
        self.created = Int(Date().timeIntervalSince1970)
        self.choices = [Choice(text: nextChunk)]
        self.model = requestedModel
        self.systemFingerprint = "fp_\(UUID().uuidString)"
    }
}
