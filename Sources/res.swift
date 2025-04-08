import Foundation
import Vapor

struct CompletionResponse: Content {
  let id: String
  let object: String = StructureConstants.completionObject
  let created: Int
  let model: String
  let choices: [CompletionChoice]
  let usage: CompletionUsage

  init(
    id: String = "cmpl-\(UUID().uuidString)", model: String, choices: [CompletionChoice],
    usage: CompletionUsage
  ) {
    self.id = id
    self.created = Int(Date().timeIntervalSince1970)
    self.model = model
    self.choices = choices
    self.usage = usage
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
}

struct CompletionUsage: Content {
  let promptTokens: Int
  let completionTokens: Int
  let totalTokens: Int
}

struct CompletionChunkResponse: Content {
  let id: String
  let object: String = StructureConstants.completionObject
  let created: Int
  let choices: [Choice]
  let model: String
  let systemFingerprint: String

  struct Choice: Content {
    let text: String
    let index: Int = 0
    var logprobs: String?
    var finishReason: String?
  }

  /// Initializes a new completion chunk.
  /// - Parameters:
  ///   - completionId: The unique ID for the completion stream.
  ///   - requestedModel: The model used for generation.
  ///   - nextChunk: The text content of this chunk.
  ///   - systemFingerprint: A unique fingerprint for the system generating the response.
  init(
    completionId: String, requestedModel: String, nextChunk: String,
    systemFingerprint: String = "fp_\(UUID().uuidString)"
  ) {
    self.id = completionId
    self.created = Int(Date().timeIntervalSince1970)
    self.choices = [Choice(text: nextChunk)]
    self.model = requestedModel
    self.systemFingerprint = systemFingerprint
  }
}

enum StructureConstants {
  static let completionObject = "text_completion"
}
