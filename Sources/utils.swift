import Vapor
import Foundation
import Logging
import Tokenizers

enum AppConstants {
  static let defaultHost = "127.0.0.1"
  static let defaultPort = 8080
  static let sseDoneMessage = "data: [DONE]\n\n"
  static let sseEventHeader = "data: "
  static let sseEventSeparator = "\n\n"
}

enum GenerationDefaults {
  static let maxTokens = 128
  static let temperature: Float = 0.8
  static let topP: Float = 1.0
  static let stream = false
  static let repetitionPenalty: Float = 1.0
  static let repetitionContextSize = 20
  static let stopSequences: [String] = []
}

struct StopCondition {
  let stopMet: Bool
  let trimLength: Int
}

func checkStoppingCriteria(tokens: [Int], stopIdSequences: [[Int]], eosTokenId: Int)
  -> StopCondition
{
  guard let lastToken = tokens.last else { return StopCondition(stopMet: false, trimLength: 0) }
  if lastToken == eosTokenId { return StopCondition(stopMet: true, trimLength: 1) }
  for stopIds in stopIdSequences {
    guard !stopIds.isEmpty else { continue }
    if tokens.count >= stopIds.count, tokens.suffix(stopIds.count) == stopIds {
      return StopCondition(stopMet: true, trimLength: stopIds.count)
    }
  }
  return StopCondition(stopMet: false, trimLength: 0)
}

func encodeSSE<T: Encodable>(response: T, logger: Logger) -> String? {
  do {
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let jsonData = try encoder.encode(response)
    guard let jsonString = String(data: jsonData, encoding: .utf8) else {
      logger.error("Failed encodeSSE utf8")
      return nil
    }
    return AppConstants.sseEventHeader + jsonString + AppConstants.sseEventSeparator
  } catch {
    logger.error("Failed encodeSSE: \(error)")
    return nil
  }
}

extension Array { func nilIfEmpty() -> Self? { self.isEmpty ? nil : self } }
