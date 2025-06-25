import Vapor
import Foundation
import Logging
import Tokenizers

enum AppConstants {
    static let defaultHost = "127.0.0.1"
    static let defaultPort = 8_080
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
    
    static let kvGroupSize: Int = 64
    static let quantizedKVStart: Int = 5_000
}

protocol MLXServerError: AbortError {
    var modelId: String? { get }
    var underlyingError: Error? { get }
    var baseReason: String { get }
    
    init(status: HTTPResponseStatus, reason: String, modelId: String?, underlyingError: Error?)
}

extension MLXServerError {
    init(status: HTTPResponseStatus, reason: String, modelId: String? = nil, underlyingError: Error? = nil) {
        self.init(status: status, reason: reason, modelId: modelId, underlyingError: underlyingError)
    }
    
    var reason: String {
        var fullReason = self.baseReason
        if let modelId = modelId, !modelId.isEmpty {
            fullReason += " (Model: \(modelId))"
        }
        if let underlyingError = underlyingError {
            fullReason += ". Underlying error: \(underlyingError.localizedDescription)"
        }
        return fullReason
    }
}

struct ModelProviderError: MLXServerError {
    let status: HTTPResponseStatus
    let baseReason: String
    let identifier: String?
    let modelId: String?
    let underlyingError: Error?
    
    init(status: HTTPResponseStatus, reason: String, modelId: String? = nil, underlyingError: Error? = nil) {
        self.status = status
        self.baseReason = reason
        self.identifier = modelId
        self.modelId = modelId
        self.underlyingError = underlyingError
    }
}

struct ProcessingError: MLXServerError {
    let status: HTTPResponseStatus
    let baseReason: String
    let identifier: String?
    let modelId: String?
    let underlyingError: Error?
    
    init(status: HTTPResponseStatus, reason: String, modelId: String? = nil, underlyingError: Error? = nil) {
        self.status = status
        self.baseReason = reason
        self.identifier = modelId
        self.modelId = modelId
        self.underlyingError = underlyingError
    }
}

struct StopCondition {
    let stopMet: Bool
    let trimLength: Int
}

func checkStoppingCriteria(tokens: [Int], stopIdSequences: [[Int]], eosTokenId: Int) -> StopCondition {
    guard let lastToken = tokens.last else { 
        return StopCondition(stopMet: false, trimLength: 0) 
    }
    
    if lastToken == eosTokenId { 
        return StopCondition(stopMet: true, trimLength: 1) 
    }
    
    for stopIds in stopIdSequences {
        guard !stopIds.isEmpty else { continue }
        if tokens.count >= stopIds.count, tokens.suffix(stopIds.count) == stopIds {
            return StopCondition(stopMet: true, trimLength: stopIds.count)
        }
    }
    
    return StopCondition(stopMet: false, trimLength: 0)
}

func stopSequencesToIds(stopWords: [String], tokenizer: Tokenizer) -> [[Int]] {
    return stopWords.compactMap { word in
        tokenizer.encode(text: word, addSpecialTokens: false).nilIfEmpty()
    }
}

func decodeTokens(_ tokens: [Int], tokenizer: Tokenizer) -> String {
    return tokenizer.decode(tokens: tokens)
}

func encodeSSE<T: Encodable>(response: T, logger: Logger) -> String? {
    do {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let jsonData = try encoder.encode(response)
        
        guard let jsonString = String(data: jsonData, encoding: .utf8) else {
            logger.error("Failed to encode SSE response to UTF-8")
            return nil
        }
        
        return AppConstants.sseEventHeader + jsonString + AppConstants.sseEventSeparator
    } catch {
        logger.error("Failed to encode SSE response: \(error)")
        return nil
    }
}

enum KVCacheValidation {
    static func validate(bits: Int?, groupSize: Int, quantizationStart: Int) throws {
        if let bits = bits {
            guard [4, 8].contains(bits) else {
                throw ProcessingError(
                    status: .badRequest,
                    reason: "kv_bits must be 4 or 8, got \(bits)"
                )
            }
        }
        
        guard groupSize > 0 && groupSize.isMultiple(of: 8) else {
            throw ProcessingError(
                status: .badRequest,
                reason: "kv_group_size must be positive and divisible by 8, got \(groupSize)"
            )
        }
        
        guard quantizationStart >= 0 else {
            throw ProcessingError(
                status: .badRequest,
                reason: "kv_quantization_start must be non-negative, got \(quantizationStart)"
            )
        }
    }
}

extension Array { 
    func nilIfEmpty() -> Self? { 
        return isEmpty ? nil : self 
    } 
}
