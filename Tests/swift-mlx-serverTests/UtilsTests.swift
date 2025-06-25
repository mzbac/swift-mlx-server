import XCTest
import Vapor
@testable import swift_mlx_server

final class UtilsTests: XCTestCase {
    
    // MARK: - StopCondition Tests
    
    func testCheckStoppingCriteriaWithEOSToken() {
        let tokens = [1, 2, 3, 4]
        let eosTokenId = 4
        let stopIdSequences: [[Int]] = []
        
        let result = checkStoppingCriteria(tokens: tokens, stopIdSequences: stopIdSequences, eosTokenId: eosTokenId)
        
        XCTAssertTrue(result.stopMet)
        XCTAssertEqual(result.trimLength, 1)
    }
    
    func testCheckStoppingCriteriaWithStopSequence() {
        let tokens = [1, 2, 3, 4, 5]
        let eosTokenId = 999
        let stopIdSequences: [[Int]] = [[3, 4, 5], [7, 8]]
        
        let result = checkStoppingCriteria(tokens: tokens, stopIdSequences: stopIdSequences, eosTokenId: eosTokenId)
        
        XCTAssertTrue(result.stopMet)
        XCTAssertEqual(result.trimLength, 3)
    }
    
    func testCheckStoppingCriteriaNoStop() {
        let tokens = [1, 2, 3, 4]
        let eosTokenId = 999
        let stopIdSequences: [[Int]] = [[7, 8], [9, 10]]
        
        let result = checkStoppingCriteria(tokens: tokens, stopIdSequences: stopIdSequences, eosTokenId: eosTokenId)
        
        XCTAssertFalse(result.stopMet)
        XCTAssertEqual(result.trimLength, 0)
    }
    
    func testCheckStoppingCriteriaEmptyTokens() {
        let tokens: [Int] = []
        let eosTokenId = 1
        let stopIdSequences: [[Int]] = []
        
        let result = checkStoppingCriteria(tokens: tokens, stopIdSequences: stopIdSequences, eosTokenId: eosTokenId)
        
        XCTAssertFalse(result.stopMet)
        XCTAssertEqual(result.trimLength, 0)
    }
    
    // MARK: - Array Extension Tests
    
    func testNilIfEmptyWithEmptyArray() {
        let emptyArray: [Int] = []
        XCTAssertNil(emptyArray.nilIfEmpty())
    }
    
    func testNilIfEmptyWithNonEmptyArray() {
        let nonEmptyArray = [1, 2, 3]
        XCTAssertNotNil(nonEmptyArray.nilIfEmpty())
        XCTAssertEqual(nonEmptyArray.nilIfEmpty(), [1, 2, 3])
    }
    
    // MARK: - SSE Encoding Tests
    
    func testEncodeSSEWithValidData() throws {
        struct TestResponse: Codable {
            let message: String
            let number: Int
        }
        
        let testResponse = TestResponse(message: "test", number: 42)
        let logger = Logger(label: "test")
        
        let result = encodeSSE(response: testResponse, logger: logger)
        
        XCTAssertNotNil(result)
        XCTAssertTrue(result!.hasPrefix("data: "))
        XCTAssertTrue(result!.hasSuffix("\n\n"))
        XCTAssertTrue(result!.contains("\"message\":\"test\""))
        XCTAssertTrue(result!.contains("\"number\":42"))
    }
    
    // MARK: - Error Types Tests
    
    func testModelProviderError() {
        let error = ModelProviderError(
            status: .internalServerError,
            reason: "Test error",
            modelId: "test-model",
            underlyingError: nil
        )
        
        XCTAssertEqual(error.status, .internalServerError)
        XCTAssertTrue(error.reason.contains("Test error"))
        XCTAssertTrue(error.reason.contains("test-model"))
        XCTAssertEqual(error.modelId, "test-model")
    }
    
    func testProcessingError() {
        let underlyingError = NSError(domain: "test", code: 1, userInfo: [NSLocalizedDescriptionKey: "Underlying error"])
        let error = ProcessingError(
            status: .badRequest,
            reason: "Processing failed",
            modelId: "model-123",
            underlyingError: underlyingError
        )
        
        XCTAssertEqual(error.status, .badRequest)
        XCTAssertTrue(error.reason.contains("Processing failed"))
        XCTAssertTrue(error.reason.contains("model-123"))
        XCTAssertTrue(error.reason.contains("Underlying error"))
        XCTAssertEqual(error.modelId, "model-123")
    }
    
    // MARK: - Constants Tests
    
    func testAppConstants() {
        XCTAssertEqual(AppConstants.defaultHost, "127.0.0.1")
        XCTAssertEqual(AppConstants.defaultPort, 8080)
        XCTAssertEqual(AppConstants.sseDoneMessage, "data: [DONE]\n\n")
        XCTAssertEqual(AppConstants.sseEventHeader, "data: ")
        XCTAssertEqual(AppConstants.sseEventSeparator, "\n\n")
    }
    
    func testGenerationDefaults() {
        XCTAssertEqual(GenerationDefaults.maxTokens, 128)
        XCTAssertEqual(GenerationDefaults.temperature, 0.8)
        XCTAssertEqual(GenerationDefaults.topP, 1.0)
        XCTAssertEqual(GenerationDefaults.stream, false)
        XCTAssertEqual(GenerationDefaults.repetitionPenalty, 1.0)
        XCTAssertEqual(GenerationDefaults.repetitionContextSize, 20)
        XCTAssertTrue(GenerationDefaults.stopSequences.isEmpty)
    }
}