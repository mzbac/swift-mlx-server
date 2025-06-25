import XCTest
@testable import swift_mlx_server

final class SimpleTests: XCTestCase {
    
    // MARK: - Basic Functionality Tests
    
    func testUtilityFunctionsWork() {
        // Test that our refactored utility functions work
        let tokens = [1, 2, 3, 4]
        let stopSequences: [[Int]] = [[3, 4]]
        let eosTokenId = 999
        
        let result = checkStoppingCriteria(tokens: tokens, stopIdSequences: stopSequences, eosTokenId: eosTokenId)
        
        XCTAssertTrue(result.stopMet)
        XCTAssertEqual(result.trimLength, 2)
    }
    
    func testArrayExtension() {
        let emptyArray: [String] = []
        let nonEmptyArray = ["test"]
        
        XCTAssertNil(emptyArray.nilIfEmpty())
        XCTAssertNotNil(nonEmptyArray.nilIfEmpty())
    }
    
    func testConstants() {
        XCTAssertEqual(AppConstants.defaultHost, "127.0.0.1")
        XCTAssertEqual(AppConstants.defaultPort, 8080)
        XCTAssertEqual(GenerationDefaults.maxTokens, 128)
        XCTAssertEqual(GenerationDefaults.temperature, 0.8)
    }
    
    func testErrorTypes() {
        let error = ModelProviderError(
            status: .badRequest,
            reason: "Test error",
            modelId: "test-model"
        )
        
        XCTAssertEqual(error.status, .badRequest)
        XCTAssertTrue(error.reason.contains("Test error"))
        XCTAssertTrue(error.reason.contains("test-model"))
    }
    
    func testCodeCompiles() {
        // This test ensures our refactored code compiles successfully
        // by importing and using the main module
        let _ = AppConstants.defaultHost
        let _ = GenerationDefaults.maxTokens
        
        // Test that stop condition works
        let result = checkStoppingCriteria(tokens: [], stopIdSequences: [], eosTokenId: 1)
        XCTAssertFalse(result.stopMet)
        
        // Test error creation
        let error = ProcessingError(status: .internalServerError, reason: "Test")
        XCTAssertEqual(error.status, .internalServerError)
        
        XCTAssertTrue(true, "All refactored code compiles and basic functionality works")
    }
}