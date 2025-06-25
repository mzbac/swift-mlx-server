import XCTest
@testable import swift_mlx_server

final class TextCompletionParametersTests: XCTestCase {
    
    func testTextCompletionParametersWithDefaults() {
        let request = CompletionRequest(
            model: "test-model",
            prompt: "Hello world",
            maxTokens: nil,
            temperature: nil,
            topP: nil,
            n: nil,
            stream: nil,
            logprobs: nil,
            stop: nil,
            repetitionPenalty: nil,
            repetitionContextSize: nil,
            kvBits: nil,
            kvGroupSize: nil,
            quantizedKVStart: nil
        )
        
        // This would be used in the actual implementation
        let maxTokens = request.maxTokens ?? GenerationDefaults.maxTokens
        let temperature = request.temperature ?? GenerationDefaults.temperature
        let topP = request.topP ?? GenerationDefaults.topP
        let stream = request.stream ?? GenerationDefaults.stream
        let stopWords = request.stop ?? GenerationDefaults.stopSequences
        let repetitionPenalty = request.repetitionPenalty ?? GenerationDefaults.repetitionPenalty
        let repetitionContextSize = request.repetitionContextSize ?? GenerationDefaults.repetitionContextSize
        
        XCTAssertEqual(maxTokens, GenerationDefaults.maxTokens)
        XCTAssertEqual(temperature, GenerationDefaults.temperature)
        XCTAssertEqual(topP, GenerationDefaults.topP)
        XCTAssertEqual(stream, GenerationDefaults.stream)
        XCTAssertEqual(stopWords, GenerationDefaults.stopSequences)
        XCTAssertEqual(repetitionPenalty, GenerationDefaults.repetitionPenalty)
        XCTAssertEqual(repetitionContextSize, GenerationDefaults.repetitionContextSize)
    }
    
    func testTextCompletionParametersWithCustomValues() {
        let request = CompletionRequest(
            model: "custom-model",
            prompt: "Custom prompt",
            maxTokens: 200,
            temperature: 0.9,
            topP: 0.8,
            n: 1,
            stream: true,
            logprobs: nil,
            stop: ["STOP", "END"],
            repetitionPenalty: 1.2,
            repetitionContextSize: 30,
            kvBits: nil,
            kvGroupSize: nil,
            quantizedKVStart: nil
        )
        
        let maxTokens = request.maxTokens ?? GenerationDefaults.maxTokens
        let temperature = request.temperature ?? GenerationDefaults.temperature
        let topP = request.topP ?? GenerationDefaults.topP
        let stream = request.stream ?? GenerationDefaults.stream
        let stopWords = request.stop ?? GenerationDefaults.stopSequences
        let repetitionPenalty = request.repetitionPenalty ?? GenerationDefaults.repetitionPenalty
        let repetitionContextSize = request.repetitionContextSize ?? GenerationDefaults.repetitionContextSize
        
        XCTAssertEqual(maxTokens, 200)
        XCTAssertEqual(temperature, 0.9)
        XCTAssertEqual(topP, 0.8)
        XCTAssertEqual(stream, true)
        XCTAssertEqual(stopWords, ["STOP", "END"])
        XCTAssertEqual(repetitionPenalty, 1.2)
        XCTAssertEqual(repetitionContextSize, 30)
    }
    
    func testChatCompletionParametersWithDefaults() {
        let messages = [
            ChatMessageRequestData(role: "user", content: .text("Hello"))
        ]
        let request = ChatCompletionRequest(
            messages: messages,
            model: nil,
            maxTokens: nil,
            temperature: nil,
            topP: nil,
            stream: nil,
            stop: nil,
            repetitionPenalty: nil,
            repetitionContextSize: nil,
            resize: nil,
            kvBits: nil,
            kvGroupSize: nil,
            quantizedKVStart: nil
        )
        
        let maxTokens = request.maxTokens ?? GenerationDefaults.maxTokens
        let temperature = request.temperature ?? GenerationDefaults.temperature
        let topP = request.topP ?? GenerationDefaults.topP
        let stream = request.stream ?? GenerationDefaults.stream
        let stopWords = request.stop ?? GenerationDefaults.stopSequences
        let repetitionPenalty = request.repetitionPenalty ?? GenerationDefaults.repetitionPenalty
        let repetitionContextSize = request.repetitionContextSize ?? GenerationDefaults.repetitionContextSize
        
        XCTAssertEqual(maxTokens, GenerationDefaults.maxTokens)
        XCTAssertEqual(temperature, GenerationDefaults.temperature)
        XCTAssertEqual(topP, GenerationDefaults.topP)
        XCTAssertEqual(stream, GenerationDefaults.stream)
        XCTAssertEqual(stopWords, GenerationDefaults.stopSequences)
        XCTAssertEqual(repetitionPenalty, GenerationDefaults.repetitionPenalty)
        XCTAssertEqual(repetitionContextSize, GenerationDefaults.repetitionContextSize)
    }
    
    func testStopSequenceTokenConversion() {
        struct MockTokenizer {
            func encode(text: String, addSpecialTokens: Bool = false) -> [Int] {
                return text.map { Int($0.asciiValue!) }
            }
        }
        
        let tokenizer = MockTokenizer()
        let stopWords = ["STOP", "END"]
        
        let stopIdSequences = stopWords.compactMap { word in
            tokenizer.encode(text: word, addSpecialTokens: false).nilIfEmpty()
        }
        
        XCTAssertEqual(stopIdSequences.count, 2)
        XCTAssertEqual(stopIdSequences[0], [83, 84, 79, 80]) // "STOP"
        XCTAssertEqual(stopIdSequences[1], [69, 78, 68]) // "END"
    }
    
    func testTextCompletionWithKVCacheParameters() {
        let request = CompletionRequest(
            model: "test-model",
            prompt: "Test KV cache",
            maxTokens: 100,
            temperature: 0.7,
            topP: 0.9,
            n: 1,
            stream: false,
            logprobs: nil,
            stop: nil,
            repetitionPenalty: 1.0,
            repetitionContextSize: 20,
            kvBits: 8,
            kvGroupSize: 64,
            quantizedKVStart: 100
        )
        
        XCTAssertEqual(request.kvBits, 8)
        XCTAssertEqual(request.kvGroupSize, 64)
        XCTAssertEqual(request.quantizedKVStart, 100)
    }
    
    func testEmptyStopSequenceFiltering() {
        struct MockTokenizer {
            func encode(text: String, addSpecialTokens: Bool = false) -> [Int] {
                return text.isEmpty ? [] : text.map { Int($0.asciiValue!) }
            }
        }
        
        let tokenizer = MockTokenizer()
        let stopWords = ["STOP", "", "END", ""]
        
        let stopIdSequences = stopWords.compactMap { word in
            tokenizer.encode(text: word, addSpecialTokens: false).nilIfEmpty()
        }
        
        XCTAssertEqual(stopIdSequences.count, 2) // Empty strings should be filtered out
        XCTAssertEqual(stopIdSequences[0], [83, 84, 79, 80]) // "STOP"
        XCTAssertEqual(stopIdSequences[1], [69, 78, 68]) // "END"
    }
}