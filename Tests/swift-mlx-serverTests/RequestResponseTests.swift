import XCTest
import Vapor
@testable import swift_mlx_server

final class RequestResponseTests: XCTestCase {
    
    // MARK: - CompletionRequest Tests
    
    func testCompletionRequestDecoding() throws {
        let jsonData = """
        {
            "model": "test-model",
            "prompt": "Hello world",
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": true,
            "stop": ["END", "###"],
            "repetition_penalty": 1.1,
            "repetition_context_size": 20,
            "kv_bits": 8,
            "kv_group_size": 64,
            "quantized_kv_start": 100
        }
        """.data(using: .utf8)!
        
        let request = try JSONDecoder().decode(CompletionRequest.self, from: jsonData)
        
        XCTAssertEqual(request.model, "test-model")
        XCTAssertEqual(request.prompt, "Hello world")
        XCTAssertEqual(request.maxTokens, 100)
        XCTAssertEqual(request.temperature, 0.7)
        XCTAssertEqual(request.topP, 0.9)
        XCTAssertEqual(request.stream, true)
        XCTAssertEqual(request.stop, ["END", "###"])
        XCTAssertEqual(request.repetitionPenalty, 1.1)
        XCTAssertEqual(request.repetitionContextSize, 20)
        XCTAssertEqual(request.kvBits, 8)
        XCTAssertEqual(request.kvGroupSize, 64)
        XCTAssertEqual(request.quantizedKVStart, 100)
    }
    
    func testCompletionRequestDecodingWithDefaults() throws {
        let jsonData = """
        {
            "prompt": "Hello world"
        }
        """.data(using: .utf8)!
        
        let request = try JSONDecoder().decode(CompletionRequest.self, from: jsonData)
        
        XCTAssertNil(request.model)
        XCTAssertEqual(request.prompt, "Hello world")
        XCTAssertNil(request.maxTokens)
        XCTAssertNil(request.temperature)
        XCTAssertNil(request.topP)
        XCTAssertNil(request.stream)
        XCTAssertNil(request.stop)
        XCTAssertNil(request.repetitionPenalty)
        XCTAssertNil(request.repetitionContextSize)
        XCTAssertNil(request.kvBits)
        XCTAssertNil(request.kvGroupSize)
        XCTAssertNil(request.quantizedKVStart)
    }
    
    // MARK: - ChatCompletionRequest Tests
    
    func testChatCompletionRequestWithTextContent() throws {
        let jsonData = """
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            "model": "test-model",
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": false
        }
        """.data(using: .utf8)!
        
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: jsonData)
        
        XCTAssertEqual(request.messages.count, 2)
        XCTAssertEqual(request.messages[0].role, "system")
        XCTAssertEqual(request.messages[0].content.asString, "You are a helpful assistant.")
        XCTAssertEqual(request.messages[1].role, "user")
        XCTAssertEqual(request.messages[1].content.asString, "Hello!")
        XCTAssertEqual(request.model, "test-model")
        XCTAssertEqual(request.maxTokens, 100)
        XCTAssertEqual(request.temperature, 0.7)
        XCTAssertEqual(request.stream, false)
        XCTAssertNil(request.kvBits)
        XCTAssertNil(request.kvGroupSize)
        XCTAssertNil(request.quantizedKVStart)
    }
    
    func testChatCompletionRequestWithKVCacheParams() throws {
        let jsonData = """
        {
            "messages": [
                {"role": "user", "content": "Test KV cache"}
            ],
            "max_tokens": 100,
            "kv_bits": 4,
            "kv_group_size": 64,
            "quantized_kv_start": 200
        }
        """.data(using: .utf8)!
        
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: jsonData)
        
        XCTAssertEqual(request.messages.count, 1)
        XCTAssertEqual(request.kvBits, 4)
        XCTAssertEqual(request.kvGroupSize, 64)
        XCTAssertEqual(request.quantizedKVStart, 200)
    }
    
    func testChatCompletionRequestWithFragmentContent() throws {
        let jsonData = """
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see?"},
                        {"type": "image", "image_url": "https://example.com/image.jpg"}
                    ]
                }
            ]
        }
        """.data(using: .utf8)!
        
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: jsonData)
        
        XCTAssertEqual(request.messages.count, 1)
        XCTAssertEqual(request.messages[0].role, "user")
        
        switch request.messages[0].content {
        case .fragments(let fragments):
            XCTAssertEqual(fragments.count, 2)
            XCTAssertEqual(fragments[0].type, "text")
            XCTAssertEqual(fragments[0].text, "What do you see?")
            XCTAssertEqual(fragments[1].type, "image")
            XCTAssertEqual(fragments[1].imageUrl?.absoluteString, "https://example.com/image.jpg")
        default:
            XCTFail("Expected fragments content")
        }
    }
    
    // MARK: - CompletionResponse Tests
    
    func testCompletionResponseEncoding() throws {
        let choice = CompletionChoice(text: "Generated text", finishReason: "stop")
        let usage = CompletionUsage(promptTokens: 10, completionTokens: 5, totalTokens: 15)
        let response = CompletionResponse(
            id: "test-id",
            model: "test-model",
            choices: [choice],
            usage: usage
        )
        
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let jsonData = try encoder.encode(response)
        let jsonString = String(data: jsonData, encoding: .utf8)!
        
        XCTAssertTrue(jsonString.contains("\"id\":\"test-id\""))
        XCTAssertTrue(jsonString.contains("\"model\":\"test-model\""))
        XCTAssertTrue(jsonString.contains("\"text\":\"Generated text\""))
        XCTAssertTrue(jsonString.contains("\"finish_reason\":\"stop\""))
        XCTAssertTrue(jsonString.contains("\"prompt_tokens\":10"))
        XCTAssertTrue(jsonString.contains("\"completion_tokens\":5"))
        XCTAssertTrue(jsonString.contains("\"total_tokens\":15"))
    }
    
    // MARK: - ChatCompletionResponse Tests
    
    func testChatCompletionResponseEncoding() throws {
        let message = ChatMessageResponseData(role: "assistant", content: "Hello! How can I help you?")
        let choice = ChatCompletionChoice(index: 0, message: message, finishReason: "stop")
        let usage = CompletionUsage(promptTokens: 8, completionTokens: 12, totalTokens: 20)
        let response = ChatCompletionResponse(
            id: "chat-test-id",
            model: "test-model",
            choices: [choice],
            usage: usage
        )
        
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let jsonData = try encoder.encode(response)
        let jsonString = String(data: jsonData, encoding: .utf8)!
        
        XCTAssertTrue(jsonString.contains("\"id\":\"chat-test-id\""))
        XCTAssertTrue(jsonString.contains("\"object\":\"chat.completion\""))
        XCTAssertTrue(jsonString.contains("\"model\":\"test-model\""))
        XCTAssertTrue(jsonString.contains("\"role\":\"assistant\""))
        XCTAssertTrue(jsonString.contains("\"content\":\"Hello! How can I help you?\""))
        XCTAssertTrue(jsonString.contains("\"finish_reason\":\"stop\""))
    }
    
    // MARK: - EmbeddingRequest Tests
    
    func testEmbeddingRequestWithStringInput() throws {
        let jsonData = """
        {
            "input": "Hello world",
            "model": "embedding-model",
            "encoding_format": "float"
        }
        """.data(using: .utf8)!
        
        let request = try JSONDecoder().decode(EmbeddingRequest.self, from: jsonData)
        
        XCTAssertEqual(request.input.values, ["Hello world"])
        XCTAssertEqual(request.model, "embedding-model")
        XCTAssertEqual(request.encoding_format, "float")
    }
    
    func testEmbeddingRequestWithArrayInput() throws {
        let jsonData = """
        {
            "input": ["First text", "Second text", "Third text"],
            "model": "embedding-model"
        }
        """.data(using: .utf8)!
        
        let request = try JSONDecoder().decode(EmbeddingRequest.self, from: jsonData)
        
        XCTAssertEqual(request.input.values, ["First text", "Second text", "Third text"])
        XCTAssertEqual(request.model, "embedding-model")
    }
    
    // MARK: - ModelInfo Tests
    
    func testModelInfoEncoding() throws {
        let modelInfo = ModelInfo(id: "test-model")
        
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let jsonData = try encoder.encode(modelInfo)
        let jsonString = String(data: jsonData, encoding: .utf8)!
        
        XCTAssertTrue(jsonString.contains("\"id\":\"test-model\""))
        XCTAssertTrue(jsonString.contains("\"object\":\"model\""))
        XCTAssertTrue(jsonString.contains("\"owned_by\":\"user\""))
        XCTAssertTrue(jsonString.contains("\"created\":"))
    }
    
    func testModelListResponseEncoding() throws {
        let models = [
            ModelInfo(id: "model-1"),
            ModelInfo(id: "model-2")
        ]
        let response = ModelListResponse(data: models)
        
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let jsonData = try encoder.encode(response)
        let jsonString = String(data: jsonData, encoding: .utf8)!
        
        XCTAssertTrue(jsonString.contains("\"object\":\"list\""))
        XCTAssertTrue(jsonString.contains("\"id\":\"model-1\""))
        XCTAssertTrue(jsonString.contains("\"id\":\"model-2\""))
    }
}