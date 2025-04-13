import Vapor
import Foundation
import Logging
import MLX
import MLXLLM
import MLXLMCommon
import Hub
import Tokenizers

actor ChatCompletionManager {
    private let modelContainer: ModelContainer
    private let tokenizer: Tokenizer
    private let eosTokenId: Int
    private let loadedModelName: String
    
    init(modelContainer: ModelContainer, tokenizer: Tokenizer, eosTokenId: Int, loadedModelName: String) {
        self.modelContainer = modelContainer
        self.tokenizer = tokenizer
        self.eosTokenId = eosTokenId
        self.loadedModelName = loadedModelName
    }
    
    func generateChatTokenStream(
        userInput: UserInput, 
        maxTokens: Int, 
        temperature: Float, 
        topP: Float,
        repetitionPenalty: Float, 
        repetitionContextSize: Int, 
        logger: Logger
    ) async throws -> AsyncStream<Int> {
        return AsyncStream { continuation in
            Task {
                var generatedTokenCount = 0
                do {
                    let generateParameters = GenerateParameters(
                        temperature: temperature, 
                        topP: topP,
                        repetitionPenalty: repetitionPenalty, 
                        repetitionContextSize: repetitionContextSize
                    )
                    _ = try await modelContainer.perform { context in
                        logger.debug("Preparing UserInput using model processor...")
                        guard let processor = context.processor as? UserInputProcessor else {
                            logger.error("Chat generation requires a UserInputProcessor in the model context.")
                            throw Abort(.internalServerError, reason: "Model processor invalid for chat input.")
                        }
                        let lmInput: LMInput = try await processor.prepare(input: userInput)
                        logger.debug("UserInput prepared into LMInput.")

                        try MLXLMCommon.generate(input: lmInput, parameters: generateParameters, context: context) { tokens in
                            guard let lastToken = tokens.last else { return .more }
                            if lastToken == self.eosTokenId { continuation.finish(); return .stop }
                            guard generatedTokenCount < maxTokens else { continuation.finish(); return .stop }
                            if lastToken == tokenizer.unknownTokenId { 
                                logger.warning("Generated unknown token ID \(lastToken). Skipping.") 
                            } else { 
                                continuation.yield(lastToken)
                                generatedTokenCount += 1 
                            }
                            return .more
                        }
                    }
                    logger.debug("Chat generate function completed or stopped.")
                    continuation.finish()
                } catch {
                    logger.error("Chat token stream error: \(error)")
                    continuation.finish()
                }
            }
        }
    }
    
    func estimatePromptTokens(messages: [ChatMessageRequestData]) -> Int {
        let combinedContent = messages.compactMap { $0.content.asString }.joined(separator: "\n")
        return tokenizer.encode(text: combinedContent).count
    }
    
    func validateProcessor() async throws {
        let hasProcessor = await modelContainer.perform { $0.processor is UserInputProcessor }
        guard hasProcessor else { 
            throw Abort(.internalServerError, reason: "Model processor invalid for chat input.") 
        }
    }
    
    func getModelName(requestedModel: String?) -> String {
        requestedModel ?? loadedModelName
    }
    
    func stopSequencesToIds(stopWords: [String]) -> [[Int]] {
        stopWords.compactMap { word in 
            tokenizer.encode(text: word, addSpecialTokens: false).nilIfEmpty() 
        }
    }
    
    func decodeTokens(_ tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }
}

func registerChatCompletionsRoute(
    _ app: Application,
    modelContainer: ModelContainer,
    tokenizer: Tokenizer,
    eosTokenId: Int,
    loadedModelName: String
) throws {
    let chatManager = ChatCompletionManager(
        modelContainer: modelContainer,
        tokenizer: tokenizer,
        eosTokenId: eosTokenId,
        loadedModelName: loadedModelName
    )

    app.post("v1", "chat", "completions") { req async throws -> Response in
        let chatRequest = try req.content.decode(ChatCompletionRequest.self)
        let logger = req.logger

        let messages: [Message] = chatRequest.messages.map {
            ["role": $0.role, "content": $0.content.asString ?? ""]
        }
        let userInput = UserInput(messages: messages)

        let estimatedPromptTokens = await chatManager.estimatePromptTokens(messages: chatRequest.messages)
        let reqModelName = await chatManager.getModelName(requestedModel: chatRequest.model)

        logger.info("Received CHAT completion request for model '\(reqModelName)', estimated prompt tokens: \(estimatedPromptTokens)")

        let maxTokens = chatRequest.maxTokens ?? GenerationDefaults.maxTokens
        let temperature = chatRequest.temperature ?? GenerationDefaults.temperature
        let topP = chatRequest.topP ?? GenerationDefaults.topP
        let streamResponse = chatRequest.stream ?? GenerationDefaults.stream
        let stopWords = chatRequest.stop ?? GenerationDefaults.stopSequences
        let stopIdSequences = await chatManager.stopSequencesToIds(stopWords: stopWords)
        let repetitionPenalty = chatRequest.repetitionPenalty ?? GenerationDefaults.repetitionPenalty
        let repetitionContextSize = chatRequest.repetitionContextSize ?? GenerationDefaults.repetitionContextSize

        try await chatManager.validateProcessor()

        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        
        if streamResponse {
            return try await handleStreamingResponse(
                chatManager: chatManager,
                userInput: userInput,
                reqModelName: reqModelName,
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                repetitionContextSize: repetitionContextSize,
                stopIdSequences: stopIdSequences,
                eosTokenId: eosTokenId,
                detokenizer: detokenizer,
                estimatedPromptTokens: estimatedPromptTokens,
                logger: logger
            )
        } else {
            return try await handleNonStreamingResponse(
                req: req,
                chatManager: chatManager,
                userInput: userInput,
                reqModelName: reqModelName,
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                repetitionContextSize: repetitionContextSize,
                stopIdSequences: stopIdSequences,
                eosTokenId: eosTokenId,
                estimatedPromptTokens: estimatedPromptTokens,
                logger: logger
            )
        }
    }
}

private func handleStreamingResponse(
    chatManager: ChatCompletionManager,
    userInput: UserInput,
    reqModelName: String,
    maxTokens: Int,
    temperature: Float,
    topP: Float,
    repetitionPenalty: Float,
    repetitionContextSize: Int,
    stopIdSequences: [[Int]],
    eosTokenId: Int,
    detokenizer: NaiveStreamingDetokenizer,
    estimatedPromptTokens: Int,
    logger: Logger
) async throws -> Response {
    let headers = HTTPHeaders([
        ("Content-Type", "text/event-stream"),
        ("Cache-Control", "no-cache"),
        ("Connection", "keep-alive"),
    ])
    
    let response = Response(status: .ok, headers: headers)
    response.body = .init(stream: { writer in
        let chatId = "chatcmpl-\(UUID().uuidString)"
        let created = Int(Date().timeIntervalSince1970)
        let systemFingerprint: String? = nil
        var streamDetokenizer = detokenizer

        Task {
            var generatedTokens: [Int] = []
            generatedTokens.reserveCapacity(maxTokens)
            var finalFinishReason: String? = nil

            do {
                logger.info("Starting CHAT stream generation (ID: \(chatId))")

                let initialDelta = ChatCompletionDelta(role: "assistant", content: "")
                let initialChoice = ChatCompletionChoiceDelta(index: 0, delta: initialDelta, finishReason: nil)
                let initialChunk = ChatCompletionChunkResponse(id: chatId, created: created, model: reqModelName, systemFingerprint: systemFingerprint, choices: [initialChoice])
                if let initialSse = encodeSSE(response: initialChunk, logger: logger) {
                    try await writer.write(.buffer(.init(string: initialSse)))
                }

                let tokenStream = try await chatManager.generateChatTokenStream(
                    userInput: userInput, 
                    maxTokens: maxTokens, 
                    temperature: temperature, 
                    topP: topP,
                    repetitionPenalty: repetitionPenalty, 
                    repetitionContextSize: repetitionContextSize, 
                    logger: logger
                )

                for try await token in tokenStream {
                    generatedTokens.append(token)
                    streamDetokenizer.append(token: token)
                    let stopCondition = checkStoppingCriteria(tokens: generatedTokens, stopIdSequences: stopIdSequences, eosTokenId: eosTokenId)
                    
                    if stopCondition.stopMet { 
                        finalFinishReason = "stop"
                        break 
                    }
                    
                    if let newTextChunk = streamDetokenizer.next() {
                        let delta = ChatCompletionDelta(role: nil, content: newTextChunk)
                        let choice = ChatCompletionChoiceDelta(index: 0, delta: delta, finishReason: nil)
                        let chunkResponse = ChatCompletionChunkResponse(
                            id: chatId, 
                            created: created, 
                            model: reqModelName, 
                            systemFingerprint: systemFingerprint, 
                            choices: [choice]
                        )
                        
                        if let sseString = encodeSSE(response: chunkResponse, logger: logger) {
                            try await writer.write(.buffer(.init(string: sseString)))
                        }
                    }
                }

                if finalFinishReason == nil { 
                    finalFinishReason = (generatedTokens.count >= maxTokens) ? "length" : "stop" 
                }

                let finalDelta = ChatCompletionDelta(role: nil, content: nil)
                let finalChoice = ChatCompletionChoiceDelta(index: 0, delta: finalDelta, finishReason: finalFinishReason)
                let finalChunk = ChatCompletionChunkResponse(
                    id: chatId, 
                    created: created, 
                    model: reqModelName, 
                    systemFingerprint: systemFingerprint, 
                    choices: [finalChoice]
                )
                
                if let finalSseString = encodeSSE(response: finalChunk, logger: logger) {
                    await writer.write(.buffer(.init(string: finalSseString)))
                }

            } catch { 
                logger.error("Chat stream error (ID: \(chatId)): \(error)")
                finalFinishReason = "error" 
            }
            
            await writer.write(.buffer(.init(string: AppConstants.sseDoneMessage)))
            logger.info("CHAT Streaming response finished sending (ID: \(chatId)). Final Reason: \(finalFinishReason ?? "unknown")")
            await writer.write(.end)
        }
    })
    
    return response
}

private func handleNonStreamingResponse(
    req: Request,
    chatManager: ChatCompletionManager,
    userInput: UserInput,
    reqModelName: String,
    maxTokens: Int,
    temperature: Float,
    topP: Float,
    repetitionPenalty: Float,
    repetitionContextSize: Int,
    stopIdSequences: [[Int]],
    eosTokenId: Int,
    estimatedPromptTokens: Int,
    logger: Logger
) async throws -> Response {
    var generatedTokens: [Int] = []
    generatedTokens.reserveCapacity(maxTokens)
    var finalFinishReason = "stop"
    let responseId = "chatcmpl-\(UUID().uuidString)"
    let created = Int(Date().timeIntervalSince1970)

    do {
        logger.info("Starting non-streaming CHAT generation (ID: \(responseId)).")
        let tokenStream = try await chatManager.generateChatTokenStream(
            userInput: userInput, 
            maxTokens: maxTokens, 
            temperature: temperature, 
            topP: topP,
            repetitionPenalty: repetitionPenalty, 
            repetitionContextSize: repetitionContextSize, 
            logger: logger
        )
        
        for try await token in tokenStream {
            generatedTokens.append(token)
            let stopCondition = checkStoppingCriteria(tokens: generatedTokens, stopIdSequences: stopIdSequences, eosTokenId: eosTokenId)
            
            if stopCondition.stopMet {
                if stopCondition.trimLength > 0 && generatedTokens.count >= stopCondition.trimLength { 
                    generatedTokens.removeLast(stopCondition.trimLength) 
                }
                finalFinishReason = "stop"
                break
            }
        }
        
        if finalFinishReason != "stop" { 
            finalFinishReason = (generatedTokens.count >= maxTokens) ? "length" : "stop" 
        }
    } catch {
        logger.error("Non-streaming chat generation error (ID: \(responseId)): \(error)")
        throw Abort(.internalServerError, reason: "Failed to generate chat completion: \(error.localizedDescription)")
    }
    
    let completionText = await chatManager.decodeTokens(generatedTokens)
    let assistantMessage = ChatMessageResponseData(role: "assistant", content: completionText)
    let chatChoice = ChatCompletionChoice(index: 0, message: assistantMessage, finishReason: finalFinishReason)
    let usage = CompletionUsage(
        promptTokens: estimatedPromptTokens, 
        completionTokens: generatedTokens.count, 
        totalTokens: estimatedPromptTokens + generatedTokens.count
    )
    
    let chatResponse = ChatCompletionResponse(
        id: responseId, 
        created: created, 
        model: reqModelName, 
        choices: [chatChoice], 
        usage: usage
    )

    logger.info("Non-streaming CHAT response generated (ID: \(responseId)). Reason: \(finalFinishReason)")
    return try await chatResponse.encodeResponse(for: req)
}
