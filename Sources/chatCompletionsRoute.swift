import Vapor
import Foundation
import Logging
import MLX
import MLXLLM
import MLXLMCommon
import Hub
import Tokenizers
import CoreImage

actor ChatCompletionManager {
    private let logger: Logger 
    init(logger: Logger) {
        self.logger = logger
    }

    func generateChatTokenStream(
        modelContainer: ModelContainer,
        tokenizer: Tokenizer,
        eosTokenId: Int,
        userInput: UserInput,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        repetitionContextSize: Int
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
                            if lastToken == eosTokenId { continuation.finish(); return .stop }
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

    func estimatePromptTokens(messages: [ChatMessageRequestData], tokenizer: Tokenizer) -> Int {
        let combinedContent = messages.compactMap { $0.content.asString }.joined(separator: "\n")
        return tokenizer.encode(text: combinedContent).count
    }

    func validateProcessor(modelContainer: ModelContainer) async throws {
        let hasProcessor = await modelContainer.perform { $0.processor is UserInputProcessor }
        guard hasProcessor else {
            throw Abort(.internalServerError, reason: "Model processor invalid for chat input.")
        }
    }

    func stopSequencesToIds(stopWords: [String], tokenizer: Tokenizer) -> [[Int]] {
        stopWords.compactMap { word in
            tokenizer.encode(text: word, addSpecialTokens: false).nilIfEmpty()
        }
    }

    func decodeTokens(_ tokens: [Int], tokenizer: Tokenizer) -> String {
        tokenizer.decode(tokens: tokens)
    }

    func processUserMessages(_ chatRequest: ChatCompletionRequest, isVLM: Bool) -> UserInput {
         if isVLM {
            return processVLMMessages(chatRequest)
        } else {
            return processTextOnlyMessages(chatRequest)
        }
    }

    private func processTextOnlyMessages(_ chatRequest: ChatCompletionRequest) -> UserInput {
        let messages: [[String: Any]] = chatRequest.messages.map {
            ["role": $0.role, "content": $0.content.asString ?? ""]
        }
        return UserInput(messages: messages)
    }

    private func processVLMMessages(_ chatRequest: ChatCompletionRequest) -> UserInput {
        var allImages: [UserInput.Image] = []
        var allVideos: [UserInput.Video] = []

        let processedMessages: [[String: Any]] = chatRequest.messages.map { message -> [String: Any] in
            switch message.content {
            case .text(let textContent):
                return ["role": message.role, "content": textContent]

            case .fragments(let fragments):
                let imageFragments = fragments.filter { $0.type == "image" }
                let videoFragments = fragments.filter { $0.type == "video" }

                let images = imageFragments.compactMap { fragment in
                    fragment.imageUrl.map { UserInput.Image.url($0) }
                }
                allImages.append(contentsOf: images)

                let videos = videoFragments.compactMap { fragment in
                    fragment.videoUrl.map { UserInput.Video.url($0) }
                }
                allVideos.append(contentsOf: videos)

                if !images.isEmpty || !videos.isEmpty {
                    var contentFragments: [[String: Any]] = []

                    fragments.forEach { fragment in
                        if fragment.type == "text", let text = fragment.text {
                            contentFragments.append(["type": "text", "text": text])
                        }
                    }

                    contentFragments.append(contentsOf: imageFragments.map { _ in ["type": "image"] })
                    contentFragments.append(contentsOf: videoFragments.map { _ in ["type": "video"] })

                    return ["role": message.role, "content": contentFragments]
                } else {
                    return ["role": message.role, "content": message.content.asString ?? ""]
                }

            case .none:
                return ["role": message.role, "content": ""]
            }
        }

        var userInput = UserInput(messages: processedMessages, images: allImages, videos: allVideos)

        if let resize = chatRequest.resize, !resize.isEmpty {
            let size: CGSize
            if resize.count == 1 {
                let v = resize[0]
                size = CGSize(width: v, height: v)
            } else if resize.count >= 2 {
                let v0 = resize[0]
                let v1 = resize[1]
                size = CGSize(width: v0, height: v1)
            } else {
                size = .zero
            }

            if size != .zero {
                userInput.processing.resize = size
            }
        }

        return userInput
    }
}

func registerChatCompletionsRoute(
    _ app: Application,
    modelProvider: ModelProvider, 
    isVLM: Bool = false
) throws {
    let chatManager = ChatCompletionManager(logger: app.logger)

    app.post("v1", "chat", "completions") { req async throws -> Response in
        let chatRequest = try req.content.decode(ChatCompletionRequest.self)
        let logger = req.logger
        let reqModelId = chatRequest.model 
        let (modelContainer, tokenizer, loadedModelName) = try await modelProvider.getModel(requestedModelId: reqModelId)
        guard let eosTokenId = tokenizer.eosTokenId else {
             throw Abort(.internalServerError, reason: "Tokenizer EOS token ID missing for model \(loadedModelName).")
        }

        let userInput = await chatManager.processUserMessages(chatRequest, isVLM: isVLM)
        let estimatedPromptTokens = await chatManager.estimatePromptTokens(messages: chatRequest.messages, tokenizer: tokenizer)

        logger.info("Received CHAT completion request for model '\(loadedModelName)', estimated prompt tokens: \(estimatedPromptTokens)")

        let maxTokens = chatRequest.maxTokens ?? GenerationDefaults.maxTokens
        let temperature = chatRequest.temperature ?? GenerationDefaults.temperature
        let topP = chatRequest.topP ?? GenerationDefaults.topP
        let streamResponse = chatRequest.stream ?? GenerationDefaults.stream
        let stopWords = chatRequest.stop ?? GenerationDefaults.stopSequences
        let stopIdSequences = await chatManager.stopSequencesToIds(stopWords: stopWords, tokenizer: tokenizer)
        let repetitionPenalty = chatRequest.repetitionPenalty ?? GenerationDefaults.repetitionPenalty
        let repetitionContextSize = chatRequest.repetitionContextSize ?? GenerationDefaults.repetitionContextSize


        try await chatManager.validateProcessor(modelContainer: modelContainer)

        let detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)

        if streamResponse {
            return try await handleStreamingChatResponse( 
                chatManager: chatManager,
                modelContainer: modelContainer,
                tokenizer: tokenizer,
                eosTokenId: eosTokenId,
                userInput: userInput,
                loadedModelName: loadedModelName, 
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                repetitionContextSize: repetitionContextSize,
                stopIdSequences: stopIdSequences,
                detokenizer: detokenizer,
                estimatedPromptTokens: estimatedPromptTokens,
                logger: logger
            )
        } else {
             return try await handleNonStreamingChatResponse( 
                req: req,
                chatManager: chatManager,
                modelContainer: modelContainer,
                tokenizer: tokenizer,
                eosTokenId: eosTokenId,
                userInput: userInput,
                loadedModelName: loadedModelName,
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                repetitionContextSize: repetitionContextSize,
                stopIdSequences: stopIdSequences,
                estimatedPromptTokens: estimatedPromptTokens,
                logger: logger
            )
        }
    }
}

private func handleStreamingChatResponse(
    chatManager: ChatCompletionManager,
    modelContainer: ModelContainer, 
    tokenizer: Tokenizer,           
    eosTokenId: Int,                
    userInput: UserInput,
    loadedModelName: String,        
    maxTokens: Int,
    temperature: Float,
    topP: Float,
    repetitionPenalty: Float,
    repetitionContextSize: Int,
    stopIdSequences: [[Int]],
    detokenizer: NaiveStreamingDetokenizer,
    estimatedPromptTokens: Int,
    logger: Logger
) async throws -> Response {
    let headers =  HTTPHeaders([
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
                logger.info("Starting CHAT stream generation (ID: \(chatId)) for model \(loadedModelName)")

                let initialDelta = ChatCompletionDelta(role: "assistant", content: "")
                let initialChoice = ChatCompletionChoiceDelta(index: 0, delta: initialDelta, finishReason: nil)

                let initialChunk = ChatCompletionChunkResponse(id: chatId, created: created, model: loadedModelName, systemFingerprint: systemFingerprint, choices: [initialChoice])
                if let initialSse = encodeSSE(response: initialChunk, logger: logger) {
                    try await writer.write(.buffer(.init(string: initialSse)))
                }


                let tokenStream = try await chatManager.generateChatTokenStream(
                    modelContainer: modelContainer,
                    tokenizer: tokenizer,
                    eosTokenId: eosTokenId,
                    userInput: userInput,
                    maxTokens: maxTokens,
                    temperature: temperature,
                    topP: topP,
                    repetitionPenalty: repetitionPenalty,
                    repetitionContextSize: repetitionContextSize
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
                            id: chatId, created: created, model: loadedModelName,
                            systemFingerprint: systemFingerprint, choices: [choice]
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
                    id: chatId, created: created, model: loadedModelName,
                    systemFingerprint: systemFingerprint, choices: [finalChoice]
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

private func handleNonStreamingChatResponse(
    req: Request,
    chatManager: ChatCompletionManager,
    modelContainer: ModelContainer, 
    tokenizer: Tokenizer,           
    eosTokenId: Int,                
    userInput: UserInput,
    loadedModelName: String,        
    maxTokens: Int,
    temperature: Float,
    topP: Float,
    repetitionPenalty: Float,
    repetitionContextSize: Int,
    stopIdSequences: [[Int]],
    estimatedPromptTokens: Int,
    logger: Logger
) async throws -> Response {
    var generatedTokens: [Int] = []
    generatedTokens.reserveCapacity(maxTokens)
    var finalFinishReason = "stop"
    let responseId = "chatcmpl-\(UUID().uuidString)"
    let created = Int(Date().timeIntervalSince1970)

    do {
        logger.info("Starting non-streaming CHAT generation (ID: \(responseId)) for model \(loadedModelName)")
        let tokenStream = try await chatManager.generateChatTokenStream(
            modelContainer: modelContainer,
            tokenizer: tokenizer,
            eosTokenId: eosTokenId,
            userInput: userInput,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            repetitionContextSize: repetitionContextSize
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

    let completionText = await chatManager.decodeTokens(generatedTokens, tokenizer: tokenizer)
    let assistantMessage = ChatMessageResponseData(role: "assistant", content: completionText)
    let chatChoice = ChatCompletionChoice(index: 0, message: assistantMessage, finishReason: finalFinishReason)
    let usage = CompletionUsage(
        promptTokens: estimatedPromptTokens,
        completionTokens: generatedTokens.count,
        totalTokens: estimatedPromptTokens + generatedTokens.count
    )

    let chatResponse = ChatCompletionResponse(
        id: responseId, created: created, model: loadedModelName,
        choices: [chatChoice], usage: usage
    )

    logger.info("Non-streaming CHAT response generated (ID: \(responseId)). Reason: \(finalFinishReason)")
    return try await chatResponse.encodeResponse(for: req)
}
