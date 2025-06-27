import CoreImage
import Foundation
import Hub
import Logging
import MLX
import MLXLLM
@preconcurrency import MLXLMCommon
@preconcurrency import Tokenizers
import Vapor

private final class AtomicCounter {
    private let lock = NSLock()
    private var _value = 0

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return _value
    }

    func increment() {
        lock.lock()
        defer { lock.unlock() }
        _value += 1
    }
}

private enum MessageProcessingKeys {
    static let role = "role"
    static let content = "content"
    static let type = "type"
    static let text = "text"
    static let imageType = "image"
    static let videoType = "video"
}

private struct ChatGenerationContext {
    let modelContainer: ModelContainer
    let tokenizer: Tokenizer
    let eosTokenId: Int
    let userInput: UserInput
    let logger: Logger
    let promptCacheManager: PromptCacheManager?
}

private struct ChatGenerationParameters {
    let maxTokens: Int
    let temperature: Float
    let topP: Float
    let repetitionPenalty: Float
    let repetitionContextSize: Int
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int
}

private struct ChatResponseContext {
    let loadedModelName: String
    let stopIdSequences: [[Int]]
    let detokenizer: NaiveStreamingDetokenizer
    let estimatedPromptTokens: Int
}

private func _processTextOnlyMessages(_ chatRequest: ChatCompletionRequest) -> UserInput {
    let messages: [[String: Any]] = chatRequest.messages.map {
        [
            MessageProcessingKeys.role: $0.role,
            MessageProcessingKeys.content: $0.content.asString ?? "",
        ]
    }
    return UserInput(messages: messages)
}

private func _processVLMMessages(_ chatRequest: ChatCompletionRequest) -> UserInput {
    var allImages: [UserInput.Image] = []
    var allVideos: [UserInput.Video] = []

    let processedMessages: [[String: Any]] = chatRequest.messages.map { message -> [String: Any] in
        switch message.content {
        case .text(let textContent):
            return [
                MessageProcessingKeys.role: message.role,
                MessageProcessingKeys.content: textContent,
            ]

        case .fragments(let fragments):
            let imageFragments = fragments.filter { $0.type == MessageProcessingKeys.imageType }
            let videoFragments = fragments.filter { $0.type == MessageProcessingKeys.videoType }

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
                    if fragment.type == MessageProcessingKeys.text, let text = fragment.text {
                        contentFragments.append([
                            MessageProcessingKeys.type: MessageProcessingKeys.text,
                            MessageProcessingKeys.text: text,
                        ])
                    }
                }

                contentFragments.append(
                    contentsOf: imageFragments.map { _ in
                        [MessageProcessingKeys.type: MessageProcessingKeys.imageType]
                    })
                contentFragments.append(
                    contentsOf: videoFragments.map { _ in
                        [MessageProcessingKeys.type: MessageProcessingKeys.videoType]
                    })

                return [
                    MessageProcessingKeys.role: message.role,
                    MessageProcessingKeys.content: contentFragments,
                ]
            } else {
                return [
                    MessageProcessingKeys.role: message.role,
                    MessageProcessingKeys.content: message.content.asString ?? "",
                ]
            }

        case .none:
            return [MessageProcessingKeys.role: message.role, MessageProcessingKeys.content: ""]
        }
    }

    var userInput = UserInput(messages: processedMessages, images: allImages, videos: allVideos)

    if let resize = chatRequest.resize, !resize.isEmpty {
        let size: CGSize
        if resize.count == 1 {
            let value = resize[0]
            size = CGSize(width: value, height: value)
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

private func _processUserMessages(_ chatRequest: ChatCompletionRequest, isVLM: Bool) -> UserInput {
    if isVLM {
        return _processVLMMessages(chatRequest)
    } else {
        return _processTextOnlyMessages(chatRequest)
    }
}

private func _estimatePromptTokens(messages: [ChatMessageRequestData], tokenizer: Tokenizer) -> Int
{
    let combinedContent = messages.compactMap { $0.content.asString }.joined(separator: "\n")
    return tokenizer.encode(text: combinedContent).count
}

private func _validateProcessor(modelContainer: ModelContainer) async throws {
    _ = await modelContainer.perform { context in
        _ = context.processor
    }
}

private func _generateChatTokenStream(
    context: ChatGenerationContext,
    parameters: ChatGenerationParameters
) async throws -> AsyncStream<Int> {
    return AsyncStream { continuation in
        Task {
            let tokenCount = AtomicCounter()
            do {
                let generateParameters = GenerateParameters(
                    kvBits: parameters.kvBits,
                    kvGroupSize: parameters.kvGroupSize,
                    quantizedKVStart: parameters.quantizedKVStart,
                    temperature: parameters.temperature,
                    topP: parameters.topP,
                    repetitionPenalty: parameters.repetitionPenalty,
                    repetitionContextSize: parameters.repetitionContextSize
                )
                
                // Store original parameters for cache manager
                let originalGenerateParameters = generateParameters
                
                _ = try await context.modelContainer.perform { modelContext in
                    let lmInput: LMInput = try await modelContext.processor.prepare(
                        input: context.userInput)

                    let promptTokens = lmInput.text.tokens.asArray(Int.self)

                    var tokensToProcess: [Int] = []
                    var existingCache: [KVCache]?
                    if let cacheManager = context.promptCacheManager {
                        tokensToProcess = promptTokens
                        let modelKey = await context.modelContainer.configuration.name
                        let cacheResult = await cacheManager.getCachedState(
                            modelKey: modelKey,
                            tokens: promptTokens,
                            parameters: generateParameters,
                            model: modelContext.model
                        )
                        tokensToProcess = cacheResult.tokensToProcess
                        existingCache = cacheResult.cache

                        if existingCache != nil {
                            context.logger.info(
                                "Using cached prompt prefix, processing \(tokensToProcess.count) new tokens"
                            )
                        }
                    }

                    let inputForGeneration =
                        tokensToProcess.isEmpty
                        ? lmInput : LMInput(tokens: MLXArray(tokensToProcess))

                    let cache =
                        existingCache ?? modelContext.model.newCache(parameters: generateParameters)

                    // Create modified parameters to prevent TokenIterator quantization
                    var iteratorParameters = generateParameters
                    iteratorParameters.quantizedKVStart = Int.max
                    
                    let iterator = try TokenIterator(
                        input: inputForGeneration,
                        model: modelContext.model,
                        cache: cache,
                        parameters: iteratorParameters  // Use modified params
                    )

                    var allGeneratedTokens: [Int] = []

                    for token in iterator {
                        if token == context.eosTokenId { break }
                        if tokenCount.value >= parameters.maxTokens { break }

                        if token == context.tokenizer.unknownTokenId {
                            context.logger.warning("Generated unknown token ID \(token). Skipping.")
                        } else {
                            continuation.yield(token)
                            tokenCount.increment()
                            allGeneratedTokens.append(token)
                        }
                    }

                    if let cacheManager = context.promptCacheManager {
                        let fullTokens = promptTokens + allGeneratedTokens
                        await cacheManager.updateCache(
                            modelKey: context.modelContainer.configuration.name,
                            tokens: fullTokens,
                            kvCaches: cache,
                            parameters: originalGenerateParameters,  // Use original params
                            model: modelContext.model
                        )
                    }
                }
                continuation.finish()
            } catch {
                context.logger.error("Chat token stream error: \(error)")
                continuation.finish()
            }
        }
    }
}

func registerChatCompletionsRoute(
    _ app: Application,
    modelProvider: ModelProvider,
    isVLM: Bool = false,
    promptCacheManager: PromptCacheManager? = nil
) throws {
    app.post("v1", "chat", "completions") { req async throws -> Response in
        let chatRequest = try req.content.decode(ChatCompletionRequest.self)
        let logger = req.logger
        let reqModelId = chatRequest.model
        let (modelContainer, tokenizer, loadedModelName) = try await modelProvider.getModel(
            requestedModelId: reqModelId)
        guard let eosTokenId = tokenizer.eosTokenId else {
            throw ProcessingError(
                status: .internalServerError, reason: "Tokenizer EOS token ID missing",
                modelId: loadedModelName)
        }

        let userInput = _processUserMessages(chatRequest, isVLM: isVLM)

        if isVLM {
            logger.info(
                "VLM: Processing request with \(userInput.images.count) images and \(userInput.videos.count) videos"
            )
        }
        let estimatedPromptTokens = _estimatePromptTokens(
            messages: chatRequest.messages, tokenizer: tokenizer)

        logger.info(
            "Received CHAT completion request for model '\(loadedModelName)', estimated prompt tokens: \(estimatedPromptTokens)"
        )

        let maxTokens = chatRequest.maxTokens ?? GenerationDefaults.maxTokens
        let temperature = chatRequest.temperature ?? GenerationDefaults.temperature
        let topP = chatRequest.topP ?? GenerationDefaults.topP
        let streamResponse = chatRequest.stream ?? GenerationDefaults.stream
        let stopWords = chatRequest.stop ?? GenerationDefaults.stopSequences
        let stopIdSequences = stopSequencesToIds(stopWords: stopWords, tokenizer: tokenizer)
        let repetitionPenalty =
            chatRequest.repetitionPenalty ?? GenerationDefaults.repetitionPenalty
        let repetitionContextSize =
            chatRequest.repetitionContextSize ?? GenerationDefaults.repetitionContextSize

        let kvBits = chatRequest.kvBits
        let kvGroupSize = chatRequest.kvGroupSize ?? GenerationDefaults.kvGroupSize
        let quantizedKVStart = chatRequest.quantizedKVStart ?? GenerationDefaults.quantizedKVStart

        try KVCacheValidation.validate(
            bits: kvBits,
            groupSize: kvGroupSize,
            quantizationStart: quantizedKVStart
        )

        try await _validateProcessor(modelContainer: modelContainer)

        let detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)

        if streamResponse {
            let generationContext = ChatGenerationContext(
                modelContainer: modelContainer,
                tokenizer: tokenizer,
                eosTokenId: eosTokenId,
                userInput: userInput,
                logger: logger,
                promptCacheManager: isVLM ? nil : promptCacheManager
            )
            let generationParameters = ChatGenerationParameters(
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                repetitionContextSize: repetitionContextSize,
                kvBits: kvBits,
                kvGroupSize: kvGroupSize,
                quantizedKVStart: quantizedKVStart
            )
            let responseContext = ChatResponseContext(
                loadedModelName: loadedModelName,
                stopIdSequences: stopIdSequences,
                detokenizer: detokenizer,
                estimatedPromptTokens: estimatedPromptTokens
            )
            return try await handleStreamingChatResponse(
                generationContext: generationContext,
                generationParameters: generationParameters,
                responseContext: responseContext
            )
        } else {
            let generationContext = ChatGenerationContext(
                modelContainer: modelContainer,
                tokenizer: tokenizer,
                eosTokenId: eosTokenId,
                userInput: userInput,
                logger: logger,
                promptCacheManager: isVLM ? nil : promptCacheManager
            )
            let generationParameters = ChatGenerationParameters(
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                repetitionContextSize: repetitionContextSize,
                kvBits: kvBits,
                kvGroupSize: kvGroupSize,
                quantizedKVStart: quantizedKVStart
            )
            let responseContext = ChatResponseContext(
                loadedModelName: loadedModelName,
                stopIdSequences: stopIdSequences,
                detokenizer: detokenizer,
                estimatedPromptTokens: estimatedPromptTokens
            )
            return try await handleNonStreamingChatResponse(
                req: req,
                generationContext: generationContext,
                generationParameters: generationParameters,
                responseContext: responseContext
            )
        }
    }
}

private func handleStreamingChatResponse(
    generationContext: ChatGenerationContext,
    generationParameters: ChatGenerationParameters,
    responseContext: ChatResponseContext
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
        var streamDetokenizer = responseContext.detokenizer

        Task {
            var generatedTokens: [Int] = []
            generatedTokens.reserveCapacity(generationParameters.maxTokens)
            var finalFinishReason: String?

            do {
                generationContext.logger.info(
                    "Starting CHAT stream generation (ID: \(chatId)) for model \(responseContext.loadedModelName)"
                )

                let initialDelta = ChatCompletionDelta(role: "assistant", content: "")
                let initialChoice = ChatCompletionChoiceDelta(
                    index: 0, delta: initialDelta, finishReason: nil)
                let initialChunk = ChatCompletionChunkResponse(
                    id: chatId, created: created, model: responseContext.loadedModelName,
                    systemFingerprint: systemFingerprint, choices: [initialChoice])
                if let initialSse = encodeSSE(
                    response: initialChunk, logger: generationContext.logger)
                {
                    writer.write(.buffer(.init(string: initialSse)))
                }

                let tokenStream = try await _generateChatTokenStream(
                    context: generationContext,
                    parameters: generationParameters
                )

                for try await token in tokenStream {
                    generatedTokens.append(token)
                    streamDetokenizer.append(token: token)
                    let stopCondition = checkStoppingCriteria(
                        tokens: generatedTokens, stopIdSequences: responseContext.stopIdSequences,
                        eosTokenId: generationContext.eosTokenId)

                    if stopCondition.stopMet {
                        finalFinishReason = "stop"
                        break
                    }

                    if let newTextChunk = streamDetokenizer.next() {
                        let delta = ChatCompletionDelta(role: nil, content: newTextChunk)
                        let choice = ChatCompletionChoiceDelta(
                            index: 0, delta: delta, finishReason: nil)
                        let chunkResponse = ChatCompletionChunkResponse(
                            id: chatId, created: created, model: responseContext.loadedModelName,
                            systemFingerprint: systemFingerprint, choices: [choice]
                        )
                        if let sseString = encodeSSE(
                            response: chunkResponse, logger: generationContext.logger)
                        {
                            writer.write(.buffer(.init(string: sseString)))
                        }
                    }
                }

                if finalFinishReason == nil {
                    finalFinishReason =
                        (generatedTokens.count >= generationParameters.maxTokens)
                        ? "length" : "stop"
                }

                let finalDelta = ChatCompletionDelta(role: nil, content: nil)
                let finalChoice = ChatCompletionChoiceDelta(
                    index: 0, delta: finalDelta, finishReason: finalFinishReason)
                let finalChunk = ChatCompletionChunkResponse(
                    id: chatId, created: created, model: responseContext.loadedModelName,
                    systemFingerprint: systemFingerprint, choices: [finalChoice]
                )
                if let finalSseString = encodeSSE(
                    response: finalChunk, logger: generationContext.logger)
                {
                    _ = try writer.write(.buffer(.init(string: finalSseString)))
                }
            } catch {
                generationContext.logger.error("Chat stream error (ID: \(chatId)): \(error)")
                finalFinishReason = "error"
            }

            _ = writer.write(.buffer(.init(string: AppConstants.sseDoneMessage)))
            generationContext.logger.info(
                "CHAT Streaming response finished sending (ID: \(chatId)). Final Reason: \(finalFinishReason ?? "unknown")"
            )
            _ = writer.write(.end)
        }
    })
    return response
}

private func handleNonStreamingChatResponse(
    req: Request,
    generationContext: ChatGenerationContext,
    generationParameters: ChatGenerationParameters,
    responseContext: ChatResponseContext
) async throws -> Response {
    var generatedTokens: [Int] = []
    generatedTokens.reserveCapacity(generationParameters.maxTokens)
    var finalFinishReason = "stop"
    let responseId = "chatcmpl-\(UUID().uuidString)"
    let created = Int(Date().timeIntervalSince1970)

    do {
        generationContext.logger.info(
            "Starting non-streaming CHAT generation (ID: \(responseId)) for model \(responseContext.loadedModelName)"
        )
        let tokenStream = try await _generateChatTokenStream(
            context: generationContext,
            parameters: generationParameters
        )

        for try await token in tokenStream {
            generatedTokens.append(token)
            let stopCondition = checkStoppingCriteria(
                tokens: generatedTokens, stopIdSequences: responseContext.stopIdSequences,
                eosTokenId: generationContext.eosTokenId)

            if stopCondition.stopMet {
                if stopCondition.trimLength > 0 && generatedTokens.count >= stopCondition.trimLength
                {
                    generatedTokens.removeLast(stopCondition.trimLength)
                }
                finalFinishReason = "stop"
                break
            }
        }

        if finalFinishReason != "stop" {
            finalFinishReason =
                (generatedTokens.count >= generationParameters.maxTokens) ? "length" : "stop"
        }
    } catch {
        generationContext.logger.error(
            "Non-streaming chat generation error (ID: \(responseId)): \(error)")
        throw ProcessingError(
            status: .internalServerError, reason: "Failed to generate chat completion",
            underlyingError: error)
    }

    let completionText = decodeTokens(generatedTokens, tokenizer: generationContext.tokenizer)

    let assistantMessage = ChatMessageResponseData(role: "assistant", content: completionText)
    let chatChoice = ChatCompletionChoice(
        index: 0, message: assistantMessage, finishReason: finalFinishReason)
    let usage = CompletionUsage(
        promptTokens: responseContext.estimatedPromptTokens,
        completionTokens: generatedTokens.count,
        totalTokens: responseContext.estimatedPromptTokens + generatedTokens.count
    )

    let chatResponse = ChatCompletionResponse(
        id: responseId, created: created, model: responseContext.loadedModelName,
        choices: [chatChoice], usage: usage
    )

    generationContext.logger.info(
        "Non-streaming CHAT response generated (ID: \(responseId)). Reason: \(finalFinishReason)")
    return try await chatResponse.encodeResponse(for: req)
}
