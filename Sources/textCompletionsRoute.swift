import Foundation
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

func registerTextCompletionsRoute(_ app: Application, modelProvider: ModelProvider, promptCacheManager: PromptCacheManager? = nil) throws {
    app.post("v1", "completions") { req async throws -> Response in
        let completionRequest = try req.content.decode(CompletionRequest.self)
        let logger = req.logger
        let reqModelName = completionRequest.model
        let baseCompletionId = "cmpl-\(UUID().uuidString)"

        let (modelContainer, tokenizer, loadedModelName) = try await modelProvider.getModel(requestedModelId: reqModelName)
        
        guard let eosTokenId = tokenizer.eosTokenId else {
            throw ProcessingError(
                status: .internalServerError, 
                reason: "Tokenizer EOS token ID missing", 
                modelId: loadedModelName
            )
        }
        
        let promptTokens = tokenizer.encode(text: completionRequest.prompt)
        logger.info("Received TEXT completion request (ID: \(baseCompletionId)) for model '\(reqModelName ?? "default")', prompt tokens: \(promptTokens.count)")

        let parameters = TextCompletionParameters(from: completionRequest)
        
        try KVCacheValidation.validate(
            bits: parameters.kvBits,
            groupSize: parameters.kvGroupSize,
            quantizationStart: parameters.quantizedKVStart
        )
        
        let stopIdSequences = stopSequencesToIds(stopWords: parameters.stopWords, tokenizer: tokenizer)

        if parameters.streamResponse {
            return try await handleStreamingTextCompletion(
                baseCompletionId: baseCompletionId,
                modelContainer: modelContainer,
                tokenizer: tokenizer,
                eosTokenId: eosTokenId,
                promptTokens: promptTokens,
                parameters: parameters,
                stopIdSequences: stopIdSequences,
                loadedModelName: loadedModelName,
                logger: logger,
                promptCacheManager: promptCacheManager
            )
        } else {
            return try await handleNonStreamingTextCompletion(
                req: req,
                baseCompletionId: baseCompletionId,
                modelContainer: modelContainer,
                tokenizer: tokenizer,
                eosTokenId: eosTokenId,
                promptTokens: promptTokens,
                parameters: parameters,
                stopIdSequences: stopIdSequences,
                loadedModelName: loadedModelName,
                logger: logger,
                promptCacheManager: promptCacheManager
            )
        }
    }
}

private struct TextCompletionParameters {
    let maxTokens: Int
    let temperature: Float
    let topP: Float
    let streamResponse: Bool
    let stopWords: [String]
    let repetitionPenalty: Float
    let repetitionContextSize: Int
    
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int
    
    init(from request: CompletionRequest) {
        self.maxTokens = request.maxTokens ?? GenerationDefaults.maxTokens
        self.temperature = request.temperature ?? GenerationDefaults.temperature
        self.topP = request.topP ?? GenerationDefaults.topP
        self.streamResponse = request.stream ?? GenerationDefaults.stream
        self.stopWords = request.stop ?? GenerationDefaults.stopSequences
        self.repetitionPenalty = request.repetitionPenalty ?? GenerationDefaults.repetitionPenalty
        self.repetitionContextSize = request.repetitionContextSize ?? GenerationDefaults.repetitionContextSize
        
        self.kvBits = request.kvBits
        self.kvGroupSize = request.kvGroupSize ?? GenerationDefaults.kvGroupSize
        self.quantizedKVStart = request.quantizedKVStart ?? GenerationDefaults.quantizedKVStart
    }
}

private func generateCompletionTokenStream(
    modelContainer: ModelContainer,
    tokenizer: Tokenizer,
    eosTokenId: Int,
    promptTokens: [Int],
    parameters: TextCompletionParameters,
    logger: Logger,
    promptCacheManager: PromptCacheManager? = nil
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
                _ = try await modelContainer.perform { context in
                    var tokensToProcess = promptTokens
                    var existingCache: [KVCache]?
                    
                    if let cacheManager = promptCacheManager {
                        let modelKey = await modelContainer.configuration.name
                        let cacheResult = await cacheManager.getCachedState(
                            modelKey: modelKey,
                            tokens: promptTokens,
                            parameters: generateParameters,
                            model: context.model
                        )
                        tokensToProcess = cacheResult.tokensToProcess
                        existingCache = cacheResult.cache
                        
                        if existingCache != nil {
                            logger.info("Using cached prompt prefix, processing \(tokensToProcess.count) new tokens")
                        }
                    }
                    
                    let input = LMInput(tokens: MLXArray(tokensToProcess))
                    
                    let cache = existingCache ?? context.model.newCache(parameters: generateParameters)
                    
                    let iterator = try TokenIterator(
                        input: input,
                        model: context.model,
                        cache: cache,
                        parameters: generateParameters
                    )
                    
                    var allGeneratedTokens: [Int] = []
                    
                    for token in iterator {
                        if token == eosTokenId { break }
                        if tokenCount.value >= parameters.maxTokens { break }
                        
                        if token == tokenizer.unknownTokenId {
                            logger.warning("Generated unknown token ID. Skipping.")
                        } else {
                            continuation.yield(token)
                            tokenCount.increment()
                            allGeneratedTokens.append(token)
                        }
                    }
                    
                    if let cacheManager = promptCacheManager {
                        let fullTokens = promptTokens + allGeneratedTokens
                        await cacheManager.updateCache(
                            modelKey: modelContainer.configuration.name,
                            tokens: fullTokens,
                            kvCaches: cache,
                            parameters: generateParameters,
                            model: context.model
                        )
                    }
                }
                logger.debug("Completion generate function completed.")
                continuation.finish()
            } catch {
                logger.error("Completion token stream error: \(error)")
                continuation.finish()
            }
        }
    }
}

private func handleStreamingTextCompletion(
    baseCompletionId: String,
    modelContainer: ModelContainer,
    tokenizer: Tokenizer,
    eosTokenId: Int,
    promptTokens: [Int],
    parameters: TextCompletionParameters,
    stopIdSequences: [[Int]],
    loadedModelName: String,
    logger: Logger,
    promptCacheManager: PromptCacheManager? = nil
) async throws -> Response {
    let headers = HTTPHeaders([
        ("Content-Type", "text/event-stream"),
        ("Cache-Control", "no-cache"),
        ("Connection", "keep-alive")
    ])
    
    let response = Response(status: .ok, headers: headers)
    response.body = .init(stream: { writer in
        Task {
            var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
            var generatedTokens: [Int] = []
            var finalFinishReason: String?
            
            do {
                logger.info("Starting TEXT stream generation (ID: \(baseCompletionId)) for model \(loadedModelName)")
                let tokenStream = try await generateCompletionTokenStream(
                    modelContainer: modelContainer,
                    tokenizer: tokenizer,
                    eosTokenId: eosTokenId,
                    promptTokens: promptTokens,
                    parameters: parameters,
                    logger: logger,
                    promptCacheManager: promptCacheManager
                )
                
                for try await token in tokenStream {
                    generatedTokens.append(token)
                    detokenizer.append(token: token)
                    
                    let stopCondition = checkStoppingCriteria(
                        tokens: generatedTokens, 
                        stopIdSequences: stopIdSequences, 
                        eosTokenId: eosTokenId
                    )
                    
                    if stopCondition.stopMet {
                        finalFinishReason = "stop"
                        break
                    }

                    if let newTextChunk = detokenizer.next() {
                        let chunkResponse = CompletionChunkResponse(
                            completionId: baseCompletionId,
                            requestedModel: loadedModelName,
                            nextChunk: newTextChunk
                        )
                        if let sseString = encodeSSE(response: chunkResponse, logger: logger) {
                            _ = try writer.write(.buffer(.init(string: sseString)))
                        }
                    }
                }
                
                if finalFinishReason == nil {
                    finalFinishReason = (generatedTokens.count >= parameters.maxTokens) ? "length" : "stop"
                }
            } catch {
                logger.error("Text stream error (ID: \(baseCompletionId)): \(error)")
            }
            
            _ = writer.write(.buffer(.init(string: AppConstants.sseDoneMessage)))
            _ = writer.write(.end)
        }
    })
    
    return response
}

private func handleNonStreamingTextCompletion(
    req: Request,
    baseCompletionId: String,
    modelContainer: ModelContainer,
    tokenizer: Tokenizer,
    eosTokenId: Int,
    promptTokens: [Int],
    parameters: TextCompletionParameters,
    stopIdSequences: [[Int]],
    loadedModelName: String,
    logger: Logger,
    promptCacheManager: PromptCacheManager? = nil
) async throws -> Response {
    var generatedTokens: [Int] = []
    var finalFinishReason = "stop"
    
    do {
        logger.info("Starting non-streaming TEXT generation (ID: \(baseCompletionId)) for model \(loadedModelName).")
        let tokenStream = try await generateCompletionTokenStream(
            modelContainer: modelContainer,
            tokenizer: tokenizer,
            eosTokenId: eosTokenId,
            promptTokens: promptTokens,
            parameters: parameters,
            logger: logger,
            promptCacheManager: promptCacheManager
        )
        
        for try await token in tokenStream {
            generatedTokens.append(token)
            let stopCondition = checkStoppingCriteria(
                tokens: generatedTokens, 
                stopIdSequences: stopIdSequences, 
                eosTokenId: eosTokenId
            )
            
            if stopCondition.stopMet {
                if stopCondition.trimLength > 0 && generatedTokens.count >= stopCondition.trimLength {
                    generatedTokens.removeLast(stopCondition.trimLength)
                }
                finalFinishReason = "stop"
                break
            }
        }
        
        if finalFinishReason != "stop" {
            finalFinishReason = (generatedTokens.count >= parameters.maxTokens) ? "length" : "stop"
        }
    } catch {
        logger.error("Non-streaming text generation error (ID: \(baseCompletionId)): \(error)")
        throw ProcessingError(
            status: .internalServerError, 
            reason: "Failed to generate completion", 
            underlyingError: error
        )
    }
    
    let completionText = decodeTokens(generatedTokens, tokenizer: tokenizer)
    
    let choice = CompletionChoice(text: completionText, finishReason: finalFinishReason)
    let usage = CompletionUsage(
        promptTokens: promptTokens.count,
        completionTokens: generatedTokens.count,
        totalTokens: promptTokens.count + generatedTokens.count
    )
    let completionResponse = CompletionResponse(
        id: baseCompletionId,
        model: loadedModelName,
        choices: [choice],
        usage: usage
    )
    
    return try await completionResponse.encodeResponse(for: req)
}
