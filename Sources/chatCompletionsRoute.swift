import Vapor
import Foundation
import Logging
import MLX
import MLXLLM
import MLXLMCommon
import Hub
import Tokenizers

func registerChatCompletionsRoute(
    _ app: Application,
    modelContainer: ModelContainer,
    tokenizer: Tokenizer,
    eosTokenId: Int,
    loadedModelName: String
) throws {

    @Sendable
    func generateChatTokenStream(
        userInput: UserInput, maxTokens: Int, temperature: Float, topP: Float,
        repetitionPenalty: Float, repetitionContextSize: Int, logger: Logger
    ) async throws -> AsyncStream<Int> {
         return AsyncStream { continuation in
             Task {
                 var generatedTokenCount = 0
                 do {
                     let generateParameters = GenerateParameters(
                         temperature: temperature, topP: topP,
                         repetitionPenalty: repetitionPenalty, repetitionContextSize: repetitionContextSize
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
                             if lastToken == tokenizer.unknownTokenId { logger.warning("Generated unknown token ID \(lastToken). Skipping.") }
                             else { continuation.yield(lastToken); generatedTokenCount += 1 }
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

    app.post("v1", "chat", "completions") { req async throws -> Response in
        let chatRequest = try req.content.decode(ChatCompletionRequest.self)
        let logger = req.logger

        let messages: [Message] = chatRequest.messages.map {
            ["role": $0.role, "content": $0.content ?? ""]
        }
        let userInput = UserInput(messages: messages)

        let combinedContent = chatRequest.messages.compactMap { $0.content }.joined(separator: "\n")
        let estimatedPromptTokens = tokenizer.encode(text: combinedContent).count
        let reqModelName = chatRequest.model ?? loadedModelName

        logger.info("Received CHAT completion request for model '\(reqModelName)', estimated prompt tokens: \(estimatedPromptTokens)")

        let maxTokens = chatRequest.maxTokens ?? GenerationDefaults.maxTokens
        let temperature = chatRequest.temperature ?? GenerationDefaults.temperature
        let topP = chatRequest.topP ?? GenerationDefaults.topP
        let streamResponse = chatRequest.stream ?? GenerationDefaults.stream
        let stopWords = chatRequest.stop ?? GenerationDefaults.stopSequences
        let stopIdSequences = stopWords.compactMap { word in tokenizer.encode(text: word, addSpecialTokens: false).nilIfEmpty() }
        let repetitionPenalty = chatRequest.repetitionPenalty ?? GenerationDefaults.repetitionPenalty
        let repetitionContextSize = chatRequest.repetitionContextSize ?? GenerationDefaults.repetitionContextSize

        let hasProcessor = await modelContainer.perform { $0.processor is UserInputProcessor }
        guard hasProcessor else { throw Abort(.internalServerError, reason: "Model processor invalid for chat input.") }

        if streamResponse {
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

                Task {
                    var generatedTokens: [Int] = []
                    var currentSentTextIndex = 0
                    var finalFinishReason: String? = nil

                    do {
                        logger.info("Starting CHAT stream generation (ID: \(chatId))")

                        let initialDelta = ChatCompletionDelta(role: "assistant", content: "")
                        let initialChoice = ChatCompletionChoiceDelta(index: 0, delta: initialDelta, finishReason: nil)
                        let initialChunk = ChatCompletionChunkResponse(id: chatId, created: created, model: reqModelName, systemFingerprint: systemFingerprint, choices: [initialChoice])
                        if let initialSse = encodeChatSSE(chunkResponse: initialChunk, logger: logger) {
                            try await writer.write(.buffer(.init(string: initialSse)))
                        }

                        let tokenStream = try await generateChatTokenStream(
                            userInput: userInput, maxTokens: maxTokens, temperature: temperature, topP: topP,
                            repetitionPenalty: repetitionPenalty, repetitionContextSize: repetitionContextSize, logger: logger
                        )

                        for try await token in tokenStream {
                             generatedTokens.append(token)
                             let stopCondition = checkStoppingCriteria(tokens: generatedTokens, stopIdSequences: stopIdSequences, eosTokenId: eosTokenId)
                             if stopCondition.stopMet { finalFinishReason = "stop"; break }

                             let decodedText = tokenizer.decode(tokens: generatedTokens)
                             if decodedText.count > currentSentTextIndex {
                                 let startIndex = decodedText.index(decodedText.startIndex, offsetBy: currentSentTextIndex)
                                 let newTextChunk = String(decodedText[startIndex...])
                                 if !newTextChunk.isEmpty && !newTextChunk.allSatisfy({ $0.unicodeScalars.first?.value == AppConstants.replacementChar.unicodeScalars.first?.value }) {
                                     let delta = ChatCompletionDelta(role: nil, content: newTextChunk)
                                     let choice = ChatCompletionChoiceDelta(index: 0, delta: delta, finishReason: nil)
                                     let chunkResponse = ChatCompletionChunkResponse(id: chatId, created: created, model: reqModelName, systemFingerprint: systemFingerprint, choices: [choice])
                                     if let sseString = encodeChatSSE(chunkResponse: chunkResponse, logger: logger) {
                                         try await writer.write(.buffer(.init(string: sseString)))
                                         currentSentTextIndex = decodedText.count
                                     }
                                 }
                             }
                        }

                        if finalFinishReason == nil { finalFinishReason = (generatedTokens.count >= maxTokens) ? "length" : "stop" }

                        let finalDelta = ChatCompletionDelta(role: nil, content: nil)
                        let finalChoice = ChatCompletionChoiceDelta(index: 0, delta: finalDelta, finishReason: finalFinishReason)
                        let finalChunk = ChatCompletionChunkResponse(id: chatId, created: created, model: reqModelName, systemFingerprint: systemFingerprint, choices: [finalChoice])
                        if let finalSseString = encodeChatSSE(chunkResponse: finalChunk, logger: logger) {
                            await writer.write(.buffer(.init(string: finalSseString)))
                        }

                    } catch { logger.error("Chat stream error (ID: \(chatId)): \(error)"); finalFinishReason = "error" }
                    await writer.write(.buffer(.init(string: AppConstants.sseDoneMessage)))
                    logger.info("CHAT Streaming response finished sending (ID: \(chatId)). Final Reason: \(finalFinishReason ?? "unknown")")
                    await writer.write(.end)
                }
            })
            return response
        } else {
            var generatedTokens: [Int] = []
            var finalFinishReason = "stop"
            let responseId = "chatcmpl-\(UUID().uuidString)"
            let created = Int(Date().timeIntervalSince1970)

            do {
                logger.info("Starting non-streaming CHAT generation (ID: \(responseId)).")
                let tokenStream = try await generateChatTokenStream(
                    userInput: userInput, maxTokens: maxTokens, temperature: temperature, topP: topP,
                    repetitionPenalty: repetitionPenalty, repetitionContextSize: repetitionContextSize, logger: logger
                )
                for try await token in tokenStream {
                    generatedTokens.append(token)
                    let stopCondition = checkStoppingCriteria(tokens: generatedTokens, stopIdSequences: stopIdSequences, eosTokenId: eosTokenId)
                     if stopCondition.stopMet {
                         if stopCondition.trimLength > 0 && generatedTokens.count >= stopCondition.trimLength { generatedTokens.removeLast(stopCondition.trimLength) }
                         finalFinishReason = "stop"; break
                     }
                }
                if finalFinishReason != "stop" { finalFinishReason = (generatedTokens.count >= maxTokens) ? "length" : "stop" }
            } catch {
                 logger.error("Non-streaming chat generation error (ID: \(responseId)): \(error)")
                 throw Abort(.internalServerError, reason: "Failed to generate chat completion: \(error.localizedDescription)")
            }
            let completionText = tokenizer.decode(tokens: generatedTokens)
            let assistantMessage = ChatMessageResponseData(role: "assistant", content: completionText)
            let chatChoice = ChatCompletionChoice(index: 0, message: assistantMessage, finishReason: finalFinishReason)
            let usage = CompletionUsage(promptTokens: estimatedPromptTokens, completionTokens: generatedTokens.count, totalTokens: estimatedPromptTokens + generatedTokens.count)
            let chatResponse = ChatCompletionResponse(id: responseId, created: created, model: reqModelName, choices: [chatChoice], usage: usage)

            logger.info("Non-streaming CHAT response generated (ID: \(responseId)). Reason: \(finalFinishReason)")
            return try await chatResponse.encodeResponse(for: req)
        }
    }
}
