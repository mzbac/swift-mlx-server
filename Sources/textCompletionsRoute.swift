import Vapor
import Foundation
import Logging
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

func registerTextCompletionsRoute(
    _ app: Application,
    modelContainer: ModelContainer,
    tokenizer: Tokenizer,
    eosTokenId: Int,
    loadedModelName: String
) throws {

    @Sendable
    func generateCompletionTokenStream(
        promptTokens: [Int], maxTokens: Int, temperature: Float, topP: Float,
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
                     let input = LMInput(tokens: MLXArray(promptTokens))
                     _ = try await modelContainer.perform { context in
                         try MLXLMCommon.generate(input: input, parameters: generateParameters, context: context) { tokens in
                             guard let lastToken = tokens.last else { return .more }
                             if lastToken == eosTokenId { continuation.finish(); return .stop }
                             guard generatedTokenCount < maxTokens else { continuation.finish(); return .stop }
                             if lastToken == tokenizer.unknownTokenId { logger.warning("Generated unknown token ID. Skipping.") }
                             else { continuation.yield(lastToken); generatedTokenCount += 1 }
                             return .more
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

    app.post("v1", "completions") { req async throws -> Response in
        let completionRequest = try req.content.decode(CompletionRequest.self)
        let logger = req.logger

        let promptTokens = tokenizer.encode(text: completionRequest.prompt)
        let reqModelName = completionRequest.model ?? loadedModelName
        logger.info("Received TEXT completion request for model '\(reqModelName)', prompt tokens: \(promptTokens.count)")

        let maxTokens = completionRequest.maxTokens ?? GenerationDefaults.maxTokens
        let temperature = completionRequest.temperature ?? GenerationDefaults.temperature
        let topP = completionRequest.topP ?? GenerationDefaults.topP
        let streamResponse = completionRequest.stream ?? GenerationDefaults.stream
        let stopWords = completionRequest.stop ?? GenerationDefaults.stopSequences
        let stopIdSequences = stopWords.compactMap { word in tokenizer.encode(text: word, addSpecialTokens: false).nilIfEmpty() }
        let repetitionPenalty = completionRequest.repetitionPenalty ?? GenerationDefaults.repetitionPenalty
        let repetitionContextSize = completionRequest.repetitionContextSize ?? GenerationDefaults.repetitionContextSize

        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        if streamResponse {
            let headers = HTTPHeaders([
                ("Content-Type", "text/event-stream"),
                ("Cache-Control", "no-cache"),
                ("Connection", "keep-alive"),
            ])
            let response = Response(status: .ok, headers: headers)
            response.body = .init(stream: { writer in
                Task {
                    let completionId = "cmpl-\(UUID().uuidString)"
                    var generatedTokens: [Int] = []
                    var finalFinishReason: String? = nil
                    do {
                        logger.info("Starting TEXT stream generation (ID: \(completionId))")
                        let tokenStream = try await generateCompletionTokenStream(
                            promptTokens: promptTokens, maxTokens: maxTokens, temperature: temperature, topP: topP,
                            repetitionPenalty: repetitionPenalty, repetitionContextSize: repetitionContextSize, logger: logger
                        )
                        for try await token in tokenStream {
                             generatedTokens.append(token)
                             detokenizer.append(token: token)
                             let stopCondition = checkStoppingCriteria(tokens: generatedTokens, stopIdSequences: stopIdSequences, eosTokenId: eosTokenId)
                             if stopCondition.stopMet { finalFinishReason = "stop"; break }

                            if let newTextChunk = detokenizer.next() {
                                let chunkResponse = CompletionChunkResponse(completionId: completionId, requestedModel: reqModelName, nextChunk: newTextChunk)
                                if let sseString = encodeSSE(response: chunkResponse, logger: logger) {
                                    try await writer.write(.buffer(.init(string: sseString)))
                                }
                            }
                        }
                        if finalFinishReason == nil { finalFinishReason = (generatedTokens.count >= maxTokens) ? "length" : "stop" }
                     } catch { logger.error("Text stream error (ID: \(completionId)): \(error)") }
                    await writer.write(.buffer(.init(string: AppConstants.sseDoneMessage)))
                    await writer.write(.end)
                }
            })
            return response
        } else {
             var generatedTokens: [Int] = []
             var finalFinishReason = "stop"
             do {
                logger.info("Starting non-streaming TEXT generation.")
                let tokenStream = try await generateCompletionTokenStream(
                    promptTokens: promptTokens, maxTokens: maxTokens, temperature: temperature, topP: topP,
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
                 logger.error("Non-streaming text generation error: \(error)")
                 throw Abort(.internalServerError, reason: "Failed to generate completion: \(error.localizedDescription)")
             }
             let completionText = tokenizer.decode(tokens: generatedTokens)
             let choice = CompletionChoice(text: completionText, finishReason: finalFinishReason)
             let usage = CompletionUsage(promptTokens: promptTokens.count, completionTokens: generatedTokens.count, totalTokens: promptTokens.count + generatedTokens.count)
             let completionResponse = CompletionResponse(model: reqModelName, choices: [choice], usage: usage)
             return try await completionResponse.encodeResponse(for: req)
        }
    }
}