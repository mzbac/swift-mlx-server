import Foundation
import Logging
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers
import Vapor

private struct TextCompletionError: AbortError {
    var status: HTTPResponseStatus
    var reason: String
    var identifier: String?

    init(status: HTTPResponseStatus, reason: String, modelName: String? = nil, underlyingError: Error? = nil) {
        self.status = status
        var fullReason = reason
        if let modelName = modelName, !modelName.isEmpty {
            fullReason += " (Model: \(modelName))"
        }
        if let underlyingError = underlyingError {
            fullReason += ". Underlying error: \(underlyingError.localizedDescription)"
        }
        self.reason = fullReason
        self.identifier = modelName
    }
}

func registerTextCompletionsRoute(
  _ app: Application,
  modelProvider: ModelProvider
) throws {

  @Sendable
  func generateCompletionTokenStream(
    modelContainer: ModelContainer,
    tokenizer: Tokenizer,
    eosTokenId: Int,
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
            try MLXLMCommon.generate(input: input, parameters: generateParameters, context: context)
            { tokens in
              guard let lastToken = tokens.last else { return .more }
              if lastToken == eosTokenId {
                continuation.finish()
                return .stop
              }
              guard generatedTokenCount < maxTokens else {
                continuation.finish()
                return .stop
              }
              if lastToken == tokenizer.unknownTokenId {
                logger.warning("Generated unknown token ID. Skipping.")
              } else {
                continuation.yield(lastToken)
                generatedTokenCount += 1
              }
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
    let reqModelName = completionRequest.model
    let baseCompletionId = "cmpl-\(UUID().uuidString)"

    let (modelContainer, tokenizer, loadedModelName) = try await modelProvider.getModel(
      requestedModelId: reqModelName)
    guard let eosTokenId = tokenizer.eosTokenId else {
      throw TextCompletionError(status: .internalServerError, reason: "Tokenizer EOS token ID missing", modelName: loadedModelName)
    }
    let promptTokens = tokenizer.encode(text: completionRequest.prompt)
    logger.info(
      "Received TEXT completion request (ID: \(baseCompletionId)) for model '\(reqModelName ?? "default")', prompt tokens: \(promptTokens.count)"
    )

    let maxTokens = completionRequest.maxTokens ?? GenerationDefaults.maxTokens
    let temperature = completionRequest.temperature ?? GenerationDefaults.temperature
    let topP = completionRequest.topP ?? GenerationDefaults.topP
    let streamResponse = completionRequest.stream ?? GenerationDefaults.stream
    let stopWords = completionRequest.stop ?? GenerationDefaults.stopSequences
    let stopIdSequences = stopWords.compactMap { word in
      tokenizer.encode(text: word, addSpecialTokens: false).nilIfEmpty()
    }
    let repetitionPenalty =
      completionRequest.repetitionPenalty ?? GenerationDefaults.repetitionPenalty
    let repetitionContextSize =
      completionRequest.repetitionContextSize ?? GenerationDefaults.repetitionContextSize

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
          let streamCompletionId = baseCompletionId
          var generatedTokens: [Int] = []
          var finalFinishReason: String? = nil
          do {
            logger.info(
              "Starting TEXT stream generation (ID: \(streamCompletionId)) for model \(loadedModelName)")
            let tokenStream = try await generateCompletionTokenStream(
              modelContainer: modelContainer,
              tokenizer: tokenizer,
              eosTokenId: eosTokenId,
              promptTokens: promptTokens, maxTokens: maxTokens, temperature: temperature,
              topP: topP,
              repetitionPenalty: repetitionPenalty, repetitionContextSize: repetitionContextSize,
              logger: logger
            )
            for try await token in tokenStream {
              generatedTokens.append(token)
              detokenizer.append(token: token)
              let stopCondition = checkStoppingCriteria(
                tokens: generatedTokens, stopIdSequences: stopIdSequences, eosTokenId: eosTokenId)
              if stopCondition.stopMet {
                finalFinishReason = "stop"
                break
              }

              if let newTextChunk = detokenizer.next() {
                let chunkResponse = CompletionChunkResponse(
                  completionId: baseCompletionId, requestedModel: loadedModelName,
                  nextChunk: newTextChunk)
                if let sseString = encodeSSE(response: chunkResponse, logger: logger) {
                  try await writer.write(.buffer(.init(string: sseString)))
                }
              }
            }
            if finalFinishReason == nil {
              finalFinishReason = (generatedTokens.count >= maxTokens) ? "length" : "stop"
            }
          } catch { logger.error("Text stream error (ID: \(streamCompletionId)): \(error)") }
          await writer.write(.buffer(.init(string: AppConstants.sseDoneMessage)))
          await writer.write(.end)
        }
      })
      return response
    } else {
      let nonStreamCompletionId = baseCompletionId
      var generatedTokens: [Int] = []
      var finalFinishReason = "stop"
      do {
        logger.info("Starting non-streaming TEXT generation (ID: \(nonStreamCompletionId)) for model \(loadedModelName).")
        let tokenStream = try await generateCompletionTokenStream(
          modelContainer: modelContainer,
          tokenizer: tokenizer,
          eosTokenId: eosTokenId,
          promptTokens: promptTokens, maxTokens: maxTokens, temperature: temperature, topP: topP,
          repetitionPenalty: repetitionPenalty, repetitionContextSize: repetitionContextSize,
          logger: logger
        )
        for try await token in tokenStream {
          generatedTokens.append(token)
          let stopCondition = checkStoppingCriteria(
            tokens: generatedTokens, stopIdSequences: stopIdSequences, eosTokenId: eosTokenId)
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
        logger.error("Non-streaming text generation error (ID: \(nonStreamCompletionId)): \(error)")
        throw TextCompletionError(status: .internalServerError, reason: "Failed to generate completion", underlyingError: error)
      }
      let completionText = tokenizer.decode(tokens: generatedTokens)
      let choice = CompletionChoice(text: completionText, finishReason: finalFinishReason)
      let usage = CompletionUsage(
        promptTokens: promptTokens.count, completionTokens: generatedTokens.count,
        totalTokens: promptTokens.count + generatedTokens.count)
      let completionResponse = CompletionResponse(
        id: nonStreamCompletionId,
        model: loadedModelName, choices: [choice], usage: usage)
      return try await completionResponse.encodeResponse(for: req)
    }
  }
}
