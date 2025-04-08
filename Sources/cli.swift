import ArgumentParser
import Foundation
import Logging
import MLX
import MLXLLM
import MLXLMCommon
import Hub
import Tokenizers
import Vapor

enum AppConstants {
  static let defaultHost = "127.0.0.1"
  static let defaultPort = 8080
  static let replacementChar = "\u{FFFD}"
  static let sseDoneMessage = "data: [DONE]\n\n"
  static let sseEventHeader = "data: "
  static let sseEventSeparator = "\n\n"
  static let defaultModelName = "default_model"
}

struct StopCondition {
  let stopMet: Bool
  let trimLength: Int
}

func checkStoppingCriteria(tokens: [Int], stopIdSequences: [[Int]], eosTokenId: Int)
  -> StopCondition
{
  guard let lastToken = tokens.last else {
    return StopCondition(stopMet: false, trimLength: 0)
  }
  if lastToken == eosTokenId {
    return StopCondition(stopMet: true, trimLength: 1)
  }
  for stopIds in stopIdSequences {
    guard !stopIds.isEmpty else { continue }
    if tokens.count >= stopIds.count, tokens.suffix(stopIds.count) == stopIds {
      return StopCondition(stopMet: true, trimLength: stopIds.count)
    }
  }
  return StopCondition(stopMet: false, trimLength: 0)
}

func encodeSSE(chunkResponse: CompletionChunkResponse, logger: Logger) -> String? {
  do {
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    let jsonData = try encoder.encode(chunkResponse)
    guard let jsonString = String(data: jsonData, encoding: .utf8) else {
      logger.error("Failed to convert chunk JSON data to UTF8 string.")
      return nil
    }
    return AppConstants.sseEventHeader + jsonString + AppConstants.sseEventSeparator
  } catch {
    logger.error("Failed to encode chunk response to JSON: \(error)")
    return nil
  }
}

func routes(_ app: Application, _ modelPath: String) async throws {
  let modelContainer: ModelContainer

  do {
    let modelFactory = LLMModelFactory.shared
    let modelConfiguration: ModelConfiguration
    
    if modelPath.hasPrefix("/") {
      modelConfiguration = ModelConfiguration(directory: URL(filePath: modelPath))
    } else {
      modelConfiguration = modelFactory.configuration(id: modelPath)
    }
    
    modelContainer = try await modelFactory.loadContainer(configuration: modelConfiguration)    
    app.logger.info("MLX Model loaded successfully from path: \(modelPath)")
  } catch {
    app.logger.critical("Failed to load MLX model from \(modelPath): \(error)")
    throw Abort(
      .internalServerError, reason: "Failed to load MLX model: \(error.localizedDescription)")
  }
  
  let tokenizer = try await modelContainer.perform { context in
    return context.tokenizer
  }
  
  guard let eosTokenId = tokenizer.eosTokenId else {
    app.logger.critical("Tokenizer does not have an EOS token ID configured. Cannot proceed.")
    throw Abort(
      .internalServerError, reason: "Tokenizer configuration error: Missing EOS token ID.")
  }

  let modelConfiguration = await modelContainer.configuration

  @Sendable
  func generateTokenStream(
    promptTokens: [Int], maxTokens: Int, temperature: Float, topP: Float,
    repetitionPenalty: Float, repetitionContextSize: Int, logger: Logger
  ) async throws -> AsyncStream<Int> {
    return AsyncStream { continuation in
      Task {
        do {
          let generateParameters = GenerateParameters(
            temperature: temperature, topP: topP,
            repetitionPenalty: repetitionPenalty, repetitionContextSize: repetitionContextSize
          )
          
          let input = LMInput(tokens: MLXArray(promptTokens))
          var generatedTokenCount = 0
          
          _ = try await modelContainer.perform { context in
            try MLXLMCommon.generate(input: input, parameters: generateParameters, context: context) { tokens in
              if let lastToken = tokens.last {
                guard generatedTokenCount < maxTokens else {
                  logger.debug("Max tokens (\(maxTokens)) reached.")
                  continuation.finish()
                  return .stop
                }
                
                if lastToken == tokenizer.unknownTokenId {
                  logger.warning("Generated unknown token ID. Skipping.")
                } else {
                  continuation.yield(lastToken)
                  generatedTokenCount += 1
                
                  if lastToken == eosTokenId {
                    logger.debug("EOS token (\(eosTokenId)) generated.")
                    continuation.finish()
                    return .stop
                  }
                }
              }
              
              return .more
            }
          }
          
          continuation.finish()
        } catch {
          logger.error("Error during token generation: \(error)")
          continuation.finish()
        }
      }
    }
  }

  app.post("v1", "completions") { req async throws -> Response in
    let completionRequest = try req.content.decode(CompletionRequest.self)
    let logger = req.logger

    let promptTokens = tokenizer.encode(text: completionRequest.prompt)
    logger.info(
      "Received completion request for model '\(completionRequest.model ?? "unspecified")', prompt tokens: \(promptTokens.count)"
    )

    let maxTokens = completionRequest.maxTokens ?? GenerationDefaults.maxTokens
    let temperature = completionRequest.temperature ?? GenerationDefaults.temperature
    let topP = completionRequest.topP ?? GenerationDefaults.topP
    let requestedModelName = completionRequest.model ?? AppConstants.defaultModelName
    let streamResponse = completionRequest.stream ?? GenerationDefaults.stream
    let stopWords = completionRequest.stop ?? GenerationDefaults.stopSequences
    let stopIdSequences = stopWords.compactMap { word -> [Int]? in
      let encoded = tokenizer.encode(text: word)
      return encoded.count > 1 ? Array(encoded.dropFirst()) : nil
    }
    logger.debug("Using stop sequences: \(stopIdSequences)")
    let repetitionPenalty =
      completionRequest.repetitionPenalty ?? GenerationDefaults.repetitionPenalty
    let repetitionContextSize =
      completionRequest.repetitionContextSize ?? GenerationDefaults.repetitionContextSize

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
          var currentSentTextIndex = 0
          var finalFinishReason: String? = nil
          do {
            logger.info("Starting stream generation (ID: \(completionId))")
            let tokenStream = try await generateTokenStream(
              promptTokens: promptTokens,
              maxTokens: maxTokens,
              temperature: temperature,
              topP: topP,
              repetitionPenalty: repetitionPenalty,
              repetitionContextSize: repetitionContextSize,
              logger: logger
            )
            for try await token in tokenStream {
              generatedTokens.append(token)
              let decodedText = tokenizer.decode(tokens: generatedTokens)
              if decodedText.count > currentSentTextIndex {
                let newTextChunk = String(
                  decodedText.suffix(
                    from: decodedText.index(decodedText.startIndex, offsetBy: currentSentTextIndex))
                )
                if !newTextChunk.isEmpty
                  && !newTextChunk.allSatisfy({
                    $0.unicodeScalars.first == Unicode.Scalar(AppConstants.replacementChar)!
                  })
                {
                  let chunkResponse = CompletionChunkResponse(
                    completionId: completionId, requestedModel: requestedModelName,
                    nextChunk: newTextChunk)
                  if let sseString = encodeSSE(chunkResponse: chunkResponse, logger: logger) {
                    await writer.write(.buffer(.init(string: sseString)))
                    currentSentTextIndex = decodedText.count
                  } else {
                    logger.error("Failed to encode or write SSE chunk string.")
                  }
                }
              }
              let stopCondition = checkStoppingCriteria(
                tokens: generatedTokens, stopIdSequences: stopIdSequences, eosTokenId: eosTokenId)
              if stopCondition.stopMet {
                logger.info("Stop condition met (stream ID: \(completionId)).")
                finalFinishReason = "stop"
                break
              }
            }
            if finalFinishReason == nil && generatedTokens.count >= maxTokens {
              finalFinishReason = "length"
              logger.info("Max tokens reached (stream ID: \(completionId)).")
            }
            await writer.write(.buffer(.init(string: AppConstants.sseDoneMessage)))
            logger.info(
              "Streaming response finished successfully (ID: \(completionId)). Finish reason: \(finalFinishReason ?? "unknown")"
            )
          } catch {
            logger.error("Error during token generation stream (ID: \(completionId)): \(error)")
            let errorPayload =
              #"{"error": {"message": "Error generating completion stream.", "type": "generation_error"}}"#
            let sseError =
              AppConstants.sseEventHeader + errorPayload + AppConstants.sseEventSeparator
            await writer.write(.buffer(.init(string: sseError)))
            await writer.write(.buffer(.init(string: AppConstants.sseDoneMessage)))
          }
          await writer.write(.end)
        }
      })
      return response
    } else {
      var generatedTokens: [Int] = []
      var finalFinishReason = "stop"
      do {
        logger.info("Starting non-streaming generation.")
        let tokenStream = try await generateTokenStream(
          promptTokens: promptTokens,
          maxTokens: maxTokens,
          temperature: temperature,
          topP: topP,
          repetitionPenalty: repetitionPenalty,
          repetitionContextSize: repetitionContextSize,
          logger: logger
        )
        for try await token in tokenStream {
          generatedTokens.append(token)
          let stopCondition = checkStoppingCriteria(
            tokens: generatedTokens, stopIdSequences: stopIdSequences, eosTokenId: eosTokenId)
          if stopCondition.stopMet {
            logger.info(
              "Stop condition met (non-stream). Trimming \(stopCondition.trimLength) tokens from final output."
            )
            if stopCondition.trimLength > 0 { generatedTokens.removeLast(stopCondition.trimLength) }
            finalFinishReason = "stop"
            break
          }
        }
        if finalFinishReason == "stop" && generatedTokens.count >= maxTokens {
          logger.info("Max tokens (\(maxTokens)) reached (non-stream).")
          finalFinishReason = "length"
        }
      } catch {
        logger.error("Error during non-streaming token generation: \(error)")
        throw Abort(
          .internalServerError,
          reason: "Failed to generate completion: \(error.localizedDescription)")
      }
      let completionText = tokenizer.decode(tokens: generatedTokens)
      let completionChoice = CompletionChoice(text: completionText, finishReason: finalFinishReason)
      let completionUsage = CompletionUsage(
        promptTokens: promptTokens.count, completionTokens: generatedTokens.count,
        totalTokens: promptTokens.count + generatedTokens.count)
      let completionResponse = CompletionResponse(
        model: requestedModelName, choices: [completionChoice], usage: completionUsage)
      logger.info(
        "Non-streaming response generated successfully. Finish reason: \(finalFinishReason)")
      return try await completionResponse.encodeResponse(for: req)
    }
  }
}

@main
struct MLXServer: AsyncParsableCommand {

  @ArgumentParser.Option(
    name: .long,
    help: "Required: Path to the MLX model directory (containing weights, tokenizer, config) or a Hugging Face model name.")
  var model: String

  @ArgumentParser.Option(name: .long, help: "Host address for the HTTP server.")
  var host: String = AppConstants.defaultHost

  @ArgumentParser.Option(name: .long, help: "Port number for the HTTP server.")
  var port: Int = AppConstants.defaultPort

  @MainActor
  func run() async throws {
    var env = Environment(name: "development", arguments: ["vapor"])
    try LoggingSystem.bootstrap(from: &env)
    let app = Application(env)
    defer {
      app.logger.info("Server shutting down.")
      app.shutdown()
    }

    try await routes(app, model)

    app.http.server.configuration.hostname = host
    app.http.server.configuration.port = port

    do {
      app.logger.info("Server starting on http://\(host):\(port)")
      app.logger.info("Using model: \(model)")
      try await app.execute()
    } catch {
      app.logger.critical("Server error: \(error)")
      throw error
    }
  }
}
