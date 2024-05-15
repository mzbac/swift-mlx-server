import ArgumentParser
import MLXLLM
import Foundation
import MLX
import Vapor

func createCompletionChunkResponse(completionId: String, requestedModel: String, nextChunk: String)
-> CompletionChunkResponse
{
    return CompletionChunkResponse(
        completionId: completionId, requestedModel: requestedModel, nextChunk: nextChunk)
}

struct StopCondition {
    let stopMet: Bool
    let trimLength: Int
}

func stoppingCriteria(tokens: [Int], stopIdSequences: [[Int]], eosTokenId: Int) -> StopCondition {
    if !tokens.isEmpty && tokens.last == eosTokenId {
        return StopCondition(stopMet: true, trimLength: 1)
    }
    for stopIds in stopIdSequences {
        if tokens.count >= stopIds.count {
            if tokens.suffix(stopIds.count) == stopIds {
                
                return StopCondition(stopMet: true, trimLength: stopIds.count)
            }
        }
    }
    return StopCondition(stopMet: false, trimLength: 0)
}

func routes(_ app: Application, _ modelPath: String) async throws {
    let modelConfiguration = ModelConfiguration.configuration(
        id: modelPath)
    let (model, tokenizer) = try await load(configuration: modelConfiguration)
    
    @Sendable @MainActor
    func generate_step_stream(
        _ promptTokens: [Int], _ maxTokens: Int, _ temperature: Float, _ topP: Float, _ repetitionPenalty: Float = 1.0, _ repetitionContextSize: Int = 20
    ) async throws -> AsyncStream<Int> {
        return AsyncStream { continuation in
            var tokens = [Int]()
            var generateParameters: GenerateParameters {
                    GenerateParameters(
                        temperature: temperature, topP: topP, repetitionPenalty: repetitionPenalty,
                        repetitionContextSize: repetitionContextSize)
                }
            for token in TokenIterator(
                prompt: MLXArray(promptTokens), model: model, parameters: generateParameters)
            {
                let t = token.item(Int.self)
                if t == tokenizer.unknownTokenId || t == tokenizer.eosTokenId {
                    break
                }
                tokens.append(t)
                continuation.yield(t)
                if tokens.count == maxTokens {
                    break
                }
            }
            continuation.finish()
        }
    }
    app.post("v1", "completions") { req async throws -> Response in
        let completionRequest = try req.content.decode(CompletionRequest.self)
        
        let prompt = modelConfiguration.prepare(prompt: completionRequest.prompt)
        
        let promptTokens = tokenizer.encode(text: prompt)
        
        let maxTokens = completionRequest.maxTokens
        let temperature = completionRequest.temperature
        let topP = completionRequest.topP ?? 1.0
        let requestedModel = completionRequest.model ?? "default_model"
        let stream = completionRequest.stream ?? false
        let stopWords: [String] = completionRequest.stop ?? []
        let stopIdSequences = stopWords.map { Array(tokenizer.encode(text: $0).dropFirst()) }
        let eosTokenId = tokenizer.eosTokenId!
        let repetitionPenalty = completionRequest.repetitionPenalty ?? 1.0
        let repetitionContextSize = completionRequest.repetitionContextSize ?? 20
        
        if stream {
            let headers = HTTPHeaders([
                ("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache"),
            ])
            let response = Response(status: .ok, headers: headers)
            
            response.body = .init(stream: { writer in
                Task {
                    do {
                        let stream_res = try await generate_step_stream(
                            promptTokens, maxTokens, temperature, topP, repetitionPenalty, repetitionContextSize)
                        let maxStopIdSequenceLen = stopIdSequences.map { $0.count }.max() ?? 0
                        var tokens: [Int] = []
                        var currentGeneratedTextIndex = 0
                        var stopSequenceBuffer: [Int] = []
                        let replacementChar = "\u{FFFD}"
                        for try await token in stream_res {
                            tokens.append(token)
                            stopSequenceBuffer.append(token)
                            if stopSequenceBuffer.count > maxStopIdSequenceLen {
                                if tokenizer.decode(tokens: [token]).contains(replacementChar) {
                                    continue
                                }
                                let stopCondition = stoppingCriteria(
                                    tokens: tokens, stopIdSequences: stopIdSequences, eosTokenId: eosTokenId)
                                if stopCondition.stopMet {
                                    if stopCondition.trimLength > 0 {
                                        tokens = Array(tokens.dropLast(stopCondition.trimLength))
                                    }
                                    break
                                }
                                let generatedText = tokenizer.decode(tokens: tokens)
                                let nextChunk = String(
                                    generatedText[
                                        generatedText.index(
                                            generatedText.startIndex, offsetBy: currentGeneratedTextIndex)...])
                                currentGeneratedTextIndex = generatedText.count
                                let chunkResponse = createCompletionChunkResponse(
                                    completionId: "cmpl-\(UUID().uuidString)", requestedModel: requestedModel,
                                    nextChunk: nextChunk)
                                let jsonData = try JSONEncoder().encode(chunkResponse)
                                let jsonString = String(data: jsonData, encoding: .utf8) ?? ""
                                let data = "data: \(jsonString)\n\n"
                                writer.write(.buffer(.init(string: data)))
                            }
                        }
                        if !stopSequenceBuffer.isEmpty {
                            let generatedText = tokenizer.decode(tokens: tokens)
                            let nextChunk = String(
                                generatedText[
                                    generatedText.index(
                                        generatedText.startIndex, offsetBy: currentGeneratedTextIndex)...])
                            let chunkResponse = createCompletionChunkResponse(
                                completionId: "cmpl-\(UUID().uuidString)", requestedModel: requestedModel,
                                nextChunk: nextChunk)
                            let jsonData = try JSONEncoder().encode(chunkResponse)
                            let jsonString = String(data: jsonData, encoding: .utf8) ?? ""
                            let data = "data: \(jsonString)\n\n"
                            writer.write(.buffer(.init(string: data)))
                        }
                        
                        writer.write(.buffer(.init(string: "data: [DONE]\n\n")))
                        writer.write(.end)
                    } catch {
                        let errorMessage = "data: Error generating completion\n\n"
                        writer.write(.buffer(.init(string: errorMessage)))
                        writer.write(.end)
                    }
                }
            })
            
            return response
        } else {
            var generatedTokens: [Int] = []
            
            let stream_res = try await generate_step_stream(promptTokens, maxTokens, temperature, topP, repetitionPenalty, repetitionContextSize)
            for try await value in stream_res {
                generatedTokens.append(value)
                let stopCondition = stoppingCriteria(
                    tokens: generatedTokens, stopIdSequences: stopIdSequences, eosTokenId: eosTokenId)
                if stopCondition.stopMet {
                    print(stopCondition)
                    
                    if stopCondition.trimLength > 0 {
                        generatedTokens = Array(generatedTokens.dropLast(stopCondition.trimLength))
                    }
                    break
                }
            }
            let completionText = tokenizer.decode(tokens: generatedTokens)
            let completionChoice = CompletionChoice(
                text: completionText, index: 0, logprobs: nil, finishReason: "stop")
            let completionUsage = CompletionUsage(
                promptTokens: promptTokens.count, completionTokens: generatedTokens.count,
                totalTokens: promptTokens.count + generatedTokens.count)
            
            let completionResponse = CompletionResponse(
                id: "cmpl-\(UUID().uuidString)", object: "text_completion",
                created: Int(Date().timeIntervalSince1970), model: requestedModel,
                choices: [completionChoice], usage: completionUsage)
            
            let jsonData = try JSONEncoder().encode(completionResponse)
            let jsonString = String(data: jsonData, encoding: .utf8) ?? ""
            
            return Response(status: .ok, body: .init(stringLiteral: jsonString))
        }
    }
}

@main
struct MLXServer: AsyncParsableCommand {
    
    @ArgumentParser.Option(
        name: .long, help: "The path to the MLX model weights, tokenizer, and config")
    var model: String
    
    @ArgumentParser.Option(name: .long, help: "Host for the HTTP server (default: 127.0.0.1)")
    var host: String = "127.0.0.1"
    @ArgumentParser.Option(name: .long, help: "Port for the HTTP server (default: 8080)")
    var port: Int = 8080
    
    @MainActor
    func run() async throws {
        let env = Environment(name: "development", arguments: ["vapor"])
        let app = Application(env)
        defer { app.shutdown() }
        try await routes(app, model)
        app.http.server.configuration.hostname = host
        app.http.server.configuration.port = port
        try await app.execute()
    }
}
