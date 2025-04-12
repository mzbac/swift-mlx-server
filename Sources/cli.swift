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
}

enum GenerationDefaults {
    static let maxTokens = 128
    static let temperature: Float = 0.8
    static let topP: Float = 1.0
    static let stream = false
    static let repetitionPenalty: Float = 1.0
    static let repetitionContextSize = 20
    static let stopSequences: [String] = []
}

struct StopCondition {
    let stopMet: Bool
    let trimLength: Int
}

func checkStoppingCriteria(tokens: [Int], stopIdSequences: [[Int]], eosTokenId: Int) -> StopCondition {
    guard let lastToken = tokens.last else { return StopCondition(stopMet: false, trimLength: 0) }
    if lastToken == eosTokenId { return StopCondition(stopMet: true, trimLength: 1) }
    for stopIds in stopIdSequences {
        guard !stopIds.isEmpty else { continue }
        if tokens.count >= stopIds.count, tokens.suffix(stopIds.count) == stopIds { return StopCondition(stopMet: true, trimLength: stopIds.count) }
    }
    return StopCondition(stopMet: false, trimLength: 0)
}

func encodeSSE<T: Encodable>(response: T, logger: Logger) -> String? {
    do {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let jsonData = try encoder.encode(response)
        guard let jsonString = String(data: jsonData, encoding: .utf8) else {
            logger.error("Failed encodeSSE utf8")
            return nil
        }
        return AppConstants.sseEventHeader + jsonString + AppConstants.sseEventSeparator
    } catch {
        logger.error("Failed encodeSSE: \(error)")
        return nil
    }
}

func routes(_ app: Application, _ modelPath: String) async throws {
    let modelContainer: ModelContainer
    let loadedModelName: String

    do {
        let modelFactory = LLMModelFactory.shared
        let modelConfiguration: ModelConfiguration
        if modelPath.hasPrefix("/") { modelConfiguration = ModelConfiguration(directory: URL(filePath: modelPath)); loadedModelName = modelConfiguration.name }
        else { modelConfiguration = modelFactory.configuration(id: modelPath); loadedModelName = modelConfiguration.name }
        modelContainer = try await modelFactory.loadContainer(configuration: modelConfiguration)
        app.logger.info("MLX Model loaded successfully. Identifier: \(loadedModelName)")
    } catch {
        app.logger.critical("Failed to load MLX model from \(modelPath): \(error)")
        throw Abort(.internalServerError, reason: "Failed to load MLX model: \(error.localizedDescription)")
    }

    let tokenizer = try await modelContainer.perform { $0.tokenizer }
    guard let eosTokenId = tokenizer.eosTokenId else { throw Abort(.internalServerError, reason: "Tokenizer EOS token ID missing.") }

    try registerTextCompletionsRoute(
        app,
        modelContainer: modelContainer,
        tokenizer: tokenizer,
        eosTokenId: eosTokenId,
        loadedModelName: loadedModelName
    )

    try registerChatCompletionsRoute(
        app,
        modelContainer: modelContainer,
        tokenizer: tokenizer,
        eosTokenId: eosTokenId,
        loadedModelName: loadedModelName
    )
}

@main
struct MLXServer: AsyncParsableCommand {
    @ArgumentParser.Option(name: .long, help: "Required: Path to MLX model dir/name.")
    var model: String
    @ArgumentParser.Option(name: .long, help: "Host address.")
    var host: String = AppConstants.defaultHost
    @ArgumentParser.Option(name: .long, help: "Port number.")
    var port: Int = AppConstants.defaultPort

    @MainActor
    func run() async throws {
        let envName = ProcessInfo.processInfo.environment["MLX_ENV"] ?? "production"
        var env = Environment(name: envName, arguments: ["vapor"])
        try LoggingSystem.bootstrap(from: &env)
        let app = Application(env)
        defer { app.logger.info("Server shutting down."); app.shutdown() }

        let corsConfiguration = CORSMiddleware.Configuration(
            allowedOrigin: .all,
            allowedMethods: [.GET, .POST, .OPTIONS],
            allowedHeaders: [
                .accept, .authorization, .contentType, .origin, 
                .xRequestedWith, .userAgent, .accessControlAllowOrigin
            ]
        )
        let corsMiddleware = CORSMiddleware(configuration: corsConfiguration)
        app.middleware.use(corsMiddleware)

        try await routes(app, model)

        app.http.server.configuration.hostname = host
        app.http.server.configuration.port = port
        do {
            app.logger.info("Server starting on http://\(host):\(port)")
            app.logger.info("Using model identifier: \(model)")
            try await app.execute()
        } catch { app.logger.critical("Server error: \(error)"); throw error }
    }
}

extension Array { func nilIfEmpty() -> Self? { self.isEmpty ? nil : self } }
