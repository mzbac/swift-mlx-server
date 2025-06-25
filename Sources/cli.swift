import ArgumentParser
import Foundation
import Hub
import Logging
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers
import Vapor
import mlx_embeddings

struct ModelInfo: Content {
    let id: String
    var object: String = "model"
    let created = Int(Date().timeIntervalSince1970)
    let ownedBy: String = "user"

    enum CodingKeys: String, CodingKey {
        case id, object, created
        case ownedBy = "owned_by"
    }
}

struct ModelListResponse: Content {
    var object: String = "list"
    let data: [ModelInfo]
}

func configureRoutes(
    _ app: Application, 
    modelProvider: ModelProvider, 
    embeddingModelProvider: EmbeddingModelProvider,
    isVLM: Bool,
    promptCacheManager: PromptCacheManager?
) async throws {
    try registerTextCompletionsRoute(app, modelProvider: modelProvider, promptCacheManager: promptCacheManager)
    try registerChatCompletionsRoute(app, modelProvider: modelProvider, isVLM: isVLM, promptCacheManager: promptCacheManager)
    registerEmbeddingsRoute(app, embeddingModelProvider: embeddingModelProvider)
    registerModelsRoute(app, modelProvider: modelProvider)
    registerCacheManagementRoutes(app, promptCacheManager: promptCacheManager)
}

private func registerModelsRoute(_ app: Application, modelProvider: ModelProvider) {
    app.get("v1", "models") { req async throws -> ModelListResponse in
        req.logger.info("Handling /v1/models request")
        let modelIds = await modelProvider.getAvailableModelIDs()
        let modelInfos = modelIds.map { ModelInfo(id: $0) }
        return ModelListResponse(data: modelInfos)
    }
}

@main
struct MLXServer: AsyncParsableCommand {
    @ArgumentParser.Option(name: .long, help: "Required: Path to MLX model dir/name.")
    var model: String

    @ArgumentParser.Option(name: .long, help: "Host address.")
    var host: String = AppConstants.defaultHost

    @ArgumentParser.Option(name: .long, help: "Port number.")
    var port: Int = AppConstants.defaultPort

    @ArgumentParser.Flag(name: .long, help: "Enable multi-modal processing for visual language models.")
    var vlm: Bool = false

    @ArgumentParser.Option(name: .long, help: "Path or identifier for embedding model.")
    var embeddingModel: String?
    
    @ArgumentParser.Flag(name: .long, help: "Enable prompt caching to reuse KV caches for common prefixes.")
    var enablePromptCache: Bool = false
    
    @ArgumentParser.Option(name: .long, help: "Maximum prompt cache size in MB (default: 1024).")
    var promptCacheSizeMB: Int = 1_024
    
    @ArgumentParser.Option(name: .long, help: "Prompt cache TTL in minutes (default: 30).")
    var promptCacheTTLMinutes: Int = 30

    @MainActor
    func run() async throws {
        let app = try await setupApplication()
        defer { shutdownApplication(app) }

        let providers = setupProviders(app: app)
        let promptCacheManager = enablePromptCache ? PromptCacheManager(
            maxSizeMB: promptCacheSizeMB,
            ttlMinutes: promptCacheTTLMinutes,
            logger: app.logger
        ) : nil
        
        try await preloadDefaultModel(providers.modelProvider, app: app)
        configureCORS(app)
        try await configureRoutes(
            app, 
            modelProvider: providers.modelProvider, 
            embeddingModelProvider: providers.embeddingModelProvider, 
            isVLM: vlm,
            promptCacheManager: promptCacheManager
        )
        try await startServer(app)
    }
    
    private func setupApplication() async throws -> Application {
        let envName = ProcessInfo.processInfo.environment["MLX_ENV"] ?? "production"
        var env = Environment(name: envName, arguments: ["vapor"])
        try LoggingSystem.bootstrap(from: &env)
        return try await Application.make(env)
    }
    
    private func setupProviders(app: Application) -> (modelProvider: ModelProvider, embeddingModelProvider: EmbeddingModelProvider) {
        let modelProvider = ModelProvider(defaultModelPath: model, isVLM: vlm, logger: app.logger)
        let embeddingModelProvider = EmbeddingModelProvider(defaultModelId: embeddingModel, logger: app.logger)
        return (modelProvider, embeddingModelProvider)
    }
    
    private func preloadDefaultModel(_ modelProvider: ModelProvider, app: Application) async throws {
        app.logger.info("Attempting to pre-load default model: \(model)")
        do {
            _ = try await modelProvider.getModel(requestedModelId: nil)
            app.logger.info("Default model pre-loaded successfully.")
        } catch {
            app.logger.error("Failed to pre-load default model \(model): \(error). Server will attempt to load on first request.")
        }
    }
    
    private func configureCORS(_ app: Application) {
        let corsConfiguration = CORSMiddleware.Configuration(
            allowedOrigin: .all,
            allowedMethods: [.GET, .POST, .OPTIONS],
            allowedHeaders: [.accept, .authorization, .contentType, .origin, .xRequestedWith, .userAgent, .accessControlAllowOrigin]
        )
        let corsMiddleware = CORSMiddleware(configuration: corsConfiguration)
        app.middleware.use(corsMiddleware)
    }
    
    private func startServer(_ app: Application) async throws {
        app.http.server.configuration.hostname = host
        app.http.server.configuration.port = port
        
        app.logger.info("Server starting on http://\(host):\(port)")
        app.logger.info("Using model identifier: \(model)")
        app.logger.info("VLM mode: \(vlm ? "enabled" : "disabled")")
        app.logger.info("Prompt cache: \(enablePromptCache ? "enabled (size: \(promptCacheSizeMB)MB, TTL: \(promptCacheTTLMinutes)min)" : "disabled")")
        
        do {
            try await app.execute()
        } catch {
            app.logger.critical("Server error: \(error)")
            throw error
        }
    }
    
    private func shutdownApplication(_ app: Application) {
        app.logger.info("Server shutting down.")
        app.shutdown()
    }
}
