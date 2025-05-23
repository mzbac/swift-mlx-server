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
  let created: Int = Int(Date().timeIntervalSince1970)
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

func routes(
  _ app: Application, modelProvider: ModelProvider, embeddingModelProvider: EmbeddingModelProvider,
  isVLM: Bool
)
  async throws
{

  try registerTextCompletionsRoute(
    app,
    modelProvider: modelProvider
  )

  try registerChatCompletionsRoute(
    app,
    modelProvider: modelProvider,
    isVLM: isVLM
  )

  registerEmbeddingsRoute(app, embeddingModelProvider: embeddingModelProvider)

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

  @ArgumentParser.Flag(
    name: .long, help: "Enable multi-modal processing for visual language models.")
  var vlm: Bool = false

  @ArgumentParser.Option(name: .long, help: "Path or identifier for embedding model.")
  var embeddingModel: String?

  @MainActor
  func run() async throws {
    let envName = ProcessInfo.processInfo.environment["MLX_ENV"] ?? "production"
    var env = Environment(name: envName, arguments: ["vapor"])
    try LoggingSystem.bootstrap(from: &env)
    let app = Application(env)
    defer {
      app.logger.info("Server shutting down.")
      app.shutdown()
    }

    let modelProvider = ModelProvider(defaultModelPath: model, isVLM: vlm, logger: app.logger)
    let embeddingModelProvider = EmbeddingModelProvider(
      defaultModelId: embeddingModel,
      logger: app.logger
    )

    app.logger.info("Attempting to pre-load default model: \(model)")
    do {
      _ = try await modelProvider.getModel(requestedModelId: nil)
      app.logger.info("Default model pre-loaded successfully.")
    } catch {
      app.logger.error(
        "Failed to pre-load default model \(model): \(error). Server will attempt to load on first request."
      )
    }
    let corsConfiguration = CORSMiddleware.Configuration(
      allowedOrigin: .all,
      allowedMethods: [.GET, .POST, .OPTIONS],
      allowedHeaders: [
        .accept, .authorization, .contentType, .origin,
        .xRequestedWith, .userAgent, .accessControlAllowOrigin,
      ]
    )
    let corsMiddleware = CORSMiddleware(configuration: corsConfiguration)
    app.middleware.use(corsMiddleware)

      try await routes(app, modelProvider: modelProvider, embeddingModelProvider: embeddingModelProvider, isVLM: vlm)

    app.http.server.configuration.hostname = host
    app.http.server.configuration.port = port
    do {
      app.logger.info("Server starting on http://\(host):\(port)")
      app.logger.info("Using model identifier: \(model)")
      app.logger.info("VLM mode: \(vlm ? "enabled" : "disabled")")
      try await app.execute()
    } catch {
      app.logger.critical("Server error: \(error)")
      throw error
    }
  }
}
