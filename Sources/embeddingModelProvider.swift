import Foundation
import Hub
import Logging
import MLX
import Tokenizers
import Vapor
import mlx_embeddings

private struct EmbeddingModelProviderError: AbortError {
    var status: HTTPResponseStatus
    var reason: String
    var identifier: String?

    init(status: HTTPResponseStatus, reason: String, modelId: String? = nil, underlyingError: Error? = nil) {
        self.status = status
        var fullReason = reason
        if let modelId = modelId, !modelId.isEmpty {
            fullReason += " (Model ID: \(modelId))"
        }
        if let underlyingError = underlyingError {
            fullReason += ". Underlying error: \(underlyingError.localizedDescription)"
        }
        self.reason = fullReason
        self.identifier = modelId
    }
}

actor EmbeddingModelProvider {
  private var currentModelContainer: mlx_embeddings.ModelContainer?
  private var currentModelIdString: String?
  private var currentLoadedModelName: String?
  private let logger: Logger
  private let defaultModelId: String?
  private let fileManager = FileManager.default

  init(defaultModelId: String?, logger: Logger) {
    self.defaultModelId = defaultModelId
    self.logger = logger
    logger.info(
      "EmbeddingModelProvider initialized. Default Embedding Model ID: \(defaultModelId ?? "None")")
  }

  func getModel(requestedModelId: String?) async throws -> (
    container: mlx_embeddings.ModelContainer, loadedName: String
  ) {
    let actualIdToLoad: String
    let originalRequestDescription: String

    if let userRequestedId = requestedModelId {
      if userRequestedId == "default_model" {
        guard let configuredDefault = self.defaultModelId else {
          logger.error(
            "Requested 'default_model' but no default embedding model is configured.")
          throw EmbeddingModelProviderError(
            status: .badRequest,
            reason: "Requested 'default_model' but no default model is configured. Please specify a model or set a default using --embedding-model."
          )
        }
        actualIdToLoad = configuredDefault
        originalRequestDescription = "default_model (resolved to: \(configuredDefault))"
      } else {
        actualIdToLoad = userRequestedId
        originalRequestDescription = userRequestedId
      }
    } else {
      guard let configuredDefault = self.defaultModelId else {
        logger.error("No embedding model requested and no default embedding model configured.")
        throw EmbeddingModelProviderError(
            status: .badRequest,
            reason: "No embedding model specified and no default is configured. Use the 'model' field in your request or start the server with a default --embedding-model."
        )
      }
      actualIdToLoad = configuredDefault
      originalRequestDescription = "nil (using default: \(configuredDefault))"
    }

    if let currentContainer = currentModelContainer,
       actualIdToLoad == self.currentModelIdString,
       let loadedName = self.currentLoadedModelName {
      logger.info(
        "Returning current embedding model container: \(loadedName) (for request: \(originalRequestDescription))"
      )
      return (currentContainer, loadedName)
    }

    if currentModelContainer != nil {
         logger.info("Requested model '\(actualIdToLoad)' (from request: \(originalRequestDescription)) is different from current '\(self.currentLoadedModelName ?? "unknown")'. Unloading current.")
         await _unloadCurrentModel()
    } else {
         logger.info("No model currently loaded.")
    }

    logger.info("Embedding model '\(actualIdToLoad)' (requested as: \(originalRequestDescription)) not loaded or is different. Loading container...")

    let modelConfiguration: ModelConfiguration
    let expandedPath = NSString(string: actualIdToLoad).expandingTildeInPath
    var isDirectory: ObjCBool = false
    var nameToReturnForThisLoad: String
    var configIdStringForLogging: String

    if fileManager.fileExists(atPath: expandedPath, isDirectory: &isDirectory),
      isDirectory.boolValue
    {
      let localURL = URL(fileURLWithPath: expandedPath)
      logger.info("Loading embedding model from local path: \(expandedPath)")
      modelConfiguration = ModelConfiguration(directory: localURL)
      nameToReturnForThisLoad = modelConfiguration.name ?? localURL.lastPathComponent
      configIdStringForLogging = localURL.path
    } else {
      logger.info("Loading embedding model from Hugging Face Hub: \(actualIdToLoad)")
      modelConfiguration = ModelConfiguration(id: actualIdToLoad)
      nameToReturnForThisLoad = modelConfiguration.name ?? actualIdToLoad
      configIdStringForLogging = actualIdToLoad
    }

    do {
      logger.info(
        "Attempting to load embedding model container for configuration: \(configIdStringForLogging)"
      )
      let newContainer = try await mlx_embeddings.loadModelContainer(
        configuration: modelConfiguration)

      let finalLoadedName = modelConfiguration.name ?? nameToReturnForThisLoad

      logger.info(
        "Successfully loaded embedding model container: \(finalLoadedName) (from target: \(actualIdToLoad), requested as: \(originalRequestDescription))"
      )

      self.currentModelContainer = newContainer
      self.currentModelIdString = actualIdToLoad
      self.currentLoadedModelName = finalLoadedName

      return (newContainer, finalLoadedName)
    } catch {
      logger.error(
        "Failed to load embedding model container for '\(actualIdToLoad)' (requested as: \(originalRequestDescription)): \(error)"
      )
      throw EmbeddingModelProviderError(
        status: .internalServerError,
        reason: "Failed to load embedding model container",
        modelId: actualIdToLoad,
        underlyingError: error
      )
    }
  }

  private func _unloadCurrentModel() async {
    if currentModelContainer != nil {
      logger.info("Unloading current model: \(currentLoadedModelName ?? "unknown") (ID: \(currentModelIdString ?? "unknown"))")
      currentModelContainer = nil
      currentModelIdString = nil
      currentLoadedModelName = nil
      MLX.GPU.clearCache()
    }
  }
}