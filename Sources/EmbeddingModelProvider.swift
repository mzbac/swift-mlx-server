import Foundation
import Hub
import Logging
import MLX
import Tokenizers
import Vapor
import mlx_embeddings

actor EmbeddingModelProvider {
  private var currentModel:
    (container: mlx_embeddings.ModelContainer, loadedName: String, modelId: String)?
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
    guard let targetModelId = requestedModelId ?? defaultModelId else {
      logger.error("No embedding model requested and no default embedding model configured.")
      throw Abort(
        .badRequest,
        reason:
          "No embedding model specified and no default is configured. Use the 'model' field in your request or start the server with a default --embedding-model."
      )
    }

    if let current = currentModel, current.modelId == targetModelId || targetModelId == "default_model" {
      logger.info(
        "Returning current embedding model container: \(current.loadedName) (for request: \(targetModelId))"
      )
      return (current.container, current.loadedName)
    }

    await unloadCurrentModel()
    logger.info("Embedding model '\(targetModelId)' not loaded. Loading container...")

    let modelConfiguration: ModelConfiguration
    let expandedPath = NSString(string: targetModelId).expandingTildeInPath
    var isDirectory: ObjCBool = false
    var nameToReturn = targetModelId
    var configIdStringForLogging: String

    if fileManager.fileExists(atPath: expandedPath, isDirectory: &isDirectory),
      isDirectory.boolValue
    {
      let localURL = URL(fileURLWithPath: expandedPath)
      logger.info("Loading embedding model from local path: \(expandedPath)")
      modelConfiguration = ModelConfiguration(directory: localURL)
      nameToReturn = modelConfiguration.name ?? localURL.lastPathComponent
      nameToReturn = targetModelId
      configIdStringForLogging = localURL.path
    } else {
      logger.info("Loading embedding model from Hugging Face Hub: \(targetModelId)")
      modelConfiguration = ModelConfiguration(id: targetModelId)
      nameToReturn = modelConfiguration.name ?? targetModelId
      configIdStringForLogging = targetModelId
    }

    do {
      logger.info(
        "Attempting to load embedding model container for configuration: \(configIdStringForLogging)"
      )
      let newContainer = try await mlx_embeddings.loadModelContainer(
        configuration: modelConfiguration)
      logger.info(
        "Successfully loaded embedding model container: \(nameToReturn) (from target: \(targetModelId))"
      )

      let loadedModel = (container: newContainer, loadedName: nameToReturn, modelId: targetModelId)
      currentModel = loadedModel

      return (loadedModel.container, loadedModel.loadedName)
    } catch {
      logger.error("Failed to load embedding model container for '\(targetModelId)': \(error)")
      throw Abort(
        .internalServerError,
        reason:
          "Failed to load embedding model container '\(targetModelId)': \(error.localizedDescription)"
      )
    }
  }

  private func unloadCurrentModel() async {
    if let current = currentModel {
      logger.info("Unloading current model: \(current.loadedName)")
      currentModel = nil
      MLX.GPU.clearCache()
    }
  }
}
