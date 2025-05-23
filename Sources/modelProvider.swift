import Vapor
import Foundation
import Logging
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers
import Hub
import MLXVLM

private struct ModelProviderError: AbortError {
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

actor ModelProvider {
    private var currentModelConfig: ModelConfiguration?
    private var currentModelContainer: ModelContainer?
    private let modelFactory: ModelFactory

    private let defaultModelPath: String
    private var currentModelIdString: String? = nil
    private var currentLoadedModelName: String? = nil

    private let logger: Logger
    private let fileManager = FileManager.default
    private let modelDownloadBaseURL: URL

    init(defaultModelPath: String, isVLM: Bool, logger: Logger) {
        self.defaultModelPath = defaultModelPath
        self.modelFactory = isVLM ? VLMModelFactory.shared : LLMModelFactory.shared
        self.logger = logger

        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        self.modelDownloadBaseURL = documents.appendingPathComponent("huggingface", isDirectory: true)

        logger.info(
            "ModelProvider initialized. Default Model Path: \(defaultModelPath). Scanning Location: \(self.modelDownloadBaseURL.path)"
        )
    }

    func getModel(requestedModelId: String?) async throws -> (ModelContainer, Tokenizer, String) {
        let actualIdToLoad: String
        if let reqId = requestedModelId, reqId.lowercased() != "default_model" {
            actualIdToLoad = reqId
        } else {
            actualIdToLoad = self.defaultModelPath
        }

        logger.debug("getModel: Requested='\(requestedModelId ?? "nil (default)")', Resolved actualIdToLoad='\(actualIdToLoad)'")

        if let currentContainer = currentModelContainer,
           actualIdToLoad == self.currentModelIdString, 
           let loadedName = self.currentLoadedModelName {
            logger.info("Returning cached model: \(loadedName) (ID: \(self.currentModelIdString ?? "unknown"))")
            let tokenizer = await currentContainer.perform { $0.tokenizer }
            return (currentContainer, tokenizer, loadedName)
        }

        logger.info(
            "Request for model to load '\(actualIdToLoad)' (requested: '\(requestedModelId ?? "default")'). Current loaded is '\(self.currentModelIdString ?? "none")'. Needs loading/reloading."
        )

        if currentModelContainer != nil {
             let unloadedModelId = self.currentModelIdString ?? "unknown"
             let unloadedModelName = self.currentLoadedModelName ?? "unknown"
             logger.info("Unloading previous model: \(unloadedModelName) (ID: \(unloadedModelId))")
             currentModelContainer = nil 
             currentModelConfig = nil
             currentModelIdString = nil
             currentLoadedModelName = nil
             MLX.GPU.clearCache() 
             logger.info("Cleared MLX GPU cache after preparing to load new model.")
        }

        let modelConfiguration: ModelConfiguration
        let expandedPath = NSString(string: actualIdToLoad).expandingTildeInPath
        var isDirectory: ObjCBool = false
        var nameToReturnForThisLoad = actualIdToLoad 

        if fileManager.fileExists(atPath: expandedPath, isDirectory: &isDirectory),
           isDirectory.boolValue
        {
            logger.info("Loading model from local path: \(expandedPath)")
            modelConfiguration = ModelConfiguration(directory: URL(fileURLWithPath: expandedPath))
            nameToReturnForThisLoad = modelConfiguration.name ?? URL(fileURLWithPath: expandedPath).lastPathComponent
        } else {
            logger.info("Loading model from Hugging Face Hub: \(actualIdToLoad)")
            modelConfiguration = modelFactory.configuration(id: actualIdToLoad)
            nameToReturnForThisLoad = modelConfiguration.name 
        }

        do {
            logger.info("Attempting to load: \(nameToReturnForThisLoad) using configuration for \(actualIdToLoad)")
            let newContainer = try await modelFactory.loadContainer(configuration: modelConfiguration)
            let newTokenizer = await newContainer.perform { $0.tokenizer }

            self.currentModelContainer = newContainer
            self.currentModelConfig = modelConfiguration
            self.currentModelIdString = actualIdToLoad 
            self.currentLoadedModelName = nameToReturnForThisLoad
            logger.info("Successfully loaded model: \(nameToReturnForThisLoad) (from ID/path: \(actualIdToLoad))")
            return (newContainer, newTokenizer, nameToReturnForThisLoad)
        } catch {
            logger.error("Failed to load model with ID/path '\(actualIdToLoad)' (resolved from request '\(requestedModelId ?? "default")'): \(error)")
            self.currentModelContainer = nil
            self.currentModelConfig = nil
            self.currentModelIdString = nil
            self.currentLoadedModelName = nil
            throw ModelProviderError(
                status: .internalServerError,
                reason: "Failed to load requested model",
                modelId: actualIdToLoad,
                underlyingError: error
            )
        }
    }

    func getCurrentLoadedModelName() -> String? {
        return currentLoadedModelName
    }

    func getAvailableModelIDs() async -> [String] {
        logger.info("Scanning for available MLX models in \(modelDownloadBaseURL.path)...")
        var foundModelIds = Set<String>()

        let modelsRootURL = modelDownloadBaseURL.appendingPathComponent("models", isDirectory: true)

        guard fileManager.fileExists(atPath: modelsRootURL.path) else {
            logger.warning(
                "Models directory does not exist at \(modelsRootURL.path). Cannot scan for cached models.")
            return Array(foundModelIds).sorted() 
        }

        logger.debug("Scanning models directory: \(modelsRootURL.path)")

        do {
            let orgDirs = try fileManager.contentsOfDirectory(
                at: modelsRootURL, includingPropertiesForKeys: [.isDirectoryKey], options: .skipsHiddenFiles
            )

            for orgDirURL in orgDirs {
                guard (try? orgDirURL.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
                else { continue }
                let orgName = orgDirURL.lastPathComponent

                let modelNameDirs = try fileManager.contentsOfDirectory(
                    at: orgDirURL, includingPropertiesForKeys: [.isDirectoryKey], options: .skipsHiddenFiles)

                for modelNameDirURL in modelNameDirs {
                    guard
                        (try? modelNameDirURL.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
                    else { continue }
                    let modelName = modelNameDirURL.lastPathComponent
                    let repoId = "\(orgName)/\(modelName)"

                    logger.debug(
                        "Checking potential model directory: \(modelNameDirURL.path) for ID: \(repoId)")

                    if _isMLXModelDirectory(modelNameDirURL) {
                        logger.debug("Found valid MLX model: \(repoId)")
                        foundModelIds.insert(repoId)
                    } else {
                        logger.debug("Directory \(modelNameDirURL.path) is not a valid MLX model.")
                    }
                }
            }
        } catch {
            logger.error("Error scanning models directory \(modelsRootURL.path): \(error)")
        }
        let sortedModels = Array(foundModelIds).sorted()
        logger.info("Scan complete. Found potential models from \(modelDownloadBaseURL.path): \(sortedModels)")
        return sortedModels
     }

    private func _isMLXModelDirectory(_ dirURL: URL) -> Bool {
        let configPath = dirURL.appendingPathComponent("config.json").path
        let tokenizerPath = dirURL.appendingPathComponent("tokenizer.json").path
        let tokenizerModelPath = dirURL.appendingPathComponent("tokenizer.model").path
        let indexPath = dirURL.appendingPathComponent("model.safetensors.index.json").path

        let configExists = fileManager.fileExists(atPath: configPath)
        let tokenizerExists =
            fileManager.fileExists(atPath: tokenizerPath)
            || fileManager.fileExists(atPath: tokenizerModelPath)
        let indexExists = fileManager.fileExists(atPath: indexPath)

        var weightsExist = false
        if !indexExists { 
            do {
                let files = try fileManager.contentsOfDirectory(atPath: dirURL.path)
                weightsExist = files.contains { $0.hasSuffix(".safetensors") }
            } catch {
                logger.error("Error checking for weights files in \(dirURL.path): \(error)")
                weightsExist = false
            }
        }

        let hasWeights = indexExists || weightsExist

        if !(configExists && tokenizerExists && hasWeights) {
            var missing: [String] = []
            if !configExists { missing.append("config.json") }
            if !tokenizerExists { missing.append("tokenizer.json/model") }
            if !hasWeights { missing.append("weights/.index or .safetensors") }
            logger.trace(
                "Directory \(dirURL.path) missing required files: \(missing.joined(separator: ", "))")
            return false
        }

        logger.trace("Valid MLX directory confirmed: \(dirURL.path)")
        return true
    }
}