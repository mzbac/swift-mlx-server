import Vapor
import Foundation
import Logging
import MLX
import MLXLLM 
import MLXLMCommon
import Tokenizers
import Hub 
import MLXVLM

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
        let targetModelId = requestedModelId ?? defaultModelPath 

        if let currentContainer = currentModelContainer, 
           (targetModelId == self.currentModelIdString || targetModelId == "default_model") {
            logger.info("Returning cached model: \(self.currentModelIdString ?? targetModelId)")
            let tokenizer = await currentContainer.perform { $0.tokenizer }
            return (currentContainer, tokenizer, self.currentLoadedModelName ?? targetModelId)
        }

        logger.info(
            "Request for model '\(targetModelId)', current is '\(self.currentModelIdString ?? "none")'. Loading..."
        )

        if currentModelContainer != nil {
             let unloadedModelId = self.currentModelIdString ?? "unknown" 
             logger.info("Unloading previous model: \(unloadedModelId)")
             currentModelContainer = nil
             currentModelConfig = nil
             currentModelIdString = nil
             currentLoadedModelName = nil
             MLX.GPU.clearCache()
             logger.info("Cleared MLX GPU cache after unloading model \(unloadedModelId).")
        }

        let modelConfiguration: ModelConfiguration
        let expandedPath = NSString(string: targetModelId).expandingTildeInPath

        var isDirectory: ObjCBool = false
        var nameToReturn = targetModelId

        if fileManager.fileExists(atPath: expandedPath, isDirectory: &isDirectory),
           isDirectory.boolValue
        {
            logger.info("Loading model from local path: \(expandedPath)")
            modelConfiguration = ModelConfiguration(directory: URL(fileURLWithPath: expandedPath))
            nameToReturn = modelConfiguration.name ?? URL(fileURLWithPath: expandedPath).lastPathComponent

        } else {
            logger.info("Loading model from Hugging Face Hub: \(targetModelId)")
            modelConfiguration = modelFactory.configuration(id: targetModelId)
            nameToReturn = modelConfiguration.name
        }

        do {
            let newContainer = try await modelFactory.loadContainer(configuration: modelConfiguration)
            let newTokenizer = await newContainer.perform { $0.tokenizer }

            self.currentModelContainer = newContainer
            self.currentModelConfig = modelConfiguration
            self.currentModelIdString = targetModelId 
            self.currentLoadedModelName = nameToReturn 
            logger.info("Successfully loaded model: \(nameToReturn) (from target: \(targetModelId))")
            return (newContainer, newTokenizer, nameToReturn)
        } catch {
            logger.error("Failed to load model \(targetModelId): \(error)")
            self.currentModelContainer = nil
            self.currentModelConfig = nil
            self.currentModelIdString = nil
            self.currentLoadedModelName = nil
            throw Abort(
                .internalServerError,
                reason: "Failed to load requested model '\(targetModelId)': \(error.localizedDescription)")
        }
    }

    func getCurrentLoadedModelName() -> String? {
        return currentLoadedModelName
    }

     func getAvailableModelIDs() async -> [String] {
        logger.info("Scanning for available MLX models in \(modelDownloadBaseURL.path)...")
        var foundModelIds = Set<String>()

        if let currentId = self.currentModelIdString {
            foundModelIds.insert(currentId)
        }

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

                    if isMLXModelDirectory(modelNameDirURL) {
                        logger.debug("Found valid MLX model: \(repoId)")
                        foundModelIds.insert(repoId)                     } else {
                        logger.debug("Directory \(modelNameDirURL.path) is not a valid MLX model.")
                    }
                }
            }
        } catch {
            logger.error("Error scanning models directory \(modelsRootURL.path): \(error)")
        }
        let sortedModels = Array(foundModelIds).sorted()
        logger.info("Scan complete. Found potential models: \(sortedModels)")
        return sortedModels
     }

    private func isMLXModelDirectory(_ dirURL: URL) -> Bool {
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
