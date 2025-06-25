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
    private var currentModelIdString: String?
    private var currentLoadedModelName: String?
    private let logger: Logger
    private let fileManager = FileManager.default
    private let modelDownloadBaseURL: URL
    
    init(defaultModelPath: String, isVLM: Bool, logger: Logger) {
        self.defaultModelPath = defaultModelPath
        self.modelFactory = isVLM ? VLMModelFactory.shared : LLMModelFactory.shared
        self.logger = logger

        guard let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            fatalError("Could not find documents directory")
        }
        self.modelDownloadBaseURL = documents.appendingPathComponent("huggingface", isDirectory: true)

        logger.info(
            "ModelProvider initialized. Default Model Path: \(defaultModelPath). Scanning Location: \(self.modelDownloadBaseURL.path)"
        )
    }
    
    func getModel(requestedModelId: String?) async throws -> (ModelContainer, Tokenizer, String) {
        let actualIdToLoad = resolveModelId(requestedModelId)
        
        logger.debug("getModel: Requested='\(requestedModelId ?? "nil (default)")', Resolved actualIdToLoad='\(actualIdToLoad)'")

        if let cachedModel = await getCachedModelIfAvailable(actualIdToLoad) {
            return cachedModel
        }

        await unloadCurrentModelIfNeeded(actualIdToLoad)
        return try await loadNewModel(actualIdToLoad, requestedModelId: requestedModelId)
    }

    func getCurrentLoadedModelName() -> String? {
        return currentLoadedModelName
    }

    func getAvailableModelIDs() async -> [String] {
        return await scanForAvailableModels()
    }
    
    private func resolveModelId(_ requestedModelId: String?) -> String {
        if let reqId = requestedModelId, reqId.lowercased() != "default_model" {
            return reqId
        } else {
            return defaultModelPath
        }
    }
    
    private func getCachedModelIfAvailable(_ actualIdToLoad: String) async -> (ModelContainer, Tokenizer, String)? {
        guard let currentContainer = currentModelContainer,
              actualIdToLoad == currentModelIdString,
              let loadedName = currentLoadedModelName else {
            return nil
        }
        
        logger.info("Returning cached model: \(loadedName) (ID: \(currentModelIdString ?? "unknown"))")
        let tokenizer = await currentContainer.perform { $0.tokenizer }
        return (currentContainer, tokenizer, loadedName)
    }
    
    private func unloadCurrentModelIfNeeded(_ actualIdToLoad: String) async {
        guard currentModelContainer != nil else { return }
        
        logger.info("Request for model to load '\(actualIdToLoad)'. Current loaded is '\(currentModelIdString ?? "none")'. Needs loading/reloading.")
        
        let unloadedModelId = currentModelIdString ?? "unknown"
        let unloadedModelName = currentLoadedModelName ?? "unknown"
        logger.info("Unloading previous model: \(unloadedModelName) (ID: \(unloadedModelId))")
        
        currentModelContainer = nil
        currentModelConfig = nil
        currentModelIdString = nil
        currentLoadedModelName = nil
        MLX.GPU.clearCache()
        
        logger.info("Cleared MLX GPU cache after preparing to load new model.")
    }
    
    private func loadNewModel(_ actualIdToLoad: String, requestedModelId: String?) async throws -> (ModelContainer, Tokenizer, String) {
        let (modelConfiguration, nameToReturnForThisLoad) = try createModelConfiguration(actualIdToLoad)
        
        do {
            logger.info("Attempting to load: \(nameToReturnForThisLoad) using configuration for \(actualIdToLoad)")
            let newContainer = try await modelFactory.loadContainer(configuration: modelConfiguration)
            let newTokenizer = await newContainer.perform { $0.tokenizer }
            
            currentModelContainer = newContainer
            currentModelConfig = modelConfiguration
            currentModelIdString = actualIdToLoad
            currentLoadedModelName = nameToReturnForThisLoad
            
            MLX.GPU.set(cacheLimit: 20 * 1_024 * 1_024)
            
            logger.info("Successfully loaded model: \(nameToReturnForThisLoad) (from ID/path: \(actualIdToLoad))")
            return (newContainer, newTokenizer, nameToReturnForThisLoad)
        } catch {
            logger.error("Failed to load model with ID/path '\(actualIdToLoad)' (resolved from request '\(requestedModelId ?? "default")'): \(error)")
            clearCurrentModelState()
            throw ModelProviderError(
                status: .internalServerError,
                reason: "Failed to load requested model",
                modelId: actualIdToLoad,
                underlyingError: error
            )
        }
    }
    
    private func createModelConfiguration(_ actualIdToLoad: String) throws -> (ModelConfiguration, String) {
        let expandedPath = NSString(string: actualIdToLoad).expandingTildeInPath
        var isDirectory: ObjCBool = false
        
        if fileManager.fileExists(atPath: expandedPath, isDirectory: &isDirectory), isDirectory.boolValue {
            logger.info("Loading model from local path: \(expandedPath)")
            let localURL = URL(fileURLWithPath: expandedPath)
            let modelConfiguration = ModelConfiguration(directory: localURL)
            let nameToReturn = modelConfiguration.name
            return (modelConfiguration, nameToReturn)
        } else {
            logger.info("Loading model from Hugging Face Hub: \(actualIdToLoad)")
            let modelConfiguration = modelFactory.configuration(id: actualIdToLoad)
            let nameToReturn = modelConfiguration.name
            return (modelConfiguration, nameToReturn)
        }
    }
    
    private func clearCurrentModelState() {
        currentModelContainer = nil
        currentModelConfig = nil
        currentModelIdString = nil
        currentLoadedModelName = nil
    }
    
    private func scanForAvailableModels() async -> [String] {
        logger.info("Scanning for available MLX models in \(modelDownloadBaseURL.path)...")
        var foundModelIds = Set<String>()
        let modelsRootURL = modelDownloadBaseURL.appendingPathComponent("models", isDirectory: true)

        guard fileManager.fileExists(atPath: modelsRootURL.path) else {
            logger.warning("Models directory does not exist at \(modelsRootURL.path). Cannot scan for cached models.")
            return []
        }

        logger.debug("Scanning models directory: \(modelsRootURL.path)")

        do {
            let orgDirs = try fileManager.contentsOfDirectory(
                at: modelsRootURL, 
                includingPropertiesForKeys: [.isDirectoryKey], 
                options: .skipsHiddenFiles
            )

            for orgDirURL in orgDirs {
                guard isDirectory(orgDirURL) else { continue }
                try scanOrganizationDirectory(orgDirURL, foundModelIds: &foundModelIds)
            }
        } catch {
            logger.error("Error scanning models directory \(modelsRootURL.path): \(error)")
        }
        
        let sortedModels = Array(foundModelIds).sorted()
        logger.info("Scan complete. Found potential models from \(modelDownloadBaseURL.path): \(sortedModels)")
        return sortedModels
    }
    
    private func scanOrganizationDirectory(_ orgDirURL: URL, foundModelIds: inout Set<String>) throws {
        let orgName = orgDirURL.lastPathComponent
        let modelNameDirs = try fileManager.contentsOfDirectory(
            at: orgDirURL, 
            includingPropertiesForKeys: [.isDirectoryKey], 
            options: .skipsHiddenFiles
        )

        for modelNameDirURL in modelNameDirs {
            guard isDirectory(modelNameDirURL) else { continue }
            
            let modelName = modelNameDirURL.lastPathComponent
            let repoId = "\(orgName)/\(modelName)"

            logger.debug("Checking potential model directory: \(modelNameDirURL.path) for ID: \(repoId)")

            if isMLXModelDirectory(modelNameDirURL) {
                logger.debug("Found valid MLX model: \(repoId)")
                foundModelIds.insert(repoId)
            } else {
                logger.debug("Directory \(modelNameDirURL.path) is not a valid MLX model.")
            }
        }
    }
    
    private func isDirectory(_ url: URL) -> Bool {
        return (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
    }

    private func isMLXModelDirectory(_ dirURL: URL) -> Bool {
        let requiredFiles = MLXModelFiles(baseURL: dirURL)
        
        let configExists = fileManager.fileExists(atPath: requiredFiles.configPath)
        let tokenizerExists = fileManager.fileExists(atPath: requiredFiles.tokenizerPath) || 
                             fileManager.fileExists(atPath: requiredFiles.tokenizerModelPath)
        let weightsExist = checkForModelWeights(in: dirURL, indexPath: requiredFiles.indexPath)

        guard configExists && tokenizerExists && weightsExist else {
            logMissingFiles(for: dirURL, config: configExists, tokenizer: tokenizerExists, weights: weightsExist)
            return false
        }

        logger.trace("Valid MLX directory confirmed: \(dirURL.path)")
        return true
    }
    
    private func checkForModelWeights(in dirURL: URL, indexPath: String) -> Bool {
        if fileManager.fileExists(atPath: indexPath) {
            return true
        }
        
        do {
            let files = try fileManager.contentsOfDirectory(atPath: dirURL.path)
            return files.contains { $0.hasSuffix(".safetensors") }
        } catch {
            logger.error("Error checking for weights files in \(dirURL.path): \(error)")
            return false
        }
    }
    
    private func logMissingFiles(for dirURL: URL, config: Bool, tokenizer: Bool, weights: Bool) {
        var missing: [String] = []
        if !config { missing.append("config.json") }
        if !tokenizer { missing.append("tokenizer.json/model") }
        if !weights { missing.append("weights/.index or .safetensors") }
        
        logger.trace("Directory \(dirURL.path) missing required files: \(missing.joined(separator: ", "))")
    }
}

private struct MLXModelFiles {
    let configPath: String
    let tokenizerPath: String
    let tokenizerModelPath: String
    let indexPath: String
    
    init(baseURL: URL) {
        self.configPath = baseURL.appendingPathComponent("config.json").path
        self.tokenizerPath = baseURL.appendingPathComponent("tokenizer.json").path
        self.tokenizerModelPath = baseURL.appendingPathComponent("tokenizer.model").path
        self.indexPath = baseURL.appendingPathComponent("model.safetensors.index.json").path
    }
}
