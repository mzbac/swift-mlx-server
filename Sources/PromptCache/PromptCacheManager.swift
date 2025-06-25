import Foundation
import MLX
import MLXLMCommon
import Logging

/// Manages prompt caching across requests with support for KV cache quantization
public actor PromptCacheManager {
    private var caches: [String: PromptCacheEntry] = [:]
    private let maxSizeMB: Int
    private let ttlMinutes: Int
    private var currentSizeBytes: Int = 0
    private var stats = PromptCacheStats()
    private let logger: Logger
    
    public init(maxSizeMB: Int = 1_024, ttlMinutes: Int = 30, logger: Logger) {
        self.maxSizeMB = maxSizeMB
        self.ttlMinutes = ttlMinutes
        self.logger = logger
    }
    
    /// Get cached state for a given prompt, returning tokens to process and existing cache
    public func getCachedState(
        modelKey: String,
        tokens: [Int],
        parameters: GenerateParameters,
        model: any LanguageModel
    ) -> (tokensToProcess: [Int], cache: [KVCache]?) {
        let cacheKey = createCacheKey(
            modelKey: modelKey,
            temperature: parameters.temperature,
            topP: parameters.topP,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize
        )
        
        cleanExpiredEntries()
        
        guard var entry = caches[cacheKey], entry.isValid(ttlMinutes: ttlMinutes) else {
            stats.misses += 1
            stats.totalTokensProcessed += tokens.count
            logger.debug("Cache miss for key: \(cacheKey)")
            return (tokens, nil)
        }
        
        let commonLength = min(commonPrefixLength(entry.tokens, tokens), tokens.count - 1)
        
        guard commonLength > 0 else {
            stats.misses += 1
            stats.totalTokensProcessed += tokens.count
            logger.debug("No common prefix found for key: \(cacheKey)")
            caches.removeValue(forKey: cacheKey)
            currentSizeBytes -= entry.estimatedSizeBytes
            return (tokens, nil)
        }
        
        stats.hits += 1
        stats.totalTokensReused += commonLength
        stats.totalTokensProcessed += tokens.count - commonLength
        
        entry.updateAccess()
        
        let tokensToTrim = entry.tokens.count - commonLength
        if tokensToTrim > 0 {
            logger.debug("Trimming \(tokensToTrim) tokens from cache")
            for cache in entry.kvCaches {
                cache.trim(count: tokensToTrim)
            }
            entry.tokens.removeLast(tokensToTrim)
            
            let oldSize = entry.estimatedSizeBytes
            entry.estimatedSizeBytes = PromptCacheEntry.estimateSize(
                tokens: entry.tokens,
                kvCaches: entry.kvCaches
            )
            currentSizeBytes = currentSizeBytes - oldSize + entry.estimatedSizeBytes
        }
        
        caches[cacheKey] = entry
        
        let kvCaches = entry.kvCaches.map { $0 as KVCache }
        
        logger.info("Cache hit: Reusing \(commonLength) tokens, processing \(tokens.count - commonLength) new tokens")
        
        return (Array(tokens[commonLength...]), kvCaches)
    }
    
    /// Update cache with new tokens and KV caches after generation
    public func updateCache(
        modelKey: String,
        tokens: [Int],
        kvCaches: [KVCache],
        parameters: GenerateParameters,
        model: any LanguageModel
    ) async {
        let cacheKey = createCacheKey(
            modelKey: modelKey,
            temperature: parameters.temperature,
            topP: parameters.topP,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize
        )
        
        let trimmableCaches = await createTrimmableCaches(from: kvCaches, parameters: parameters, model: model)
        
        let entry = PromptCacheEntry(
            key: cacheKey,
            tokens: tokens,
            kvCaches: trimmableCaches
        )
        
        let maxSizeBytes = maxSizeMB * 1_024 * 1_024
        var sizeAfterAdding = currentSizeBytes + entry.estimatedSizeBytes
        
        if let oldEntry = caches[cacheKey] {
            sizeAfterAdding -= oldEntry.estimatedSizeBytes
        }
        
        while sizeAfterAdding > maxSizeBytes && !caches.isEmpty {
            if let lruKey = findLeastRecentlyUsedKey() {
                if let removedEntry = caches.removeValue(forKey: lruKey) {
                    currentSizeBytes -= removedEntry.estimatedSizeBytes
                    sizeAfterAdding -= removedEntry.estimatedSizeBytes
                    stats.evictions += 1
                    logger.debug("Evicted cache entry: \(lruKey)")
                }
            } else {
                break
            }
        }
        
        caches[cacheKey] = entry
        currentSizeBytes = sizeAfterAdding
        
        logger.debug("Updated cache for key: \(cacheKey), size: \(entry.estimatedSizeBytes) bytes")
    }
    
    /// Get current cache statistics
    public func getStats() -> PromptCacheStats {
        return stats
    }
    
    /// Clear all cached entries
    public func clearCache() {
        caches.removeAll()
        currentSizeBytes = 0
        logger.info("Cleared all prompt cache entries")
    }
    
    /// Get cache status information
    public func getCacheStatus() -> (entryCount: Int, sizeBytes: Int, sizeMB: Double) {
        return (
            entryCount: caches.count,
            sizeBytes: currentSizeBytes,
            sizeMB: Double(currentSizeBytes) / (1_024 * 1_024)
        )
    }
    
    private func createCacheKey(
        modelKey: String,
        temperature: Float,
        topP: Float,
        kvBits: Int?,
        kvGroupSize: Int
    ) -> String {
        let kvConfig = kvBits.map { "kv\($0)g\(kvGroupSize)" } ?? "nokv"
        return "\(modelKey)-t\(temperature)-p\(topP)-\(kvConfig)"
    }
    
    private func commonPrefixLength(_ first: [Int], _ second: [Int]) -> Int {
        let len = min(first.count, second.count)
        for i in 0..<len where first[i] != second[i] {
            return i
        }
        return len
    }
    
    private func cleanExpiredEntries() {
        var keysToRemove: [String] = []
        var removedSize = 0
        
        for (key, entry) in caches where !entry.isValid(ttlMinutes: ttlMinutes) {
            keysToRemove.append(key)
            removedSize += entry.estimatedSizeBytes
        }
        
        for key in keysToRemove {
            caches.removeValue(forKey: key)
        }
        
        currentSizeBytes -= removedSize
        
        if !keysToRemove.isEmpty {
            logger.debug("Cleaned \(keysToRemove.count) expired cache entries")
        }
    }
    
    private func findLeastRecentlyUsedKey() -> String? {
        var oldestKey: String?
        var oldestTime = Date()
        
        for (key, entry) in caches where entry.lastAccessedAt < oldestTime {
            oldestTime = entry.lastAccessedAt
            oldestKey = key
        }
        
        return oldestKey
    }
    
    private func createTrimmableCaches(
        from kvCaches: [KVCache],
        parameters: GenerateParameters,
        model: any LanguageModel
    ) async -> [TrimmableKVCache] {
        var trimmableCaches: [TrimmableKVCache] = []
        
        for (index, cache) in kvCaches.enumerated() {
            if let trimmable = cache as? TrimmableKVCache {
                trimmableCaches.append(trimmable)
                continue
            }
            
            if cache is QuantizedKVCacheProtocol,
               let kvBits = parameters.kvBits {
                let wrapper = CachedQuantizedKVCache(
                    groupSize: parameters.kvGroupSize,
                    bits: kvBits
                )
                let evaluatable = cache as any Evaluatable
                _ = evaluatable.innerState()
                logger.debug("Wrapped quantized cache at index \(index)")
                trimmableCaches.append(wrapper)
            } else {
                let wrapper = CachedKVCacheSimple()
                let evaluatable = cache as any Evaluatable
                _ = evaluatable.innerState()
                logger.debug("Wrapped simple cache at index \(index)")
                trimmableCaches.append(wrapper)
            }
        }
        
        return trimmableCaches
    }
}

public extension PromptCacheManager {
    /// Check if prompt caching would be beneficial for given tokens
    func shouldUseCache(tokenCount: Int) -> Bool {
        return tokenCount > 10
    }
    
    /// Estimate memory usage for a potential cache entry
    func estimateCacheSizeForTokens(_ tokenCount: Int, model: any LanguageModel) -> Int {
        return tokenCount * 4 * 1_024
    }
    
    /// Get the maximum cache size in MB
    var maxCacheSizeMB: Int {
        maxSizeMB
    }
    
    /// Get the cache TTL in minutes
    var cacheTTLMinutes: Int {
        ttlMinutes
    }
}
