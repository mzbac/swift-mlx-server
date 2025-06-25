import Foundation
import MLX
import MLXLMCommon

/// Represents a cached prompt entry with its tokens and KV cache
struct PromptCacheEntry {
    /// Unique key for this cache entry (model + temperature + topP)
    let key: String
    
    /// The tokens that have been processed and cached
    var tokens: [Int]
    
    /// The KV cache objects (trimmable versions)
    var kvCaches: [TrimmableKVCache]
    
    /// When this cache entry was created
    let createdAt: Date
    
    /// When this cache entry was last accessed
    var lastAccessedAt: Date
    
    /// Size in bytes (estimated)
    var estimatedSizeBytes: Int
    
    /// Check if the cache is still valid (based on TTL)
    func isValid(ttlMinutes: Int) -> Bool {
        let ttlSeconds = TimeInterval(ttlMinutes * 60)
        return Date().timeIntervalSince(lastAccessedAt) < ttlSeconds
    }
    
    /// Update the last accessed time
    mutating func updateAccess() {
        lastAccessedAt = Date()
    }
    
    /// Calculate the estimated size of this cache entry
    static func estimateSize(tokens: [Int], kvCaches: [TrimmableKVCache]) -> Int {
        var size = tokens.count * MemoryLayout<Int>.size
        
        for cache in kvCaches {
            let evaluatable = cache as any Evaluatable
            let states = evaluatable.innerState()
                for state in states {
                    let elements = state.shape.reduce(1, *)
                    let elementSize = state.dtype == .float16 ? 2 : 4
                    size += elements * elementSize
                }
        }
        
        return size
    }
    
    /// Create a new cache entry
    init(key: String, tokens: [Int], kvCaches: [TrimmableKVCache]) {
        self.key = key
        self.tokens = tokens
        self.kvCaches = kvCaches
        self.createdAt = Date()
        self.lastAccessedAt = Date()
        self.estimatedSizeBytes = Self.estimateSize(tokens: tokens, kvCaches: kvCaches)
    }
}

/// Statistics for prompt cache performance
public struct PromptCacheStats: Sendable {
    public var hits: Int = 0
    public var misses: Int = 0
    public var evictions: Int = 0
    public var totalTokensReused: Int = 0
    public var totalTokensProcessed: Int = 0
    
    public var hitRate: Double {
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : 0
    }
    
    var averageTokensReused: Double {
        return hits > 0 ? Double(totalTokensReused) / Double(hits) : 0
    }
}
