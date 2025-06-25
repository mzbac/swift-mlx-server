import Foundation
import MLX
import MLXLMCommon

/// Protocol that adds trimming capability to KVCache implementations
protocol TrimmableKVCache: KVCache {
    /// Remove the specified number of tokens from the end of the cache
    func trim(count: Int)
    
    /// Get the current number of cached tokens
    var currentTokenCount: Int { get }
}

/// Extension to provide default implementation where possible
extension TrimmableKVCache {
    var currentTokenCount: Int {
        let evaluatable = self as any Evaluatable
        if let state = evaluatable.innerState().first {
            return state.dim(2)
        }
        return 0
    }
}
