import Foundation
import MLX
import MLXLMCommon

/// Simplified wrapper that makes QuantizedKVCache trimmable by delegating to a new instance
public final class CachedQuantizedKVCache: TrimmableKVCache, QuantizedKVCacheProtocol {
    private var cache: QuantizedKVCache
    private var savedTokens: [Int] = []
    
    public var offset: Int { cache.offset }
    public var maxSize: Int? { cache.maxSize }
    public let groupSize: Int
    public let bits: Int
    
    public init(groupSize: Int, bits: Int) {
        self.groupSize = groupSize
        self.bits = bits
        self.cache = QuantizedKVCache(groupSize: groupSize, bits: bits)
    }
    
    public func innerState() -> [MLXArray] {
        cache.innerState()
    }
    
    public var currentTokenCount: Int {
        return offset
    }
    
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        return cache.update(keys: keys, values: values)
    }
    
    public func updateQuantized(keys: MLXArray, values: MLXArray) -> ((MLXArray, MLXArray, MLXArray), (MLXArray, MLXArray, MLXArray)) {
        return cache.updateQuantized(keys: keys, values: values)
    }
    
    public func getQuantizedState() -> ((MLXArray, MLXArray, MLXArray), (MLXArray, MLXArray, MLXArray))? {
        return cache.getQuantizedState()
    }
    
    public var state: [MLXArray] {
        get { cache.state }
        set { cache.state = newValue }
    }
    
    public var metaState: [String] {
        get { cache.metaState }
        set { cache.metaState = newValue }
    }
    
    public var isTrimmable: Bool { true }
    
    public var isEmpty: Bool { offset == 0 }
    
    @discardableResult
    public func trim(_ n: Int) -> Int {
        let actualTrim = min(n, offset)
        trim(count: actualTrim)
        return actualTrim
    }
    
    public func trim(count: Int) {
        guard !isEmpty, offset >= count else { return }
        
        let newCache = QuantizedKVCache(groupSize: groupSize, bits: bits)
        
        let currentState = cache.state
        
        if currentState.count >= 6 {
            let keysWq = currentState[0]
            let keysScales = currentState[1]
            let keysBiases = currentState[2]
            let valuesWq = currentState[3]
            let valuesScales = currentState[4]
            let valuesBiases = currentState[5]
            
            let newOffset = offset - count
            
            let trimmedKeysWq = keysWq[.ellipsis, count..., 0...]
            let trimmedValuesWq = valuesWq[.ellipsis, count..., 0...]
            
            let scaleOffset = count / groupSize
            let trimmedKeysScales = keysScales[.ellipsis, scaleOffset..., 0...]
            let trimmedValuesScales = valuesScales[.ellipsis, scaleOffset..., 0...]
            
            let trimmedKeysBiases = keysBiases[.ellipsis, count..., 0...]
            let trimmedValuesBiases = valuesBiases[.ellipsis, count..., 0...]
            
            newCache.state = [
                trimmedKeysWq, trimmedKeysScales, trimmedKeysBiases,
                trimmedValuesWq, trimmedValuesScales, trimmedValuesBiases
            ]
            
            newCache.metaState = ["offset:\(newOffset)"]
        }
        
        self.cache = newCache
    }
}
