import Foundation
import MLX
import MLXLMCommon

/// Wrapper around KVCacheSimple that adds trimming capability for prompt caching
public final class CachedKVCacheSimple: TrimmableKVCache, Evaluatable {
    private var keys: MLXArray?
    private var values: MLXArray?
    public private(set) var offset = 0
    private let step = 256
    
    public var maxSize: Int? { nil }
    
    public func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }
    
    public var currentTokenCount: Int {
        return offset
    }
    
    public var state: [MLXArray] {
        get { innerState() }
        set { 
            if newValue.count >= 2 {
                self.keys = newValue[0]
                self.values = newValue[1]
                self.offset = newValue[0].dim(2)
            }
        }
    }
    
    public var metaState: [String] {
        get { ["offset:\(offset)", "step:\(step)"] }
        set {
            for item in newValue where item.hasPrefix("offset:") {
                self.offset = Int(item.dropFirst(7)) ?? 0
            }
        }
    }
    
    public var isTrimmable: Bool { true }
    
    public var isEmpty: Bool { offset == 0 }
    
    @discardableResult
    public func trim(_ n: Int) -> Int {
        let actualTrim = min(n, offset)
        trim(count: actualTrim)
        return actualTrim
    }
    
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previousOffset = self.offset
        
        let shouldExpand = if let currentKeys = self.keys, (previousOffset + keys.dim(2)) > currentKeys.dim(2) {
            true
        } else {
            self.keys == nil
        }
        
        if shouldExpand {
            expandCache(keyTemplate: keys, valueTemplate: values)
        }
        
        self.offset += keys.dim(2)
        
        self.keys?[0..., 0..., previousOffset ..< self.offset, 0...] = keys
        self.values?[0..., 0..., previousOffset ..< self.offset, 0...] = values
        
        guard let keys = self.keys, let values = self.values else {
            return (MLXArray(), MLXArray())
        }
        return (
            keys[0..., 0..., ..<self.offset, 0...],
            values[0..., 0..., ..<self.offset, 0...]
        )
    }
    
    public func trim(count: Int) {
        guard !isEmpty, offset >= count else { return }
        
        let newOffset = offset - count
        
        if let keys = self.keys, let values = self.values {
            let trimmedKeys = keys[0..., 0..., ..<newOffset, 0...]
            let trimmedValues = values[0..., 0..., ..<newOffset, 0...]
            
            eval(trimmedKeys, trimmedValues)
            
            self.keys = trimmedKeys
            self.values = trimmedValues
        }
        
        self.offset = newOffset
    }
    
    private func expandCache(keyTemplate: MLXArray, valueTemplate: MLXArray) {
        let (B, kvHeads, S, kHeadDim) = keyTemplate.shape4
        let vHeadDim = valueTemplate.dim(3)
        
        let nSteps = (step + S - 1) / step
        let kShape = [B, kvHeads, nSteps * step, kHeadDim]
        let vShape = [B, kvHeads, nSteps * step, vHeadDim]
        
        let newK = MLXArray.zeros(kShape, dtype: keyTemplate.dtype)
        let newV = MLXArray.zeros(vShape, dtype: valueTemplate.dtype)
        
        if var currentKeys = self.keys, var currentValues = self.values {
            if !offset.isMultiple(of: step) {
                currentKeys = currentKeys[0..., 0..., ..<offset, 0...]
                currentValues = currentValues[0..., 0..., ..<offset, 0...]
            }
            self.keys = concatenated([currentKeys, newK], axis: 2)
            self.values = concatenated([currentValues, newV], axis: 2)
        } else {
            self.keys = newK
            self.values = newV
        }
    }
}
