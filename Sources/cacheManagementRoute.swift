import Vapor
import Foundation
import Logging

/// Response model for cache status
struct CacheStatusResponse: Content {
    let enabled: Bool
    let entryCount: Int
    let currentSizeMB: Double
    let maxSizeMB: Int
    let ttlMinutes: Int
    let stats: CacheStatsResponse
}

/// Response model for cache statistics
struct CacheStatsResponse: Content {
    let hits: Int
    let misses: Int
    let evictions: Int
    let hitRate: Double
    let totalTokensReused: Int
    let totalTokensProcessed: Int
    let averageTokensReused: Double
}

/// Response model for cache clear operation
struct CacheClearResponse: Content {
    let success: Bool
    let message: String
}

/// Register cache management endpoints
func registerCacheManagementRoutes(_ app: Application, promptCacheManager: PromptCacheManager?) {
    app.get("v1", "cache", "status") { req async throws -> CacheStatusResponse in
        req.logger.info("Handling /v1/cache/status request")
        
        guard let cacheManager = promptCacheManager else {
            return CacheStatusResponse(
                enabled: false,
                entryCount: 0,
                currentSizeMB: 0,
                maxSizeMB: 0,
                ttlMinutes: 0,
                stats: CacheStatsResponse(
                    hits: 0,
                    misses: 0,
                    evictions: 0,
                    hitRate: 0,
                    totalTokensReused: 0,
                    totalTokensProcessed: 0,
                    averageTokensReused: 0
                )
            )
        }
        
        let stats = await cacheManager.getStats()
        let status = await cacheManager.getCacheStatus()
        let maxSizeMB = await cacheManager.maxCacheSizeMB
        let ttlMinutes = await cacheManager.cacheTTLMinutes
        
        return CacheStatusResponse(
            enabled: true,
            entryCount: status.entryCount,
            currentSizeMB: status.sizeMB,
            maxSizeMB: maxSizeMB,
            ttlMinutes: ttlMinutes,
            stats: CacheStatsResponse(
                hits: stats.hits,
                misses: stats.misses,
                evictions: stats.evictions,
                hitRate: stats.hitRate,
                totalTokensReused: stats.totalTokensReused,
                totalTokensProcessed: stats.totalTokensProcessed,
                averageTokensReused: stats.averageTokensReused
            )
        )
    }
    
    app.delete("v1", "cache") { req async throws -> CacheClearResponse in
        req.logger.info("Handling DELETE /v1/cache request")
        
        guard let cacheManager = promptCacheManager else {
            return CacheClearResponse(
                success: false,
                message: "Prompt cache is not enabled"
            )
        }
        
        await cacheManager.clearCache()
        
        return CacheClearResponse(
            success: true,
            message: "Cache cleared successfully"
        )
    }
}
