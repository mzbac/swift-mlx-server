// swift-tools-version: 5.10.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "swift-mlx-server",
  platforms: [.macOS(.v14), .iOS(.v16)],
  dependencies: [
    .package(
      url: "https://github.com/apple/swift-argument-parser.git", .upToNextMajor(from: "1.3.0")),
    .package(url: "https://github.com/vapor/vapor.git", from: "4.0.0"),
    .package(url: "https://github.com/ml-explore/mlx-swift-examples.git", branch: "main"),
    .package(url: "https://github.com/mzbac/mlx.embeddings.git", branch: "main"),
    .package(url: "https://github.com/realm/SwiftLint.git", .upToNextMajor(from: "0.59.1")),
  ],
  targets: [
    .executableTarget(
      name: "swift-mlx-server",
      dependencies: [
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "Vapor", package: "vapor"),
        .product(name: "MLXLLM", package: "mlx-swift-examples"),
        .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
        .product(name: "MLXVLM", package: "mlx-swift-examples"),
        .product(name: "mlx_embeddings", package: "mlx.embeddings"),
      ]
    ),
    .testTarget(
      name: "swift-mlx-serverTests",
      dependencies: [
        "swift-mlx-server",
        .product(name: "XCTVapor", package: "vapor"),
      ]
    )
  ]
)
