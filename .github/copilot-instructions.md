# GitHub Copilot Instructions

## Project Context
- This project is a Swift server implementation for MLX models
- Focus on performance and type safety
- Target platforms: macOS and iOS with Apple Silicon

## Code Style Guidelines

### Swift Best Practices
- Use Swift 6.0+ features
- Prefer value types over reference types where appropriate
- Use structured concurrency with async/await patterns
- Leverage Swift's strong type system (avoid forced unwrapping)
- Use property wrappers for repeated patterns
- Apply the Actor model for concurrency management
- Use Swift Distributed Actors for networked components
- Implement Swift Macros for repetitive code patterns
- Follow Swift's official style guide for naming conventions

### Comments
- Do NOT add any comments

### Architecture
- Follow SOLID principles
- Use protocol-oriented design
- Implement dependency injection
- Separate data, business logic, and presentation layers
- Use proper error handling with Swift's Result type or try/catch
- Design for testability

### Performance
- Optimize for Apple Silicon
- Consider memory footprint for ML operations
- Use lazy loading where appropriate
- Implement proper caching strategies
- Profile and optimize hot paths

### Testing
- Write unit tests for core functionality
- Use XCTest framework
- Practice TDD where possible
- Mock external dependencies appropriately