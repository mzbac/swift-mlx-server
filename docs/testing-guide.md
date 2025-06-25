# Swift MLX Server Testing Guide

This guide provides comprehensive testing procedures for the Swift MLX Server, including all endpoints and KV cache quantization features.

## Prerequisites

1. Build the server using the instructions in CLAUDE.md
2. Ensure the executable is located at: `./dist/Build/Products/Release/swift-mlx-server`
3. Have `jq` installed for JSON formatting (optional but recommended)

## Configuring Log Level

The server uses Vapor's logging system which can be configured via the `LOG_LEVEL` environment variable. Available log levels:
- `trace` - Most verbose, includes all log messages
- `debug` - Detailed debugging information
- `info` - General informational messages (default)
- `notice` - Normal but significant events
- `warning` - Warning messages
- `error` - Error conditions
- `critical` - Critical conditions

To run the server with debug logging:
```bash
LOG_LEVEL=debug ./dist/Build/Products/Release/swift-mlx-server --model mlx-community/Qwen3-0.6B-4bit-DWQ-053125 --host 127.0.0.1 --port 8080
```

## Recommended Test Models

- **Text Generation**: `mlx-community/Qwen3-0.6B-4bit-DWQ-053125`
- **Vision Language Model**: `mlx-community/Qwen2.5-VL-3B-Instruct-8bit`
- **Embeddings**: `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`

## Quick Test Script

Save this as `test-server.sh` in the project root:

```bash
#!/bin/bash

# Configuration
SERVER_URL="http://127.0.0.1:8080"
MODEL="${1:-mlx-community/Llama-3.2-1B-Instruct-8bit}"
LOG_LEVEL="${2:-info}"
EXECUTABLE_PATH="./dist/Build/Products/Release/swift-mlx-server"

# Kill any existing server on port 8080
echo "Stopping any existing server on port 8080..."
lsof -ti:8080 | xargs kill -9 2>/dev/null || true

# Start the server with specified log level and any additional arguments
echo "Starting server with $MODEL (log level: $LOG_LEVEL)..."
# Pass through any additional arguments (like --enable-prompt-cache)
LOG_LEVEL=$LOG_LEVEL "$EXECUTABLE_PATH" --model "$MODEL" --host 127.0.0.1 --port 8080 "${@:3}" > server.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start (this may take a while to download the model)..."
MAX_WAIT=60
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s -o /dev/null -w "%{http_code}" "$SERVER_URL/v1/models" | grep -q "200"; then
        echo "Server is ready!"
        break
    fi
    echo -n "."
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done
echo ""

# Check if server is running
if ! ps -p $SERVER_PID > /dev/null; then
    echo "Server failed to start. Check server.log for details:"
    tail -20 server.log
    exit 1
fi

echo "Running tests..."

# Test 1: Basic chat completion
echo -e "\n1. Testing basic chat completion..."
curl -s -X POST "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50
  }' | jq -r '.choices[0].message.content' || echo "Failed"

# Test 2: KV cache quantization
echo -e "\n2. Testing KV cache quantization (4-bit)..."
curl -s -X POST "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain AI briefly."}],
    "max_tokens": 100,
    "kv_bits": 4,
    "kv_group_size": 64,
    "kv_quantization_start": 50
  }' | jq -r '.choices[0].message.content' || echo "Failed"

# Test 3: Prompt cache (if enabled)
if [[ "$@" == *"--enable-prompt-cache"* ]]; then
  echo -e "\n3. Testing prompt cache..."
  
  # First request with a system prompt
  echo "  3a. First request (cache miss)..."
  curl -s -X POST "$SERVER_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
      ],
      "max_tokens": 20
    }' > /dev/null
  
  # Second request with same system prompt (should hit cache)
  echo "  3b. Second request (cache hit)..."
  curl -s -X POST "$SERVER_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 3+3?"}
      ],
      "max_tokens": 20
    }' > /dev/null
  
  # Check cache status
  echo "  3c. Cache status:"
  curl -s -X GET "$SERVER_URL/v1/cache/status" | jq '.stats | {hits, misses, hitRate}'
fi

# Clean up
echo -e "\nCleaning up..."
kill $SERVER_PID 2>/dev/null || true
echo "Tests completed!"
```

### Using the Test Script

Run with default settings (info log level):
```bash
./test-server.sh
```

Run with a specific model and debug logging:
```bash
./test-server.sh mlx-community/Qwen3-0.6B-4bit-DWQ-053125 debug
```

Run with trace logging for maximum verbosity:
```bash
./test-server.sh mlx-community/Qwen3-0.6B-4bit-DWQ-053125 trace
```

Run with prompt cache enabled:
```bash
./test-server.sh mlx-community/Qwen3-0.6B-4bit-DWQ-053125 info --enable-prompt-cache --prompt-cache-size-mb 512
```

## Comprehensive Testing

### 1. Chat Completions Endpoint

#### Basic Test
```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

#### With KV Cache Quantization
```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a detailed explanation about neural networks."}
    ],
    "max_tokens": 500,
    "temperature": 0.7,
    "kv_bits": 4,
    "kv_group_size": 64,
    "kv_quantization_start": 100
  }'
```

#### Streaming Response
```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a story."}],
    "max_tokens": 200,
    "stream": true,
    "kv_bits": 8,
    "kv_group_size": 32,
    "kv_quantization_start": 50
  }'
```

### 2. Text Completions Endpoint

#### Basic Test
```bash
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

#### With KV Cache Quantization
```bash
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "In a world where technology has advanced beyond our wildest dreams,",
    "max_tokens": 400,
    "temperature": 0.8,
    "kv_bits": 4,
    "kv_group_size": 64,
    "kv_quantization_start": 100
  }'
```

### 3. Embeddings Endpoint

The embeddings endpoint follows the OpenAI API specification with the following supported parameters:

- **input** (required): String or array of strings to embed
- **model** (optional): Model ID to use (defaults to the server's embedding model)
- **encoding_format** (optional): "float" or "base64" (defaults to "float")
- **dimensions** (optional): Number of dimensions for the output embeddings
- **user** (optional): Unique identifier for the end-user
- **batch_size** (optional): Custom batch size for processing multiple inputs

#### Single Text Embedding
```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "encoding_format": "float"
  }'
```

#### Batch Embeddings
```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["First text", "Second text", "Third text"],
    "encoding_format": "float",
    "batch_size": 2
  }'
```

#### With All Parameters (OpenAI Compatible)
```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Generate embedding for this text",
    "model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    "encoding_format": "float",
    "dimensions": 1024,
    "user": "user-123"
  }'
```

#### Base64 Encoding Format
```bash
# Test with base64 encoding for binary compatibility
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Test embedding with base64 encoding",
    "encoding_format": "base64"
  }'
```

### 4. Vision Language Model (VLM) Testing

To test VLM capabilities, start the server with the `--vlm` flag:

```bash
./dist/Build/Products/Release/swift-mlx-server --model mlx-community/Qwen2.5-VL-3B-Instruct-8bit --host 127.0.0.1 --port 8080 --vlm
```

#### Test Image Understanding
```bash
# Basic image description
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What do you see in this image? Describe it in detail."},
          {"type": "image", "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/640px-Cat03.jpg"}
        ]
      }
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'

# Image analysis with specific questions
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Count the number of people in this image and describe what they are doing."},
          {"type": "image", "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Frog_on_palm_frond.jpg/640px-Frog_on_palm_frond.jpg"}
        ]
      }
    ],
    "max_tokens": 150,
    "temperature": 0.5
  }'

# Multiple images comparison
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Compare these two images and describe the differences."},
          {"type": "image", "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"},
          {"type": "image", "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Six_weeks_old_cat_(aka).jpg/320px-Six_weeks_old_cat_(aka).jpg"}
        ]
      }
    ],
    "max_tokens": 250,
    "temperature": 0.7
  }'
```

#### Test with Image Resizing
```bash
# Custom image resize dimensions
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Analyze this image and identify any text or numbers visible."},
          {"type": "image", "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Visual_Acuity_Test.png/640px-Visual_Acuity_Test.png"}
        ]
      }
    ],
    "max_tokens": 200,
    "temperature": 0.3,
    "resize": [512, 512]
  }'
```

**Note**: KV cache quantization is only implemented for LLM (text generation) models and is not applicable to VLM (Vision Language Models). Do not use KV cache parameters (`kv_bits`, `kv_group_size`, `kv_quantization_start`) with VLM requests.

### 5. Models Endpoint

```bash
curl -X GET http://127.0.0.1:8080/v1/models
```

## KV Cache Quantization Parameters

The KV cache quantization feature helps reduce memory usage for long contexts. Available parameters:

- **kv_bits**: Number of bits for quantization (2, 4, or 8)
- **kv_group_size**: Group size for quantization (16, 32, 64, etc.)
- **kv_quantization_start**: Token position to start quantization

### Recommended Configurations

1. **Memory-Efficient (4-bit)**:
   ```json
   {
     "kv_bits": 4,
     "kv_group_size": 64,
     "kv_quantization_start": 100
   }
   ```

2. **Balanced (8-bit)**:
   ```json
   {
     "kv_bits": 8,
     "kv_group_size": 32,
     "kv_quantization_start": 200
   }
   ```

3. **Aggressive (2-bit)**:
   ```json
   {
     "kv_bits": 2,
     "kv_group_size": 16,
     "kv_quantization_start": 50
   }
   ```

## Prompt Cache Testing

The prompt cache feature allows the server to reuse KV caches for common prompt prefixes, significantly improving performance for requests with shared context.

### Prompt Cache Configuration

To enable prompt caching, use the following CLI flags:

```bash
./dist/Build/Products/Release/swift-mlx-server \
  --model mlx-community/Qwen3-0.6B-4bit-DWQ-053125 \
  --host 127.0.0.1 \
  --port 8080 \
  --enable-prompt-cache \
  --prompt-cache-size-mb 1024 \
  --prompt-cache-ttl-minutes 30
```

**Configuration Options:**
- `--enable-prompt-cache`: Enable the prompt caching feature
- `--prompt-cache-size-mb`: Maximum cache size in MB (default: 1024)
- `--prompt-cache-ttl-minutes`: Cache entry time-to-live in minutes (default: 30)

### Best Practices for Prompt Cache

1. **Minimum Prompt Length**: The cache is most effective with prompts containing at least 50-100 tokens. Short prompts (< 10 tokens) may actually show negative performance due to cache management overhead.

2. **Optimal Use Cases**:
   - Long system prompts that remain constant across multiple requests
   - Common instruction prefixes in chat applications
   - Repeated context in document analysis or code review scenarios
   - Multi-turn conversations with substantial shared context

3. **Performance Expectations**:
   - With prompts of 100+ tokens: 20-50% performance improvement
   - With prompts of 50-100 tokens: 10-20% performance improvement
   - With prompts < 50 tokens: Minimal or negative improvement

4. **Memory Considerations**:
   - Each cached token requires approximately 4KB of memory (varies by model)
   - Monitor cache evictions in stats to ensure cache size is adequate
   - Consider reducing cache size if memory pressure is high

### Basic Prompt Cache Tests

#### 1. Test Cache Hit Performance

```bash
# Define a long system prompt for effective caching
SYSTEM_PROMPT="You are an advanced AI assistant with extensive expertise in technology, science, and mathematics. Your responses should be detailed, accurate, and helpful. You have deep knowledge in programming languages, software architecture, algorithms, and data structures. When answering questions, provide clear explanations with relevant examples. You excel at breaking down complex concepts into understandable parts."

# First request - cache miss (slower)
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$SYSTEM_PROMPT\"},
      {\"role\": \"user\", \"content\": \"What is machine learning?\"}
    ],
    \"max_tokens\": 100
  }"

# Second request with same system prompt - cache hit (faster)
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$SYSTEM_PROMPT\"},
      {\"role\": \"user\", \"content\": \"What is deep learning?\"}
    ],
    \"max_tokens\": 100
  }"
```

#### 2. Test with Text Completions

```bash
# Common prefix for story generation
# First request - establishes cache
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time in a magical kingdom far away, there lived a wise old wizard named Merlin. He had spent centuries studying the ancient arts and had discovered many secrets. One day, a young apprentice came to him seeking knowledge about",
    "max_tokens": 50
  }'

# Second request - reuses cached prefix
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time in a magical kingdom far away, there lived a wise old wizard named Merlin. He had spent centuries studying the ancient arts and had discovered many secrets. One day, a mysterious stranger arrived at",
    "max_tokens": 50
  }'
```

#### 3. Test Cache with Streaming

```bash
# Streaming with cache enabled
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a creative storyteller."},
      {"role": "user", "content": "Write a short story about AI."}
    ],
    "max_tokens": 200,
    "stream": true
  }'
```

### Cache Management API Tests

#### 1. Check Cache Status

```bash
# Get current cache status and statistics
curl -X GET http://127.0.0.1:8080/v1/cache/status | jq .
```

Expected response:
```json
{
  "enabled": true,
  "entryCount": 2,
  "currentSizeMB": 45.3,
  "maxSizeMB": 1024,
  "ttlMinutes": 30,
  "stats": {
    "hits": 5,
    "misses": 2,
    "evictions": 0,
    "hitRate": 0.714,
    "totalTokensReused": 450,
    "totalTokensProcessed": 650,
    "averageTokensReused": 90.0
  }
}
```

#### 2. Clear Cache

```bash
# Clear all cache entries
curl -X DELETE http://127.0.0.1:8080/v1/cache | jq .
```

Expected response:
```json
{
  "success": true,
  "message": "Cache cleared successfully"
}
```

### Advanced Cache Testing

#### 1. Test with KV Cache Quantization

Prompt cache works seamlessly with KV cache quantization:

```bash
# Request with both prompt cache and KV quantization
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are an AI assistant specialized in explaining complex topics simply."},
      {"role": "user", "content": "Explain quantum computing."}
    ],
    "max_tokens": 300,
    "kv_bits": 4,
    "kv_group_size": 64,
    "kv_quantization_start": 100
  }'
```

#### 2. Test Cache Eviction

```bash
# Generate multiple unique prompts to exceed cache size
for i in {1..20}; do
  curl -X POST http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"messages\": [
        {\"role\": \"system\", \"content\": \"You are test assistant number $i with unique characteristics.\"},
        {\"role\": \"user\", \"content\": \"Hello\"}
      ],
      \"max_tokens\": 500
    }"
done

# Check cache status to see evictions
curl -X GET http://127.0.0.1:8080/v1/cache/status | jq .stats.evictions
```

#### 3. Performance Comparison Script

Save as `test-cache-performance.sh`:

```bash
#!/bin/bash

SERVER_URL="http://127.0.0.1:8080"
ITERATIONS=5

echo "Testing prompt cache performance..."

# Clear cache first
curl -s -X DELETE "$SERVER_URL/v1/cache" > /dev/null

# Test without cache hits (different prompts)
echo -e "\n1. Testing without cache hits (different prompts):"
START_TIME=$(date +%s%N)
for i in $(seq 1 $ITERATIONS); do
  curl -s -X POST "$SERVER_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"messages\": [
        {\"role\": \"system\", \"content\": \"You are assistant number $i.\"},
        {\"role\": \"user\", \"content\": \"Hello\"}
      ],
      \"max_tokens\": 100
    }" > /dev/null
done
END_TIME=$(date +%s%N)
NO_CACHE_TIME=$((($END_TIME - $START_TIME) / 1000000))
echo "Time without cache: ${NO_CACHE_TIME}ms"

# Clear cache
curl -s -X DELETE "$SERVER_URL/v1/cache" > /dev/null

# Test with cache hits (same prompt prefix)
echo -e "\n2. Testing with cache hits (same prompt prefix):"
START_TIME=$(date +%s%N)
for i in $(seq 1 $ITERATIONS); do
  curl -s -X POST "$SERVER_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"messages\": [
        {\"role\": \"system\", \"content\": \"You are a helpful AI assistant.\"},
        {\"role\": \"user\", \"content\": \"Question $i: Tell me a fact.\"}
      ],
      \"max_tokens\": 100
    }" > /dev/null
done
END_TIME=$(date +%s%N)
CACHE_TIME=$((($END_TIME - $START_TIME) / 1000000))
echo "Time with cache: ${CACHE_TIME}ms"

# Calculate improvement
IMPROVEMENT=$(echo "scale=2; (($NO_CACHE_TIME - $CACHE_TIME) * 100) / $NO_CACHE_TIME" | bc)
echo -e "\nPerformance improvement: ${IMPROVEMENT}%"

# Show cache statistics
echo -e "\nCache statistics:"
curl -s -X GET "$SERVER_URL/v1/cache/status" | jq '.stats'
```

## VLM Testing Script

For comprehensive VLM testing, use the dedicated script:

```bash
./test-vlm.sh mlx-community/Qwen2.5-VL-3B-Instruct-8bit
```

This script tests:
- Basic image description
- Specific image analysis questions
- Image resizing capabilities
- VLM with KV cache quantization
- Multiple image comparison
- Complex scene understanding

## Prompt Cache Testing Scripts

The repository includes several comprehensive prompt cache testing scripts:

### 1. Basic Prompt Cache Test (`test-prompt-cache.sh`)

This script tests prompt cache functionality with realistic long prompts that demonstrate actual performance improvements:

```bash
#!/bin/bash

# Configuration
SERVER_URL="http://127.0.0.1:8080"
MODEL="${1:-mlx-community/Qwen3-0.6B-4bit-DWQ-053125}"
EXECUTABLE_PATH="./dist/Build/Products/Release/swift-mlx-server"

echo "=== Prompt Cache Testing Script ==="
echo "Testing model: $MODEL"
echo ""

# Kill any existing server
echo "Stopping any existing server..."
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
sleep 2

# Start server with prompt cache enabled
echo "Starting server with prompt cache enabled..."
"$EXECUTABLE_PATH" --model "$MODEL" --host 127.0.0.1 --port 8080 \
  --enable-prompt-cache --prompt-cache-size-mb 256 --prompt-cache-ttl-minutes 15 > cache-test.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
MAX_WAIT=60
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s -o /dev/null -w "%{http_code}" "$SERVER_URL/v1/models" | grep -q "200"; then
        echo "Server is ready!"
        break
    fi
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [ $WAIT_COUNT -eq $MAX_WAIT ]; then
    echo "Server failed to start. Check cache-test.log"
    exit 1
fi

# Warm-up request to ensure model is loaded
echo -e "\nRunning warm-up request to load model..."
curl -s -X POST "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }' > /dev/null
echo "Warm-up complete."

echo -e "\n=== Test 1: Basic Cache Functionality ==="

# Clear cache to start fresh
echo "Clearing cache..."
curl -s -X DELETE "$SERVER_URL/v1/cache" | jq .

# Test with long system prompt for effective caching
SCIENCE_SYSTEM_PROMPT="You are a helpful AI assistant specialized in science with deep expertise in biology, chemistry, physics, astronomy, and earth sciences. You have extensive knowledge of scientific principles, research methods, and can explain complex scientific concepts in clear, accessible language. When answering questions, you provide accurate, evidence-based information and can discuss both fundamental concepts and cutting-edge research. You're particularly skilled at explaining how different scientific fields interconnect and influence each other."

echo -e "\n1a. First request with long system prompt (cache miss):"
TIME_START=$(date +%s%N)
RESPONSE=$(curl -s -X POST "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$SCIENCE_SYSTEM_PROMPT\"},
      {\"role\": \"user\", \"content\": \"What is photosynthesis?\"}
    ],
    \"max_tokens\": 50
  }")
TIME_END=$(date +%s%N)
TIME_TAKEN=$((($TIME_END - $TIME_START) / 1000000))
echo "Response time: ${TIME_TAKEN}ms"
echo "$RESPONSE" | jq -r '.choices[0].message.content' | head -2

# Test cache hit with same system prompt
echo -e "\n1b. Second request with same long system prompt (cache hit):"
TIME_START=$(date +%s%N)
RESPONSE=$(curl -s -X POST "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$SCIENCE_SYSTEM_PROMPT\"},
      {\"role\": \"user\", \"content\": \"What is cellular respiration?\"}
    ],
    \"max_tokens\": 50
  }")
TIME_END=$(date +%s%N)
TIME_TAKEN=$((($TIME_END - $TIME_START) / 1000000))
echo "Response time: ${TIME_TAKEN}ms (should be faster)"
echo "$RESPONSE" | jq -r '.choices[0].message.content' | head -2

# Check cache stats
echo -e "\n1c. Cache statistics after 2 requests:"
curl -s -X GET "$SERVER_URL/v1/cache/status" | jq '{
  entryCount: .entryCount,
  currentSizeMB: .currentSizeMB,
  stats: .stats
}'

echo -e "\n=== Test 2: Text Completions with Common Prefix ==="

# Test with story prefix
STORY_PREFIX="In the year 2150, humanity had finally achieved faster-than-light travel. The first interstellar colony ship, named Hope's Horizon, was preparing for its maiden voyage to Proxima Centauri. Captain Sarah Chen stood on the bridge, looking at the stars through the viewscreen."

echo -e "\n2a. First completion request:"
TIME_START=$(date +%s%N)
curl -s -X POST "$SERVER_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"$STORY_PREFIX She thought about\",
    \"max_tokens\": 30
  }" | jq -r '.choices[0].text' | tr '\n' ' '
TIME_END=$(date +%s%N)
TIME_TAKEN=$((($TIME_END - $TIME_START) / 1000000))
echo -e "\nTime: ${TIME_TAKEN}ms"

echo -e "\n2b. Second completion with same prefix:"
TIME_START=$(date +%s%N)
curl -s -X POST "$SERVER_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"$STORY_PREFIX The crew was\",
    \"max_tokens\": 30
  }" | jq -r '.choices[0].text' | tr '\n' ' '
TIME_END=$(date +%s%N)
TIME_TAKEN=$((($TIME_END - $TIME_START) / 1000000))
echo -e "\nTime: ${TIME_TAKEN}ms (should be faster)"

echo -e "\n=== Test 3: Cache with KV Quantization ==="

echo -e "\n3. Testing cache with KV quantization enabled:"
AI_SYSTEM_PROMPT="You are an expert in artificial intelligence, machine learning, deep learning, and neural networks. You have comprehensive knowledge of AI history, current state-of-the-art techniques, and emerging trends. You can explain complex AI concepts at various levels of detail, from beginner-friendly overviews to technical deep-dives. You're familiar with popular frameworks like TensorFlow, PyTorch, and MLX, as well as the mathematical foundations underlying AI algorithms. You can discuss practical applications, ethical considerations, and the future potential of AI technology."

curl -s -X POST "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$AI_SYSTEM_PROMPT\"},
      {\"role\": \"user\", \"content\": \"Explain neural networks in simple terms.\"}
    ],
    \"max_tokens\": 100,
    \"kv_bits\": 4,
    \"kv_group_size\": 64,
    \"kv_quantization_start\": 50
  }" > /dev/null

echo "Request completed. Checking cache status:"
curl -s -X GET "$SERVER_URL/v1/cache/status" | jq '.stats'

echo -e "\n=== Test 4: Cache Performance Comparison ==="

# Clear cache for fair comparison
curl -s -X DELETE "$SERVER_URL/v1/cache" > /dev/null

# Test without cache benefit (different prompts each time)
echo -e "\n4a. Ten requests with different prompts (no cache benefit):"
TOTAL_TIME=0
for i in {1..10}; do
    TIME_START=$(date +%s%N)
    curl -s -X POST "$SERVER_URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"messages\": [
          {\"role\": \"user\", \"content\": \"Random question $i: What is $RANDOM?\"}
        ],
        \"max_tokens\": 20
      }" > /dev/null
    TIME_END=$(date +%s%N)
    TIME_TAKEN=$((($TIME_END - $TIME_START) / 1000000))
    TOTAL_TIME=$(($TOTAL_TIME + $TIME_TAKEN))
    echo "  Request $i: ${TIME_TAKEN}ms"
done
AVG_NO_CACHE=$(($TOTAL_TIME / 10))
echo "Average time without cache benefit: ${AVG_NO_CACHE}ms"

# Clear cache again
curl -s -X DELETE "$SERVER_URL/v1/cache" > /dev/null

# Test with cache benefit (same long prefix)
echo -e "\n4b. Ten requests with same long system prompt (cache benefit):"
LONG_SYSTEM_PROMPT="You are an advanced AI assistant with extensive expertise across multiple domains including technology, science, mathematics, and programming. Your responses should be detailed, accurate, and helpful. You have deep knowledge in programming languages including Python, JavaScript, Swift, C++, Java, and many others. You can assist with code reviews, debugging complex issues, algorithm design, software architecture, system design, and performance optimization. When answering questions, you should provide clear explanations with relevant examples when appropriate. You excel at breaking down complex concepts into understandable parts and can adapt your explanations based on the user's level of expertise. Your goal is to be as helpful as possible while maintaining accuracy and clarity in all your responses."

TOTAL_TIME=0
for i in {1..10}; do
    TIME_START=$(date +%s%N)
    curl -s -X POST "$SERVER_URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"messages\": [
          {\"role\": \"system\", \"content\": \"$LONG_SYSTEM_PROMPT\"},
          {\"role\": \"user\", \"content\": \"Question $i: Tell me about item $i\"}
        ],
        \"max_tokens\": 20
      }" > /dev/null
    TIME_END=$(date +%s%N)
    TIME_TAKEN=$((($TIME_END - $TIME_START) / 1000000))
    TOTAL_TIME=$(($TOTAL_TIME + $TIME_TAKEN))
    echo "  Request $i: ${TIME_TAKEN}ms"
done
AVG_WITH_CACHE=$(($TOTAL_TIME / 10))
echo "Average time with cache benefit: ${AVG_WITH_CACHE}ms"

# Calculate improvement
if [ $AVG_NO_CACHE -gt 0 ]; then
    IMPROVEMENT=$(echo "scale=2; (($AVG_NO_CACHE - $AVG_WITH_CACHE) * 100) / $AVG_NO_CACHE" | bc)
    echo -e "\nPerformance improvement: ${IMPROVEMENT}%"
fi

echo -e "\n=== Final Cache Statistics ==="
curl -s -X GET "$SERVER_URL/v1/cache/status" | jq .

# Cleanup
echo -e "\nCleaning up..."
kill $SERVER_PID 2>/dev/null || true
echo "Cache testing completed!"
```

### Using the Prompt Cache Test Script

Run with default model:
```bash
chmod +x test-prompt-cache.sh
./test-prompt-cache.sh
```

Run with a specific model:
```bash
./test-prompt-cache.sh mlx-community/Qwen3-0.6B-4bit-DWQ-053125
```

### 2. Extreme Cache Test (`test-extreme-cache.sh`)

This script tests prompt cache with very long prompts (200+ tokens) to demonstrate maximum performance gains:

```bash
#!/bin/bash
./test-extreme-cache.sh mlx-community/Qwen3-0.6B-4bit-DWQ-053125
```

Key features:
- Tests with extremely long system prompts (200+ tokens)
- Shows per-request cache statistics
- Automatically manages server lifecycle
- Demonstrates 20-50% performance improvements
- Compares cached vs non-cached performance

### 3. Cache Quality Test (`test-cache-quality.sh`)

This script validates that responses maintain quality when using the cache:

```bash
#!/bin/bash
./test-cache-quality.sh mlx-community/Qwen3-0.6B-4bit-DWQ-053125
```

Tests include:
- Factual questions to verify accuracy
- Multiple diverse questions with same system prompt
- Per-request cache hit statistics
- Response quality validation
- Ensures cache doesn't produce "garbage" output

## Automated Test Suite

Create `run-all-tests.sh`:

```bash
#!/bin/bash

# Run text generation tests with different models
echo "Testing text generation models..."
./test-server.sh mlx-community/Qwen3-0.6B-4bit-DWQ-053125
./test-server.sh mlx-community/Qwen3-0.6B-4bit-DWQ-053125

# Run VLM tests
echo -e "\nTesting Vision Language Model..."
./test-vlm.sh mlx-community/Qwen2.5-VL-3B-Instruct-8bit

# Run prompt cache tests
echo -e "\nTesting prompt cache functionality..."
./test-prompt-cache.sh mlx-community/Qwen3-0.6B-4bit-DWQ-053125

# Test embedding server
echo -e "\nTesting embeddings..."
EXECUTABLE_PATH="./dist/Build/Products/Release/swift-mlx-server"
LOG_LEVEL=info $EXECUTABLE_PATH --model mlx-community/Qwen3-0.6B-4bit-DWQ-053125 \
  --embedding-model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \
  --host 127.0.0.1 --port 8080 > embedding-server.log 2>&1 &
SERVER_PID=$!
sleep 10

# Test embeddings
curl -s -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Test embedding"}' | jq .

kill $SERVER_PID

# Test with prompt cache and embeddings together
echo -e "\nTesting server with both prompt cache and embeddings..."
LOG_LEVEL=info $EXECUTABLE_PATH --model mlx-community/Qwen3-0.6B-4bit-DWQ-053125 \
  --embedding-model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \
  --enable-prompt-cache --prompt-cache-size-mb 256 \
  --host 127.0.0.1 --port 8080 > combined-server.log 2>&1 &
SERVER_PID=$!
sleep 10

# Test both features
echo "Testing chat with cache..."
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 20
  }' | jq -r '.choices[0].message.content'

echo "Testing embeddings..."
curl -s -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Test embedding with cache"}' | jq '.data[0].embedding[:5]'

echo "Cache status:"
curl -s -X GET http://127.0.0.1:8080/v1/cache/status | jq '.stats'

kill $SERVER_PID
echo "All tests completed!"
```

## Troubleshooting

1. **Server fails to start**: Check `server.log` for errors
   - If the log is empty or has minimal information, run with `LOG_LEVEL=debug` or `LOG_LEVEL=trace`
   - Example: `LOG_LEVEL=debug ./dist/Build/Products/Release/swift-mlx-server --model mlx-community/Qwen3-0.6B-4bit-DWQ-053125 --host 127.0.0.1 --port 8080`
2. **Model download issues**: Ensure internet connectivity and sufficient disk space
3. **Port already in use**: Run `lsof -ti:8080 | xargs kill -9` to free the port
4. **Memory issues**: Use smaller models or enable KV cache quantization
5. **Debugging API requests**: Enable debug logging to see detailed request/response information
   - The server logs will show incoming requests, processing steps, and any errors
   - Use `LOG_LEVEL=trace` for the most detailed output

### Prompt Cache Troubleshooting

1. **Cache not showing hits**:
   - Ensure `--enable-prompt-cache` flag is set when starting the server
   - Check that prompts share a common prefix (system prompts work well)
   - Verify cache status endpoint is accessible: `curl http://127.0.0.1:8080/v1/cache/status`
   - Temperature and other generation parameters must match for cache hits

2. **High memory usage with cache**:
   - Reduce cache size with `--prompt-cache-size-mb` (default: 1024MB)
   - Monitor cache evictions in stats to see if size limit is being hit
   - Use shorter TTL with `--prompt-cache-ttl-minutes` to expire old entries

3. **Cache performance not improving**:
   - Ensure prompts have substantial common prefixes (at least 10-20 tokens)
   - Check cache hit rate in statistics - low hit rate indicates prompts are too diverse
   - For best results, use consistent system prompts across requests

4. **Cache API returns "not enabled"**:
   - The server must be started with `--enable-prompt-cache` flag
   - Check server logs to confirm cache initialization
   - Verify with: `curl http://127.0.0.1:8080/v1/cache/status | jq .enabled`

5. **Unexpected cache behavior**:
   - Clear cache with `curl -X DELETE http://127.0.0.1:8080/v1/cache`
   - Enable debug logging to see cache decisions: `LOG_LEVEL=debug`
   - Check that model parameters (temperature, top_p) are consistent

## Performance Testing

For load testing, use `wrk`:

```bash
# Install wrk if needed: brew install wrk

# Test throughput
wrk -t4 -c20 -d30s -s - http://127.0.0.1:8080/v1/chat/completions <<EOF
wrk.method = "POST"
wrk.body = '{"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'
wrk.headers["Content-Type"] = "application/json"
EOF
```