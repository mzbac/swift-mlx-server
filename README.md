# swift-mlx-server

A Swift-based server application designed to provide an OpenAI-compatible text completion API with support for both text-only and vision language models (VLM). It also supports generating text embeddings.

## Usage

To run the server:

```
swift-mlx-server --model hf/model/id --host 127.0.0.1 --port 8080 [--vlm] [--embedding-model hf/embedding/model/id]
```
Replace `hf/model/id` with the Hugging Face model ID for text generation. Adjust the host and port as necessary to fit your setup. Use the `--vlm` flag when running with vision language models that support multi-modal inputs. Use `--embedding-model` to specify a model for the embeddings endpoint.

# API Endpoints

The server provides three main endpoints:

## Text Completions

- `POST /v1/completions`: Generates and returns a text completion for the given prompt.

### Request Body

```json
{
  "model": "model-name",
  "prompt": "Your text prompt goes here",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false,
  "stop": ["###", "END"],
  "repetition_penalty": 1.1,
  "repetition_context_size": 20,
  "kv_bits": 8,
  "kv_group_size": 64,
  "kv_quantization_start": 1000
}
```

### Response

```json
{
  "model": "model-name",
  "choices": [
    {
      "text": "Generated completion text",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 42,
    "total_tokens": 52
  }
}
```

## Chat Completions

- `POST /v1/chat/completions`: Generates a response based on a series of messages in a conversation.

### Request Body

For text-only models:

```json
{
  "model": "model-name",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, what can you help me with?"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false,
  "stop": ["###", "END"],
  "repetition_penalty": 1.1,
  "repetition_context_size": 20,
  "kv_bits": 4,
  "kv_quantization_start": 500
}
```

For visual language models (requires `--vlm` flag):

```json
{
  "model": "model-name",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
      {"type": "text", "text": "What do you see in this image?"},
      {"type": "image", "image_url": "https://example.com/image.jpg"},
      {"type": "video", "video_url": "https://example.com/video.mp4"},
      {"type": "text", "text": "Please describe it in detail."}
    ]}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false,
  "stop": ["###", "END"],
  "repetition_penalty": 1.1,
  "repetition_context_size": 20,
  "resize": [512, 512]
}
```

### Response

```json
{
  "id": "chatcmpl-123456",
  "created": 1689809600,
  "model": "model-name",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm an AI assistant and I can help with a variety of tasks..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 23,
    "completion_tokens": 42,
    "total_tokens": 65
  }
}
```

## Embeddings

- `POST /v1/embeddings`: Generates embedding vectors for the given input text(s). Requires the server to be started with the `--embedding-model` option.

### Request Body

```json
{
  "input": "Your text string goes here",
  "model": "embedding-model-name",
  "encoding_format": "float",
  "dimensions": 1024,
  "user": "user-id-123",
  "batch_size": 32
}
```

Alternatively, `input` can be an array of strings:

```json
{
  "input": ["First text string", "Second text string"],
  "model": "embedding-model-name"
}
```

### Response

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0023, -0.0034, /* ... */, 0.0123],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": "UklGR...",
      "index": 1
    }
  ],
  "model": "embedding-model-name",
  "usage": {
    "prompt_tokens": 15,
    "total_tokens": 15
  }
}
```

## Request Parameters

The endpoints accept the following parameters:

### Common Parameters (Completions & Chat)

- `model` (Optional): The model to use for generation. If not provided, the server uses the model specified at startup. For embeddings, this refers to the embedding model.

- `stop` (Optional, Completions & Chat): An array of strings or a single string. These are sequences of tokens on which the generation should stop.

- `max_tokens` (Optional, Completions & Chat): An integer specifying the maximum number of tokens to generate. Defaults to 100.

- `stream` (Optional, Completions & Chat): A boolean indicating if the response should be streamed. If true, responses are sent as they are generated. Defaults to false.

- `temperature` (Optional, Completions & Chat): A float specifying the sampling temperature. Higher values like 0.8 make output more random, lower values like 0.2 make it more deterministic. Defaults to 0.7.

- `top_p` (Optional, Completions & Chat): A float specifying the nucleus sampling parameter. Defaults to 0.9.

- `repetition_penalty` (Optional, Completions & Chat): Applies a penalty to repeated tokens to reduce repetition. Defaults to 1.0.

- `repetition_context_size` (Optional, Completions & Chat): The size of the context window for applying repetition penalty. Defaults to 20.

### KV Cache Quantization Parameters (Completions & Chat)

These parameters enable memory-efficient generation for long contexts by quantizing the key-value cache:

- `kv_bits` (Optional): Number of bits for KV cache quantization. Supported values are 4 or 8. When not specified, no quantization is applied. 4-bit quantization saves ~75% memory, 8-bit saves ~50%.

- `kv_group_size` (Optional): Group size for quantization. Must be positive and divisible by 8. Defaults to 64.

- `kv_quantization_start` (Optional): Number of tokens after which to start quantizing the KV cache. Defaults to 5000. Set to 0 to quantize from the beginning.

Example usage for long-context generation:
```json
{
  "prompt": "Long text here...",
  "max_tokens": 2000,
  "kv_bits": 4,
  "kv_group_size": 64,
  "kv_quantization_start": 1000
}
```

### Embeddings Parameters

- `input` (Required, Embeddings): The input text or array of texts to embed.

- `encoding_format` (Optional, Embeddings): The format to return the embeddings in. Can be `float` (default) or `base64`.

- `dimensions` (Optional, Embeddings): The desired number of dimensions for the output embeddings. (Note: Currently not implemented, model's default dimension is used).

- `user` (Optional, Embeddings): A unique identifier representing your end-user, which can help OpenAI monitor and detect abuse.

- `batch_size` (Optional, Embeddings): The number of input texts to process in a single batch. Defaults to processing all inputs in one batch.

### VLM-Specific Parameters (Chat)

When running in VLM mode with the `--vlm` flag:

- `resize` (Optional): An array of one or two integers specifying the dimensions to resize images to. If one value is provided, it's used for both width and height. If two values are provided, they represent [width, height].

## Models Endpoint

- `GET /v1/models`: Returns a list of available models on the server.

### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "model-id",
      "object": "model",
      "created": 1686935002,
      "owned_by": "organization-owner"
    },
    {
      "id": "embedding-model-id",
      "object": "model",
      "created": 1686935002,
      "owned_by": "organization-owner"
    }
  ]
}
```

## Multi-Modal Content Format (Chat)

When using VLM mode, message content can be structured in different ways:

- Simple text string: `"content": "Your text here"`
- Array of content fragments:
  ```json
  "content": [
    {"type": "text", "text": "Text content here"},
    {"type": "image", "image_url": "https://example.com/image.jpg"},
    {"type": "video", "video_url": "https://example.com/video.mp4"}
  ]
  ```

The server will process these fragments appropriately for supported visual language models.
