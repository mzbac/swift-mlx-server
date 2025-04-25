# swift-mlx-server

A Swift-based server application designed to provide an OpenAI-compatible text completion API with support for both text-only and visual language models (VLM).

## Usage

To run the server:

```
swift-mlx-server --model hf/model/id --host 127.0.0.1 --port 8080 [--vlm]
```
Replace `hf/model/id` with the Hugging Face model ID. Adjust the host and port as necessary to fit your setup. Use the `--vlm` flag when running with vision language models that support multi-modal inputs.

# API Endpoints

The server provides two main endpoints:

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
  "repetition_context_size": 20
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
  "repetition_context_size": 20
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

## Request Parameters

Both endpoints accept the following parameters:

- `model` (Optional): The model to use for generation. If not provided, the server uses the model specified at startup.

- `stop` (Optional): An array of strings or a single string. These are sequences of tokens on which the generation should stop.

- `max_tokens` (Optional): An integer specifying the maximum number of tokens to generate. Defaults to 100.

- `stream` (Optional): A boolean indicating if the response should be streamed. If true, responses are sent as they are generated. Defaults to false.

- `temperature` (Optional): A float specifying the sampling temperature. Higher values like 0.8 make output more random, lower values like 0.2 make it more deterministic. Defaults to 0.7.

- `top_p` (Optional): A float specifying the nucleus sampling parameter. Defaults to 0.9.

- `repetition_penalty` (Optional): Applies a penalty to repeated tokens to reduce repetition. Defaults to 1.0.

- `repetition_context_size` (Optional): The size of the context window for applying repetition penalty. Defaults to 20.

### VLM-Specific Parameters

When running in VLM mode with the `--vlm` flag:

- `resize` (Optional): An array of one or two integers specifying the dimensions to resize images to. If one value is provided, it's used for both width and height. If two values are provided, they represent [width, height].

## Multi-Modal Content Format

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
