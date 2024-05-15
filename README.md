# swift-mlx-server

A Swift-based server application designed to provide an OpenAI-compatible text completion API.

## Usage

To run the server:

```
swift-mlx-server --model hf/model/id --host 127.0.0.1 --port 8080
```
Replace `hf/model/id` with the Hugging Face model ID. Adjust the host and port as necessary to fit your setup.


# API Endpoints

- `POST /v1/completions`:Generates and returns a text completion for the given prompt. For request details and parameters, refer to the OpenAI API Completions documentation https://platform.openai.com/docs/api-reference/completions/create.

## Request Fields

- stop: (Optional) An array of strings or a single string. Thesse are sequences of tokens on which the generation should stop.

- max_tokens: (Optional) An integer specifying the maximum number of tokens to generate. Defaults to 100.

- stream: (Optional) A boolean indicating if the response should be streamed. If true, responses are sent as they are generated. Defaults to false.

- temperature: (Optional) A float specifying the sampling temperature. Defaults to 1.0.

- top_p: (Optional) A float specifying the nucleus sampling parameter. Defaults to 1.0.

- repetition_penalty: (Optional) Applies a penalty to repeated tokens. Defaults to 1.0.

- repetition_context_size: (Optional) The size of the context window for applying repetition penalty. Defaults to 20.