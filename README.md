# swift-mlx-server

A Swift-based server application designed to provide an OpenAI-compatible text completion API.

## Usage

To run the server:

```
swift-mlx-server --model path/to/your/model --host 127.0.0.1 --port 8080
```
Replace `path/to/your/model` with the actual path to your model files or an Hugging Face model ID. Adjust the host and port as necessary to fit your setup.


# API Endpoints

- `POST /v1/completions`:Generates and returns a text completion for the given prompt. For request details and parameters, refer to the OpenAI API Completions documentation https://platform.openai.com/docs/api-reference/completions/create.
