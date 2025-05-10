# Chat Models

The chat module provides a high-level interface for working with various language models, enabling natural language interactions and conversation management. This module supports multiple model architectures and provides utilities for context management and response streaming.

## Quick Start

```python
from OPENtransformer import LLMAPI, ModelConversation

# Initialize the API
chat = LLMAPI(model_name="meta-llama/Llama-2-7b-chat-hf")   ###random example model fit one that fits in your memory constraints 
chat.load_model()

# Create a conversation
conversation = ModelConversation(chat)

# Chat with the model
response = conversation.chat("What is machine learning?")
print(response)
```

## API Reference

### LLMAPI

The main class for interacting with language models.

#### Methods

##### `__init__(self, model_name=None, config=None)`
Initialize the chat API with optional model name and configuration.

Parameters:
- `model_name` (str, optional): Name or path of the model to use
- `config` (dict, optional): Configuration dictionary for the API

##### `load_model(self, model_name=None)`
Load a language model.

Parameters:
- `model_name` (str, optional): Name or path of the model to load

##### `generate(self, prompt, max_length=100, temperature=0.7)`
Generate text from a prompt.

Parameters:
- `prompt` (str): Input text prompt
- `max_length` (int, optional): Maximum length of generated text
- `temperature` (float, optional): Sampling temperature

Returns:
- `str`: Generated text

### ModelConversation

Class for managing conversations with language models.

#### Methods

##### `__init__(self, model)`
Initialize a conversation with a language model.

Parameters:
- `model` (LLMAPI): Initialized language model API

##### `chat(self, message, system_prompt=None)`
Send a message and get a response.

Parameters:
- `message` (str): User message
- `system_prompt` (str, optional): System prompt to guide the conversation

Returns:
- `str`: Model's response

##### `reset(self)`
Reset the conversation history.

## Advanced Usage

### Streaming Responses

```python
from OPENtransformer import LLMAPI

chat = LLMAPI()
chat.load_model()

# Enable streaming
for token in chat.generate_stream("Tell me a story about a robot"):
    print(token, end="", flush=True)
```

### Custom Model Configuration

```python
chat = LLMAPI(
    config={
        "temperature": 0.8,
        "top_p": 0.9,
        "max_length": 200,
        "device": "cuda"
    }
)
```

### Conversation Management

```python
from OPENtransformer import LLMAPI, ModelConversation

chat = LLMAPI()
conversation = ModelConversation(chat)

# Set system prompt
conversation.set_system_prompt("You are a helpful AI assistant.")

# Chat with context
response1 = conversation.chat("What is Python?")
response2 = conversation.chat("Can you give me an example?")

# Reset conversation
conversation.reset()
```

## Best Practices

1. **Context Management**
   - Use system prompts to guide model behavior
   - Maintain conversation history appropriately
   - Clear context when switching topics

2. **Performance Optimization**
   - Use appropriate batch sizes
   - Enable model quantization when possible
   - Implement caching for frequent queries

3. **Error Handling**
   - Handle model loading errors
   - Implement retry logic for API calls
   - Validate input parameters

4. **Resource Management**
   - Clean up resources when done
   - Monitor memory usage
   - Use context managers for automatic cleanup

## Examples

### Basic Chat

```python
from OPENtransformer import LLMAPI, ModelConversation

chat = LLMAPI(model_name="meta-llama/Llama-2-7b-chat-hf")
chat.load_model()

conversation = ModelConversation(chat)
response = conversation.chat(
    "Explain the concept of neural networks in simple terms",
    system_prompt="You are a helpful AI tutor."
)
print(response)
```

### Multi-turn Conversation

```python
from OPENtransformer import LLMAPI, ModelConversation

chat = LLMAPI()
conversation = ModelConversation(chat)

# First turn
response1 = conversation.chat("Let's talk about artificial intelligence.")
print("AI:", response1)

# Second turn
response2 = conversation.chat("What are its main applications?")
print("AI:", response2)

# Third turn
response3 = conversation.chat("Can you elaborate on machine learning?")
print("AI:", response3)
```

### Custom Model Integration

```python
from OPENtransformer import LLMAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load custom model
model = AutoModelForCausalLM.from_pretrained("your-custom-model")
tokenizer = AutoTokenizer.from_pretrained("your-custom-model")

# Initialize API with custom model
chat = LLMAPI(
    config={
        "model": model,
        "tokenizer": tokenizer,
        "device": "cuda"
    }
)
```

## Troubleshooting

Common issues and solutions:

1. **Model Loading Errors**
   - Check model path
   - Verify model compatibility
   - Ensure sufficient disk space

2. **Memory Issues**
   - Reduce batch size
   - Enable model quantization
   - Use model offloading

3. **Response Quality**
   - Adjust temperature
   - Improve prompt quality
   - Use appropriate system prompts

4. **Performance Issues**
   - Enable model optimization
   - Use appropriate hardware
   - Implement caching

For more information and examples, please refer to the [examples directory](../examples/README.md). 