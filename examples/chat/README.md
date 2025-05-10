# Chat Model Examples

This directory contains examples for using the LLM API to interact with various language models.

## Files
- `llm_api.py`: The main LLM API implementation
- `model_conversation.py`: Example of model-to-model conversation

## Usage Examples

### Basic Chat
```python
from llm_api import LLMAPI

# Initialize API with a model
api = LLMAPI(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    hf_token="your_hf_token"
)

# Load model
api.load_model()

# Chat with the model
response = api.chat("What is the capital of France?")
print(response)
```

### Model-to-Model Conversation
```python
from model_conversation import ModelConversation

# Initialize conversation
conversation = ModelConversation()

# Setup models
conversation.setup_tinyllama()

# Start conversation
conversation.start_conversation()
```

## Supported Models
- Llama 2 (7B, 13B, 70B)
- TinyLlama
- Mistral
- Phi-2

## Features
1. Easy model loading and initialization
2. Support for multiple model architectures
3. Configurable generation parameters
4. Model-to-model conversation capabilities
5. Streaming response support

## Best Practices
1. Use appropriate model size for your hardware
2. Set proper temperature and top_p values
3. Monitor token usage for cost control
4. Use system prompts for better control
5. Implement proper error handling 