#!/usr/bin/env python3

import sys
import os
import time
from pathlib import Path

# Add project root to Python path (now two levels up from test file)
project_root = str(Path(__file__).absolute().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from OPENtransformer.chat.llm_api import LLMAPI
    print("Successfully imported LLMAPI")
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    api = LLMAPI(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", hf_token=hf_token)
    print("Initialized LLMAPI with TinyLlama 1.1B Chat v1.0")
    api.load_model()  # Load the model before using it
    print("Loaded model")
    # Long prompt for testing
    long_prompt = (
        "You are a helpful assistant. Please write a detailed, step-by-step explanation of how a rocket launches into space, "
        "including the physics of thrust, gravity, atmospheric drag, staging, and orbital insertion. "
        "Make the explanation suitable for a high school student, and include analogies and examples where appropriate. "
        "The explanation should be at least 300 words."
    )
    max_length = 256
    print(f"Sending long prompt (max_length={max_length})...")
    start_time = time.time()
    response = api.chat(long_prompt, max_length=max_length)
    end_time = time.time()
    duration = end_time - start_time
    num_tokens = len(response.split())
    tps = num_tokens / duration if duration > 0 else 0
    print(f"\nLLMAPI chat response (first 500 chars):\n{response[:500]}")
    print(f"\nOutput length (words): {num_tokens}")
    print(f"Time taken: {duration:.2f} seconds")
    print(f"Tokens per second (TPS): {tps:.2f}")
except Exception as e:
    print(f"Error testing LLMAPI: {e}")
    raise 