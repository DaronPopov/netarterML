#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to Python path (now two levels up from test file)
project_root = str(Path(__file__).absolute().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from OPENtransformer.chat.llm_api import LLMAPI
    print("Successfully imported LLMAPI")
    api = LLMAPI(model_name="gpt2")  # Added model_name parameter
    print("Initialized LLMAPI")
    # Optionally, try a simple generation to trigger any lazy downloads
    response = api.generate("Hello, world!")
    print(f"LLMAPI generate response: {response}")
except Exception as e:
    print(f"Error testing LLMAPI: {e}")
    raise 