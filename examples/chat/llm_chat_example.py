import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the C/ASM-based LLMAPI
try:
    from engines.llm_api import LLMAPI
    print("Successfully imported C/ASM-based LLMAPI")
except ImportError as e:
    print(f"Error importing LLMAPI: {e}")
    sys.exit(1)

def main():
    print("Loading TinyLlama 1.1B model with C/ASM backend...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Using TinyLlama 1.1B Chat model
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        print("Error: HUGGINGFACE_TOKEN environment variable not set")
        sys.exit(1)

    # Initialize the API with C/ASM backend
    chat = LLMAPI(model_name=model_name, hf_token=hf_token, use_simd=True)
    
    # Load model with C/ASM backend
    print("Loading model with C/ASM backend...")
    chat.load_model(
        device_map="cpu",
        torch_dtype=None,  # Let C/ASM backend handle dtype
        use_simd=True
    )
    print("\nTinyLlama Chat initialized with C/ASM backend!")
    print("----------------------------------------")

    # Chat loop
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
        try:
            response = chat.chat(
                message=user_input,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9
            )
            print("\nTinyLlama:", response)
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main() 