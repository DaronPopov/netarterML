#!/usr/bin/env python3
import os
import sys
import time
import threading
import queue
from pathlib import Path
import subprocess
import json
import requests
import signal



###example of how to run a conversation between two models using the OPENtransformer framework both models are running on the same machine offline###


# Add the project root to Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# from chat_with_tinyllama import chat_loop as tinyllama_chat
from llm_api import LLMAPI

class ModelConversation:
    def __init__(self):
        self.llama_process = None
        self.tinyllama_api = None
        self.conversation_history = []
        self.message_queue = queue.Queue()
        self.model_path = None
        self.server_url = "http://localhost:8080/completion"
        
    def setup_llama_cpp(self, model_path):
        """Setup llama.cpp server with Metal acceleration"""
        self.model_path = model_path
        llama_cpp_path = os.path.join(project_root, "llama.cpp")
        server_path = os.path.join(llama_cpp_path, "build", "bin", "llama-server")
        
        if not os.path.exists(server_path):
            raise FileNotFoundError(f"llama.cpp server executable not found at {server_path}")
            
        # Kill any existing server
        os.system("pkill -f llama-server")
        time.sleep(1)
            
        # Start llama.cpp server with Metal acceleration
        self.llama_process = subprocess.Popen(
            [server_path, 
             "--model", model_path,
             "--n-gpu-layers", "35",  # Use GPU for all layers
             "--ctx-size", "2048",
             "--temp", "0.7",
             "--repeat-penalty", "1.1",
             "--host", "127.0.0.1",
             "--port", "8080",
             "--threads", "8"],  # Use multiple threads for better performance
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        time.sleep(5)
        
        # Check if server started successfully
        if self.llama_process.poll() is not None:
            stderr = self.llama_process.stderr.read()
            raise RuntimeError(f"Failed to start llama.cpp server: {stderr}")
            
        print("llama.cpp server started")
        
    def send_to_llama(self, message):
        """Send message to llama.cpp server"""
        try:
            # Format the message as a chat
            formatted_message = f"Human: {message}\n\nAssistant:"
            
            response = requests.post(self.server_url, json={
                "prompt": formatted_message,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "n_predict": 256,
                "stop": ["\n\nHuman:", "\n\nAssistant:", "\n\nSystem:"],
                "stream": False
            })
            response_json = response.json()
            if "content" in response_json:
                return response_json["content"].strip()
            elif "response" in response_json:
                return response_json["response"].strip()
            else:
                print(f"Unexpected response format: {response_json}")
                return None
        except Exception as e:
            print(f"Error sending message to llama.cpp server: {e}")
            return None
            
    def setup_tinyllama(self):
        """Setup TinyLlama using OPENtransformer"""
        hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        if not hf_token:
            print("Warning: HUGGINGFACE_TOKEN environment variable not set")
        self.tinyllama_api = LLMAPI(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            hf_token=hf_token
        )
        self.tinyllama_api.load_model()
        
    def run_conversation(self, num_exchanges=5):
        """Run a conversation between the two models"""
        print("\nStarting conversation between llama.cpp (GPU) and TinyLlama (CPU)...")
        print("="*50)
        
        # Initial prompt
        initial_prompt = "Let's discuss the future of artificial intelligence and its potential impact on society. What are your thoughts?"
        print(f"\nInitial prompt: {initial_prompt}")
        
        # Get initial response from llama.cpp
        llama_response = self.send_to_llama(initial_prompt)
        if llama_response:
            print(f"\nllama.cpp (GPU): {llama_response}")
        else:
            print("\nError: No response from llama.cpp")
            return
        
        for i in range(num_exchanges):
            print(f"\nExchange {i+1}/{num_exchanges}")
            print("-"*30)
            
            # Format TinyLlama's prompt
            tinyllama_prompt = f"Previous response: {llama_response}\n\nPlease continue the discussion about AI and provide your perspective."
            
            # Get TinyLlama response
            tinyllama_response = self.tinyllama_api.chat(tinyllama_prompt)
            print(f"\nTinyLlama (CPU): {tinyllama_response}")
            
            # Get llama.cpp response
            llama_response = self.send_to_llama(tinyllama_response)
            if llama_response:
                print(f"\nllama.cpp (GPU): {llama_response}")
            else:
                print("\nError: No response from llama.cpp")
                break
            
            # Add to conversation history
            self.conversation_history.append({
                "llama": llama_response,
                "tinyllama": tinyllama_response
            })
            
            # Small delay between exchanges
            time.sleep(1)
            
    def cleanup(self):
        """Cleanup resources"""
        if self.llama_process:
            self.llama_process.terminate()
            self.llama_process.wait()
            
        # Kill any remaining server process
        os.system("pkill -f llama-server")
            
        # Save conversation history
        with open("conversation_history.json", "w") as f:
            json.dump(self.conversation_history, f, indent=2)

def main():
    # Initialize conversation
    conversation = ModelConversation()
    
    try:
        # Setup models
        print("Setting up llama.cpp with GPU acceleration...")
        conversation.setup_llama_cpp("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        
        print("Setting up TinyLlama with CPU...")
        conversation.setup_tinyllama()
        
        # Run conversation
        conversation.run_conversation(num_exchanges=5)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        conversation.cleanup()

if __name__ == "__main__":
    main() 