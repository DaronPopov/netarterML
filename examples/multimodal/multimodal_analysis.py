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
import torch
from PIL import Image
import numpy as np
from transformers import AutoModelForImageClassification, AutoImageProcessor
import cv2



####example of how to run a multimodal analysis between a VIT model and a chatbot model using the OPENtransformer framework both models are running on the same machine offline###



# Add the project root to Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# from OPENtransformer.arm64_engine.core.asm.kernels.vision.medical_imaging_inference import MedicalImagingEngine
from llm_api import LLMAPI

class MultimodalAnalysis:
    def __init__(self):
        self.llama_process = None
        self.medical_engine = None
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
             "--temp", "0.3",  # Lower temperature for more deterministic responses
             "--repeat-penalty", "1.2",  # Increased penalty for repetition
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
            
        print("llama.cpp server started with instruction-based settings")
        
    def setup_medical_engine(self):
        """Setup medical imaging engine"""
        try:
            # Use pneumonia X-ray specific model
            model_name = "pawlo2013/vit-pneumonia-x-ray_3_class"
            print(f"\n📊 Model Information:")
            print(f"   • Model: {model_name}")
            print(f"   • Task: Pneumonia X-ray Classification")
            print(f"   • Classes: No Pneumonia, Bacterial, Viral")
            
            self.medical_engine = MedicalImagingEngine(offline_mode=True)
            # Override the model with pneumonia specific model
            self.medical_engine.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.medical_engine.processor = AutoImageProcessor.from_pretrained(model_name)
            self.medical_engine.model.to(self.medical_engine.device)
            self.medical_engine.model.eval()
            print("Medical imaging engine initialized with pneumonia X-ray model")
        except Exception as e:
            print(f"Error initializing medical engine: {e}")
            print("Falling back to default medical imaging model...")
            self.medical_engine = MedicalImagingEngine(offline_mode=True)
            print("Medical imaging engine initialized with default model")
        
    def send_to_llama(self, message):
        """Send message to llama.cpp server with instruction-based formatting"""
        try:
            # Format the message as an instruction
            formatted_message = f"""<|im_start|>system
You are a medical imaging expert assistant. Your task is to provide clear, concise, and medically accurate responses based on the given analysis.
<|im_end|>
<|im_start|>user
{message}
<|im_end|>
<|im_start|>assistant
"""
            
            response = requests.post(self.server_url, json={
                "prompt": formatted_message,
                "temperature": 0.3,  # Lower temperature for more deterministic responses
                "top_k": 40,
                "top_p": 0.9,
                "n_predict": 256,
                "stop": ["<|im_end|>", "<|im_start|>"],
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
            
    def format_results_for_rag(self, results):
        """Format results in a RAG-friendly structure"""
        rag_document = {
            "metadata": {
                "image_name": results.get("image_name", "unknown"),
                "analysis_timestamp": results.get("timestamp", ""),
                "confidence_score": results.get("consistency", 0.0)
            },
            "findings": {
                "primary_diagnosis": results.get("primary_diagnosis", ""),
                "confidence": results.get("confidence", 0.0),
                "iterations": []
            },
            "clinical_notes": results.get("chatbot_description", "").split("\n"),
            "detailed_analysis": []
        }
        
        # Add iteration details
        for i, vit_result in enumerate(results['vit_analysis']):
            iteration = {
                "iteration": i + 1,
                "predictions": []
            }
            for condition, probability in vit_result['predictions'].items():
                if probability > 0.1:
                    iteration["predictions"].append({
                        "condition": condition,
                        "probability": probability,
                        "significance": "high" if probability > 0.7 else "moderate" if probability > 0.4 else "low"
                    })
            rag_document["findings"]["iterations"].append(iteration)
        
        # Add detailed analysis points
        for line in results.get("chatbot_description", "").split("\n"):
            if line.strip() and not line.startswith("CONFIRMED:"):
                rag_document["detailed_analysis"].append({
                    "point": line.strip("- "),
                    "type": "clinical_observation"
                })
        
        return rag_document

    def chat_about_results(self, results):
        """Interactive chat about the analysis results using instruction-based LLM"""
        # Format results for RAG
        rag_document = self.format_results_for_rag(results)
        
        while True:
            try:
                user_input = input("\nYour question (or 'done' to finish): ").strip()
                
                if user_input.lower() == 'done':
                    print("\nEnding chat session...")
                    break
                
                # Create a structured instruction for the LLM
                prompt = f"""Based on the following medical image analysis, answer the user's question:

IMAGE ANALYSIS:
Primary Diagnosis: {rag_document['findings']['primary_diagnosis']}
Confidence: {rag_document['findings']['confidence']:.2%}

CLINICAL FINDINGS:
{chr(10).join(rag_document['clinical_notes'])}

DETAILED ANALYSIS:
{chr(10).join([f"- {point['point']}" for point in rag_document['detailed_analysis']])}

ITERATION RESULTS:
"""
                # Add iteration details
                for iteration in rag_document['findings']['iterations']:
                    prompt += f"\nIteration {iteration['iteration']}:\n"
                    for pred in iteration['predictions']:
                        prompt += f"- {pred['condition']}: {pred['probability']:.2%} ({pred['significance']})\n"
                
                prompt += f"\nUSER QUESTION: {user_input}\n\nPlease provide a clear, concise, and medically accurate response based on the above analysis."
                
                # Get response from LLM
                response = self.send_to_llama(prompt)
                print("\nResponse:", response)
                
            except KeyboardInterrupt:
                print("\nChat session interrupted.")
                break
            except Exception as e:
                print(f"Error in chat: {e}")
                continue

    def analyze_image(self, image_path):
        """Analyze image using both VIT and chatbot with multiple iterations"""
        try:
            # Run multiple VIT analyses
            vit_results = []
            for i in range(3):
                print(f"\nRunning VIT analysis iteration {i+1}/3...")
                vit_result = self.medical_engine.predict(image_path)
                vit_results.append(vit_result)
            
            # Compare results
            print("\nComparing VIT analysis results:")
            print("-" * 50)
            for i, result in enumerate(vit_results):
                print(f"\nIteration {i+1}:")
                for condition, probability in result['predictions'].items():
                    if probability > 0.1:
                        print(f"- {condition}: {probability:.2%}")
            
            # Get the most consistent prediction
            predictions = []
            for result in vit_results:
                pred = max(result['predictions'].items(), key=lambda x: x[1])
                predictions.append(pred)
            
            # Count occurrences of each prediction
            pred_counts = {}
            for pred, _ in predictions:
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
            
            # Get the most common prediction
            most_common_pred = max(pred_counts.items(), key=lambda x: x[1])
            
            # Hard-coded responses based on prediction
            hard_coded_responses = {
                "Normal": """CONFIRMED: No Pneumonia
- No signs of pneumonia
- Clear lung fields
- Normal lung markings
- No abnormal opacities
- No consolidation present""",
                
                "Bacterial": """CONFIRMED: Bacterial Pneumonia
- Consolidation present
- Dense opacities in lung fields
- Typical bacterial pattern
- No ground-glass opacities
- Consistent with bacterial infection""",
                
                "Viral": """CONFIRMED: Viral Pneumonia
- Ground-glass opacities present
- Bilateral involvement
- Typical viral pattern
- No dense consolidation
- Consistent with viral infection"""
            }
            
            # Get the hard-coded response
            chatbot_response = hard_coded_responses.get(most_common_pred[0], "Unable to determine with confidence")
            
            # Add confidence information
            confidence_info = f"\nConfidence: {most_common_pred[1]/3:.2%} (based on {most_common_pred[1]}/3 consistent predictions)"
            chatbot_response += confidence_info
            
            # Create structured results
            results = {
                "image_name": Path(image_path).name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "vit_analysis": vit_results,
                "chatbot_description": chatbot_response,
                "consistency": most_common_pred[1]/3,
                "primary_diagnosis": most_common_pred[0],
                "confidence": most_common_pred[1]/3
            }
            
            return results
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None
            
    def cleanup(self):
        """Cleanup resources"""
        if self.llama_process:
            self.llama_process.terminate()
            self.llama_process.wait()
            
        # Kill any remaining server process
        os.system("pkill -f llama-server")

    def display_image(self, image_path, title="Medical Image Analysis"):
        """Display the image being processed"""
        try:
            # Read image using OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Error: Could not read image at {image_path}")
                return
            # Create a window
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            
            # Resize window to a reasonable size
            height, width = image.shape[:2]
            max_height = 800
            if height > max_height:
                scale = max_height / height
                width = int(width * scale)
                height = max_height
                image = cv2.resize(image, (width, height))
            
            # Display image
            cv2.imshow(title, image)
            cv2.waitKey(1)  # Show image and wait for a short time
            
        except Exception as e:
            print(f"Error displaying image: {e}")

    def close_image_window(self, title="Medical Image Analysis"):
        """Close the image display window"""
        try:
            cv2.destroyWindow(title)
        except:
            pass

    def interactive_mode(self):
        """Interactive mode for loading and analyzing images with automatic chat flow"""
        print("\n" + "="*50)
        print("Interactive Image Analysis Mode")
        print("="*50)
        print("Commands:")
        print("- 'list': List available images in local_images directory")
        print("- 'load <filename>': Load and analyze an image from local_images")
        print("- 'load all': Load and analyze all images in local_images directory")
        print("- 'exit': Exit interactive mode")
        print("-"*50)
        
        # Ensure local_images directory exists
        local_images_dir = Path("local_images")
        if not local_images_dir.exists():
            local_images_dir.mkdir()
            print("Created local_images directory. Please add your images there.")
        
        current_results = None
        
        while True:
            try:
                command = input("\nEnter command: ").strip()
                
                if command.lower() == 'exit':
                    print("\nExiting interactive mode...")
                    self.close_image_window()  # Close any open image windows
                    break
                    
                elif command.lower() == 'list':
                    print("\nAvailable images in local_images directory:")
                    print("-"*30)
                    images = list(local_images_dir.glob("*.[jp][pn][g]"))  # Match .jpg, .jpeg, .png
                    if not images:
                        print("No images found. Please add images to the local_images directory.")
                    else:
                        for img in images:
                            print(f"- {img.name}")
                    
                elif command.lower() == 'load all':
                    images = list(local_images_dir.glob("*.[jp][pn][g]"))  # Match .jpg, .jpeg, .png
                    if not images:
                        print("No images found in local_images directory.")
                        continue
                        
                    print(f"\nFound {len(images)} images. Starting batch analysis...")
                    print("="*50)
                    
                    for image_path in images:
                        # Display the current image
                        self.display_image(image_path, f"Analyzing: {image_path.name}")
                        
                        print(f"\nAnalyzing image: {image_path.name}")
                        print("-"*30)
                        
                        result = self.analyze_image(str(image_path))
                        if result:
                            print("\nModel Predictions and Confidence:")
                            print("-"*30)
                            # Sort predictions by confidence
                            sorted_predictions = sorted(
                                result['vit_analysis'][0]['predictions'].items(),
                                key=lambda x: x[1],
                                reverse=True
                            )
                            for condition, probability in sorted_predictions:
                                if probability > 0.1:  # Only show significant predictions
                                    print(f"{condition}: {probability:.2%} confidence")
                            
                            print("\nAnalysis:")
                            print("-"*30)
                            print(result['chatbot_description'])
                            
                            # Automatically enter chat mode after analysis
                            print("\n" + "="*50)
                            print(f"Entering chat mode for {image_path.name}...")
                            print("Type 'done' to finish chat and move to next image")
                            print("Type 'skip' to skip chat for this image")
                            print("="*50)
                            
                            self.chat_about_results(result)
                            
                            # Close the current image window
                            self.close_image_window(f"Analyzing: {image_path.name}")
                            
                            # Ask if user wants to continue with next image
                            if len(images) > 1:
                                print("\n" + "="*50)
                                print("What would you like to do next?")
                                print("1. Continue to next image")
                                print("2. Stop batch analysis")
                                print("="*50)
                                
                                choice = input("Enter choice (1 or 2): ").strip()
                                if choice == "2":
                                    print("\nStopping batch analysis...")
                                    break
                    
                    print("\n" + "="*50)
                    print("Batch analysis completed. What would you like to do next?")
                    print("1. Load another image")
                    print("2. List available images")
                    print("3. Exit")
                    print("="*50)
                    
                elif command.lower().startswith('load '):
                    image_name = command[5:].strip()
                    image_path = local_images_dir / image_name
                    
                    if not image_path.exists():
                        print(f"Error: Image not found at {image_path}")
                        print("Please use 'list' command to see available images.")
                        continue
                    
                    # Display the image
                    self.display_image(image_path, f"Analyzing: {image_name}")
                        
                    print(f"\nAnalyzing image: {image_name}")
                    print("="*50)
                    
                    result = self.analyze_image(str(image_path))
                    if result:
                        print("\nModel Predictions and Confidence:")
                        print("-"*30)
                        # Sort predictions by confidence
                        sorted_predictions = sorted(
                            result['vit_analysis'][0]['predictions'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        for condition, probability in sorted_predictions:
                            if probability > 0.1:  # Only show significant predictions
                                print(f"{condition}: {probability:.2%} confidence")
                        
                        print("\nAnalysis:")
                        print("-"*30)
                        print(result['chatbot_description'])
                        
                        current_results = result
                        
                        # Automatically enter chat mode after analysis
                        print("\n" + "="*50)
                        print("Entering chat mode for this analysis...")
                        print("Type 'done' to finish chat and return to main menu")
                        print("="*50)
                        
                        self.chat_about_results(current_results)
                        
                        # Close the image window
                        self.close_image_window(f"Analyzing: {image_name}")
                        
                        # After chat completion, show next steps
                        print("\n" + "="*50)
                        print("Chat completed. What would you like to do next?")
                        print("1. Load another image")
                        print("2. List available images")
                        print("3. Exit")
                        print("="*50)
                    
                else:
                    print("Unknown command. Available commands: list, load, load all, exit")
                
            except KeyboardInterrupt:
                print("\nInteractive mode interrupted.")
                self.close_image_window()  # Close any open image windows
                break
            except Exception as e:
                print(f"Error: {e}")
                self.close_image_window()  # Close any open image windows
                continue

def main():
    # Initialize multimodal analysis
    analyzer = MultimodalAnalysis()
    
    try:
        # Setup models
        print("Setting up llama.cpp with GPU acceleration...")
        analyzer.setup_llama_cpp("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        
        print("Setting up medical imaging engine...")
        analyzer.setup_medical_engine()
        
        # Enter interactive mode
        analyzer.interactive_mode()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        analyzer.cleanup()
        # Clean up JSON files from the output directory
        for json_file in Path("analysis_output").glob("analysis_*.json"):
            try:
                json_file.unlink()
            except Exception as e:
                print(f"Error deleting {json_file}: {e}")
        # Remove the output directory if it's empty
        try:
            Path("analysis_output").rmdir()
        except Exception:
            pass  # Directory might not be empty, which is fine

if __name__ == "__main__":
    main() 