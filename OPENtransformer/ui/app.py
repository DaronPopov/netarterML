import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import torch
import gc
import socket
import threading
import queue

# Add project root to Python path
project_root = str(Path(__file__).absolute().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add OPENtransformer to Python path
open_transformer_path = os.path.join(project_root, "OPENtransformer")
if open_transformer_path not in sys.path:
    sys.path.insert(0, open_transformer_path)

# Import our components
from components.token_frame import TokenFrame
from components.webcam_tab import WebcamTab
from components.chat_tab import ChatTab
from components.image_tab import ImageTab
from components.medical_tab import MedicalTab

# Import our engines
from engines.webcam_blip import WebcamBlipEngine
from engines.llm_api import LLMAPI
from engines.medical_engine import MedicalImageEngine
from ui.engines.arbitrary_image_engine import SIMDOptimizedPipeline as ArbitraryImageEngine


# Import theme configuration
from styles.theme import configure_styles

class AIStudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Studio")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.webcam_engine = None
        self.chat_engine = None
        self.diffusion_engine = None
        self.medical_engine = None
        self.model_loading = False
        self.loading_thread = None
        
        # Add model cache directory
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "models")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Add cached models tracking
        self.cached_models = {
            'chat': [],
            'webcam': [],
            'image': [],
            'medical': []
        }
        
        # Create token frame
        self.token_frame = TokenFrame(root, self.on_token_set)
        
        # Create main notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Create tabs
        self.webcam_tab = WebcamTab(self.notebook, self.load_webcam_model)
        self.chat_tab = ChatTab(self.notebook, self.load_chat_model)
        self.image_tab = ImageTab(self.notebook, self.load_image_model, self)
        self.medical_tab = MedicalTab(self.notebook, self.load_medical_model)
        
        self.notebook.add(self.webcam_tab.frame, text="🎥 Webcam")
        self.notebook.add(self.chat_tab.frame, text="💬 Chat")
        self.notebook.add(self.image_tab.frame, text="🎨 Image Generation")
        self.notebook.add(self.medical_tab.frame, text="🏥 Medical Imaging")
        
        # Bind tab change event
        self.notebook.bind('<<NotebookTabChanged>>', self.on_tab_change)
        
        # Scan for cached models
        self.scan_cached_models()
        
        # Initialize Tcl event loop
        self.root.update_idletasks()
    
    def on_token_set(self):
        """Callback when token is set"""
        self.scan_cached_models()
    
    def scan_cached_models(self):
        """Scan for cached models in the Hugging Face cache directory"""
        try:
            # Clear existing cached models
            for key in self.cached_models:
                self.cached_models[key] = []
            
            # Get the correct cache directory
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "models")
            if not os.path.exists(cache_dir):
                print(f"Cache directory not found: {cache_dir}")
                return
            
            # Define model type patterns and their corresponding modalities
            model_patterns = {
                'image': {
                    'patterns': [
                        "stable-diffusion", "sd-", "sdxl", "controlnet", "diffusion",
                        "latent", "pixart", "kandinsky", "deepfloyd", "wuerstchen",
                        "sd-turbo", "sd_turbo", "sd turbo"
                    ],
                    'config_types': ['stable-diffusion', 'diffusion', 'latent-diffusion', 'controlnet', 'sd-turbo'],
                    'model_files': ['model_index.json', 'config.json', 'scheduler_config.json', 'model.safetensors']
                },
                'webcam': {
                    'patterns': [
                        "blip", "clip", "vit", "owl", "detr", "yolo", "faster-rcnn",
                        "mask-rcnn", "retinanet", "efficientdet"
                    ],
                    'config_types': ['blip', 'clip', 'vision-transformer', 'detr'],
                    'model_files': ['config.json', 'preprocessor_config.json']
                },
                'chat': {
                    'patterns': [
                        "gemma", "llama", "mistral", "phi", "falcon", "gpt", "opt",
                        "bloom", "t5", "bart", "pegasus", "mpt", "qwen", "yi"
                    ],
                    'config_types': ['llm', 'causal', 'seq2seq', 'text-generation'],
                    'model_files': ['config.json', 'tokenizer_config.json']
                },
                'medical': {
                    'patterns': [
                        "resnet", "densenet", "efficientnet", "medical", "unet",
                        "vnet", "swin", "transunet", "medseg", "nnunet"
                    ],
                    'config_types': ['resnet', 'unet', 'swin', 'medical'],
                    'model_files': ['config.json', 'preprocessor_config.json']
                }
            }
            
            # Scan cache directory
            for model_dir in os.listdir(cache_dir):
                # Skip hidden directories and .locks
                if model_dir.startswith('.') or model_dir == '.locks':
                    continue
                    
                model_path = os.path.join(cache_dir, model_dir)
                if os.path.isdir(model_path):
                    # Convert directory name back to model name
                    model_name = model_dir.replace("models--", "").replace("--", "/")
                    print(f"Found model: {model_name}")
                    
                    # First try to determine type from directory name
                    model_added = False
                    for modality, info in model_patterns.items():
                        if any(x in model_dir.lower() for x in info['patterns']):
                            if model_name not in self.cached_models[modality]:
                                self.cached_models[modality].append(model_name)
                                model_added = True
                                print(f"Added {model_name} to {modality} based on directory name")
                                break
                    
                    # If not found by directory name, check snapshots
                    if not model_added:
                        snapshots_dir = os.path.join(model_path, "snapshots")
                        if os.path.exists(snapshots_dir):
                            for snapshot in os.listdir(snapshots_dir):
                                snapshot_path = os.path.join(snapshots_dir, snapshot)
                                if os.path.isdir(snapshot_path):
                                    # Check for model files
                                    has_model_files = False
                                    for file in os.listdir(snapshot_path):
                                        if file.endswith(('.bin', '.safetensors', '.pt', '.pth', '.json', '.model')):
                                            has_model_files = True
                                            break
                                    
                                    if has_model_files:
                                        # Try to determine type from config files
                                        for modality, info in model_patterns.items():
                                            config_found = False
                                            for config_file in info['model_files']:
                                                config_path = os.path.join(snapshot_path, config_file)
                                                if os.path.exists(config_path):
                                                    try:
                                                        import json
                                                        with open(config_path, 'r') as f:
                                                            config = json.load(f)
                                                        
                                                        # Check model type from config
                                                        model_type = config.get('model_type', '').lower()
                                                        if any(t in model_type for t in info['config_types']):
                                                            if model_name not in self.cached_models[modality]:
                                                                self.cached_models[modality].append(model_name)
                                                                model_added = True
                                                                config_found = True
                                                                print(f"Added {model_name} to {modality} based on config")
                                                                break
                                                    except Exception as e:
                                                        print(f"Error reading config for {model_name}: {e}")
                                            
                                            if config_found:
                                                break
                                    
                                    break  # Found a valid snapshot, no need to check others
            
            # Sort models alphabetically
            for category in self.cached_models:
                self.cached_models[category].sort()
            
            # Update dropdowns
            self.update_model_dropdowns()
            
            print("\nCached models found:")
            for category, models in self.cached_models.items():
                print(f"{category}: {models}")
                
        except Exception as e:
            print(f"Error scanning cached models: {e}")
            import traceback
            traceback.print_exc()
    
    def update_model_dropdowns(self):
        """Update all model dropdowns with cached models"""
        try:
            # Update webcam dropdown
            if hasattr(self.webcam_tab, 'update_model_dropdown'):
                self.webcam_tab.update_model_dropdown(self.cached_models['webcam'])
            
            # Update chat dropdown
            if hasattr(self.chat_tab, 'update_model_dropdown'):
                self.chat_tab.update_model_dropdown(self.cached_models['chat'])
            
            # Update image dropdown
            if hasattr(self.image_tab, 'update_model_dropdown'):
                self.image_tab.update_model_dropdown(self.cached_models['image'])
            
            # Update medical dropdown
            if hasattr(self.medical_tab, 'update_model_dropdown'):
                self.medical_tab.update_model_dropdown(self.cached_models['medical'])
            
        except Exception as e:
            print(f"Error updating model dropdowns: {e}")
            import traceback
            traceback.print_exc()
    
    def check_internet_connection(self, host="8.8.8.8", port=53, timeout=3):
        """Check for internet connection by trying to connect to Google DNS."""
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except socket.error as ex:
            print(f"Internet check failed: {ex}")
            return False
    
    def load_webcam_model(self, model_name, callback):
        """Load webcam model"""
        if self.model_loading:
            callback(False, "A model is already being loaded")
            return
        
        self.model_loading = True
        is_online = self.check_internet_connection()
        
        def load_thread():
            try:
                # Clear previous model if exists
                if self.webcam_engine is not None:
                    self.webcam_engine = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Load new model
                self.webcam_engine = WebcamBlipEngine(model_name, self.token_frame.get_token())
                
                # Add to cached models if not already present
                if model_name not in self.cached_models['webcam']:
                    self.cached_models['webcam'].append(model_name)
                    self.update_model_dropdowns()
                
                callback(True)
                
            except Exception as e:
                callback(False, str(e))
            finally:
                self.model_loading = False
        
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
    
    def load_chat_model(self, model_name, callback):
        """Load chat model"""
        if self.model_loading:
            callback(False, "A model is already being loaded")
            return
        
        self.model_loading = True
        is_online = self.check_internet_connection()
        
        def load_thread():
            try:
                # Clear previous model if exists
                if self.chat_engine is not None:
                    self.chat_engine = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Load new model
                self.chat_engine = LLMAPI(model_name, hf_token=self.token_frame.get_token())
                self.chat_engine.load_model()
                
                # Set the chat engine in the chat tab
                self.chat_tab.chat_engine = self.chat_engine
                
                # Add to cached models if not already present
                if model_name not in self.cached_models['chat']:
                    self.cached_models['chat'].append(model_name)
                    self.update_model_dropdowns()
                
                callback(True)
                
            except Exception as e:
                callback(False, str(e))
            finally:
                self.model_loading = False
        
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
    
    def load_image_model(self, model_name, callback):
        """Load the specified image generation model."""
        if self.model_loading:
            callback(False, "Another model is currently loading")
            return
        
        self.model_loading = True
        
        def load_thread():
            try:
                # Initialize the SIMD-optimized engine
                self.diffusion_engine = ArbitraryImageEngine(
                    model_id=model_name,
                    device="cpu"  # Using CPU for SIMD optimization
                )
                
                # Call the callback with success
                self.root.after(0, lambda: callback(True))
                
            except Exception as e:
                error_msg = str(e) if str(e) else "Unknown error occurred"
                print(f"Error loading image model: {error_msg}")
                self.root.after(0, lambda: callback(False, error_msg))
            
            finally:
                self.model_loading = False
        
        # Start loading in a separate thread
        threading.Thread(target=load_thread, daemon=True).start()
    
    def load_medical_model(self, model_name, callback):
        """Load medical imaging model"""
        if self.model_loading:
            callback(False, "A model is already being loaded")
            return
        
        self.model_loading = True
        is_online = self.check_internet_connection()
        
        def load_thread():
            try:
                # Clear previous model if exists
                if self.medical_engine is not None:
                    self.medical_engine.cleanup()
                    self.medical_engine = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Load new model
                self.medical_engine = MedicalImageEngine(model_name, self.token_frame.get_token())
                if not self.medical_engine.load_model():
                    raise RuntimeError("Failed to load model")
                
                # Set the medical engine in the medical tab
                self.medical_tab.medical_engine = self.medical_engine
                
                # Add to cached models if not already present
                if model_name not in self.cached_models['medical']:
                    self.cached_models['medical'].append(model_name)
                    self.update_model_dropdowns()
                
                callback(True)
                
            except Exception as e:
                callback(False, str(e))
            finally:
                self.model_loading = False
        
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
    
    def on_tab_change(self, event):
        """Handle tab changes and update model selection"""
        # Clear previous model and memory
        self.clear_current_model()
        
        # Get new tab
        tab = self.notebook.select()
        tab_name = self.notebook.tab(tab, "text")
        
        # Scan for models when changing tabs
        self.scan_cached_models()
    
    def clear_current_model(self):
        """Clear the current model and free memory"""
        if self.model_loading:
            # Wait for current loading to complete
            if self.loading_thread and self.loading_thread.is_alive():
                self.loading_thread.join(timeout=1.0)
        
        # Clear webcam resources
        if self.webcam_engine is not None:
            self.webcam_tab.cleanup()
            self.webcam_engine = None
        
        # Clear chat resources
        if self.chat_engine is not None:
            self.chat_tab.cleanup()
            self.chat_engine = None
        
        # Clear diffusion resources
        if self.diffusion_engine is not None:
            self.image_tab.cleanup()
            self.diffusion_engine = None
        
        # Clear medical resources
        if self.medical_engine is not None:
            self.medical_tab.cleanup()
            self.medical_engine = None
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def cleanup(self):
        """Clean up resources before closing"""
        self.clear_current_model()

def main():
    root = tk.Tk()
    
    # Configure styles before creating the UI
    configure_styles()
    
    # Set window icon and title
    root.title("AI Studio")
    root.geometry("1200x800")
    
    # Set window background color
    root.configure(bg='#2b2b2b')
    
    # Create the main application
    app = AIStudioApp(root)
    
    # Set up window close handler
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    
    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    main() 