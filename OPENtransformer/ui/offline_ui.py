#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import queue
import time
import torch
import gc
import socket
import io
from contextlib import redirect_stdout
from transformers import AutoTokenizer, AutoModelForCausalLM
import multiprocessing

# Set multiprocessing start method to 'spawn' for better compatibility
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

# Add project root to Python path
project_root = str(Path(__file__).absolute().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add OPENtransformer to Python path
open_transformer_path = os.path.join(project_root, "OPENtransformer")
if open_transformer_path not in sys.path:
    sys.path.insert(0, open_transformer_path)

# Import our existing modules
from ui.engines.arbitrary_image_engine import SIMDOptimizedPipeline as ArbitraryImageEngine
from engines.webcam_blip import WebcamBlipEngine
from engines.llm_api import LLMAPI
from engines.medical_engine import MedicalImageEngine

# Configure ttk styles for dark theme
def configure_styles():
    style = ttk.Style()
    
    # Configure the main theme
    style.theme_use('clam')  # Use clam theme as base
    
    # Configure colors for dark theme
    style.configure('.',
        background='#2b2b2b',
        foreground='#ffffff',
        font=('Helvetica', 10)
    )
    
    # Configure frames
    style.configure('TLabelframe',
        background='#333333',
        borderwidth=2,
        relief='solid'
    )
    style.configure('TLabelframe.Label',
        background='#333333',
        foreground='#ffffff',
        font=('Helvetica', 10, 'bold')
    )
    
    # Configure buttons
    style.configure('TButton',
        padding=5,
        font=('Helvetica', 10),
        background='#4a90e2',
        foreground='#ffffff'
    )
    style.map('TButton',
        background=[('active', '#357abd'), ('disabled', '#666666')],
        foreground=[('disabled', '#999999')]
    )
    
    # Configure entry fields
    style.configure('TEntry',
        padding=5,
        fieldbackground='#404040',
        foreground='#ffffff',
        borderwidth=1
    )
    
    # Configure notebook tabs
    style.configure('TNotebook',
        background='#2b2b2b',
        borderwidth=0
    )
    style.configure('TNotebook.Tab',
        padding=[10, 5],
        font=('Helvetica', 10),
        background='#404040',
        foreground='#ffffff'
    )
    style.map('TNotebook.Tab',
        background=[('selected', '#4a90e2')],
        foreground=[('selected', '#ffffff')]
    )
    
    # Configure text widgets
    style.configure('TText',
        background='#404040',
        foreground='#ffffff',
        fieldbackground='#404040'
    )
    
    # Configure listbox
    style.configure('TListbox',
        background='#404040',
        foreground='#ffffff',
        fieldbackground='#404040'
    )
    
    # Configure scrollbar
    style.configure('TScrollbar',
        background='#404040',
        troughcolor='#2b2b2b',
        borderwidth=0
    )

# Helper function to run model loading in a thread
def _load_model_thread(target_object, model_name, hf_token, online):
    try:
        # Initial status update
        target_object.root.after(0, lambda: target_object.chat_status_label.config(text=f"Status: Starting load for {model_name}..."))
        target_object.root.after(0, lambda: target_object.load_model_button.config(state='disabled'))
        target_object.root.after(0, lambda: target_object.model_name_input.config(state='disabled'))

        # --- Clear previous model --- 
        if target_object.chat_engine:
            target_object.root.after(0, lambda: target_object.chat_status_label.config(text="Status: Clearing previous model..."))
            # Short sleep to allow UI to potentially update before heavy gc
            time.sleep(0.1) 
            del target_object.chat_engine
            target_object.chat_engine = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            target_object.chat_model_loaded = False
            # Short sleep after gc
            time.sleep(0.1) 

        # --- Instantiate LLMAPI --- 
        target_object.root.after(0, lambda: target_object.chat_status_label.config(text="Status: Initializing backend..."))
        time.sleep(0.1)
        try:
            engine = LLMAPI(model_name, hf_token=hf_token)
            print(f"LLMAPI backend initialized for {model_name}.")
        except Exception as init_error:
            error_message = f"Error initializing LLMAPI backend: {str(init_error)[:200]}..."
            print(error_message)
            target_object.root.after(0, lambda: target_object.show_error(error_message))
            target_object.root.after(0, lambda: target_object.chat_status_label.config(text="Status: Error initializing backend"))
            target_object.chat_engine = None
            target_object.chat_model_loaded = False
            # Need to exit the thread here if init fails
            target_object.model_loading = False 
            target_object.root.after(0, lambda: target_object.load_model_button.config(state='normal'))
            target_object.root.after(0, lambda: target_object.model_name_input.config(state='normal'))
            print("Model loading thread aborted due to backend init error.")
            return # Exit thread early

        # --- Core Loading via LLMAPI.load_model() --- 
        try:
            mode = "online" if online else "offline"
            print(f"Attempting LLMAPI.load_model() for {model_name}. Mode: {mode}")
            target_object.root.after(0, lambda: target_object.chat_status_label.config(text=f"Status: Loading {model_name} via backend ({mode})..."))
            time.sleep(0.1) # Allow UI update
            
            # Call the backend's load method
            # We assume load_model handles online/offline logic internally based on its implementation
            engine.load_model()
            
            print(f"LLMAPI.load_model() completed for {model_name}.")

            # --- Finalizing --- 
            target_object.root.after(0, lambda: target_object.chat_status_label.config(text="Status: Finalizing setup..."))
            time.sleep(0.1) # Allow UI update
            target_object.chat_engine = engine
            target_object.chat_model_loaded = True
            target_object.root.after(0, lambda: target_object.chat_status_label.config(text=f"Status: {model_name} loaded successfully via backend"))
            print(f"Successfully loaded and set up {model_name} via LLMAPI.")

        except Exception as load_error:
            # Error during engine.load_model()
            error_message = f"Error in LLMAPI.load_model() for {model_name}: {str(load_error)[:200]}..."
            print(error_message)
            target_object.root.after(0, lambda: target_object.show_error(error_message))
            target_object.root.after(0, lambda: target_object.chat_status_label.config(text="Status: Error loading model via backend"))
            target_object.chat_engine = None
            target_object.chat_model_loaded = False
        # --- End Core Loading ---

    except Exception as e:
        # Catch errors in the thread logic itself
        err_msg = f"Critical error in loading thread: {str(e)[:200]}..."
        print(err_msg)
        target_object.root.after(0, lambda: target_object.show_error(err_msg))
        target_object.root.after(0, lambda: target_object.chat_status_label.config(text="Status: Critical Error"))
        target_object.chat_engine = None
        target_object.chat_model_loaded = False
    finally:
        # Reset loading flag and re-enable UI elements via root.after
        target_object.model_loading = False # Reset the flag
        target_object.root.after(0, lambda: target_object.load_model_button.config(state='normal'))
        target_object.root.after(0, lambda: target_object.model_name_input.config(state='normal'))
        print("Model loading thread finished.")

class AIStudioUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Studio")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.hf_token = None
        self.webcam_engine = None
        self.chat_engine = None
        self.diffusion_engine = None
        self.medical_engine = None
        self.webcam_running = False
        self.chat_history = []
        self.frame_queue = queue.Queue(maxsize=1)
        self.caption_queue = queue.Queue(maxsize=1)
        self.current_tab = None
        self.chat_model_loaded = False
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
        
        # Scan for cached models
        self.scan_cached_models()
        
        # Re-add streaming variable
        self.current_response = "" 
        
        # Add webcam display size
        self.webcam_width = 640
        self.webcam_height = 480
        
        # Create token input frame at the top
        self.setup_token_frame()
        
        # Create main notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Create tabs
        self.webcam_tab = ttk.Frame(self.notebook)
        self.chat_tab = ttk.Frame(self.notebook)
        self.image_tab = ttk.Frame(self.notebook)
        self.medical_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.webcam_tab, text="🎥 Webcam")
        self.notebook.add(self.chat_tab, text="💬 Chat")
        self.notebook.add(self.image_tab, text="🎨 Image Generation")
        self.notebook.add(self.medical_tab, text="🏥 Medical Imaging")
        
        # Initialize UI components
        self.setup_webcam_tab()
        self.setup_chat_tab()
        self.setup_image_tab()
        self.setup_medical_tab()
        
        # Bind tab change event
        self.notebook.bind('<<NotebookTabChanged>>', self.on_tab_change)
        
        # Initially disable all model-related buttons until token is set
        self.disable_model_buttons()
        
        # Try to load token from environment variable
        self.try_load_token_from_env()
        
        # Initialize Tcl event loop
        self.root.update_idletasks()
    
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
                    # Format: models--org--model-name -> org/model-name
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
            if hasattr(self, 'webcam_model_dropdown'):
                self.webcam_model_dropdown['values'] = self.cached_models['webcam']
                if self.cached_models['webcam']:
                    self.webcam_model_dropdown.set(self.cached_models['webcam'][0])
                    self.webcam_model_input.delete(0, tk.END)
                    self.webcam_model_input.insert(0, self.cached_models['webcam'][0])
                    print(f"Updated webcam models: {self.cached_models['webcam']}")
            
            # Update chat dropdown
            if hasattr(self, 'chat_model_dropdown'):
                self.chat_model_dropdown['values'] = self.cached_models['chat']
                if self.cached_models['chat']:
                    self.chat_model_dropdown.set(self.cached_models['chat'][0])
                    self.model_name_input.delete(0, tk.END)
                    self.model_name_input.insert(0, self.cached_models['chat'][0])
                    print(f"Updated chat models: {self.cached_models['chat']}")
            
            # Update image dropdown
            if hasattr(self, 'image_model_dropdown'):
                self.image_model_dropdown['values'] = self.cached_models['image']
                if self.cached_models['image']:
                    self.image_model_dropdown.set(self.cached_models['image'][0])
                    self.image_model_input.delete(0, tk.END)
                    self.image_model_input.insert(0, self.cached_models['image'][0])
                    print(f"Updated image models: {self.cached_models['image']}")
            
            # Update medical dropdown
            if hasattr(self, 'medical_model_dropdown'):
                self.medical_model_dropdown['values'] = self.cached_models['medical']
                if self.cached_models['medical']:
                    self.medical_model_dropdown.set(self.cached_models['medical'][0])
                    self.medical_model_input.delete(0, tk.END)
                    self.medical_model_input.insert(0, self.cached_models['medical'][0])
                    print(f"Updated medical models: {self.cached_models['medical']}")
                    
        except Exception as e:
            print(f"Error updating model dropdowns: {e}")
            import traceback
            traceback.print_exc()

    def try_load_token_from_env(self):
        """Try to load token from environment variable"""
        token = os.environ.get('HUGGINGFACE_TOKEN')
        if token and token.startswith('hf_'):
            self.hf_token = token
            self.token_input.delete(0, tk.END)
            self.token_input.insert(0, token)
            self.token_status_label.config(text="Status: Token loaded from environment")
            self.enable_model_buttons()
            print("Token loaded from environment")
    
    def setup_token_frame(self):
        """Setup the token input frame at the top of the window"""
        token_frame = ttk.LabelFrame(self.root, text="HuggingFace Token")
        token_frame.pack(fill='x', padx=10, pady=5)
        
        # Token input
        self.token_input = ttk.Entry(token_frame, show="*")  # Show as asterisks for security
        self.token_input.pack(side='left', expand=True, fill='x', padx=5, pady=5)
        
        # Set token button
        self.set_token_button = ttk.Button(token_frame, text="Set Token", command=self.set_token)
        self.set_token_button.pack(side='right', padx=5, pady=5)
        
        # Status label for token
        self.token_status_label = ttk.Label(token_frame, text="Status: No token set")
        self.token_status_label.pack(side='right', padx=5, pady=5)
    
    def set_token(self):
        """Set the HuggingFace token and enable model operations"""
        token = self.token_input.get().strip()
        if not token:
            self.show_error("Please enter a HuggingFace token")
            return
        
        if not token.startswith('hf_'):
            self.show_error("Invalid token format. Token should start with 'hf_'")
            return
        
        # Set token in environment variable for persistence
        os.environ['HUGGINGFACE_TOKEN'] = token
        self.hf_token = token
        self.token_status_label.config(text="Status: Token set")
        self.enable_model_buttons()
        print("Token set successfully")
    
    def disable_model_buttons(self):
        """Disable all model-related buttons"""
        self.load_model_button.config(state='disabled')
        self.start_button.config(state='disabled')
        self.generate_button.config(state='disabled')
    
    def enable_model_buttons(self):
        """Enable all model-related buttons"""
        self.load_model_button.config(state='normal')
        self.start_button.config(state='normal')
        self.generate_button.config(state='normal')
    
    def check_token(self):
        """Check if token is set before performing model operations"""
        if not self.hf_token:
            self.show_error("Please set your HuggingFace token first")
            return False
        return True
    
    def setup_webcam_tab(self):
        # Webcam frame
        webcam_frame = ttk.LabelFrame(self.webcam_tab, text="Webcam Feed")
        webcam_frame.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Model selection frame
        model_frame = ttk.Frame(webcam_frame)
        model_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(model_frame, text="Model:").pack(side='left', padx=5)
        
        # Add dropdown for cached models
        self.webcam_model_dropdown = ttk.Combobox(model_frame, values=self.cached_models['webcam'])
        self.webcam_model_dropdown.pack(side='left', expand=True, fill='x', padx=5)
        self.webcam_model_dropdown.bind('<<ComboboxSelected>>', lambda e: self.webcam_model_input.delete(0, tk.END) or self.webcam_model_input.insert(0, self.webcam_model_dropdown.get()))
        
        # Add search entry
        ttk.Label(model_frame, text="Search:").pack(side='left', padx=5)
        self.webcam_model_input = ttk.Entry(model_frame)
        self.webcam_model_input.insert(0, "Salesforce/blip-image-captioning-base") # Default model
        self.webcam_model_input.pack(side='left', expand=True, fill='x', padx=5)
        
        self.load_webcam_model_button = ttk.Button(model_frame, text="Load Model", command=self.load_specified_webcam_model)
        self.load_webcam_model_button.pack(side='right', padx=5)
        
        # Status label
        self.webcam_status_label = ttk.Label(webcam_frame, text="Status: No model loaded")
        self.webcam_status_label.pack(fill='x', padx=5, pady=5)
        
        # Create a canvas for the video display
        self.video_canvas = tk.Canvas(webcam_frame, width=self.webcam_width, height=self.webcam_height)
        self.video_canvas.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Create a label for the video that will be placed on the canvas
        self.video_label = ttk.Label(self.video_canvas)
        self.video_label.place(relx=0.5, rely=0.5, anchor='center')
        
        # Create a label for captions that will overlay the video
        self.caption_label = ttk.Label(
            self.video_canvas,
            text="",
            wraplength=self.webcam_width - 20,
            background='black',
            foreground='white',
            font=('Arial', 12)
        )
        self.caption_label.place(relx=0.5, rely=0.95, anchor='s')
        
        # Control buttons
        button_frame = ttk.Frame(webcam_frame)
        button_frame.pack(fill='x', padx=5, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Webcam", command=self.start_webcam)
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Webcam", command=self.stop_webcam, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        # Initially disable webcam buttons until model is loaded
        self.start_button.config(state='disabled')
    
    def load_specified_webcam_model(self):
        """Load the webcam model specified in the input field."""
        if not self.check_token():
            return
            
        model_name = self.webcam_model_input.get().strip()
        if not model_name:
            self.show_error("Please enter a model name.")
            return

        if self.model_loading: # Prevent multiple loads
            self.show_error("A model is already being loaded. Please wait.")
            return
            
        self.model_loading = True
        is_online = self.check_internet_connection()
        print(f"Attempting to load: {model_name}. Online status: {is_online}")

        # Update status
        self.webcam_status_label.config(text=f"Status: Loading {model_name}...")
        self.load_webcam_model_button.config(state='disabled')
        self.webcam_model_input.config(state='disabled')
        self.webcam_model_dropdown.config(state='disabled')

        def load_thread():
            try:
                # Clear previous model if exists
                if self.webcam_engine is not None:
                    self.webcam_engine = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Load new model
                self.webcam_engine = WebcamBlipEngine(model_name, self.hf_token)
                
                # Add to cached models if not already present
                if model_name not in self.cached_models['webcam']:
                    self.cached_models['webcam'].append(model_name)
                    self.update_model_dropdowns()
                
                # Update UI
                def update_ui():
                    self.webcam_status_label.config(text=f"Status: {model_name} loaded successfully")
                    self.load_webcam_model_button.config(state='normal')
                    self.webcam_model_input.config(state='normal')
                    self.webcam_model_dropdown.config(state='normal')
                    self.start_button.config(state='normal')
                    self.model_loading = False
                self.root.after(0, update_ui)
                
            except Exception as e:
                error_msg = str(e)
                def error_update():
                    self.show_error(f"Error loading webcam model: {error_msg}")
                    self.webcam_status_label.config(text="Status: Error loading model")
                    self.load_webcam_model_button.config(state='normal')
                    self.webcam_model_input.config(state='normal')
                    self.webcam_model_dropdown.config(state='normal')
                    self.model_loading = False
                self.root.after(0, error_update)

        # Start loading in background thread
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
    
    def start_webcam(self):
        if not self.check_token():
            return
            
        if self.webcam_engine is None:
            self.show_error("Please load a model first")
            return
        
        self.webcam_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.load_webcam_model_button.config(state='disabled')
        self.webcam_model_input.config(state='disabled')
        
        # Start webcam thread
        threading.Thread(target=self.webcam_thread, daemon=True).start()
        # Start caption thread
        threading.Thread(target=self.caption_thread, daemon=True).start()
    
    def stop_webcam(self):
        self.webcam_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.load_webcam_model_button.config(state='normal')
        self.webcam_model_input.config(state='normal')
    
    def webcam_thread(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.webcam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webcam_height)
        
        while self.webcam_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update frame queue
            if self.frame_queue.empty():
                self.frame_queue.put(rgb_frame)
            
            # Resize frame to fit canvas
            image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update video display using root.after for thread safety
            def update_ui():
                self.video_label.config(image=photo)
                self.video_label.image = photo
            self.root.after(0, update_ui)
            
            time.sleep(0.03)  # ~30 FPS
        
        cap.release()
    
    def caption_thread(self):
        while self.webcam_running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                try:
                    caption = self.webcam_engine.generate_caption(frame)
                    # Update caption using root.after for thread safety
                    def update_caption():
                        self.caption_label.config(text=caption)
                    self.root.after(0, update_caption)
                except Exception as e:
                    self.show_error(f"Error generating caption: {str(e)}")
                finally:
                    self.frame_queue.task_done()
            time.sleep(0.1)
    
    def setup_chat_tab(self):
        # Chat frame
        chat_frame = ttk.LabelFrame(self.chat_tab, text="Chat")
        chat_frame.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Status label
        self.chat_status_label = ttk.Label(chat_frame, text="Status: No model loaded")
        self.chat_status_label.pack(fill='x', padx=5, pady=5)

        # Model selection frame
        model_frame = ttk.Frame(chat_frame)
        model_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(model_frame, text="Model:").pack(side='left', padx=5)
        
        # Add dropdown for cached models
        self.chat_model_dropdown = ttk.Combobox(model_frame, values=self.cached_models['chat'])
        self.chat_model_dropdown.pack(side='left', expand=True, fill='x', padx=5)
        self.chat_model_dropdown.bind('<<ComboboxSelected>>', lambda e: self.model_name_input.delete(0, tk.END) or self.model_name_input.insert(0, self.chat_model_dropdown.get()))
        
        # Add search entry
        ttk.Label(model_frame, text="Search:").pack(side='left', padx=5)
        self.model_name_input = ttk.Entry(model_frame)
        self.model_name_input.insert(0, "google/gemma-2-2b-it") # Default model
        self.model_name_input.pack(side='left', expand=True, fill='x', padx=5)

        self.load_model_button = ttk.Button(model_frame, text="Load Model", command=self.load_specified_chat_model)
        self.load_model_button.pack(side='right', padx=5)
        
        # Chat history
        self.chat_history_text = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, height=20)
        self.chat_history_text.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Control buttons frame
        control_frame = ttk.Frame(chat_frame)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Clear chat button
        self.clear_button = ttk.Button(control_frame, text="Clear Chat", command=self.clear_chat)
        self.clear_button.pack(side='left', padx=5)
        
        # Stop generation button (initially hidden)
        self.stop_button = ttk.Button(control_frame, text="Stop Generation", command=self.stop_generation, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        # Input frame
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill='x', padx=5, pady=5)
        
        self.chat_input = ttk.Entry(input_frame)
        self.chat_input.pack(side='left', expand=True, fill='x', padx=5)
        
        # Add send button
        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side='right', padx=5)
        
        # Bind Enter key to send message
        self.chat_input.bind('<Return>', lambda e: self.send_message())
        
        # Set focus to chat input
        self.chat_input.focus_set()
        
        # Add flag for generation control
        self.generation_running = False
        self.generation_stopped = False
    
    def setup_image_tab(self):
        # Main container with padding
        main_frame = ttk.Frame(self.image_tab, padding="10")
        main_frame.pack(expand=True, fill='both')
        
        # Left panel for controls
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        
        # Model section
        model_frame = ttk.LabelFrame(left_panel, text="Model", padding="5")
        model_frame.pack(fill='x', pady=(0, 10))
        
        # Model dropdown
        self.image_model_dropdown = ttk.Combobox(model_frame, values=self.cached_models['image'], width=30)
        self.image_model_dropdown.pack(fill='x', pady=2)
        self.image_model_dropdown.set("runwayml/stable-diffusion-v1-5")
        self.image_model_dropdown.bind('<<ComboboxSelected>>', lambda e: self.image_model_input.delete(0, tk.END) or self.image_model_input.insert(0, self.image_model_dropdown.get()))
        
        # Model input
        self.image_model_input = ttk.Entry(model_frame)
        self.image_model_input.insert(0, "runwayml/stable-diffusion-v1-5")
        self.image_model_input.pack(fill='x', pady=2)
        
        # Load button
        self.load_image_model_button = ttk.Button(model_frame, text="Load Model", command=self.load_specified_image_model)
        self.load_image_model_button.pack(fill='x', pady=2)
        
        # Status label
        self.image_status_label = ttk.Label(model_frame, text="Status: No model loaded")
        self.image_status_label.pack(fill='x', pady=2)
        
        # Prompt section
        prompt_frame = ttk.LabelFrame(left_panel, text="Prompts", padding="5")
        prompt_frame.pack(fill='x', pady=(0, 10))
        
        # Positive prompt
        ttk.Label(prompt_frame, text="Positive:").pack(anchor='w')
        self.image_prompt_var = tk.StringVar()
        self.image_prompt_input = ttk.Entry(prompt_frame, textvariable=self.image_prompt_var)
        self.image_prompt_input.pack(fill='x', pady=2)
        
        # Negative prompt
        ttk.Label(prompt_frame, text="Negative:").pack(anchor='w')
        self.negative_prompt_var = tk.StringVar()
        self.negative_prompt_input = ttk.Entry(prompt_frame, textvariable=self.negative_prompt_var)
        self.negative_prompt_input.pack(fill='x', pady=2)
        
        # Parameters section
        params_frame = ttk.LabelFrame(left_panel, text="Parameters", padding="5")
        params_frame.pack(fill='x', pady=(0, 10))
        
        # Steps
        steps_frame = ttk.Frame(params_frame)
        steps_frame.pack(fill='x', pady=2)
        ttk.Label(steps_frame, text="Steps:").pack(side='left')
        self.steps_var = tk.IntVar(value=25)
        steps_spinbox = ttk.Spinbox(steps_frame, from_=1, to=50, textvariable=self.steps_var, width=5)
        steps_spinbox.pack(side='right')
        self.create_tooltip(steps_spinbox, "Number of denoising steps (1-50)")
        
        # Guidance
        guidance_frame = ttk.Frame(params_frame)
        guidance_frame.pack(fill='x', pady=2)
        ttk.Label(guidance_frame, text="Guidance:").pack(side='left')
        self.guidance_var = tk.DoubleVar(value=7.5)
        guidance_spinbox = ttk.Spinbox(guidance_frame, from_=1.0, to=20.0, increment=0.1, textvariable=self.guidance_var, width=5)
        guidance_spinbox.pack(side='right')
        self.create_tooltip(guidance_spinbox, "How closely to follow the prompt (1.0-20.0)")
        
        # Size
        size_frame = ttk.Frame(params_frame)
        size_frame.pack(fill='x', pady=2)
        ttk.Label(size_frame, text="Size:").pack(side='left')
        self.width_var = tk.IntVar(value=512)
        self.height_var = tk.IntVar(value=512)
        width_spinbox = ttk.Spinbox(size_frame, from_=64, to=1024, increment=64, textvariable=self.width_var, width=5)
        width_spinbox.pack(side='left', padx=2)
        ttk.Label(size_frame, text="x").pack(side='left')
        height_spinbox = ttk.Spinbox(size_frame, from_=64, to=1024, increment=64, textvariable=self.height_var, width=5)
        height_spinbox.pack(side='left', padx=2)
        self.create_tooltip(size_frame, "Image dimensions (multiples of 64)")
        
        # Control buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(fill='x', pady=(0, 10))
        
        self.generate_button = ttk.Button(button_frame, text="Generate", command=self.generate_image, state='disabled')
        self.generate_button.pack(side='left', expand=True, padx=2)
        
        self.save_image_button = ttk.Button(button_frame, text="Save", command=self.save_generated_image, state='disabled')
        self.save_image_button.pack(side='left', expand=True, padx=2)
        
        self.clear_image_button = ttk.Button(button_frame, text="Clear", command=self.clear_image)
        self.clear_image_button.pack(side='left', expand=True, padx=2)
        
        # Right panel for image display
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side='right', expand=True, fill='both')
        
        # Image display
        self.image_canvas = tk.Canvas(right_panel, width=512, height=512, bg='black')
        self.image_canvas.pack(expand=True, fill='both')
        
        # Initialize variables
        self.image_engine = None
        self.image_model_loaded = False
        self.generated_image = None
        self.image_model_loading = False
    
    def load_specified_image_model(self):
        """Load the specified image generation model."""
        if not self.check_token():
            return
            
        model_name = self.image_model_input.get().strip()
        if not model_name:
            self.show_error("Please enter a model name.")
            return

        if self.image_model_loading:
            self.show_error("A model is already being loaded. Please wait.")
            return
            
        self.image_model_loading = True
        
        def load_thread():
            try:
                # Update UI
                def update_ui():
                    self.image_status_label.config(text=f"Status: Loading {model_name}...")
                    self.load_image_model_button.config(state='disabled')
                    self.image_model_input.config(state='disabled')
                    self.image_model_dropdown.config(state='disabled')
                
                self.root.after(0, update_ui)
                
                # Initialize engine
                self.image_engine = ArbitraryImageEngine(model_id=model_name, device="cpu")
                self.image_model_loaded = True
                
                # Add to cached models if not already present
                if model_name not in self.cached_models['image']:
                    self.cached_models['image'].append(model_name)
                    self.update_model_dropdowns()
                
                # Update UI
                def final_update():
                    self.image_status_label.config(text=f"Status: {model_name} loaded successfully")
                    self.load_image_model_button.config(state='normal')
                    self.image_model_input.config(state='normal')
                    self.image_model_dropdown.config(state='normal')
                    self.generate_button.config(state='normal')
                
                self.root.after(0, final_update)
                
            except Exception as e:
                def error_update():
                    self.image_status_label.config(text=f"Status: Error loading model - {str(e)}")
                    self.load_image_model_button.config(state='normal')
                    self.image_model_input.config(state='normal')
                    self.image_model_dropdown.config(state='normal')
                    self.show_error(f"Error loading model: {str(e)}")
                
                self.root.after(0, error_update)
            
            finally:
                self.image_model_loading = False
        
        # Start loading thread
        threading.Thread(target=load_thread, daemon=True).start()
    
    def generate_image(self):
        """Generate an image using the loaded model."""
        if not self.image_model_loaded:
            self.show_error("Please load a model first")
            return
            
        prompt = self.image_prompt_var.get().strip()
        if not prompt:
            self.show_error("Please enter a prompt")
            return
        
        def generation_thread():
            try:
                # Update UI
                def update_ui():
                    self.image_status_label.config(text="Status: Generating image...")
                    self.generate_button.config(state='disabled')
                    self.load_image_model_button.config(state='disabled')
                
                self.root.after(0, update_ui)
                
                # Get parameters
                steps = self.steps_var.get()
                guidance = self.guidance_var.get()
                width = self.width_var.get()
                height = self.height_var.get()
                negative_prompt = self.negative_prompt_var.get().strip()
                
                # Generate image
                image = self.image_engine(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height
                )[0]
                
                # Convert to PhotoImage
                self.generated_image = ImageTk.PhotoImage(image)
                
                # Update canvas
                def update_canvas():
                    self.image_canvas.delete("all")  # Clear previous image
                    self.image_canvas.create_image(256, 256, image=self.generated_image)
                    self.image_status_label.config(text="Status: Image generated successfully")
                    self.generate_button.config(state='normal')
                    self.load_image_model_button.config(state='normal')
                    self.save_image_button.config(state='normal')
                
                self.root.after(0, update_canvas)
                
            except Exception as e:
                def error_update():
                    self.image_status_label.config(text=f"Status: Error generating image - {str(e)}")
                    self.generate_button.config(state='normal')
                    self.load_image_model_button.config(state='normal')
                    self.show_error(f"Error generating image: {str(e)}")
                
                self.root.after(0, error_update)
        
        # Start generation thread
        threading.Thread(target=generation_thread, daemon=True).start()
    
    def save_generated_image(self):
        """Save the generated image to a file."""
        if not self.generated_image:
            self.show_error("No image to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Get the PIL Image from the PhotoImage
                image = Image.open(self.generated_image)
                image.save(file_path)
                self.image_status_label.config(text=f"Status: Image saved to {file_path}")
            except Exception as e:
                self.show_error(f"Error saving image: {str(e)}")
    
    def setup_medical_tab(self):
        # Medical imaging frame
        medical_frame = ttk.LabelFrame(self.medical_tab, text="Medical Image Analysis")
        medical_frame.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Model selection frame
        model_frame = ttk.Frame(medical_frame)
        model_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(model_frame, text="Model:").pack(side='left', padx=5)
        
        # Add dropdown for cached models
        self.medical_model_dropdown = ttk.Combobox(model_frame, values=self.cached_models['medical'])
        self.medical_model_dropdown.pack(side='left', expand=True, fill='x', padx=5)
        self.medical_model_dropdown.bind('<<ComboboxSelected>>', lambda e: self.medical_model_input.delete(0, tk.END) or self.medical_model_input.insert(0, self.medical_model_dropdown.get()))
        
        # Add search entry
        ttk.Label(model_frame, text="Search:").pack(side='left', padx=5)
        self.medical_model_input = ttk.Entry(model_frame)
        self.medical_model_input.insert(0, "microsoft/resnet-50") # Default model
        self.medical_model_input.pack(side='left', expand=True, fill='x', padx=5)

        self.load_medical_model_button = ttk.Button(model_frame, text="Load Model", command=self.load_specified_medical_model)
        self.load_medical_model_button.pack(side='right', padx=5)
        
        # Status label
        self.medical_status_label = ttk.Label(medical_frame, text="Status: No model loaded")
        self.medical_status_label.pack(fill='x', padx=5, pady=5)
        
        # File upload frame
        upload_frame = ttk.LabelFrame(medical_frame, text="Upload Images")
        upload_frame.pack(fill='x', padx=5, pady=5)
        
        # Supported formats
        supported_formats = "Supported formats: PNG, JPEG, GIF, BMP, TIFF, WebP, DICOM, NIfTI, Analyze, MGH, MINC, PFS"
        ttk.Label(upload_frame, text=supported_formats, wraplength=600).pack(padx=5, pady=5)
        
        # File list
        self.file_list_frame = ttk.Frame(upload_frame)
        self.file_list_frame.pack(fill='x', padx=5, pady=5)
        
        self.file_listbox = tk.Listbox(self.file_list_frame, height=5, bg='#404040', fg='#ffffff')
        self.file_listbox.pack(side='left', fill='x', expand=True)
        
        scrollbar = ttk.Scrollbar(self.file_list_frame, orient="vertical", command=self.file_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.file_listbox.configure(yscrollcommand=scrollbar.set)
        
        # File control buttons
        file_buttons_frame = ttk.Frame(upload_frame)
        file_buttons_frame.pack(fill='x', padx=5, pady=5)
        
        self.add_file_button = ttk.Button(file_buttons_frame, text="Add Files", command=self.add_medical_files)
        self.add_file_button.pack(side='left', padx=5)
        
        self.remove_file_button = ttk.Button(file_buttons_frame, text="Remove Selected", command=self.remove_medical_file)
        self.remove_file_button.pack(side='left', padx=5)
        
        self.clear_files_button = ttk.Button(file_buttons_frame, text="Clear All", command=self.clear_medical_files)
        self.clear_files_button.pack(side='left', padx=5)
        
        # Analysis frame
        analysis_frame = ttk.LabelFrame(medical_frame, text="Image Analysis")
        analysis_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Image display
        self.medical_canvas = tk.Canvas(analysis_frame, bg='#404040')
        self.medical_canvas.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Analysis results
        results_frame = ttk.Frame(analysis_frame)
        results_frame.pack(side='right', fill='y', padx=5, pady=5)
        
        ttk.Label(results_frame, text="Analysis Results:").pack(anchor='w', padx=5, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, width=40, height=20, bg='#404040', fg='#ffffff')
        self.results_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Analysis button
        self.analyze_button = ttk.Button(analysis_frame, text="Analyze Images", command=self.analyze_medical_images, state='disabled')
        self.analyze_button.pack(side='bottom', padx=5, pady=5)
        
        # Initialize file list
        self.medical_files = []
        
        # Initially disable analysis button
        self.analyze_button.config(state='disabled')
    
    def add_medical_files(self):
        """Add medical image files to the list"""
        # Define file types for all common image formats
        filetypes = [
            ('All Images', '*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.tiff;*.webp;*.dcm;*.nii;*.nii.gz;*.hdr;*.img;*.mgh;*.mnc;*.pfs'),
            ('PNG Files', '*.png'),
            ('JPEG Files', '*.jpg;*.jpeg'),
            ('GIF Files', '*.gif'),
            ('BMP Files', '*.bmp'),
            ('TIFF Files', '*.tiff'),
            ('WebP Files', '*.webp'),
            ('DICOM Files', '*.dcm'),
            ('NIfTI Files', '*.nii;*.nii.gz'),
            ('Analyze Files', '*.hdr;*.img'),
            ('MGH Files', '*.mgh'),
            ('MINC Files', '*.mnc'),
            ('PFS Files', '*.pfs'),
            ('All Files', '*')
        ]
        
        try:
            # Use a simpler file dialog approach for macOS
            if sys.platform == 'darwin':
                files = filedialog.askopenfilenames(
                    title="Select Images",
                    filetypes=[('All Files', '*')],  # Show all files on macOS
                    multiple=True
                )
            else:
                files = filedialog.askopenfilenames(
                    title="Select Images",
                    filetypes=filetypes,
                    multiple=True
                )
            
            if files:
                for file in files:
                    # Check if file has a supported extension
                    ext = os.path.splitext(file)[1].lower()
                    if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.dcm', '.nii', '.gz', '.hdr', '.img', '.mgh', '.mnc', '.pfs']:
                        if file not in self.medical_files:
                            self.medical_files.append(file)
                            self.file_listbox.insert(tk.END, os.path.basename(file))
                    else:
                        self.show_error(f"Unsupported file format: {ext}")
                
                # Enable analyze button if we have files and a model
                if self.medical_files and self.medical_engine is not None:
                    self.analyze_button.config(state='normal')
        except Exception as e:
            self.show_error(f"Error selecting files: {str(e)}")
            print(f"File dialog error: {str(e)}")

    def remove_medical_file(self):
        """Remove selected file from the list"""
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
            self.file_listbox.delete(index)
            self.medical_files.pop(index)
            
            # Disable analyze button if no files left
            if not self.medical_files:
                self.analyze_button.config(state='disabled')

    def clear_medical_files(self):
        """Clear all files from the list"""
        self.file_listbox.delete(0, tk.END)
        self.medical_files.clear()
        self.analyze_button.config(state='disabled')
        self.medical_canvas.delete("all")
        self.results_text.delete(1.0, tk.END)

    def load_specified_medical_model(self):
        """Load the medical imaging model specified in the input field."""
        if not self.check_token():
            return
            
        model_name = self.medical_model_input.get().strip()
        if not model_name:
            self.show_error("Please enter a model name.")
            return

        if self.model_loading: # Prevent multiple loads
            self.show_error("A model is already being loaded. Please wait.")
            return
            
        self.model_loading = True
        is_online = self.check_internet_connection()
        print(f"Attempting to load: {model_name}. Online status: {is_online}")

        # Update status
        self.medical_status_label.config(text=f"Status: Loading {model_name}...")
        self.load_medical_model_button.config(state='disabled')
        self.medical_model_input.config(state='disabled')
        self.medical_model_dropdown.config(state='disabled')

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
                self.medical_engine = MedicalImageEngine(model_name, self.hf_token)
                if not self.medical_engine.load_model():
                    raise RuntimeError("Failed to load model")
                
                # Add to cached models if not already present
                if model_name not in self.cached_models['medical']:
                    self.cached_models['medical'].append(model_name)
                    self.update_model_dropdowns()
                
                # Update UI
                def update_ui():
                    self.medical_status_label.config(text=f"Status: {model_name} loaded successfully")
                    self.load_medical_model_button.config(state='normal')
                    self.medical_model_input.config(state='normal')
                    self.medical_model_dropdown.config(state='normal')
                    # Enable analyze button if we have files
                    if self.medical_files:
                        self.analyze_button.config(state='normal')
                    self.model_loading = False
                self.root.after(0, update_ui)
                
            except Exception as e:
                error_msg = str(e)
                def error_update():
                    self.show_error(f"Error loading medical model: {error_msg}")
                    self.medical_status_label.config(text="Status: Error loading model")
                    self.load_medical_model_button.config(state='normal')
                    self.medical_model_input.config(state='normal')
                    self.medical_model_dropdown.config(state='normal')
                    self.model_loading = False
                self.root.after(0, error_update)

        # Start loading in background thread
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()

    def analyze_medical_images(self):
        """Analyze the uploaded medical images"""
        if not self.medical_engine:
            self.show_error("Please load a model first")
            return
            
        if not self.medical_files:
            self.show_error("Please add some medical images first")
            return
        
        # Disable buttons during analysis
        self.analyze_button.config(state='disabled')
        self.add_file_button.config(state='disabled')
        self.remove_file_button.config(state='disabled')
        self.clear_files_button.config(state='disabled')
        
        def analysis_thread():
            try:
                self.medical_status_label.config(text="Status: Analyzing images...")
                self.results_text.delete(1.0, tk.END)
                
                for file in self.medical_files:
                    try:
                        # Load and analyze image
                        image, results = self.medical_engine.analyze_image(file)
                        
                        # Display image
                        if image is not None:
                            # Convert to PhotoImage
                            photo = ImageTk.PhotoImage(image)
                            
                            # Update canvas
                            def update_canvas():
                                self.medical_canvas.delete("all")
                                self.medical_canvas.create_image(
                                    self.medical_canvas.winfo_width()//2,
                                    self.medical_canvas.winfo_height()//2,
                                    image=photo
                                )
                                self.medical_canvas.image = photo  # Keep reference
                            self.root.after(0, update_canvas)
                        
                        # Display results
                        def update_results():
                            self.results_text.insert(tk.END, f"\nResults for {os.path.basename(file)}:\n")
                            self.results_text.insert(tk.END, results + "\n")
                            self.results_text.see(tk.END)
                        self.root.after(0, update_results)
                        
                    except Exception as img_error:
                        error_msg = str(img_error)
                        def show_error():
                            self.results_text.insert(tk.END, f"\nError analyzing {os.path.basename(file)}: {error_msg}\n")
                            self.results_text.see(tk.END)
                        self.root.after(0, show_error)
                
                def final_update():
                    self.medical_status_label.config(text="Status: Analysis complete")
                    self.analyze_button.config(state='normal')
                    self.add_file_button.config(state='normal')
                    self.remove_file_button.config(state='normal')
                    self.clear_files_button.config(state='normal')
                self.root.after(0, final_update)
                
            except Exception as e:
                error_msg = str(e)
                def error_update():
                    self.show_error(f"Error during analysis: {error_msg}")
                    self.medical_status_label.config(text="Status: Error during analysis")
                    self.analyze_button.config(state='normal')
                    self.add_file_button.config(state='normal')
                    self.remove_file_button.config(state='normal')
                    self.clear_files_button.config(state='normal')
                self.root.after(0, error_update)

        # Start analysis in background thread
        thread = threading.Thread(target=analysis_thread, daemon=True)
        thread.start()

    def on_tab_change(self, event):
        """Handle tab changes and update model selection"""
        # Clear previous model and memory
        self.clear_current_model()
        
        # Get new tab
        tab = self.notebook.select()
        tab_name = self.notebook.tab(tab, "text")
        self.current_tab = tab_name
        
        # Scan for models when changing tabs
        self.scan_cached_models()
        
        # Focus appropriate input
        if tab_name == "🎥 Webcam":
            self.webcam_model_input.focus_set()
        elif tab_name == "💬 Chat":
            self.model_name_input.focus_set()
        elif tab_name == "🎨 Image Generation":
            self.image_prompt_input.focus_set()
        elif tab_name == "🏥 Medical Imaging":
            self.medical_model_input.focus_set()
    
    def clear_current_model(self):
        """Clear the current model and free memory"""
        if self.model_loading:
            # Wait for current loading to complete
            if self.loading_thread and self.loading_thread.is_alive():
                self.loading_thread.join(timeout=1.0)
        
        # Clear webcam resources
        if self.webcam_engine is not None:
            self.stop_webcam()
            self.webcam_engine = None
            self.webcam_status_label.config(text="Status: No model loaded")
            self.start_button.config(state='disabled')
            self.stop_button.config(state='disabled')
            self.load_webcam_model_button.config(state='normal')
            self.webcam_model_input.config(state='normal')
        
        # Clear chat resources
        if self.chat_engine is not None:
            self.chat_engine = None
            self.chat_model_loaded = False
            self.chat_status_label.config(text="Status: No model loaded")
            self.load_model_button.config(state='normal')
            self.model_name_input.config(state='normal')
            self.clear_chat()
        
        # Clear diffusion resources
        if self.diffusion_engine is not None:
            self.diffusion_engine = None
            self.image_label.config(text="")
            self.generate_button.config(state='normal')
            self.image_model_input.config(state='normal')
        
        # Clear medical resources
        if self.medical_engine is not None:
            self.medical_engine = None
            self.medical_status_label.config(text="Status: No model loaded")
            self.load_medical_model_button.config(state='normal')
            self.medical_model_input.config(state='normal')
            self.analyze_button.config(state='disabled')
            self.medical_canvas.delete("all")
            self.results_text.delete(1.0, tk.END)
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Clear any queued frames
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
            except queue.Empty:
                break
        
        # Clear any queued captions
        while not self.caption_queue.empty():
            try:
                self.caption_queue.get_nowait()
                self.caption_queue.task_done()
            except queue.Empty:
                break
        
        # Reset generation flags
        self.generation_running = False
        self.generation_stopped = False
        
        # Clear video display
        self.video_label.config(image='')
        self.caption_label.config(text='')
        
        print("Cleared all models and memory")
    
    def check_internet_connection(self, host="8.8.8.8", port=53, timeout=3):
        """Check for internet connection by trying to connect to Google DNS."""
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except socket.error as ex:
            print(f"Internet check failed: {ex}")
            return False
    
    def load_specified_chat_model(self):
        """Load the chat model specified in the input field."""
        if not self.check_token():
            return
            
        model_name = self.model_name_input.get().strip()
        if not model_name:
            self.show_error("Please enter a model name.")
            return

        if self.model_loading: # Prevent multiple loads
            self.show_error("A model is already being loaded. Please wait.")
            return
            
        self.model_loading = True
        is_online = self.check_internet_connection()
        print(f"Attempting to load: {model_name}. Online status: {is_online}")

        # Clear previous chat history when loading a new model
        self.clear_chat()

        # Update UI
        self.chat_status_label.config(text=f"Status: Loading {model_name}...")
        self.load_model_button.config(state='disabled')
        self.model_name_input.config(state='disabled')
        self.chat_model_dropdown.config(state='disabled')

        # Run loading in a background thread
        self.loading_thread = threading.Thread(
            target=_load_model_thread, 
            args=(self, model_name, self.hf_token, is_online),
            daemon=True
        )
        self.loading_thread.start()

        # Add to cached models if not already present
        if model_name not in self.cached_models['chat']:
            self.cached_models['chat'].append(model_name)
            self.update_model_dropdowns()
    
    def load_diffusion_model(self):
        """Load diffusion model on demand"""
        if self.diffusion_engine is None:
            try:
                self.diffusion_engine = SIMDOptimizedPipeline(
                    self.image_model_input.get().strip(),
                    cache_dir=self.cache_dir
                )
                return True
            except Exception as e:
                self.show_error(f"Error loading diffusion model: {str(e)}")
                return False
        return True
    
    def clear_chat(self):
        """Clear the chat history"""
        self.chat_history_text.delete(1.0, tk.END)
        self.current_response = "" # Reset current response on clear
        self.chat_input.focus_set()
    
    def _update_chat_text(self, text_to_add):
        """Helper method to insert text into chat history (for use with root.after)"""
        self.chat_history_text.insert(tk.END, text_to_add)
        self.chat_history_text.see(tk.END)

    def stream_token(self, token):
        """Handle streaming tokens immediately for a fluid experience"""
        if token: # Ensure token is not empty
            self.current_response += token
            
            # Schedule UI update for this token using the helper method via root.after
            self.root.after(0, self._update_chat_text, token)

    def stop_generation(self):
        """Stop the current generation process"""
        if self.generation_running:
            self.generation_stopped = True
            self.chat_status_label.config(text="Status: Stopping generation...")
            self.stop_button.config(state='disabled')
            self.send_button.config(state='normal')
            self.chat_input.config(state='normal')
            self.clear_button.config(state='normal')
            self.chat_input.focus_set()

    def send_message(self):
        if self.model_loading:
            self.show_error("Please wait for the model to finish loading")
            return
        
        if not self.chat_model_loaded:
            self.show_error("Please wait for the model to load")
            return
        
        message = self.chat_input.get()
        if message:
            self.chat_input.config(state='disabled')
            self.send_button.config(state='disabled')
            self.clear_button.config(state='disabled')
            self.stop_button.config(state='normal')  # Enable stop button
            
            self.chat_history_text.insert(tk.END, f"You: {message}\n")
            self.chat_history_text.insert(tk.END, "AI: ")
            self.chat_history_text.see(tk.END)
            self.chat_input.delete(0, tk.END)
            
            # Reset generation flags
            self.generation_running = True
            self.generation_stopped = False
            
            def generation_thread():
                try:
                    self.root.after(0, lambda: self.chat_status_label.config(text="Status: Generating response..."))
                    
                    # Create a custom streamer that writes to the UI
                    class UITextStreamer:
                        def __init__(self, text_widget, parent):
                            self.text_widget = text_widget
                            self.root = text_widget.master
                            self.parent = parent
                            self.buffer = ""
                        
                        def write(self, text):
                            # Check if generation was stopped
                            if self.parent.generation_stopped:
                                return
                                
                            # Filter out TPS display and cursor control characters
                            if text.startswith('\033[') or text.startswith('[TPS:'):
                                return
                            
                            # Clean up any remaining control characters
                            text = text.replace('\033[s', '').replace('\033[u', '')
                            text = text.replace('\033[3;1H', '').replace('\033[K', '')
                            
                            # Only update UI if we have actual text
                            if text.strip():
                                # Schedule UI update
                                def update():
                                    if not self.parent.generation_stopped:
                                        self.text_widget.insert(tk.END, text)
                                        self.text_widget.see(tk.END)
                                self.root.after(0, update)
                        
                        def flush(self):
                            pass
                    
                    # Create streamer instance
                    streamer = UITextStreamer(self.chat_history_text, self)
                    
                    # Call backend chat method with streaming
                    start_time = time.time()
                    # Redirect stdout to our custom streamer
                    old_stdout = sys.stdout
                    sys.stdout = streamer
                    try:
                        response = self.chat_engine.chat(message)
                    finally:
                        sys.stdout = old_stdout
                    end_time = time.time()
                    # Only log timing to console
                    print(f"Response generated in {end_time - start_time:.2f} seconds")
                    
                    # Schedule final UI updates
                    def final_update():
                        # Add newline after stream
                        if not self.generation_stopped:
                            self.chat_history_text.insert(tk.END, "\n")
                            self.chat_history_text.see(tk.END)
                        
                        self.chat_status_label.config(text="Status: Ready")
                        self.chat_input.config(state='normal')
                        self.send_button.config(state='normal')
                        self.clear_button.config(state='normal')
                        self.stop_button.config(state='disabled')
                        self.chat_input.focus_set()
                        self.generation_running = False
                    
                    self.root.after(0, final_update)
                    
                except Exception as e:
                    # Error Handling: Update status and re-enable UI
                    error_msg = str(e)  # Capture the error message
                    def error_update():
                        self.chat_status_label.config(text="Status: Error generating response")
                        self.show_error(f"Error getting chat response: {error_msg}") 
                        self.chat_input.config(state='normal')
                        self.send_button.config(state='normal')
                        self.clear_button.config(state='normal')
                        self.stop_button.config(state='disabled')
                        self.chat_input.focus_set()
                        self.generation_running = False
                    self.root.after(0, error_update)
                finally:
                    pass # No change needed to model_loading flag here.

            # Start the generation thread
            thread = threading.Thread(target=generation_thread, daemon=True)
            thread.start()
    
    def show_error(self, message):
        messagebox.showerror("Error", message)
    
    def cleanup(self):
        self.webcam_running = False
        self.clear_current_model()

    def create_tooltip(self, widget, text):
        """Create a tooltip for a given widget"""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(tooltip, text=text, justify='left',
                            background="#ffffe0", relief='solid', borderwidth=1,
                            wraplength=300)
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
            
            widget.tooltip = tooltip
            widget.bind('<Leave>', lambda e: hide_tooltip())
            tooltip.bind('<Leave>', lambda e: hide_tooltip())
        
        widget.bind('<Enter>', show_tooltip)

    def clear_image(self):
        """Clear the generated image and reset the canvas"""
        self.image_canvas.delete("all")
        self.generated_image = None
        self.save_image_button.config(state='disabled')
        self.image_status_label.config(text="Status: Image cleared")
        # Don't clear prompts or parameters to maintain persistence

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
    app = AIStudioUI(root)
    
    # Set up window close handler
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    
    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    main() 