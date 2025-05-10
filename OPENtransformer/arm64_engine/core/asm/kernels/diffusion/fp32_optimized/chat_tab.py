import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import sys
import gc
import torch
from typing import Callable

class ChatTab:
    def __init__(self, parent, load_model_callback: Callable):
        self.frame = ttk.Frame(parent)
        self.load_model_callback = load_model_callback
        self.root = parent.winfo_toplevel()  # Get the root window
        
        # Initialize variables
        self.chat_engine = None
        self.chat_model_loaded = False
        self.generation_running = False
        self.generation_stopped = False
        self.current_response = ""
        self.model_loading = False
        self.loading_thread = None
        
        # Create chat frame
        chat_frame = ttk.LabelFrame(self.frame, text="Chat")
        chat_frame.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Status label
        self.chat_status_label = ttk.Label(chat_frame, text="Status: No model loaded")
        self.chat_status_label.pack(fill='x', padx=5, pady=5)

        # Model selection frame
        model_frame = ttk.Frame(chat_frame)
        model_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(model_frame, text="Model:").pack(side='left', padx=5)
        
        # Add dropdown for cached models
        self.chat_model_dropdown = ttk.Combobox(model_frame)
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
    
    def update_model_dropdown(self, models):
        """Update the model dropdown with available models"""
        self.chat_model_dropdown['values'] = models
        if models:
            self.chat_model_dropdown.set(models[0])
            self.model_name_input.delete(0, tk.END)
            self.model_name_input.insert(0, models[0])
    
    def load_specified_chat_model(self):
        """Load the chat model specified in the input field"""
        model_name = self.model_name_input.get().strip()
        if not model_name:
            self.show_error("Please enter a model name")
            return
        
        if self.model_loading:  # Prevent multiple loads
            self.show_error("A model is already being loaded. Please wait.")
            return
            
        self.model_loading = True
        
        # Update UI
        self.chat_status_label.config(text=f"Status: Loading {model_name}...")
        self.load_model_button.config(state='disabled')
        self.model_name_input.config(state='disabled')
        self.chat_model_dropdown.config(state='disabled')
        
        def on_model_loaded(success, error=None):
            if success:
                self.chat_model_loaded = True
                self.chat_status_label.config(text=f"Status: {model_name} loaded successfully")
            else:
                self.show_error(f"Error loading model: {error}")
                self.chat_status_label.config(text="Status: Error loading model")
            
            self.load_model_button.config(state='normal')
            self.model_name_input.config(state='normal')
            self.chat_model_dropdown.config(state='normal')
            self.model_loading = False
        
        # Call the parent's load model callback
        self.load_model_callback(model_name, on_model_loaded)
    
    def clear_chat(self):
        """Clear the chat history"""
        self.chat_history_text.delete(1.0, tk.END)
        self.current_response = "" # Reset current response on clear
        self.chat_input.focus_set()
    
    def _update_chat_text(self, text_to_add):
        """Helper method to insert text into chat history"""
        self.chat_history_text.insert(tk.END, text_to_add)
        self.chat_history_text.see(tk.END)

    def stream_token(self, token):
        """Handle streaming tokens immediately for a fluid experience"""
        if token: # Ensure token is not empty
            self.current_response += token
            self.chat_history_text.insert(tk.END, token)
            self.chat_history_text.see(tk.END)

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
        """Send a message to the chat model"""
        if self.model_loading:
            self.show_error("Please wait for the model to finish loading")
            return
            
        if not self.chat_model_loaded:
            self.show_error("Please load a model first")
            return
            
        if not self.chat_engine:
            self.show_error("Chat engine not initialized. Please try loading the model again.")
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
                    self.chat_status_label.config(text="Status: Generating response...")
                    
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
                        if not hasattr(self.chat_engine, 'chat'):
                            raise AttributeError("Chat engine does not have a chat method")
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
                    
                    self.chat_history_text.after(0, final_update)
                    
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
                    self.chat_history_text.after(0, error_update)

            # Start the generation thread
            thread = threading.Thread(target=generation_thread, daemon=True)
            thread.start()
    
    def show_error(self, message):
        """Show an error message box"""
        messagebox.showerror("Error", message)
    
    def cleanup(self):
        """Clean up resources"""
        if self.chat_engine:
            self.chat_engine = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        self.chat_model_loaded = False
        self.chat_status_label.config(text="Status: No model loaded")
        self.load_model_button.config(state='normal')
        self.model_name_input.config(state='normal')
        self.clear_chat() 