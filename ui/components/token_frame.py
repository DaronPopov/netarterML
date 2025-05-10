import tkinter as tk
from tkinter import ttk
import os

class TokenFrame:
    def __init__(self, parent, on_token_set):
        self.parent = parent
        self.on_token_set = on_token_set
        self.hf_token = None
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="HuggingFace Token")
        self.frame.pack(fill='x', padx=10, pady=5)
        
        # Token input
        self.token_input = ttk.Entry(self.frame, show="*")  # Show as asterisks for security
        self.token_input.pack(side='left', expand=True, fill='x', padx=5, pady=5)
        
        # Set token button
        self.set_token_button = ttk.Button(self.frame, text="Set Token", command=self.set_token)
        self.set_token_button.pack(side='right', padx=5, pady=5)
        
        # Status label for token
        self.token_status_label = ttk.Label(self.frame, text="Status: No token set")
        self.token_status_label.pack(side='right', padx=5, pady=5)
        
        # Try to load token from environment
        self.try_load_token_from_env()
    
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
        self.on_token_set()
        print("Token set successfully")
    
    def try_load_token_from_env(self):
        """Try to load token from environment variable"""
        token = os.environ.get('HUGGINGFACE_TOKEN')
        if token and token.startswith('hf_'):
            self.hf_token = token
            self.token_input.delete(0, tk.END)
            self.token_input.insert(0, token)
            self.token_status_label.config(text="Status: Token loaded from environment")
            self.on_token_set()
            print("Token loaded from environment")
    
    def show_error(self, message):
        """Show error message"""
        from tkinter import messagebox
        messagebox.showerror("Error", message)
    
    def get_token(self):
        """Get the current token"""
        return self.hf_token 