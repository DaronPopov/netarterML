import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).absolute().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add OPENtransformer to Python path
open_transformer_path = os.path.join(project_root, "OPENtransformer")
if open_transformer_path not in sys.path:
    sys.path.insert(0, open_transformer_path)

# Import our arbitrary image engine
from ui.engines.arbitrary_image_engine import SIMDOptimizedPipeline as ArbitraryImageEngine

class ImageTab:
    def __init__(self, parent, on_model_load, app):
        self.parent = parent
        self.on_model_load = on_model_load
        self.app = app  # Store the app reference
        self.image_engine = None
        self.generated_image = None
        
        # Create main container with padding
        self.frame = ttk.Frame(parent, padding="10")
        self.frame.pack(expand=True, fill='both')
        
        # Left panel for controls
        self.left_panel = ttk.Frame(self.frame)
        self.left_panel.pack(side='left', fill='y', padx=(0, 10))
        
        # Model section
        self.setup_model_frame()
        
        # Prompt section
        self.setup_prompt_frame()
        
        # Parameters section
        self.setup_parameters_frame()
        
        # Control buttons
        self.setup_control_buttons()
        
        # Right panel for image display
        self.right_panel = ttk.Frame(self.frame)
        self.right_panel.pack(side='right', expand=True, fill='both')
        
        # Image display
        self.setup_image_display()
    
    def setup_model_frame(self):
        model_frame = ttk.LabelFrame(self.left_panel, text="Model", padding="5")
        model_frame.pack(fill='x', pady=(0, 10))
        
        # Model dropdown with search
        model_input_frame = ttk.Frame(model_frame)
        model_input_frame.pack(fill='x', pady=2)
        
        self.image_model_dropdown = ttk.Combobox(model_input_frame, values=[], width=30)
        self.image_model_dropdown.pack(side='left', expand=True, fill='x', padx=(0, 5))
        self.image_model_dropdown.set("runwayml/stable-diffusion-v1-5")
        self.image_model_dropdown.bind('<<ComboboxSelected>>', 
            lambda e: self.image_model_input.delete(0, tk.END) or 
                     self.image_model_input.insert(0, self.image_model_dropdown.get()))
        
        # Model input
        self.image_model_input = ttk.Entry(model_input_frame)
        self.image_model_input.insert(0, "runwayml/stable-diffusion-v1-5")
        self.image_model_input.pack(side='left', expand=True, fill='x')
        
        # Load button
        self.load_image_model_button = ttk.Button(model_frame, text="Load Model", 
                                                command=self.load_specified_image_model)
        self.load_image_model_button.pack(fill='x', pady=2)
        
        # Status label
        self.image_status_label = ttk.Label(model_frame, text="Status: No model loaded")
        self.image_status_label.pack(fill='x', pady=2)
    
    def setup_prompt_frame(self):
        prompt_frame = ttk.LabelFrame(self.left_panel, text="Prompts", padding="5")
        prompt_frame.pack(fill='x', pady=(0, 10))
        
        # Positive prompt
        ttk.Label(prompt_frame, text="Positive:").pack(anchor='w')
        self.prompt_var = tk.StringVar()
        self.prompt_input = ttk.Entry(prompt_frame, textvariable=self.prompt_var)
        self.prompt_input.pack(fill='x', pady=2)
        self.create_tooltip(self.prompt_input, "Enter your prompt here. Be descriptive!")
        
        # Negative prompt
        ttk.Label(prompt_frame, text="Negative:").pack(anchor='w')
        self.negative_prompt_var = tk.StringVar()
        self.negative_prompt_input = ttk.Entry(prompt_frame, textvariable=self.negative_prompt_var)
        self.negative_prompt_input.pack(fill='x', pady=2)
        self.create_tooltip(self.negative_prompt_input, "Enter what you don't want in the image")
    
    def setup_parameters_frame(self):
        params_frame = ttk.LabelFrame(self.left_panel, text="Parameters", padding="5")
        params_frame.pack(fill='x', pady=(0, 10))
        
        # Steps
        steps_frame = ttk.Frame(params_frame)
        steps_frame.pack(fill='x', pady=2)
        ttk.Label(steps_frame, text="Steps:").pack(side='left')
        self.steps_var = tk.IntVar(value=25)
        steps_spinbox = ttk.Spinbox(steps_frame, from_=1, to=50, textvariable=self.steps_var, width=5)
        steps_spinbox.pack(side='right')
        self.create_tooltip(steps_spinbox, "Number of denoising steps (1-50)\nHigher = better quality but slower")
        
        # Guidance
        guidance_frame = ttk.Frame(params_frame)
        guidance_frame.pack(fill='x', pady=2)
        ttk.Label(guidance_frame, text="Guidance:").pack(side='left')
        self.guidance_var = tk.DoubleVar(value=7.5)
        guidance_spinbox = ttk.Spinbox(guidance_frame, from_=1.0, to=20.0, increment=0.1, textvariable=self.guidance_var, width=5)
        guidance_spinbox.pack(side='right')
        self.create_tooltip(guidance_spinbox, "How closely to follow the prompt (1.0-20.0)\nHigher = more prompt influence")
        
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
        self.create_tooltip(size_frame, "Image dimensions (multiples of 64)\nLarger = more detail but slower")
    
    def setup_control_buttons(self):
        button_frame = ttk.Frame(self.left_panel)
        button_frame.pack(fill='x', pady=(0, 10))
        
        # Generate button
        self.generate_button = ttk.Button(button_frame, text="Generate", 
                                        command=self.generate_image, state='disabled')
        self.generate_button.pack(side='left', expand=True, padx=2)
        self.create_tooltip(self.generate_button, "Generate image with current settings")
        
        # Save button
        self.save_image_button = ttk.Button(button_frame, text="Save", 
                                          command=self.save_generated_image, state='disabled')
        self.save_image_button.pack(side='left', expand=True, padx=2)
        self.create_tooltip(self.save_image_button, "Save the generated image")
        
        # Clear button
        self.clear_image_button = ttk.Button(button_frame, text="Clear", 
                                           command=self.clear_image)
        self.clear_image_button.pack(side='left', expand=True, padx=2)
        self.create_tooltip(self.clear_image_button, "Clear the current image")
    
    def setup_image_display(self):
        # Create a frame for the image display with a border
        display_frame = ttk.Frame(self.right_panel, style='ImageDisplay.TFrame')
        display_frame.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Create canvas with black background
        self.image_canvas = tk.Canvas(display_frame, width=512, height=512, bg='black',
                                    highlightthickness=0)
        self.image_canvas.pack(expand=True, fill='both')
        
        # Add placeholder text
        self.image_canvas.create_text(256, 256, text="Generated image will appear here",
                                    fill='gray', font=('Arial', 14))
    
    def load_specified_image_model(self):
        """Load the image generation model specified in the input field."""
        model_name = self.image_model_input.get().strip()
        if not model_name:
            self.show_error("Please enter a model name.")
            return
        
        # Update status
        self.image_status_label.config(text=f"Status: Loading {model_name}...")
        self.load_image_model_button.config(state='disabled')
        self.image_model_input.config(state='disabled')
        self.image_model_dropdown.config(state='disabled')
        
        # Call the parent's model loading function
        self.on_model_load(model_name, self.on_model_loaded)
    
    def on_model_loaded(self, success, error=None):
        """Callback for when model loading is complete"""
        def update_ui():
            if success:
                # Get the engine from the app instance
                self.image_engine = self.app.diffusion_engine
                if not self.image_engine:
                    self.show_error("Model loaded but engine reference is missing")
                    return
                
                self.image_status_label.config(text=f"Status: Model loaded successfully")
                self.generate_button.config(state='normal')
            else:
                self.image_status_label.config(text="Status: Error loading model")
                self.show_error(f"Error loading image model: {error}")
            
            self.load_image_model_button.config(state='normal')
            self.image_model_input.config(state='normal')
            self.image_model_dropdown.config(state='normal')
        
        # Schedule UI update on main thread
        self.app.root.after(0, update_ui)  # Use app.root for scheduling updates
    
    def generate_image(self):
        """Generate an image using the current settings."""
        if not self.image_engine:
            self.show_error("Please load a model first")
            return
        
        # Get current settings
        prompt = self.prompt_var.get().strip()
        if not prompt:
            self.show_error("Please enter a prompt")
            return
        
        negative_prompt = self.negative_prompt_var.get().strip()
        height = self.height_var.get()
        width = self.width_var.get()
        steps = self.steps_var.get()
        guidance = self.guidance_var.get()
        
        # Disable controls during generation
        self.generate_button.config(state='disabled')
        self.image_status_label.config(text="Status: Generating image...")
        
        def generation_thread():
            try:
                # Generate the image
                image = self.image_engine.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance
                )
                
                # Store the generated image
                self.generated_image = image
                
                # Update the UI
                def update_canvas():
                    # Convert PIL image to PhotoImage
                    photo = ImageTk.PhotoImage(image)
                    
                    # Update canvas
                    self.image_canvas.delete("all")
                    self.image_canvas.create_image(
                        self.image_canvas.winfo_width() // 2,
                        self.image_canvas.winfo_height() // 2,
                        image=photo,
                        anchor='center'
                    )
                    self.image_canvas.image = photo  # Keep a reference
                    
                    # Enable save button
                    self.save_image_button.config(state='normal')
                    
                    # Update status
                    self.image_status_label.config(text="Status: Image generated")
                    self.generate_button.config(state='normal')
                
                # Schedule UI update on main thread
                self.parent.after(0, update_canvas)
                
            except Exception as error:
                def error_update():
                    self.show_error(f"Error generating image: {str(error)}")
                    self.image_status_label.config(text="Status: Generation failed")
                    self.generate_button.config(state='normal')
                
                # Schedule error update on main thread
                self.parent.after(0, error_update)
        
        # Start generation in a separate thread
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
    
    def clear_image(self):
        """Clear the generated image and reset the canvas"""
        self.image_canvas.delete("all")
        self.generated_image = None
        self.save_image_button.config(state='disabled')
        self.image_status_label.config(text="Status: Image cleared")
        # Don't clear prompts or parameters to maintain persistence
        # Add placeholder text back
        self.image_canvas.create_text(256, 256, text="Generated image will appear here",
                                    fill='gray', font=('Arial', 14))
    
    def show_error(self, message):
        """Show error message"""
        def show_error_dialog():
            from tkinter import messagebox
            messagebox.showerror("Error", message)
        
        # Schedule error dialog on main thread
        self.app.root.after(0, show_error_dialog)
    
    def update_model_dropdown(self, models):
        """Update the model dropdown with available models"""
        self.image_model_dropdown['values'] = models
        if models:
            self.image_model_dropdown.set(models[0])
            self.image_model_input.delete(0, tk.END)
            self.image_model_input.insert(0, models[0])
    
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
    
    def cleanup(self):
        """Clean up resources"""
        if self.image_engine:
            self.image_engine = None 