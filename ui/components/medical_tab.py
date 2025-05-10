import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading
import os

class MedicalTab:
    def __init__(self, parent, on_model_load):
        self.parent = parent
        self.on_model_load = on_model_load
        self.medical_engine = None
        self.current_image = None
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="Medical Image Analysis")
        self.frame.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Create main content frame
        self.content_frame = ttk.Frame(self.frame)
        self.content_frame.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Left panel for model and config
        self.left_panel = ttk.Frame(self.content_frame)
        self.left_panel.pack(side='left', fill='y', padx=5, pady=5)
        
        # Model selection frame
        self.setup_model_frame()
        
        # Configuration frame
        self.setup_config_frame()
        
        # Right panel for image and results
        self.right_panel = ttk.Frame(self.content_frame)
        self.right_panel.pack(side='right', expand=True, fill='both', padx=5, pady=5)
        
        # Image selection and display
        self.setup_image_frame()
        
        # Analysis results frame
        self.setup_results_frame()
        
        # Status label
        self.medical_status_label = ttk.Label(self.frame, text="Status: No model loaded")
        self.medical_status_label.pack(fill='x', padx=5, pady=5)
    
    def setup_model_frame(self):
        """Setup model selection frame"""
        model_frame = ttk.LabelFrame(self.left_panel, text="Model Selection")
        model_frame.pack(fill='x', padx=5, pady=5)
        
        # Model input
        ttk.Label(model_frame, text="Model:").pack(side='left', padx=5)
        self.medical_model_input = ttk.Entry(model_frame)
        self.medical_model_input.pack(side='left', expand=True, fill='x', padx=5)
        
        # Load model button
        self.load_medical_model_button = ttk.Button(
            model_frame,
            text="Load Model",
            command=self.load_specified_medical_model
        )
        self.load_medical_model_button.pack(side='right', padx=5)
    
    def setup_config_frame(self):
        """Setup configuration frame for confidence thresholds"""
        config_frame = ttk.LabelFrame(self.left_panel, text="Diagnosis Configuration")
        config_frame.pack(fill='x', padx=5, pady=5)
        
        # Create a frame for the confidence thresholds
        thresholds_frame = ttk.Frame(config_frame)
        thresholds_frame.pack(fill='x', padx=5, pady=5)
        
        # High confidence threshold
        high_frame = ttk.Frame(thresholds_frame)
        high_frame.pack(fill='x', pady=2)
        ttk.Label(high_frame, text="High Confidence:").pack(side='left', padx=5)
        self.high_conf_var = tk.StringVar(value="0.75")
        self.high_conf_entry = ttk.Entry(high_frame, textvariable=self.high_conf_var, width=6)
        self.high_conf_entry.pack(side='left', padx=5)
        ttk.Label(high_frame, text="(Positive Diagnosis)").pack(side='left')
        
        # Moderate confidence threshold
        mod_frame = ttk.Frame(thresholds_frame)
        mod_frame.pack(fill='x', pady=2)
        ttk.Label(mod_frame, text="Moderate Confidence:").pack(side='left', padx=5)
        self.mod_conf_var = tk.StringVar(value="0.60")
        self.mod_conf_entry = ttk.Entry(mod_frame, textvariable=self.mod_conf_var, width=6)
        self.mod_conf_entry.pack(side='left', padx=5)
        ttk.Label(mod_frame, text="(Possible Diagnosis)").pack(side='left')
        
        # Minimum confidence threshold
        min_frame = ttk.Frame(thresholds_frame)
        min_frame.pack(fill='x', pady=2)
        ttk.Label(min_frame, text="Minimum Confidence:").pack(side='left', padx=5)
        self.min_conf_var = tk.StringVar(value="0.50")
        self.min_conf_entry = ttk.Entry(min_frame, textvariable=self.min_conf_var, width=6)
        self.min_conf_entry.pack(side='left', padx=5)
        ttk.Label(min_frame, text="(Detection)").pack(side='left')
        
        # Add validation for the entries
        for entry in [self.high_conf_entry, self.mod_conf_entry, self.min_conf_entry]:
            entry.bind('<FocusOut>', self.validate_confidence_values)
    
    def validate_confidence_values(self, event=None):
        """Validate and adjust confidence threshold values"""
        try:
            high = float(self.high_conf_var.get())
            mod = float(self.mod_conf_var.get())
            min_conf = float(self.min_conf_var.get())
            
            # Ensure values are between 0 and 1
            high = max(0.0, min(1.0, high))
            mod = max(0.0, min(1.0, mod))
            min_conf = max(0.0, min(1.0, min_conf))
            
            # Ensure proper ordering
            if high < mod:
                high = mod
            if mod < min_conf:
                mod = min_conf
            
            # Update values
            self.high_conf_var.set(f"{high:.2f}")
            self.mod_conf_var.set(f"{mod:.2f}")
            self.min_conf_var.set(f"{min_conf:.2f}")
            
        except ValueError:
            # Reset to defaults if invalid input
            self.high_conf_var.set("0.75")
            self.mod_conf_var.set("0.60")
            self.min_conf_var.set("0.50")
    
    def setup_image_frame(self):
        """Setup image selection and display frame"""
        image_frame = ttk.LabelFrame(self.right_panel, text="Image")
        image_frame.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Image selection button
        self.select_image_button = ttk.Button(
            image_frame,
            text="Select Image",
            command=self.select_image
        )
        self.select_image_button.pack(side='top', padx=5, pady=5)
        
        # Image path label
        self.image_path_label = ttk.Label(image_frame, text="No image selected")
        self.image_path_label.pack(side='top', padx=5)
        
        # Image display canvas
        self.medical_canvas = tk.Canvas(image_frame, bg='white')
        self.medical_canvas.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Image label
        self.medical_image_label = ttk.Label(self.medical_canvas)
        self.medical_image_label.place(relx=0.5, rely=0.5, anchor='center')
    
    def setup_results_frame(self):
        """Setup analysis results frame"""
        results_frame = ttk.LabelFrame(self.right_panel, text="Analysis Results")
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Analysis text
        self.analysis_text = tk.Text(results_frame, height=10, wrap=tk.WORD)
        self.analysis_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Analyze button
        self.analyze_button = ttk.Button(
            results_frame,
            text="Analyze Image",
            command=self.analyze_image,
            state='disabled'
        )
        self.analyze_button.pack(side='bottom', padx=5, pady=5)
    
    def select_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select Medical Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.dcm *.nii *.gz *.hdr *.img *.mgh *.mnc *.pfs"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Disable button during loading
                self.select_image_button.config(state='disabled')
                
                # Load and display image using medical engine if available
                if self.medical_engine:
                    self.current_image = self.medical_engine._load_image(file_path)
                else:
                    # Fallback to PIL for standard image formats
                    self.current_image = Image.open(file_path)
                
                if self.current_image:
                    self.display_image(self.current_image)
                    # Update path label
                    self.image_path_label.config(text=os.path.basename(file_path))
                    # Enable analyze button if model is loaded
                    if self.medical_engine:
                        self.analyze_button.config(state='normal')
                else:
                    self.show_error("Failed to load image")
                
            except Exception as e:
                self.show_error(f"Error loading image: {str(e)}")
            finally:
                # Always re-enable the select button
                self.select_image_button.config(state='normal')
    
    def display_image(self, image):
        """Display the image in the canvas"""
        # Resize image to fit the canvas while maintaining aspect ratio
        canvas_width = self.medical_canvas.winfo_width()
        canvas_height = self.medical_canvas.winfo_height()
        
        # If canvas size is not yet available, use default size
        if canvas_width <= 1:
            canvas_width = 512
        if canvas_height <= 1:
            canvas_height = 512
        
        # Calculate resize dimensions
        img_width, img_height = image.size
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Resize image
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        
        # Update image label
        self.medical_image_label.config(image=photo)
        self.medical_image_label.image = photo  # Keep a reference
    
    def load_specified_medical_model(self):
        """Load the medical model specified in the input field."""
        model_name = self.medical_model_input.get().strip()
        if not model_name:
            self.show_error("Please enter a model name.")
            return
        
        # Update status
        self.medical_status_label.config(text=f"Status: Loading {model_name}...")
        self.load_medical_model_button.config(state='disabled')
        self.medical_model_input.config(state='disabled')
        
        # Call the parent's model loading function
        self.on_model_load(model_name, self.on_model_loaded)
    
    def on_model_loaded(self, success, error=None):
        """Callback for when model loading is complete"""
        if success:
            self.medical_status_label.config(text=f"Status: Model loaded successfully")
            if self.current_image:
                self.analyze_button.config(state='normal')
        else:
            self.medical_status_label.config(text="Status: Error loading model")
            self.show_error(f"Error loading medical model: {error}")
        
        self.load_medical_model_button.config(state='normal')
        self.medical_model_input.config(state='normal')
    
    def analyze_image(self):
        """Analyze the current image"""
        if not self.medical_engine:
            self.show_error("Please load a model first")
            return
        
        if not self.current_image:
            self.show_error("Please select an image first")
            return
        
        # Get confidence thresholds
        try:
            self.validate_confidence_values()
            high_conf = float(self.high_conf_var.get())
            mod_conf = float(self.mod_conf_var.get())
            min_conf = float(self.min_conf_var.get())
            
            # Update model configuration
            self.medical_engine.model_config.update({
                'high_confidence_threshold': high_conf,
                'moderate_confidence_threshold': mod_conf,
                'confidence_threshold': min_conf
            })
        except ValueError as e:
            self.show_error(f"Invalid confidence threshold values: {str(e)}")
            return
        
        # Disable buttons and clear previous analysis
        self.analyze_button.config(state='disabled')
        self.select_image_button.config(state='disabled')
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "Analyzing image...")
        self.parent.update_idletasks()
        
        def analysis_thread():
            try:
                # Analyze image
                image, analysis = self.medical_engine.analyze_image(self.current_image)
                
                # Update UI with results
                def update_ui():
                    self.analysis_text.delete(1.0, tk.END)
                    self.analysis_text.insert(tk.END, analysis)
                    self.analyze_button.config(state='normal')
                    self.select_image_button.config(state='normal')
                self.parent.after(0, update_ui)
                
            except Exception as e:
                error_msg = str(e)
                def error_update():
                    self.show_error(f"Error analyzing image: {error_msg}")
                    self.analysis_text.delete(1.0, tk.END)
                    self.analysis_text.insert(tk.END, "Error analyzing image")
                    self.analyze_button.config(state='normal')
                    self.select_image_button.config(state='normal')
                self.parent.after(0, error_update)
        
        # Start analysis thread
        thread = threading.Thread(target=analysis_thread, daemon=True)
        thread.start()
    
    def show_error(self, message):
        """Show error message"""
        from tkinter import messagebox
        messagebox.showerror("Error", message)
    
    def cleanup(self):
        """Clean up resources"""
        if self.medical_engine:
            self.medical_engine.cleanup()
            self.medical_engine = None 