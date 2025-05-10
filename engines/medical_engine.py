#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification
import pydicom
import nibabel as nib
import SimpleITK as sitk
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalImageEngine:
    """Engine for analyzing medical images using various AI models"""
    
    def __init__(self, model_name: str, hf_token: str = None):
        """
        Initialize the medical image analysis engine
        
        Args:
            model_name: Name of the model to use
            hf_token: Optional HuggingFace token for authentication
        """
        self.model_name = model_name
        self.hf_token = hf_token
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model configuration
        self.model_config = {
            'input_size': 224,  # Default input size
            'normalize': True,  # Default normalization
            'mean': [0.485, 0.456, 0.406],  # Default ImageNet mean
            'std': [0.229, 0.224, 0.225],   # Default ImageNet std
            'num_classes': None,  # Will be set after model load
            'class_labels': None,  # Will be set after model load
            'model_type': None,   # Will be set after model load
            'confidence_threshold': 0.50,    # Minimum confidence for detection
            'high_confidence_threshold': 0.75,  # High confidence for positive diagnosis
            'moderate_confidence_threshold': 0.60,  # Moderate confidence for possible diagnosis
            'num_predictions': 5
        }
        
        logger.info(f"Initializing MedicalImageEngine with model: {model_name}")
        logger.info(f"Using device: {self.device}")

    def _detect_model_type(self):
        """Detect the type of model and its configuration"""
        try:
            # Check model architecture
            if hasattr(self.model, 'config'):
                config = self.model.config
                
                # Detect model type
                if hasattr(config, 'model_type'):
                    self.model_config['model_type'] = config.model_type
                
                # Get number of classes
                if hasattr(config, 'num_labels'):
                    self.model_config['num_classes'] = config.num_labels
                
                # Get class labels
                if hasattr(config, 'id2label'):
                    self.model_config['class_labels'] = list(config.id2label.values())
                
                # Get input size if available
                if hasattr(config, 'image_size'):
                    self.model_config['input_size'] = config.image_size
                
                logger.info(f"Detected model type: {self.model_config['model_type']}")
                logger.info(f"Number of classes: {self.model_config['num_classes']}")
                logger.info(f"Class labels: {self.model_config['class_labels']}")
                
                return True
            return False
        except Exception as e:
            logger.error(f"Error detecting model type: {str(e)}")
            return False

    def _adapt_processor(self):
        """Adapt the processor based on model configuration"""
        try:
            if self.processor is None:
                return False
                
            # Set processor parameters based on model config
            if hasattr(self.processor, 'size'):
                # Handle different size configurations
                if isinstance(self.processor.size, dict):
                    # If size is a dict with shortest_edge, convert to height/width
                    if 'shortest_edge' in self.processor.size:
                        size = self.model_config['input_size']
                        self.processor.size = {'height': size, 'width': size}
                    # If size is a dict with height/width, use those values
                    elif 'height' in self.processor.size and 'width' in self.processor.size:
                        self.processor.size = {
                            'height': self.model_config['input_size'],
                            'width': self.model_config['input_size']
                        }
                else:
                    # If size is a single value, use it for both dimensions
                    self.processor.size = self.model_config['input_size']
            
            if hasattr(self.processor, 'do_normalize'):
                self.processor.do_normalize = self.model_config['normalize']
            
            if hasattr(self.processor, 'image_mean'):
                self.processor.image_mean = self.model_config['mean']
            
            if hasattr(self.processor, 'image_std'):
                self.processor.image_std = self.model_config['std']
            
            return True
        except Exception as e:
            logger.error(f"Error adapting processor: {str(e)}")
            return False

    def load_model(self):
        """Load the model and processor"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # First try loading the processor
            logger.info("Loading image processor...")
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )
            logger.info("Image processor loaded successfully")
            
            # Then load the model
            logger.info("Loading model...")
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                ignore_mismatched_sizes=True
            )
            
            # Move model to appropriate device
            logger.info(f"Moving model to {self.device}")
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Detect model type and configuration
            if not self._detect_model_type():
                logger.warning("Could not detect model type, using default configuration")
            
            # Adapt processor to model configuration
            if not self._adapt_processor():
                logger.warning("Could not adapt processor, using default configuration")
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Clean up any partially loaded components
            if self.processor:
                del self.processor
                self.processor = None
            if self.model:
                del self.model
                self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
    
    def _load_image(self, file_path: str) -> Image.Image:
        """Load an image file in various formats"""
        ext = os.path.splitext(file_path)[1].lower()
        logger.info(f"Loading image: {file_path}")
        
        try:
            if ext in ['.dcm']:
                # Load DICOM file
                logger.info("Loading DICOM file")
                try:
                    ds = pydicom.dcmread(file_path)
                    # Get pixel data and normalize
                    pixel_array = ds.pixel_array
                    # Normalize to 0-255 range
                    pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
                    # Convert to PIL Image
                    image = Image.fromarray(pixel_array)
                    logger.info("DICOM file loaded successfully")
                except Exception as dicom_error:
                    logger.error(f"Error loading DICOM file: {str(dicom_error)}")
                    raise RuntimeError(f"Failed to load DICOM file: {str(dicom_error)}")
            elif ext in ['.nii', '.gz']:
                # Load NIfTI file
                logger.info("Loading NIfTI file")
                nii_img = nib.load(file_path)
                data = nii_img.get_fdata()
                # Get middle slice if 3D
                if len(data.shape) > 2:
                    data = data[:, :, data.shape[2]//2]
                # Normalize to 0-255
                data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
                image = Image.fromarray(data)
            elif ext in ['.hdr', '.img']:
                # Load Analyze file
                logger.info("Loading Analyze file")
                analyze_img = nib.load(file_path)
                data = analyze_img.get_fdata()
                # Get middle slice if 3D
                if len(data.shape) > 2:
                    data = data[:, :, data.shape[2]//2]
                # Normalize to 0-255
                data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
                image = Image.fromarray(data)
            elif ext in ['.mgh']:
                # Load MGH file
                logger.info("Loading MGH file")
                mgh_img = nib.load(file_path)
                data = mgh_img.get_fdata()
                # Get middle slice if 3D
                if len(data.shape) > 2:
                    data = data[:, :, data.shape[2]//2]
                # Normalize to 0-255
                data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
                image = Image.fromarray(data)
            elif ext in ['.mnc']:
                # Load MINC file
                logger.info("Loading MINC file")
                minc_img = sitk.ReadImage(file_path)
                data = sitk.GetArrayFromImage(minc_img)
                # Get middle slice if 3D
                if len(data.shape) > 2:
                    data = data[data.shape[0]//2]
                # Normalize to 0-255
                data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
                image = Image.fromarray(data)
            elif ext in ['.pfs']:
                # Load PFS file
                logger.info("Loading PFS file")
                pfs_img = sitk.ReadImage(file_path)
                data = sitk.GetArrayFromImage(pfs_img)
                # Get middle slice if 3D
                if len(data.shape) > 2:
                    data = data[data.shape[0]//2]
                # Normalize to 0-255
                data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
                image = Image.fromarray(data)
            else:
                # Load standard image formats
                logger.info("Loading standard image file")
                image = Image.open(file_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                logger.info("Converting image to RGB")
                image = image.convert('RGB')
            
            # Resize image to model's expected size
            logger.info("Resizing image to model's expected size")
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            logger.info("Image loaded successfully")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to load image: {str(e)}")
    
    def analyze_image(self, image_input) -> tuple:
        """
        Analyze a medical image
        
        Args:
            image_input: Either a file path (str) or a PIL Image object
            
        Returns:
            Tuple of (PIL Image, analysis results)
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            logger.info("Analyzing image")
            
            # Handle both file paths and PIL Image objects
            if isinstance(image_input, str):
                # Load image from file path
                image = self._load_image(image_input)
                if image is None:
                    return None, "Error: Could not load image"
            elif isinstance(image_input, Image.Image):
                # Use provided PIL Image
                image = image_input
            else:
                raise ValueError("Input must be either a file path (str) or PIL Image object")
            
            # Process image for model
            logger.info("Processing image for model")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            logger.info("Getting model predictions")
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Validate number of predictions
            num_classes = probs.shape[-1]
            if num_classes == 0:
                return image, "Error: Model returned no predictions"
            
            # Adjust number of predictions if needed
            num_predictions = min(self.model_config['num_predictions'], num_classes)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probs[0], num_predictions)
            
            # Format results
            results = []
            
            # Add model information
            results.append(f"Model: {self.model_name}")
            results.append(f"Type: {self.model_config['model_type']}")
            
            # Add class information if available
            if self.model_config['class_labels']:
                results.append(f"Classes: {', '.join(self.model_config['class_labels'])}")
            else:
                results.append(f"Number of classes: {num_classes}")
            
            results.append("\nDetailed Predictions:")
            
            # Add predictions
            max_prob = 0.0
            max_label = None
            for prob, idx in zip(top_probs, top_indices):
                try:
                    # Try to get label from model config
                    if hasattr(self.model.config, 'id2label'):
                        label = self.model.config.id2label[idx.item()]
                    else:
                        # Fallback to index if no label mapping
                        label = f"Class {idx.item()}"
                    
                    confidence = prob.item()
                    
                    # Track highest probability prediction
                    if confidence > max_prob:
                        max_prob = confidence
                        max_label = label
                    
                    if confidence >= self.model_config['confidence_threshold']:
                        results.append(f"{label}: {confidence:.2%}")
                except Exception as e:
                    logger.warning(f"Error processing prediction {idx.item()}: {str(e)}")
                    continue
            
            # Add binary diagnosis
            results.append("\nDiagnosis:")
            if max_prob >= self.model_config['confidence_threshold']:
                # Use custom confidence thresholds for diagnosis
                if max_prob >= self.model_config['high_confidence_threshold']:
                    results.append(f"Positive for {max_label} ({max_prob:.2%} confidence)")
                elif max_prob >= self.model_config['moderate_confidence_threshold']:
                    results.append(f"Possible {max_label} ({max_prob:.2%} confidence)")
                else:
                    results.append(f"Uncertain - {max_label} detected but confidence too low ({max_prob:.2%})")
            else:
                results.append("No significant findings detected")
            
            # Add clinical note
            results.append("\nClinical Note:")
            results.append("This is an AI-assisted analysis. Please consult with a healthcare professional for definitive diagnosis.")
            
            logger.info("Analysis complete")
            return image, "\n".join(results)
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return None, f"Error analyzing image: {str(e)}"
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources")
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleanup complete") 