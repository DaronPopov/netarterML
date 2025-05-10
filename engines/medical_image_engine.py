#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import pydicom
import nibabel as nib
import SimpleITK as sitk
from transformers import AutoModelForImageClassification, AutoProcessor

class MedicalImageEngine:
    """
    Engine for processing medical images using vision transformer models.
    Supports multiple medical image formats and provides analysis capabilities.
    """
    
    def __init__(self, model_name: str, hf_token: str = None):
        """
        Initialize the medical image processing engine.
        
        Args:
            model_name: Name of the model to use
            hf_token: HuggingFace token for model access
        """
        self.model_name = model_name
        self.hf_token = hf_token
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the model and processor"""
        try:
            # Load model and processor
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def _load_dicom(self, file_path: str) -> Image.Image:
        """Load a DICOM file and convert to PIL Image"""
        try:
            ds = pydicom.dcmread(file_path)
            image = ds.pixel_array
            
            # Normalize to 0-255
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
            return Image.fromarray(image)
        except Exception as e:
            print(f"Error loading DICOM file: {str(e)}")
            return None
    
    def _load_nifti(self, file_path: str) -> Image.Image:
        """Load a NIfTI file and convert to PIL Image"""
        try:
            nii = nib.load(file_path)
            data = nii.get_fdata()
            
            # Get middle slice
            middle_slice = data[:, :, data.shape[2]//2]
            
            # Normalize to 0-255
            middle_slice = ((middle_slice - middle_slice.min()) / (middle_slice.max() - middle_slice.min()) * 255).astype(np.uint8)
            
            return Image.fromarray(middle_slice)
        except Exception as e:
            print(f"Error loading NIfTI file: {str(e)}")
            return None
    
    def _load_analyze(self, file_path: str) -> Image.Image:
        """Load an Analyze file and convert to PIL Image"""
        try:
            # Load using SimpleITK
            image = sitk.ReadImage(file_path)
            array = sitk.GetArrayFromImage(image)
            
            # Get middle slice
            middle_slice = array[array.shape[0]//2]
            
            # Normalize to 0-255
            middle_slice = ((middle_slice - middle_slice.min()) / (middle_slice.max() - middle_slice.min()) * 255).astype(np.uint8)
            
            return Image.fromarray(middle_slice)
        except Exception as e:
            print(f"Error loading Analyze file: {str(e)}")
            return None
    
    def _load_mgh(self, file_path: str) -> Image.Image:
        """Load an MGH file and convert to PIL Image"""
        try:
            # Load using SimpleITK
            image = sitk.ReadImage(file_path)
            array = sitk.GetArrayFromImage(image)
            
            # Get middle slice
            middle_slice = array[array.shape[0]//2]
            
            # Normalize to 0-255
            middle_slice = ((middle_slice - middle_slice.min()) / (middle_slice.max() - middle_slice.min()) * 255).astype(np.uint8)
            
            return Image.fromarray(middle_slice)
        except Exception as e:
            print(f"Error loading MGH file: {str(e)}")
            return None
    
    def _load_minc(self, file_path: str) -> Image.Image:
        """Load a MINC file and convert to PIL Image"""
        try:
            # Load using SimpleITK
            image = sitk.ReadImage(file_path)
            array = sitk.GetArrayFromImage(image)
            
            # Get middle slice
            middle_slice = array[array.shape[0]//2]
            
            # Normalize to 0-255
            middle_slice = ((middle_slice - middle_slice.min()) / (middle_slice.max() - middle_slice.min()) * 255).astype(np.uint8)
            
            return Image.fromarray(middle_slice)
        except Exception as e:
            print(f"Error loading MINC file: {str(e)}")
            return None
    
    def _load_pfs(self, file_path: str) -> Image.Image:
        """Load a PFS file and convert to PIL Image"""
        try:
            # Load using SimpleITK
            image = sitk.ReadImage(file_path)
            array = sitk.GetArrayFromImage(image)
            
            # Get middle slice
            middle_slice = array[array.shape[0]//2]
            
            # Normalize to 0-255
            middle_slice = ((middle_slice - middle_slice.min()) / (middle_slice.max() - middle_slice.min()) * 255).astype(np.uint8)
            
            return Image.fromarray(middle_slice)
        except Exception as e:
            print(f"Error loading PFS file: {str(e)}")
            return None
    
    def load_image(self, file_path: str) -> Image.Image:
        """
        Load a medical image file and convert to PIL Image.
        Supports multiple medical image formats.
        
        Args:
            file_path: Path to the medical image file
            
        Returns:
            PIL Image object or None if loading fails
        """
        # Get file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        # Load based on file type
        if ext == '.dcm':
            return self._load_dicom(file_path)
        elif ext in ['.nii', '.gz']:
            return self._load_nifti(file_path)
        elif ext in ['.hdr', '.img']:
            return self._load_analyze(file_path)
        elif ext == '.mgh':
            return self._load_mgh(file_path)
        elif ext == '.mnc':
            return self._load_minc(file_path)
        elif ext == '.pfs':
            return self._load_pfs(file_path)
        else:
            print(f"Unsupported file format: {ext}")
            return None
    
    def analyze_image(self, file_path: str) -> tuple:
        """
        Analyze a medical image and return results.
        
        Args:
            file_path: Path to the medical image file
            
        Returns:
            Tuple of (PIL Image, analysis results string)
        """
        # Load image
        image = self.load_image(file_path)
        if image is None:
            return None, "Error: Could not load image"
        
        try:
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
            # Get top predictions
            top_probs, top_indices = torch.topk(probs[0], 5)
            
            # Format results
            results = "Analysis Results:\n"
            for prob, idx in zip(top_probs, top_indices):
                label = self.model.config.id2label[idx.item()]
                results += f"{label}: {prob.item():.2%}\n"
            
            return image, results
            
        except Exception as e:
            return image, f"Error during analysis: {str(e)}" 