
#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Optional, Dict, Any
from .security_camera import SecurityCamera

class SecurityCameraAPI:
    """Simple API for setting up and managing security cameras"""
    
    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize the security camera API
        
        Args:
            hf_token: Optional HuggingFace token. If not provided, will look for HUGGINGFACE_TOKEN env var
        """
        self.hf_token = hf_token or os.environ.get('HUGGINGFACE_TOKEN')
        if not self.hf_token:
            raise ValueError("HuggingFace token is required. Set HUGGINGFACE_TOKEN env var or pass token to constructor")
        
        self.cameras = {}
    
    def add_camera(self, 
                  camera_id: int = 0,
                  name: str = "default",
                  resolution: tuple = (1280, 720),
                  fps: int = 20,
                  yolo_skip_frames: int = 5,
                  detection_threshold: float = 0.3) -> str:
        """
        Add a new camera to the system
        
        Args:
            camera_id: Camera device ID (default: 0 for built-in webcam)
            name: Unique name for this camera
            resolution: Camera resolution (width, height)
            fps: Frames per second
            yolo_skip_frames: How often to run YOLO detection
            detection_threshold: Confidence threshold for detections
            
        Returns:
            str: Camera name
        """
        if name in self.cameras:
            raise ValueError(f"Camera with name '{name}' already exists")
        
        # Create camera instance
        camera = SecurityCamera(self.hf_token)
        
        # Configure camera settings
        camera.vision_analyzer.yolo_skip_frames = yolo_skip_frames
        camera.vision_analyzer.detection_threshold = detection_threshold
        
        # Store camera settings
        self.cameras[name] = {
            'instance': camera,
            'device_id': camera_id,
            'resolution': resolution,
            'fps': fps,
            'running': False
        }
        
        return name
    
    def start_camera(self, name: str) -> None:
        """Start a specific camera"""
        if name not in self.cameras:
            raise ValueError(f"Camera '{name}' not found")
        
        camera = self.cameras[name]
        if camera['running']:
            raise ValueError(f"Camera '{name}' is already running")
        
        # Configure camera
        camera['instance'].run(camera['device_id'])
        camera['running'] = True
    
    def stop_camera(self, name: str) -> None:
        """Stop a specific camera"""
        if name not in self.cameras:
            raise ValueError(f"Camera '{name}' not found")
        
        camera = self.cameras[name]
        if not camera['running']:
            raise ValueError(f"Camera '{name}' is not running")
        
        camera['instance'].running = False
        camera['running'] = False
    
    def get_camera_status(self, name: str) -> Dict[str, Any]:
        """Get status of a specific camera"""
        if name not in self.cameras:
            raise ValueError(f"Camera '{name}' not found")
        
        camera = self.cameras[name]
        return {
            'name': name,
            'running': camera['running'],
            'resolution': camera['resolution'],
            'fps': camera['fps'],
            'cpu_usage': camera['instance'].cpu_percent,
            'ram_usage': camera['instance'].ram_percent
        }
    
    def list_cameras(self) -> list:
        """List all configured cameras"""
        return list(self.cameras.keys())

# Example usage:
if __name__ == "__main__":
    # Create API instance
    api = SecurityCameraAPI()
    
    # Add a camera
    camera_name = api.add_camera(
        camera_id=0,  # Built-in webcam
        name="main_camera",
        resolution=(1280, 720),
        fps=20
    )
    
    # Start the camera
    api.start_camera(camera_name)
    
    # Get camera status
    status = api.get_camera_status(camera_name)
    print(f"Camera status: {status}")
    
    # List all cameras
    cameras = api.list_cameras()
    print(f"Configured cameras: {cameras}")
