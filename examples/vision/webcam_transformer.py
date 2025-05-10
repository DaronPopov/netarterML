#!/usr/bin/env python3
"""
Real-time webcam transformer example using the Vision API
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the Vision API
from vision_api import VisionAPI

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Real-time webcam transformer")
    parser.add_argument("--model", type=str, default="microsoft/resnet-50",
                      help="Model to use for inference")
    parser.add_argument("--camera", type=int, default=0,
                      help="Camera device ID")
    parser.add_argument("--fps", type=int, default=30,
                      help="Maximum frames per second")
    parser.add_argument("--save", action="store_true",
                      help="Save processed frames")
    parser.add_argument("--output-dir", type=str, default="processed_frames",
                      help="Directory to save processed frames")
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if saving frames
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Vision API
    api = VisionAPI(model_name=args.model)
    
    try:
        # Load model
        print(f"\nLoading model: {args.model}")
        api.load_model()
        
        # Open webcam
        print(f"\nOpening camera {args.camera}")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {args.camera}")
        
        # Set FPS
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        
        # Frame counter for saving
        frame_count = 0
        
        print("\nStarting real-time processing...")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame
            results = api.process_frame(frame)
            
            # Draw predictions on frame
            for i, pred in enumerate(results["predictions"]):
                text = f"{pred['label']}: {pred['probability']:.2f}"
                cv2.putText(frame, text, (10, 30 + i*30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Real-time Vision Transformer", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save frame
                output_path = os.path.join(args.output_dir, f"frame_{frame_count:04d}.jpg")
                api.save_frame(frame, output_path)
                frame_count += 1
        
    except KeyboardInterrupt:
        print("\nProcessing stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 