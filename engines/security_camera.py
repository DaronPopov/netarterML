#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import (
    ViltProcessor, ViltForQuestionAnswering,
    AutoProcessor, AutoModelForVision2Seq,
    CLIPProcessor, CLIPModel
)
import time
import threading
from queue import Queue
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO  # Add YOLO import
import psutil  # Add psutil for system monitoring

# Add project root to Python path
project_root = str(Path(__file__).absolute().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import LLM API
from engines.llm_api import LLMAPI, APIKeyManager

class MultiVisionAnalyzer:
    """Class to handle multiple vision models in parallel"""
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up logging first
        self.setup_logging()
        
        # Set cache directory for models
        cache_dir = Path("model_cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Initialize YOLO frame skip counter
        self.yolo_frame_skip = 0
        self.yolo_skip_frames = 5  # Run YOLO every 5 frames instead of 3
        
        # Initialize tracking variables
        self.last_boxes = {}  # Store last detected boxes
        self.box_velocities = {}  # Store box velocities for prediction
        self.box_history = {}  # Store box history for smoothing
        self.max_history = 3  # Reduce history length from 5 to 3 frames
        
        # Initialize models with local cache
        self.logger.info("Loading VQA model (ViLT) from cache...")
        try:
            self.vilt_processor = ViltProcessor.from_pretrained(
                "dandelin/vilt-b32-finetuned-vqa",
                token=hf_token,
                cache_dir=cache_dir,
                local_files_only=True
            )
            self.vilt_model = ViltForQuestionAnswering.from_pretrained(
                "dandelin/vilt-b32-finetuned-vqa",
                token=hf_token,
                cache_dir=cache_dir,
                local_files_only=True
            )
            self.vilt_model = self.vilt_model.to(self.device)
            self.vilt_model.eval()
        except Exception as e:
            self.logger.error(f"Failed to load ViLT model from cache: {e}")
            raise
        
        self.logger.info("Loading Scene Understanding model (BLIP) from cache...")
        try:
            self.blip_processor = AutoProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                token=hf_token,
                cache_dir=cache_dir,
                local_files_only=True
            )
            self.blip_model = AutoModelForVision2Seq.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                token=hf_token,
                cache_dir=cache_dir,
                local_files_only=True
            )
            self.blip_model = self.blip_model.to(self.device)
            self.blip_model.eval()
        except Exception as e:
            self.logger.error(f"Failed to load BLIP model from cache: {e}")
            raise
        
        self.logger.info("Loading Object Detection model (CLIP) from cache...")
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                token=hf_token,
                cache_dir=cache_dir,
                local_files_only=True
            )
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                token=hf_token,
                cache_dir=cache_dir,
                local_files_only=True
            )
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model from cache: {e}")
            raise
        
        # Initialize thread pool with fewer workers for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)  # Reduce from 4 to 2 workers
        
        # Frame buffer for recursive processing
        self.frame_buffer = []
        self.max_buffer_size = 1  # Process 1 frame at a time instead of 2
        self.last_analysis = None
        self.analysis_lock = threading.Lock()
        
        # Comprehensive detection classes
        self.detection_classes = {
            'people': [
                'person', 'man', 'woman', 'child', 'baby', 'elderly person',
                'security guard', 'visitor', 'employee', 'crowd', 'group of people'
            ],
            'vehicles': [
                'car', 'truck', 'van', 'bus', 'motorcycle', 'bicycle',
                'emergency vehicle', 'delivery vehicle', 'moving vehicle', 'parked vehicle'
            ],
            'objects': [
                'backpack', 'bag', 'suitcase', 'box', 'package', 'container',
                'tool', 'equipment', 'device', 'electronics', 'furniture'
            ],
            'security': [
                'camera', 'security camera', 'alarm', 'sensor', 'lock',
                'security system', 'access control', 'surveillance equipment'
            ],
            'environment': [
                'door', 'window', 'entrance', 'exit', 'hallway', 'room',
                'building', 'structure', 'light', 'shadow', 'dark area'
            ],
            'activity': [
                'walking', 'running', 'standing', 'sitting', 'carrying',
                'moving', 'entering', 'exiting', 'waiting', 'talking'
            ],
            'suspicious': [
                'masked person', 'hidden face', 'suspicious behavior',
                'unauthorized access', 'tampering', 'forced entry'
            ]
        }
        
        # Flatten detection classes for CLIP
        self.clip_labels = [label for category in self.detection_classes.values() for label in category]
        
        # Initialize YOLO model for pre-annotation
        self.logger.info("Loading YOLO model for pre-annotation...")
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # Use nano model for speed
            self.yolo_model.to(self.device)
            # Set model to half precision for better performance
            if torch.cuda.is_available():
                self.yolo_model.half()
            self.logger.info("YOLO model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def setup_logging(self):
        """Set up logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger('MultiVisionAnalyzer')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"vision_analyzer_{timestamp}.log"
        
        # File handler with rotation (10MB per file, keep 5 backup files)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # Create formatters and add them to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def analyze_with_vilt(self, image, questions):
        """Analyze scene using ViLT VQA model"""
        with torch.no_grad():
            answers = {}
            for question in questions:
                encoding = self.vilt_processor(image, question, return_tensors="pt").to(self.device)
                outputs = self.vilt_model(**encoding)
                logits = outputs.logits
                idx = logits.argmax(-1).item()
                answer = self.vilt_model.config.id2label[idx]
                answers[question] = answer
            return answers
    
    def analyze_with_blip(self, image):
        """Generate detailed scene description using BLIP"""
        with torch.no_grad():
            inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
            generated_ids = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
            return caption
    
    def analyze_with_clip(self, image, candidate_labels):
        """Detect objects using CLIP"""
        with torch.no_grad():
            inputs = self.clip_processor(
                images=image,
                text=candidate_labels,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Get top 5 detected objects with confidence > 0.3
            top_probs, top_indices = torch.topk(probs, 5)
            detected_objects = {
                candidate_labels[idx]: float(prob)
                for prob, idx in zip(top_probs[0], top_indices[0])
                if float(prob) > 0.3
            }
            return detected_objects
    
    def generate_scene_description(self, vilt_results, blip_caption, detected_objects):
        """Generate a coherent scene description from all model outputs"""
        # Start with BLIP's natural description
        description = blip_caption
        
        # Add VQA insights
        vqa_insights = []
        if isinstance(vilt_results, dict):
            for question, answers in vilt_results.items():
                if isinstance(answers, list):
                    # Take most common answer if we have multiple frames
                    most_common = max(set(answers), key=answers.count)
                    if most_common == "yes":
                        vqa_insights.append(question.replace("Is there ", "").replace(" in the image?", ""))
        
        # Add detected objects with confidence
        if detected_objects:
            objects_str = ", ".join([f"{obj} ({conf:.2f})" for obj, conf in detected_objects.items() if conf > 0.3])
            if objects_str:
                description += f" Detected objects: {objects_str}."
        
        # Add VQA insights
        if vqa_insights:
            description += f" Additional observations: {'; '.join(vqa_insights)}."
        
        return description
    
    def process_frame_pair(self, frames):
        """Process frames in parallel with all models"""
        if len(frames) != self.max_buffer_size:
            self.logger.warning(f"Expected {self.max_buffer_size} frames, got {len(frames)}")
            return None
        
        # Convert frames to PIL Images
        images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
        
        # Define questions for ViLT
        vilt_questions = [
            "Is there a person in the image?",
            "Is there any movement in the image?",
            "Is the scene well lit?",
            "Are there any unusual objects in the image?",
            "Is the scene normal or suspicious?"
        ]
        
        # Process all frames with all models in parallel
        futures = []
        for image in images:
            # Submit all model tasks for this frame
            futures.extend([
                self.executor.submit(self.analyze_with_vilt, image, vilt_questions),
                self.executor.submit(self.analyze_with_blip, image),
                self.executor.submit(self.analyze_with_clip, image, self.clip_labels)
            ])
        
        # Get results as they complete
        results = [future.result() for future in futures]
        
        # Combine results from all frames
        combined_vqa = {}
        combined_blip = []
        combined_clip = {}
        
        # Process results in groups of 3 (one group per frame)
        for i in range(0, len(results), 3):
            vqa_result = results[i]
            blip_result = results[i + 1]
            clip_result = results[i + 2]
            
            # Combine VQA results
            for q, a in vqa_result.items():
                if q not in combined_vqa:
                    combined_vqa[q] = []
                combined_vqa[q].append(a)
            
            # Add BLIP result
            combined_blip.append(blip_result)
            
            # Combine CLIP results
            for obj, conf in clip_result.items():
                if obj not in combined_clip:
                    combined_clip[obj] = []
                combined_clip[obj].append(conf)
        
        # Average the confidence scores for CLIP
        final_clip = {obj: sum(confs) / len(confs) for obj, confs in combined_clip.items()}
        
        # Get most common BLIP description
        blip_descriptions = {}
        for desc in combined_blip:
            if desc not in blip_descriptions:
                blip_descriptions[desc] = 0
            blip_descriptions[desc] += 1
        most_common_blip = max(blip_descriptions.items(), key=lambda x: x[1])[0]
        
        # Get most common VQA answers
        vqa_insights = []
        for question, answers in combined_vqa.items():
            most_common = max(set(answers), key=answers.count)
            if most_common == "yes":
                vqa_insights.append(question.replace("Is there ", "").replace(" in the image?", ""))
        
        # Generate final scene description
        scene_description = f"{most_common_blip}"
        
        # Add detected objects with confidence
        detected_objects = {obj: conf for obj, conf in final_clip.items() if conf > 0.3}
        if detected_objects:
            objects_str = ", ".join([f"{obj} ({conf:.2f})" for obj, conf in detected_objects.items()])
            scene_description += f" Detected objects: {objects_str}."
        
        # Add VQA insights
        if vqa_insights:
            scene_description += f" Additional observations: {'; '.join(vqa_insights)}."
        
        return {
            'scene_description': scene_description,
            'raw_analysis': {
                'vqa_results': combined_vqa,
                'blip_caption': most_common_blip,
                'detected_objects': detected_objects,
                'vqa_insights': vqa_insights
            }
        }
    
    def analyze_scene(self, frame):
        """Analyze scene using recursive frame processing"""
        # Add frame to buffer
        self.frame_buffer.append(frame.copy())
        
        # Process frames when buffer is full
        if len(self.frame_buffer) == self.max_buffer_size:
            analysis = self.process_frame_pair(self.frame_buffer)
            self.frame_buffer = []  # Clear buffer after processing
            
            # Update last analysis with thread safety
            with self.analysis_lock:
                self.last_analysis = analysis
            
            return analysis
        
        # Return last analysis if available
        with self.analysis_lock:
            return self.last_analysis

    def pre_annotate_frame(self, frame):
        """Pre-annotate frame with YOLO detections"""
        try:
            # Only run YOLO detection every few frames
            self.yolo_frame_skip = (self.yolo_frame_skip + 1) % self.yolo_skip_frames
            should_detect = self.yolo_frame_skip == 0
            
            # Run YOLO detection if it's time
            if should_detect:
                # Resize frame to smaller size for faster processing
                height, width = frame.shape[:2]
                scale = 0.5  # Reduce to 50% size
                small_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                
                results = self.yolo_model(small_frame, conf=0.3)[0]  # Get first result
                
                # Update box history and velocities
                current_boxes = {}
                for box in results.boxes:
                    # Scale coordinates back to original size
                    x1, y1, x2, y2 = map(int, box.xyxy[0] / scale)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = results.names[cls]
                    
                    # Create unique ID for this detection
                    box_id = f"{label}_{x1}_{y1}"
                    
                    # Store current box
                    current_boxes[box_id] = {
                        'box': (x1, y1, x2, y2),
                        'label': label,
                        'conf': conf,
                        'cls': cls
                    }
                    
                    # Update velocity if we have previous position
                    if box_id in self.last_boxes:
                        prev_box = self.last_boxes[box_id]['box']
                        dx = (x1 - prev_box[0]) / self.yolo_skip_frames
                        dy = (y1 - prev_box[1]) / self.yolo_skip_frames
                        self.box_velocities[box_id] = (dx, dy)
                    
                    # Update history
                    if box_id not in self.box_history:
                        self.box_history[box_id] = []
                    self.box_history[box_id].append((x1, y1, x2, y2))
                    if len(self.box_history[box_id]) > self.max_history:
                        self.box_history[box_id].pop(0)
                
                # Update last boxes
                self.last_boxes = current_boxes
            
            # Draw boxes with interpolation
            annotated_frame = frame.copy()
            for box_id, box_info in self.last_boxes.items():
                # Get base box
                x1, y1, x2, y2 = box_info['box']
                label = box_info['label']
                conf = box_info['conf']
                
                # Apply velocity prediction if available
                if box_id in self.box_velocities:
                    dx, dy = self.box_velocities[box_id]
                    x1 = int(x1 + dx * self.yolo_frame_skip)
                    y1 = int(y1 + dy * self.yolo_frame_skip)
                    x2 = int(x2 + dx * self.yolo_frame_skip)
                    y2 = int(y2 + dy * self.yolo_frame_skip)
                
                # Ensure box stays within frame bounds
                height, width = frame.shape[:2]
                x1 = max(0, min(x1, width-1))
                y1 = max(0, min(y1, height-1))
                x2 = max(0, min(x2, width-1))
                y2 = max(0, min(y2, height-1))
                
                # Draw box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with confidence
                text = f"{label} {conf:.2f}"
                cv2.putText(annotated_frame, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return annotated_frame
        except Exception as e:
            self.logger.error(f"Error in pre-annotation: {e}")
            return frame

class SecurityCamera:
    def __init__(self, hf_token: str):
        """
        Initialize the security camera system with multiple vision models.
        """
        # Initialize running flag first
        self.running = True
        
        # Set up logging
        self.setup_logging()
        
        # Initialize system monitoring
        self.cpu_percent = 0
        self.ram_percent = 0
        self.monitoring_thread = threading.Thread(target=self._monitor_system_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Initialize vision analyzer
        self.logger.info("Initializing multi-vision analyzer...")
        self.vision_analyzer = MultiVisionAnalyzer(hf_token)
        
        # Initialize thread pool executor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize object tracking
        self.trackers = []
        self.tracked_objects = []
        self.tracking_history = {}  # Store tracking history for each object
        self.track_id = 0
        self.tracking_threshold = 0.3  # Minimum confidence to start tracking
        self.max_tracking_age = 30  # Maximum frames to track an object
        
        # Initialize YOLO frame skip counter
        self.yolo_frame_skip = 0
        self.yolo_skip_frames = 5  # Run YOLO every 5 frames instead of 3
        
        # Initialize event summarization
        self.event_history = []
        self.max_event_history = 10
        self.event_threshold = 0.7  # Confidence threshold for events
        self.last_event_time = time.time()
        self.event_interval = 5.0  # Summarize events every 5 seconds
        
        # Initialize variables
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_display = 0
        self.last_analysis = ""
        self.analysis_queue = Queue(maxsize=1)  # Queue size of 1 for single frame processing
        
        # Initialize scene analysis variables
        self.scene_history = []
        self.max_history = 10  # Keep last 10 scene descriptions
        self.alert_threshold = 0.8  # Confidence threshold for alerts
        self.last_analysis_time = time.time()
        self.analysis_interval = 0.1  # Analyze every 100ms
        
        # Initialize annotation logging
        self.setup_annotation_logging()
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self._analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def setup_logging(self):
        """Set up logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger('SecurityCamera')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"security_camera_{timestamp}.log"
        
        # File handler with rotation (10MB per file, keep 5 backup files)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # Create formatters and add them to handlers
        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def setup_annotation_logging(self):
        """Set up annotation logging to file"""
        # Create annotations directory if it doesn't exist
        self.annotations_dir = Path("annotations")
        self.annotations_dir.mkdir(exist_ok=True)
        
        # Clear previous annotation files
        for file in self.annotations_dir.glob("scene_annotations_*.log"):
            try:
                file.unlink()
                print(f"Cleared previous annotation file: {file}")
            except Exception as e:
                print(f"Error clearing file {file}: {e}")
        
        # Create a new log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.annotation_file = self.annotations_dir / f"scene_annotations_{timestamp}.log"
        
        # Write header to the file
        with open(self.annotation_file, 'w') as f:
            f.write("Time,Scene Description,Detected Objects,Alerts\n")
        
        print(f"\nStarting new annotation log: {self.annotation_file}")
        print("Previous annotation files have been cleared.")

    def update_trackers(self, frame):
        """Update object trackers and manage tracking history"""
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Update existing trackers
        new_trackers = []
        new_tracked_objects = []
        
        for tracker, obj_info in zip(self.trackers, self.tracked_objects):
            success, bbox = tracker.update(frame)
            if success:
                # Update tracking history
                obj_id = obj_info['id']
                if obj_id not in self.tracking_history:
                    self.tracking_history[obj_id] = []
                self.tracking_history[obj_id].append(bbox)
                
                # Keep only recent history
                if len(self.tracking_history[obj_id]) > self.max_tracking_age:
                    self.tracking_history[obj_id] = self.tracking_history[obj_id][-self.max_tracking_age:]
                
                # Update object info
                obj_info['bbox'] = bbox
                obj_info['age'] += 1
                
                # Only keep trackers that haven't exceeded max age
                if obj_info['age'] < self.max_tracking_age:
                    new_trackers.append(tracker)
                    new_tracked_objects.append(obj_info)
            else:
                # Remove tracking history for lost objects
                if obj_info['id'] in self.tracking_history:
                    del self.tracking_history[obj_info['id']]
        
        self.trackers = new_trackers
        self.tracked_objects = new_tracked_objects
        
        return self.tracked_objects

    def add_new_trackers(self, frame, detected_objects):
        """Add new trackers for detected objects"""
        for obj, conf in detected_objects.items():
            if conf > self.tracking_threshold:
                # Use CLIP's detection to initialize tracker
                # For now, we'll use a simple bounding box in the center
                # In a real implementation, you'd want to use the actual detection box
                height, width = frame.shape[:2]
                bbox = (width//4, height//4, width//2, height//2)
                
                # Create new tracker using KCF instead of CSRT
                try:
                    tracker = cv2.TrackerKCF_create()
                    success = tracker.init(frame, bbox)
                    
                    if success:
                        self.trackers.append(tracker)
                        self.tracked_objects.append({
                            'id': self.track_id,
                            'object': obj,
                            'confidence': conf,
                            'bbox': bbox,
                            'age': 0
                        })
                        self.track_id += 1
                except Exception as e:
                    self.logger.warning(f"Failed to initialize tracker: {e}")
                    continue

    def process_frame(self, frame):
        """Process a single frame with all models in parallel"""
        # Update object trackers
        tracked_objects = self.update_trackers(frame)
        
        # Convert frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Define specialized questions for each model
        vilt_activity_questions = [
            "What activity is happening in the scene?",
            "Is there any movement?",
            "Is the scene well lit?",
            "Are there multiple people?",
            "Is there any unusual behavior?"
        ]
        
        blip_scene_questions = [
            "Describe the scene in detail",
            "What objects are visible?",
            "What is the person doing?",
            "Describe the environment"
        ]
        
        clip_security_objects = [
            'person', 'multiple people', 'suspicious person', 'security camera',
            'unauthorized access', 'forced entry', 'suspicious package',
            'weapon', 'tool', 'backpack', 'suitcase', 'vehicle',
            'dark area', 'blind spot', 'security breach',
            'headphones', 'laptop', 'phone', 'chair', 'bed',
            'door', 'window', 'entrance', 'exit'
        ]
        
        # Process frame with all models in parallel
        futures = [
            self.executor.submit(self.vision_analyzer.analyze_with_vilt, image, vilt_activity_questions),
            self.executor.submit(self.vision_analyzer.analyze_with_blip, image),
            self.executor.submit(self.vision_analyzer.analyze_with_clip, image, clip_security_objects)
        ]
        
        # Get results
        vqa_result, blip_result, clip_result = [future.result() for future in futures]
        
        # Add new trackers for detected objects
        self.add_new_trackers(frame, clip_result)
        
        # Process VQA results for activity description
        activity_insights = []
        for question, answer in vqa_result.items():
            if answer != "no":
                activity_insights.append(answer)
        
        # Get detected objects with confidence
        detected_objects = {obj: conf for obj, conf in clip_result.items() if conf > 0.3}
        
        # Generate comprehensive scene description
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Combine all insights into a clear description
        scene_description = []
        
        # Add BLIP's scene description
        scene_description.append(blip_result)
        
        # Add activity insights from ViLT
        if activity_insights:
            scene_description.append(f"Activity: {'; '.join(activity_insights)}")
        
        # Add tracked objects information
        if tracked_objects:
            tracked_str = ", ".join([f"{obj['object']} (ID: {obj['id']}, Age: {obj['age']})" 
                                   for obj in tracked_objects])
            scene_description.append(f"Tracked: {tracked_str}")
        
        # Add detected objects from CLIP
        if detected_objects:
            objects_str = ", ".join([f"{obj} ({conf:.2f})" for obj, conf in detected_objects.items()])
            scene_description.append(f"Detected: {objects_str}")
        
        # Combine all parts into final description
        final_description = " | ".join(scene_description)
        
        # Check for security concerns
        security_alerts = []
        if any("suspicious" in obj.lower() for obj in detected_objects.keys()):
            security_alerts.append("SUSPICIOUS ACTIVITY DETECTED")
        if any("unauthorized" in obj.lower() for obj in detected_objects.keys()):
            security_alerts.append("UNAUTHORIZED ACCESS ATTEMPT")
        if any("weapon" in obj.lower() for obj in detected_objects.keys()):
            security_alerts.append("POTENTIAL WEAPON DETECTED")
        
        # Add alerts for tracked objects that have been present for too long
        for obj in tracked_objects:
            if obj['age'] > self.max_tracking_age // 2:
                security_alerts.append(f"PROLONGED PRESENCE: {obj['object'].upper()}")
        
        output = {
            'timestamp': timestamp,
            'scene_description': final_description,
            'security_status': {
                'alerts': security_alerts,
                'activity': activity_insights,
                'detected_objects': detected_objects,
                'tracked_objects': tracked_objects
            },
            'raw_analysis': {
                'vqa_results': vqa_result,
                'blip_caption': blip_result,
                'detected_objects': detected_objects
            }
        }
        
        return output

    def summarize_events(self, scene_info):
        """Summarize recent events and suggest actions"""
        current_time = time.time()
        
        # Add current scene to event history
        self.event_history.append({
            'timestamp': scene_info['timestamp'],
            'description': scene_info['scene_description'],
            'alerts': scene_info['security_status']['alerts'],
            'activity': scene_info['security_status']['activity'],
            'tracked_objects': scene_info['security_status']['tracked_objects']
        })
        
        # Keep only recent history
        if len(self.event_history) > self.max_event_history:
            self.event_history = self.event_history[-self.max_event_history:]
        
        # Generate summary if enough time has passed
        if current_time - self.last_event_time >= self.event_interval:
            self.last_event_time = current_time
            
            # Analyze recent events
            summary = self._analyze_events()
            return summary
        
        return None

    def _analyze_events(self):
        """Analyze recent events and generate summary with actions"""
        if not self.event_history:
            return None
        
        # Collect all alerts and activities
        all_alerts = []
        all_activities = []
        tracked_objects = {}
        
        for event in self.event_history:
            all_alerts.extend(event['alerts'])
            all_activities.extend(event['activity'])
            
            # Track object persistence
            for obj in event['tracked_objects']:
                obj_id = obj['id']
                if obj_id not in tracked_objects:
                    tracked_objects[obj_id] = {
                        'object': obj['object'],
                        'first_seen': event['timestamp'],
                        'last_seen': event['timestamp'],
                        'age': obj['age']
                    }
                else:
                    tracked_objects[obj_id]['last_seen'] = event['timestamp']
                    tracked_objects[obj_id]['age'] = max(tracked_objects[obj_id]['age'], obj['age'])
        
        # Generate summary
        summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'time_period': f"{self.event_history[0]['timestamp']} to {self.event_history[-1]['timestamp']}",
            'summary': [],
            'actions': []
        }
        
        # Add persistent objects to summary
        persistent_objects = [obj for obj in tracked_objects.values() if obj['age'] > self.max_tracking_age // 2]
        if persistent_objects:
            summary['summary'].append("Persistent Objects:")
            for obj in persistent_objects:
                summary['summary'].append(f"• {obj['object']} present for {obj['age']} frames")
                summary['actions'].append(f"Verify if {obj['object']} should be present")
        
        # Add alerts to summary
        if all_alerts:
            summary['summary'].append("\nSecurity Alerts:")
            for alert in set(all_alerts):  # Remove duplicates
                summary['summary'].append(f"• {alert}")
                if "SUSPICIOUS" in alert:
                    summary['actions'].append("Investigate suspicious activity")
                elif "UNAUTHORIZED" in alert:
                    summary['actions'].append("Check access logs and verify authorization")
                elif "WEAPON" in alert:
                    summary['actions'].append("Immediate security response required")
        
        # Add activities to summary
        if all_activities:
            summary['summary'].append("\nRecent Activities:")
            for activity in set(all_activities):  # Remove duplicates
                summary['summary'].append(f"• {activity}")
                if "sleeping" in activity.lower():
                    summary['actions'].append("Verify if sleeping is allowed in this area")
                elif "unauthorized" in activity.lower():
                    summary['actions'].append("Review access permissions")
        
        return summary

    def log_annotation(self, scene_info):
        """Log scene annotation to file and print to console in security monitoring style"""
        if not scene_info:
            return
        
        timestamp = scene_info['timestamp']
        scene_description = scene_info['scene_description']
        security_status = scene_info['security_status']
        
        # Get event summary
        event_summary = self.summarize_events(scene_info)
        
        # Format detected objects with confidence
        objects_str = "; ".join([f"{obj} ({conf:.2f})" for obj, conf in security_status['detected_objects'].items()])
        
        # Write to file
        with open(self.annotation_file, 'a') as f:
            f.write(f"{timestamp},{scene_description},{objects_str},{'; '.join(security_status['alerts']) if security_status['alerts'] else 'None'}\n")
        
        # Print to console with security monitoring style
        print(f"\n[{timestamp}]")
        print("=" * 50)
        print("SECURITY MONITORING SYSTEM")
        print("=" * 50)
        
        # Print alerts if any
        if security_status['alerts']:
            print("\n⚠️  ACTIVE ALERTS:")
            for alert in security_status['alerts']:
                print(f"• {alert}")
        
        # Print scene analysis
        print("\n📹 SCENE ANALYSIS:")
        print(f"• {scene_description}")
        
        # Print activity insights
        if security_status['activity']:
            print(f"\n🎯 ACTIVITY DETECTED:")
            for activity in security_status['activity']:
                print(f"• {activity}")
        
        # Print tracked objects
        if security_status['tracked_objects']:
            print(f"\n🎯 TRACKED OBJECTS:")
            for obj in security_status['tracked_objects']:
                print(f"• {obj['object']} (ID: {obj['id']}, Age: {obj['age']} frames)")
        
        # Print event summary if available
        if event_summary:
            print("\n📊 EVENT SUMMARY:")
            print(f"Time Period: {event_summary['time_period']}")
            for line in event_summary['summary']:
                print(line)
            
            if event_summary['actions']:
                print("\n🛡️  RECOMMENDED ACTIONS:")
                for action in event_summary['actions']:
                    print(f"• {action}")
        
        print("=" * 50)

    def _analysis_loop(self):
        """Background thread for continuous scene analysis"""
        while self.running:
            current_time = time.time()
            if current_time - self.last_analysis_time >= self.analysis_interval:
                if not self.analysis_queue.empty():
                    frame = self.analysis_queue.get()
                    try:
                        scene_info = self.process_frame(frame)
                        if scene_info:
                            self.last_analysis = scene_info['scene_description']
                            
                            # Log the annotation
                            self.log_annotation(scene_info)
                            
                            # Update analysis time
                            self.last_analysis_time = current_time
                    except Exception as e:
                        self.logger.error(f"Error analyzing scene: {e}")
                    finally:
                        self.analysis_queue.task_done()
            time.sleep(0.01)  # Small sleep to prevent CPU overload
    
    def _monitor_system_resources(self):
        """Background thread to monitor system resources"""
        while self.running:
            try:
                # Get CPU usage
                self.cpu_percent = psutil.cpu_percent()
                
                # Get RAM usage
                self.ram_percent = psutil.virtual_memory().percent
                
                time.sleep(1)  # Update every second
            except Exception as e:
                self.logger.error(f"Error monitoring system resources: {e}")
                time.sleep(1)

    def run(self, device_id: int = 0):
        """Run the security camera system"""
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            self.logger.error(f"Error: Could not open camera {device_id}")
            return
        
        # Set camera properties for 720p at 20 FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 20)
        
        self.logger.info("Starting security camera system...")
        print("\nStarting security camera system...")
        print("Press 'q' to quit")
        print("Press 's' to save current scene analysis")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.error("Error: Could not read frame")
                break
            
            # Pre-annotate frame with YOLO detections
            annotated_frame = self.vision_analyzer.pre_annotate_frame(frame)
            
            # Update frame count and FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 1.0:
                self.fps_display = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()
            
            # Add FPS and status to frame
            cv2.putText(annotated_frame, f"FPS: {self.fps_display:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add timestamp to frame
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(annotated_frame, timestamp, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add system resource info to top right corner
            height, width = annotated_frame.shape[:2]
            resource_text = [
                f"CPU: {self.cpu_percent:.1f}%",
                f"RAM: {self.ram_percent:.1f}%"
            ]
            
            # Draw semi-transparent background for resource info
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (width - 200, 10), (width - 10, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            
            # Add resource text
            for i, text in enumerate(resource_text):
                y_pos = 35 + i * 25
                cv2.putText(annotated_frame, text, (width - 190, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add latest analysis to frame
            if self.last_analysis:
                analysis_lines = str(self.last_analysis).split('\n')
                for i, line in enumerate(analysis_lines[:3]):  # Show first 3 lines
                    cv2.putText(annotated_frame, line, (10, frame.shape[0] - 100 + i * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Security Camera', annotated_frame)
            
            # Add frame to analysis queue if not full
            if not self.analysis_queue.full():
                self.analysis_queue.put(frame.copy())
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if self.last_analysis:
                    # Save latest scene analysis to file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"scene_analysis_{timestamp}.json"
                    with open(filename, 'w') as f:
                        json.dump(self.last_analysis, f, indent=2)
                    self.logger.info(f"Scene analysis saved to {filename}")
                    print(f"\nScene analysis saved to {filename}")
        
        # Cleanup
        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Security camera system stopped")

    def __del__(self):
        """Cleanup when the object is destroyed"""
        self.running = False
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

def main():
    # ===== CONFIGURE YOUR MODELS HERE =====
    HF_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')
    if not HF_TOKEN:
        print("Error: HUGGINGFACE_TOKEN environment variable not set")
        print("Please set your HuggingFace token using:")
        print("export HUGGINGFACE_TOKEN='your_token_here'")
        sys.exit(1)
    
    # Initialize and run security camera with Mac's built-in webcam (device ID 0)
    camera = SecurityCamera(HF_TOKEN)
    camera.run(device_id=0)  # Use Mac's built-in webcam

if __name__ == "__main__":
    main() 