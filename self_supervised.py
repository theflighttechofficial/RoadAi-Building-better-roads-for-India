"""
Self-Supervised Learning Pipeline
Generate training data from high-confidence detections
Expand dataset without manual labeling
"""

import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict
import json
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)


class SelfSupervisedTrainer:
    """
    Auto-generate pseudo-labels from unlabeled videos
    """
    
    def __init__(
        self,
        teacher_model_path: str,
        confidence_threshold: float = 0.85,
        output_dir: str = "pseudo_labeled_data"
    ):
        """
        Args:
            teacher_model_path: Path to your best trained model
            confidence_threshold: Only use very confident predictions (0.8-0.9)
            output_dir: Where to save pseudo-labeled dataset
        """
        self.teacher_model = YOLO(teacher_model_path)
        self.threshold = confidence_threshold
        self.output_dir = Path(output_dir)
        
        # Create dataset structure
        self.train_dir = self.output_dir / "train"
        self.train_images = self.train_dir / "images"
        self.train_labels = self.train_dir / "labels"
        
        for dir in [self.train_images, self.train_labels]:
            dir.mkdir(exist_ok=True, parents=True)
        
        self.pseudo_dataset = []
        
        log.info(f"Self-supervised trainer initialized (conf >= {confidence_threshold})")
    
    def generate_pseudo_labels_from_video(
        self,
        video_path: str,
        frame_skip: int = 30,
        max_frames: int = 1000
    ) -> int:
        """
        Extract frames and generate pseudo-labels
        
        Args:
            video_path: Path to unlabeled dashcam video
            frame_skip: Process every Nth frame
            max_frames: Maximum frames to process
            
        Returns:
            Number of frames added to dataset
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_added = 0
        frame_idx = 0
        
        pbar = tqdm(total=min(total_frames, max_frames), desc="Generating pseudo-labels")
        
        while cap.isOpened() and frames_added < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            # Run teacher model
            results = self.teacher_model.predict(
                frame,
                conf=self.threshold,
                verbose=False
            )
            
            # Only keep if high confidence detections exist
            if len(results[0].boxes) > 0:
                high_conf_boxes = results[0].boxes[
                    results[0].boxes.conf >= self.threshold
                ]
                
                if len(high_conf_boxes) > 0:
                    # Save image and label
                    img_name = f"pseudo_{Path(video_path).stem}_frame_{frame_idx:06d}"
                    self._save_pseudo_label(frame, high_conf_boxes, img_name)
                    frames_added += 1
            
            frame_idx += 1
            pbar.update(frame_skip)
        
        cap.release()
        pbar.close()
        
        log.info(f"Generated {frames_added} pseudo-labeled frames from {video_path}")
        
        return frames_added
    
    def _save_pseudo_label(self, frame: np.ndarray, boxes, img_name: str):
        """Save image and YOLO format label"""
        # Save image
        img_path = self.train_images / f"{img_name}.jpg"
        cv2.imwrite(str(img_path), frame)
        
        # Save label in YOLO format
        label_path = self.train_labels / f"{img_name}.txt"
        
        h, w = frame.shape[:2]
        
        with open(label_path, 'w') as f:
            for box in boxes:
                cls = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Convert to YOLO format (class, x_center, y_center, width, height)
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                box_width = (x2 - x1) / w
                box_height = (y2 - y1) / h
                conf = box.conf.item()
                
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
        
        # Track in dataset
        self.pseudo_dataset.append({
            'image': str(img_path),
            'label': str(label_path),
            'source': 'pseudo_label',
            'confidence': float(boxes.conf.mean().item())
        })
    
    def create_dataset_yaml(self, class_names: List[str]) -> str:
        """
        Create data.yaml for training
        
        Args:
            class_names: List of class names
            
        Returns:
            Path to data.yaml
        """
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'train/images',  # Use same for validation initially
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        
        with open(yaml_path, 'w') as f:
            import yaml
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        log.info(f"Created data.yaml: {yaml_path}")
        
        return str(yaml_path)
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            'total_images': len(self.pseudo_dataset),
            'avg_confidence': np.mean([item['confidence'] for item in self.pseudo_dataset]) if self.pseudo_dataset else 0,
            'output_dir': str(self.output_dir)
        }