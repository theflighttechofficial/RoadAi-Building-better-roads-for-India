"""
Active Learning Pipeline
Automatically identifies hard examples for manual review
Improves model over time with minimal labeling effort
"""

import numpy as np
import cv2
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging

log = logging.getLogger(__name__)


class ActiveLearningQueue:
    """
    Intelligent review queue for uncertain detections
    """
    
    def __init__(
        self, 
        queue_dir: str = "active_learning_queue",
        uncertainty_threshold: float = 0.4,
        max_queue_size: int = 1000
    ):
        """
        Args:
            queue_dir: Directory to store review queue
            uncertainty_threshold: Threshold for flagging uncertain detections
            max_queue_size: Maximum items in queue
        """
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(exist_ok=True, parents=True)
        
        self.uncertainty_threshold = uncertainty_threshold
        self.max_queue_size = max_queue_size
        
        self.review_queue = []
        self.queue_file = self.queue_dir / "review_queue.json"
        
        # Load existing queue
        self._load_queue()
        
        log.info(f"Active learning queue initialized: {len(self.review_queue)} items")
    
    def add_to_queue_if_uncertain(
        self,
        frame: np.ndarray,
        detection_result: Dict,
        frame_metadata: Dict = None
    ):
        """
        Add frame to review queue if it meets criteria
        
        Args:
            frame: Input image
            detection_result: Detection results with uncertainty
            frame_metadata: Additional metadata (GPS, timestamp, etc.)
        """
        if frame_metadata is None:
            frame_metadata = {}
        
        # Check criteria for review
        criteria = {
            'high_uncertainty': self._check_high_uncertainty(detection_result),
            'edge_case': self._is_edge_case(detection_result),
            'conflicting_predictions': self._has_conflicts(detection_result),
            'low_confidence': self._check_low_confidence(detection_result),
        }
        
        # If any criterion is met, add to queue
        if any(criteria.values()):
            priority = self._calculate_priority(criteria, detection_result)
            
            # Generate unique ID
            item_id = f"review_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Save frame image
            frame_path = self.queue_dir / f"{item_id}.jpg"
            cv2.imwrite(str(frame_path), frame)
            
            # Create queue item
            queue_item = {
                'id': item_id,
                'frame_path': str(frame_path),
                'detection': self._serialize_detection(detection_result),
                'metadata': frame_metadata,
                'criteria': criteria,
                'priority': priority,
                'timestamp': datetime.now().isoformat(),
                'status': 'pending'  # pending, reviewed, labeled
            }
            
            self.review_queue.append(queue_item)
            
            # Trim queue if too large
            if len(self.review_queue) > self.max_queue_size:
                self._trim_queue()
            
            self._save_queue()
            
            log.info(f"Added to review queue: {item_id} (priority: {priority:.2f})")
    
    def _check_high_uncertainty(self, detection_result: Dict) -> bool:
        """Check if uncertainty exceeds threshold"""
        if 'uncertainty' in detection_result:
            max_uncertainty = detection_result['uncertainty'].max() if len(detection_result['uncertainty']) > 0 else 0
            return max_uncertainty > self.uncertainty_threshold
        return False
    
    def _is_edge_case(self, detection_result: Dict) -> bool:
        """Detect edge cases (unusual damage patterns)"""
        # Check for unusual aspect ratios, sizes, etc.
        if len(detection_result.get('boxes', [])) == 0:
            return False
        
        boxes = detection_result['boxes']
        
        # Check for extremely large or small boxes
        areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in boxes]
        if areas:
            return max(areas) > 200000 or min(areas) < 100  # Pixel area thresholds
        
        return False
    
    def _has_conflicts(self, detection_result: Dict) -> bool:
        """Check for conflicting predictions"""
        # If ensemble was used, check for disagreement
        if 'raw_detections' in detection_result and 'boxes' in detection_result:
            raw_count = detection_result['raw_detections']
            fused_count = len(detection_result['boxes'])
            
            # High disagreement between models
            return abs(raw_count - fused_count) > 3
        
        return False
    
    def _check_low_confidence(self, detection_result: Dict) -> bool:
        """Check for low confidence detections"""
        if 'scores' in detection_result and len(detection_result['scores']) > 0:
            return detection_result['scores'].min() < 0.4
        return False
    
    def _calculate_priority(self, criteria: Dict, detection_result: Dict) -> float:
        """
        Calculate review priority (0-1, higher = more urgent)
        """
        priority = 0.0
        
        # Weight each criterion
        weights = {
            'high_uncertainty': 0.4,
            'edge_case': 0.3,
            'conflicting_predictions': 0.2,
            'low_confidence': 0.1
        }
        
        for criterion, is_met in criteria.items():
            if is_met:
                priority += weights.get(criterion, 0.1)
        
        # Boost priority for multiple detections
        num_detections = len(detection_result.get('boxes', []))
        if num_detections > 3:
            priority += 0.1
        
        return min(priority, 1.0)
    
    def export_for_labeling(
        self, 
        output_format: str = 'roboflow',
        top_n: int = 100
    ) -> str:
        """
        Export highest priority items for labeling
        
        Args:
            output_format: 'roboflow', 'cvat', or 'labelme'
            top_n: Number of items to export
            
        Returns:
            Path to exported dataset
        """
        # Sort by priority
        sorted_queue = sorted(
            [item for item in self.review_queue if item['status'] == 'pending'],
            key=lambda x: x['priority'],
            reverse=True
        )[:top_n]
        
        export_dir = self.queue_dir / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        export_dir.mkdir(exist_ok=True)
        
        images_dir = export_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Export images
        for i, item in enumerate(sorted_queue):
            # Copy image
            src = Path(item['frame_path'])
            dst = images_dir / f"{i:04d}_{item['id']}.jpg"
            
            if src.exists():
                cv2.imwrite(str(dst), cv2.imread(str(src)))
        
        # Create metadata file
        metadata = {
            'export_date': datetime.now().isoformat(),
            'num_images': len(sorted_queue),
            'format': output_format,
            'items': sorted_queue
        }
        
        with open(export_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log.info(f"Exported {len(sorted_queue)} items to {export_dir}")
        
        return str(export_dir)
    
    def mark_as_reviewed(self, item_id: str):
        """Mark an item as reviewed"""
        for item in self.review_queue:
            if item['id'] == item_id:
                item['status'] = 'reviewed'
                self._save_queue()
                break
    
    def _serialize_detection(self, detection_result: Dict) -> Dict:
        """Convert numpy arrays to JSON-serializable format"""
        serialized = {}
        for key, value in detection_result.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            else:
                serialized[key] = value
        return serialized
    
    def _trim_queue(self):
        """Remove lowest priority items"""
        self.review_queue = sorted(
            self.review_queue,
            key=lambda x: x['priority'],
            reverse=True
        )[:self.max_queue_size]
    
    def _save_queue(self):
        """Save queue to disk"""
        with open(self.queue_file, 'w') as f:
            json.dump(self.review_queue, f, indent=2)
    
    def _load_queue(self):
        """Load queue from disk"""
        if self.queue_file.exists():
            with open(self.queue_file, 'r') as f:
                self.review_queue = json.load(f)
            log.info(f"Loaded {len(self.review_queue)} items from queue")
        else:
            self.review_queue = []
    
    def get_stats(self) -> Dict:
        """Get queue statistics"""
        pending = sum(1 for item in self.review_queue if item['status'] == 'pending')
        reviewed = sum(1 for item in self.review_queue if item['status'] == 'reviewed')
        
        return {
            'total': len(self.review_queue),
            'pending': pending,
            'reviewed': reviewed,
            'avg_priority': np.mean([item['priority'] for item in self.review_queue]) if self.review_queue else 0
        }