"""
Uncertainty Estimation using Monte Carlo Dropout
Identifies unreliable detections for manual review
"""

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Dict, List
import logging

log = logging.getLogger(__name__)


class UncertaintyEstimator:
    """
    Bayesian uncertainty estimation for YOLO detections
    """
    
    def __init__(self, model_path: str, num_samples: int = 10):
        """
        Args:
            model_path: Path to trained YOLO model
            num_samples: Number of MC dropout samples (10-30 recommended)
        """
        self.model = YOLO(model_path)
        self.num_samples = num_samples
        self.dropout_rate = 0.2
        
        # Enable dropout at inference
        self._enable_dropout()
        
        log.info(f"Uncertainty estimator initialized with {num_samples} MC samples")
    
    def _enable_dropout(self):
        """
        Enable dropout layers during inference for Monte Carlo sampling
        """
        for module in self.model.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep in training mode to enable dropout
                log.debug(f"Enabled dropout: {module}")
    
    def predict_with_uncertainty(
        self, 
        image: np.ndarray,
        conf_threshold: float = 0.25
    ) -> Dict:
        """
        Run multiple forward passes and estimate uncertainty
        
        Args:
            image: Input image
            conf_threshold: Confidence threshold
            
        Returns:
            Dict with predictions and uncertainty metrics
        """
        all_predictions = []
        
        # Run multiple forward passes with dropout
        for i in range(self.num_samples):
            results = self.model.predict(
                image, 
                conf=conf_threshold,
                verbose=False
            )
            
            if len(results[0].boxes) > 0:
                all_predictions.append({
                    'boxes': results[0].boxes.xyxy.cpu().numpy(),
                    'scores': results[0].boxes.conf.cpu().numpy(),
                    'labels': results[0].boxes.cls.cpu().numpy()
                })
        
        # Calculate statistics
        uncertainty_results = self._compute_uncertainty(all_predictions)
        
        return uncertainty_results
    
    def _compute_uncertainty(self, predictions: List[Dict]) -> Dict:
        """
        Compute mean, variance, and epistemic uncertainty
        
        Args:
            predictions: List of predictions from MC sampling
            
        Returns:
            Aggregated predictions with uncertainty metrics
        """
        if not predictions:
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': np.array([]),
                'uncertainty': np.array([]),
                'needs_review': np.array([])
            }
        
        # Stack all predictions
        all_boxes = np.concatenate([p['boxes'] for p in predictions if len(p['boxes']) > 0])
        all_scores = np.concatenate([p['scores'] for p in predictions if len(p['scores']) > 0])
        all_labels = np.concatenate([p['labels'] for p in predictions if len(p['labels']) > 0])
        
        # Cluster predictions using IoU
        clustered = self._cluster_predictions(all_boxes, all_scores, all_labels)
        
        return clustered
    
    def _cluster_predictions(
        self, 
        boxes: np.ndarray, 
        scores: np.ndarray, 
        labels: np.ndarray,
        iou_threshold: float = 0.5
    ) -> Dict:
        """
        Cluster overlapping predictions and compute uncertainty
        """
        from scipy.cluster.hierarchy import fclusterdata
        
        if len(boxes) == 0:
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': np.array([]),
                'uncertainty': np.array([]),
                'needs_review': np.array([])
            }
        
        # Cluster by IoU
        clusters = fclusterdata(boxes, t=1-iou_threshold, criterion='distance', metric=self._box_distance)
        
        unique_clusters = np.unique(clusters)
        
        final_boxes = []
        final_scores = []
        final_labels = []
        uncertainties = []
        needs_review = []
        
        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            cluster_boxes = boxes[mask]
            cluster_scores = scores[mask]
            cluster_labels = labels[mask]
            
            # Mean box
            mean_box = cluster_boxes.mean(axis=0)
            
            # Mean score
            mean_score = cluster_scores.mean()
            
            # Variance (epistemic uncertainty)
            score_variance = cluster_scores.var()
            box_variance = cluster_boxes.var(axis=0).mean()
            
            # Combined uncertainty
            uncertainty = score_variance + box_variance * 0.1
            
            # Most common label
            label = int(np.bincount(cluster_labels.astype(int)).argmax())
            
            # Flag if uncertainty is high
            review_flag = uncertainty > 0.3 or len(cluster_boxes) < self.num_samples * 0.3
            
            final_boxes.append(mean_box)
            final_scores.append(mean_score)
            final_labels.append(label)
            uncertainties.append(uncertainty)
            needs_review.append(review_flag)
        
        return {
            'boxes': np.array(final_boxes),
            'scores': np.array(final_scores),
            'labels': np.array(final_labels),
            'uncertainty': np.array(uncertainties),
            'needs_review': np.array(needs_review),
            'num_samples': self.num_samples
        }
    
    def _box_distance(self, box1, box2):
        """Compute distance between boxes for clustering"""
        iou = self._compute_iou(box1, box2)
        return 1 - iou
    
    @staticmethod
    def _compute_iou(box1, box2):
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0