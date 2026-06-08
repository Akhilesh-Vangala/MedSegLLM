import numpy as np
import torch
from typing import Dict, List, Tuple
from sklearn.metrics import jaccard_score
import logging

logger = logging.getLogger(__name__)


class SegmentationEvaluator:
    def __init__(self):
        self.metrics_history = []
    
    def calculate_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        intersection = np.logical_and(pred_mask > 0.5, gt_mask > 0.5).sum()
        union = np.logical_or(pred_mask > 0.5, gt_mask > 0.5).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        iou = intersection / union
        return float(iou)
    
    def calculate_dice(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        intersection = np.logical_and(pred_mask > 0.5, gt_mask > 0.5).sum()
        pred_sum = (pred_mask > 0.5).sum()
        gt_sum = (gt_mask > 0.5).sum()
        
        if pred_sum + gt_sum == 0:
            return 1.0
        
        dice = 2 * intersection / (pred_sum + gt_sum)
        return float(dice)
    
    def evaluate_batch(self, pred_masks: List[np.ndarray],
                      gt_masks: List[np.ndarray]) -> Dict:
        ious = []
        dices = []
        
        for pred, gt in zip(pred_masks, gt_masks):
            iou = self.calculate_iou(pred, gt)
            dice = self.calculate_dice(pred, gt)
            ious.append(iou)
            dices.append(dice)
        
        metrics = {
            'mean_iou': float(np.mean(ious)),
            'std_iou': float(np.std(ious)),
            'mean_dice': float(np.mean(dices)),
            'std_dice': float(np.std(dices)),
            'min_iou': float(np.min(ious)),
            'max_iou': float(np.max(ious))
        }
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def calculate_hausdorff_distance(self, pred_mask: np.ndarray,
                                    gt_mask: np.ndarray) -> float:
        from scipy.spatial.distance import directed_hausdorff
        
        pred_coords = np.argwhere(pred_mask > 0.5)
        gt_coords = np.argwhere(gt_mask > 0.5)
        
        if len(pred_coords) == 0 or len(gt_coords) == 0:
            return float('inf')
        
        h1 = directed_hausdorff(pred_coords, gt_coords)[0]
        h2 = directed_hausdorff(gt_coords, pred_coords)[0]
        
        return max(h1, h2)
