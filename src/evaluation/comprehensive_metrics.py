import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix
import logging
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff

logger = logging.getLogger(__name__)


class ComprehensiveSegmentationEvaluator:
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
    
    def calculate_hausdorff_distance(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        pred_coords = np.argwhere(pred_mask > 0.5)
        gt_coords = np.argwhere(gt_mask > 0.5)
        
        if len(pred_coords) == 0 or len(gt_coords) == 0:
            return float('inf')
        
        h1 = directed_hausdorff(pred_coords, gt_coords)[0]
        h2 = directed_hausdorff(gt_coords, pred_coords)[0]
        
        return max(h1, h2)
    
    def calculate_surface_distance(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict:
        pred_surface = self._get_surface(pred_mask)
        gt_surface = self._get_surface(gt_mask)
        
        if len(pred_surface) == 0 or len(gt_surface) == 0:
            return {
                'mean_surface_distance': float('inf'),
                'rms_surface_distance': float('inf'),
                'max_surface_distance': float('inf')
            }
        
        distances = []
        for p_point in pred_surface:
            min_dist = np.min([np.linalg.norm(p_point - g_point) for g_point in gt_surface])
            distances.append(min_dist)
        
        for g_point in gt_surface:
            min_dist = np.min([np.linalg.norm(g_point - p_point) for p_point in pred_surface])
            distances.append(min_dist)
        
        distances = np.array(distances)
        
        return {
            'mean_surface_distance': float(np.mean(distances)),
            'rms_surface_distance': float(np.sqrt(np.mean(distances ** 2))),
            'max_surface_distance': float(np.max(distances))
        }
    
    def _get_surface(self, mask: np.ndarray) -> np.ndarray:
        from scipy.ndimage import binary_erosion
        
        eroded = binary_erosion(mask > 0.5)
        surface = (mask > 0.5) & ~eroded
        
        coords = np.argwhere(surface)
        return coords
    
    def calculate_volume_similarity(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        pred_volume = (pred_mask > 0.5).sum()
        gt_volume = (gt_mask > 0.5).sum()
        
        if gt_volume == 0:
            return 1.0 if pred_volume == 0 else 0.0
        
        vs = 1 - abs(pred_volume - gt_volume) / gt_volume
        return float(vs)
    
    def evaluate_comprehensive(self, pred_masks: List[np.ndarray],
                              gt_masks: List[np.ndarray]) -> Dict:
        ious = []
        dices = []
        hausdorffs = []
        volume_similarities = []
        surface_distances = []
        
        for pred, gt in zip(pred_masks, gt_masks):
            iou = self.calculate_iou(pred, gt)
            dice = self.calculate_dice(pred, gt)
            hausdorff = self.calculate_hausdorff_distance(pred, gt)
            vs = self.calculate_volume_similarity(pred, gt)
            sd = self.calculate_surface_distance(pred, gt)
            
            ious.append(iou)
            dices.append(dice)
            hausdorffs.append(hausdorff if hausdorff != float('inf') else 0)
            volume_similarities.append(vs)
            surface_distances.append(sd)
        
        metrics = {
            'mean_iou': float(np.mean(ious)),
            'std_iou': float(np.std(ious)),
            'mean_dice': float(np.mean(dices)),
            'std_dice': float(np.std(dices)),
            'mean_hausdorff': float(np.mean(hausdorffs)),
            'mean_volume_similarity': float(np.mean(volume_similarities)),
            'mean_surface_distance': float(np.mean([sd['mean_surface_distance'] for sd in surface_distances if sd['mean_surface_distance'] != float('inf')])),
            'min_iou': float(np.min(ious)),
            'max_iou': float(np.max(ious))
        }
        
        self.metrics_history.append(metrics)
        
        return metrics
