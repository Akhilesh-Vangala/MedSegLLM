import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from scipy import ndimage
from scipy.ndimage import label, find_objects
import logging

logger = logging.getLogger(__name__)


class SegmentationPostProcessor:
    def __init__(self, config: Dict):
        self.config = config
    
    def apply_morphology(self, mask: np.ndarray, operation: str = 'closing', kernel_size: int = 5) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if operation == 'opening':
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif operation == 'dilation':
            mask = cv2.dilate(mask, kernel, iterations=1)
        elif operation == 'erosion':
            mask = cv2.erode(mask, kernel, iterations=1)
        
        return mask
    
    def remove_small_objects(self, mask: np.ndarray, min_size: int = 100) -> np.ndarray:
        labeled_mask, num_features = label(mask)
        
        for i in range(1, num_features + 1):
            component_size = np.sum(labeled_mask == i)
            if component_size < min_size:
                mask[labeled_mask == i] = 0
        
        return mask
    
    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        return mask
    
    def smooth_boundaries(self, mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        smoothed = ndimage.gaussian_filter(mask.astype(float), sigma=sigma)
        return (smoothed > 0.5).astype(np.uint8)
    
    def refine_mask(self, mask: np.ndarray, image: Optional[np.ndarray] = None) -> np.ndarray:
        refined = mask.copy()
        
        refined = self.remove_small_objects(refined, min_size=self.config.get('min_object_size', 100))
        refined = self.apply_morphology(refined, operation='closing', kernel_size=5)
        refined = self.fill_holes(refined)
        refined = self.smooth_boundaries(refined, sigma=1.0)
        
        if image is not None:
            refined = self._refine_with_image(refined, image)
        
        return refined
    
    def _refine_with_image(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        edges = cv2.Canny(image.astype(np.uint8), 50, 150)
        
        mask_edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        
        combined_edges = cv2.bitwise_or(edges, mask_edges)
        
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        refined_mask = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                cv2.fillPoly(refined_mask, [contour], 1)
        
        return refined_mask
    
    def extract_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours
    
    def calculate_region_properties(self, mask: np.ndarray) -> List[Dict]:
        labeled_mask, num_features = label(mask)
        properties = []
        
        for i in range(1, num_features + 1):
            region_mask = (labeled_mask == i).astype(np.uint8)
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = contours[0]
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                x, y, w, h = cv2.boundingRect(contour)
                
                properties.append({
                    'region_id': i,
                    'area': area,
                    'perimeter': perimeter,
                    'bbox': (x, y, w, h),
                    'centroid': (x + w // 2, y + h // 2),
                    'contour': contour
                })
        
        return properties
