import numpy as np
import cv2
from typing import Tuple, Optional, Dict
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
import logging

logger = logging.getLogger(__name__)


class MedicalImagePreprocessor:
    def __init__(self, config: Dict):
        self.config = config
    
    def normalize_intensity(self, image: np.ndarray, method: str = 'zscore') -> np.ndarray:
        if method == 'zscore':
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                image = (image - mean) / std
        elif method == 'minmax':
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
        elif method == 'percentile':
            p2, p98 = np.percentile(image, [2, 98])
            image = np.clip(image, p2, p98)
            image = (image - p2) / (p98 - p2 + 1e-8)
        
        return image
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image.astype(np.uint8))
        return enhanced.astype(np.float32) / 255.0
    
    def denoise(self, image: np.ndarray, method: str = 'gaussian', sigma: float = 1.0) -> np.ndarray:
        if method == 'gaussian':
            return gaussian_filter(image, sigma=sigma)
        elif method == 'median':
            return median_filter(image, size=3)
        elif method == 'bilateral':
            if len(image.shape) == 2:
                image = (image * 255).astype(np.uint8)
                denoised = cv2.bilateralFilter(image, 9, 75, 75)
                return denoised.astype(np.float32) / 255.0
            else:
                return image
        return image
    
    def enhance_contrast(self, image: np.ndarray, alpha: float = 1.5, beta: float = 0) -> np.ndarray:
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced.astype(np.float32) / 255.0
    
    def apply_morphology(self, image: np.ndarray, operation: str = 'opening', kernel_size: int = 3) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        image = (image * 255).astype(np.uint8)
        
        if operation == 'opening':
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'gradient':
            result = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        else:
            result = image
        
        return result.astype(np.float32) / 255.0
    
    def extract_features(self, image: np.ndarray) -> Dict:
        features = {}
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        features['mean_intensity'] = np.mean(image)
        features['std_intensity'] = np.std(image)
        features['min_intensity'] = np.min(image)
        features['max_intensity'] = np.max(image)
        features['median_intensity'] = np.median(image)
        
        hist, _ = np.histogram(image, bins=256)
        features['histogram_entropy'] = -np.sum(hist * np.log(hist + 1e-8))
        
        edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        return features
    
    def preprocess_pipeline(self, image: np.ndarray, steps: List[str]) -> np.ndarray:
        processed = image.copy()
        
        for step in steps:
            if step == 'normalize':
                processed = self.normalize_intensity(processed)
            elif step == 'clahe':
                processed = self.apply_clahe(processed)
            elif step == 'denoise':
                processed = self.denoise(processed)
            elif step == 'contrast':
                processed = self.enhance_contrast(processed)
        
        return processed
