import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class MaskRCNNSegmenter:
    def __init__(self,
                 num_classes: int = 2,
                 pretrained: bool = True,
                 model_path: Optional[str] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.num_classes = num_classes
        
        if model_path:
            self.model = self._load_custom_model(model_path)
        else:
            self.model = self._create_model(pretrained)
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Mask R-CNN model loaded on {device}")
    
    def _create_model(self, pretrained: bool) -> nn.Module:
        model = maskrcnn_resnet50_fpn(pretrained=pretrained)
        
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, self.num_classes)
        
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            self.num_classes
        )
        
        return model
    
    def _load_custom_model(self, model_path: str) -> nn.Module:
        model = self._create_model(pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded custom model from {model_path}")
        return model
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        image = cv2.imread(image_path)
        if image is None:
            image = np.array(Image.open(image_path))
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image_tensor.to(self.device)
    
    def segment(self,
               image_path: str,
               confidence_threshold: float = 0.5) -> Tuple[List[np.ndarray], List[float], List[Dict]]:
        image_tensor = self.preprocess_image(image_path)
        
        with torch.no_grad():
            predictions = self.model([image_tensor])
        
        masks = []
        scores = []
        boxes = []
        
        pred = predictions[0]
        for i in range(len(pred['scores'])):
            if pred['scores'][i] >= confidence_threshold:
                mask = pred['masks'][i, 0].cpu().numpy()
                masks.append(mask)
                scores.append(pred['scores'][i].item())
                boxes.append({
                    'box': pred['boxes'][i].cpu().numpy().tolist(),
                    'label': pred['labels'][i].item()
                })
        
        logger.info(f"Detected {len(masks)} segments with confidence >= {confidence_threshold}")
        return masks, scores, boxes
    
    def calculate_iou(self,
                     pred_mask: np.ndarray,
                     gt_mask: np.ndarray) -> float:
        intersection = np.logical_and(pred_mask > 0.5, gt_mask > 0.5).sum()
        union = np.logical_or(pred_mask > 0.5, gt_mask > 0.5).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        iou = intersection / union
        return float(iou)
    
    def batch_segment(self,
                     image_paths: List[str],
                     confidence_threshold: float = 0.5) -> List[Tuple[List[np.ndarray], List[float], List[Dict]]]:
        results = []
        for path in image_paths:
            masks, scores, boxes = self.segment(path, confidence_threshold)
            results.append((masks, scores, boxes))
        return results
