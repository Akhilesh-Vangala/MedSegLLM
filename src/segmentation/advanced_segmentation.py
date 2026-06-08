import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import logging

logger = logging.getLogger(__name__)


class AdvancedMaskRCNN(nn.Module):
    def __init__(self,
                 num_classes: int = 2,
                 backbone: str = "R50-FPN",
                 pretrained: bool = True):
        super().__init__()
        
        cfg = get_cfg()
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        
        if backbone == "R50-FPN":
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        elif backbone == "R101-FPN":
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        else:
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        
        if pretrained:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        self.model = build_model(cfg)
        
        logger.info(f"Advanced Mask R-CNN initialized with {backbone} backbone")
    
    def segment(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
        outputs = self.predictor(image)
        
        instances = outputs["instances"]
        
        masks = instances.pred_masks.cpu().numpy()
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()
        
        filtered_indices = scores >= confidence_threshold
        
        return {
            'masks': masks[filtered_indices],
            'boxes': boxes[filtered_indices],
            'scores': scores[filtered_indices],
            'classes': classes[filtered_indices]
        }


class UNetSegmenter(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        
        self.encoder1 = self._conv_block(in_channels, 64)
        self.encoder2 = self._conv_block(64, 128)
        self.encoder3 = self._conv_block(128, 256)
        self.encoder4 = self._conv_block(256, 512)
        
        self.bottleneck = self._conv_block(512, 1024)
        
        self.decoder4 = self._conv_block(1024 + 512, 512)
        self.decoder3 = self._conv_block(512 + 256, 256)
        self.decoder2 = self._conv_block(256 + 128, 128)
        self.decoder1 = self._conv_block(128 + 64, 64)
        
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        bottleneck = self.bottleneck(self.pool(e4))
        
        d4 = self.decoder4(torch.cat([self.upsample(bottleneck), e4], dim=1))
        d3 = self.decoder3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upsample(d2), e1], dim=1))
        
        return self.final(d1)


class DeepLabV3Segmenter(nn.Module):
    def __init__(self, num_classes: int = 2, backbone: str = "resnet50"):
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
        
        if backbone == "resnet50":
            self.model = deeplabv3_resnet50(pretrained=True, num_classes=num_classes)
        else:
            self.model = deeplabv3_resnet101(pretrained=True, num_classes=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)['out']


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        
        self.encoder1 = self._conv_block(in_channels, 64)
        self.encoder2 = self._conv_block(64, 128)
        self.encoder3 = self._conv_block(128, 256)
        self.encoder4 = self._conv_block(256, 512)
        
        self.bottleneck = self._conv_block(512, 1024)
        
        self.attention4 = AttentionBlock(512, 1024, 512)
        self.attention3 = AttentionBlock(256, 512, 256)
        self.attention2 = AttentionBlock(128, 256, 128)
        self.attention1 = AttentionBlock(64, 128, 64)
        
        self.decoder4 = self._conv_block(1024 + 512, 512)
        self.decoder3 = self._conv_block(512 + 256, 256)
        self.decoder2 = self._conv_block(256 + 128, 128)
        self.decoder1 = self._conv_block(128 + 64, 64)
        
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        bottleneck = self.bottleneck(self.pool(e4))
        
        d4 = self.upsample(bottleneck)
        att4 = self.attention4(e4, d4)
        d4 = self.decoder4(torch.cat([d4, att4], dim=1))
        
        d3 = self.upsample(d4)
        att3 = self.attention3(e3, d3)
        d3 = self.decoder3(torch.cat([d3, att3], dim=1))
        
        d2 = self.upsample(d3)
        att2 = self.attention2(e2, d2)
        d2 = self.decoder2(torch.cat([d2, att2], dim=1))
        
        d1 = self.upsample(d2)
        att1 = self.attention1(e1, d1)
        d1 = self.decoder1(torch.cat([d1, att1], dim=1))
        
        return self.final(d1)


class AttentionBlock(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class EnsembleSegmenter:
    def __init__(self, segmenters: List[nn.Module], fusion_method: str = "voting"):
        self.segmenters = nn.ModuleList(segmenters)
        self.fusion_method = fusion_method
    
    def segment(self, image: torch.Tensor) -> torch.Tensor:
        all_predictions = []
        
        for segmenter in self.segmenters:
            with torch.no_grad():
                pred = segmenter(image)
                if isinstance(pred, dict):
                    pred = pred['out']
                all_predictions.append(pred.argmax(dim=1))
        
        all_predictions = torch.stack(all_predictions, dim=0)
        
        if self.fusion_method == "voting":
            final_pred = torch.mode(all_predictions, dim=0)[0]
        elif self.fusion_method == "average":
            final_pred = (all_predictions.float().mean(dim=0) > 0.5).long()
        else:
            final_pred = all_predictions[0]
        
        return final_pred
