import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo
import logging

logger = logging.getLogger(__name__)


class PanopticFPN(nn.Module):
    def __init__(self, num_classes: int = 2, num_thing_classes: int = 1, num_stuff_classes: int = 1):
        super().__init__()
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_stuff_classes
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_thing_classes
        
        self.model = build_model(cfg)
        self.num_classes = num_classes
    
    def forward(self, images: List[torch.Tensor]) -> Dict:
        outputs = self.model(images)
        return outputs


class PanopticDeepLab(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_resnet50
        
        self.backbone = deeplabv3_resnet50(pretrained=True, num_classes=num_classes)
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone(x)
        if isinstance(output, dict):
            return output['out']
        return output
