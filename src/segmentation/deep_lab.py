import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    deeplabv3_resnet50, deeplabv3_resnet101,
    deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large
)
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int = 2, backbone: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        
        if backbone == 'resnet50':
            self.backbone = deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)
        elif backbone == 'resnet101':
            self.backbone = deeplabv3_resnet101(pretrained=pretrained, num_classes=num_classes)
        elif backbone == 'mobilenet':
            self.backbone = deeplabv3_mobilenet_v3_large(pretrained=pretrained, num_classes=num_classes)
        else:
            self.backbone = deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone(x)
        if isinstance(output, dict):
            return output['out']
        return output


class LRASPP(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.model = lraspp_mobilenet_v3_large(pretrained=pretrained, num_classes=num_classes)
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if isinstance(output, dict):
            return output['out']
        return output


class FCN(nn.Module):
    def __init__(self, num_classes: int = 2, backbone: str = 'resnet50'):
        super().__init__()
        from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101
        
        if backbone == 'resnet50':
            self.model = fcn_resnet50(pretrained=True, num_classes=num_classes)
        else:
            self.model = fcn_resnet101(pretrained=True, num_classes=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if isinstance(output, dict):
            return output['out']
        return output
