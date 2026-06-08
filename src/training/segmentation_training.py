import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional
import numpy as np
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import logging

logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    def __init__(self, images: List[np.ndarray], masks: List[np.ndarray], transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
        mask = torch.LongTensor(mask)
        
        return image, mask


class AdvancedSegmentationTrainer:
    def __init__(self, model: nn.Module, config: Dict, device: str = "cuda"):
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 10)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.combined_loss = CombinedLoss(self.criterion, self.dice_loss, alpha=0.5)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            loss = self.combined_loss(outputs, masks)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                pred_masks = outputs.argmax(dim=1)
                dice = self.dice_loss(pred_masks.float(), masks.float())
                total_dice += (1 - dice.item())
        
        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        
        return {'loss': avg_loss, 'dice': avg_dice}
    
    def validate(self, val_loader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                
                loss = self.combined_loss(outputs, masks)
                total_loss += loss.item()
                
                pred_masks = outputs.argmax(dim=1)
                
                for pred, gt in zip(pred_masks, masks):
                    pred_np = pred.cpu().numpy()
                    gt_np = gt.cpu().numpy()
                    
                    intersection = np.logical_and(pred_np > 0.5, gt_np > 0.5).sum()
                    union = np.logical_or(pred_np > 0.5, gt_np > 0.5).sum()
                    
                    if union > 0:
                        iou = intersection / union
                        total_iou += iou
                    
                    dice = 2 * intersection / (pred_np.sum() + gt_np.sum() + 1e-8)
                    total_dice += dice
        
        avg_loss = total_loss / len(val_loader)
        avg_iou = total_iou / (len(val_loader) * val_loader.batch_size)
        avg_dice = total_dice / (len(val_loader) * val_loader.batch_size)
        
        return {'loss': avg_loss, 'iou': avg_iou, 'dice': avg_dice}


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, ce_loss: nn.Module, dice_loss: nn.Module, alpha: float = 0.5):
        super().__init__()
        self.ce_loss = ce_loss
        self.dice_loss = dice_loss
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.alpha * ce + (1 - self.alpha) * dice


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_probs = F.softmax(pred, dim=1)
        pred_probs = pred_probs[:, 1]
        
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).float()
        target_probs = target_one_hot[:, 1]
        
        pt = pred_probs * target_probs + (1 - pred_probs) * (1 - target_probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        intersection = (pred_probs * target_probs).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_probs.sum() + target_probs.sum() + self.smooth
        )
        
        focal_dice = focal_weight.mean() * (1 - dice)
        
        return focal_dice
