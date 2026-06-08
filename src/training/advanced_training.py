import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import copy
from collections import defaultdict

logger = logging.getLogger(__name__)


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


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_probs = torch.sigmoid(pred)
        
        pt = pred_probs * target + (1 - pred_probs) * (1 - target)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        pred_flat = pred_probs.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        focal_dice = focal_weight.mean() * (1 - dice)
        
        return focal_dice


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        true_positives = (pred_flat * target_flat).sum()
        false_positives = (pred_flat * (1 - target_flat)).sum()
        false_negatives = ((1 - pred_flat) * target_flat).sum()
        
        tversky = (true_positives + self.smooth) / (
            true_positives + self.alpha * false_positives + self.beta * false_negatives + self.smooth
        )
        
        return 1 - tversky


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.3, focal_weight: float = 0.2):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.focal_dice_loss = FocalDiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        
        pred_probs = F.softmax(pred, dim=1)
        pred_binary = pred_probs[:, 1]
        target_binary = (target > 0).float()
        
        dice = self.dice_loss(pred_binary, target_binary)
        focal = self.focal_dice_loss(pred_binary, target_binary)
        
        total_loss = self.ce_weight * ce + self.dice_weight * dice + self.focal_weight * focal
        
        return total_loss


class AdvancedSegmentationTrainer:
    def __init__(self, model: nn.Module, config: Dict, device: str = "cuda"):
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        
        self.scaler = GradScaler() if config.get('use_mixed_precision', False) else None
        self.use_mixed_precision = config.get('use_mixed_precision', False)
        
        self.best_model_state = None
        self.best_score = -float('inf')
        self.training_history = defaultdict(list)
    
    def _create_optimizer(self):
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adamw')
        lr = optimizer_config.get('learning_rate', 1e-4)
        weight_decay = optimizer_config.get('weight_decay', 0.01)
        
        if optimizer_type == 'adamw':
            return AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adam':
            return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    def _create_scheduler(self):
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', 10)
            return CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3)
        elif scheduler_type == 'onecycle':
            max_lr = scheduler_config.get('max_lr', 1e-3)
            steps_per_epoch = scheduler_config.get('steps_per_epoch', 100)
            epochs = scheduler_config.get('epochs', 10)
            return OneCycleLR(self.optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
        else:
            return None
    
    def _create_criterion(self):
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'combined')
        
        if loss_type == 'dice':
            return DiceLoss()
        elif loss_type == 'focal_dice':
            return FocalDiceLoss()
        elif loss_type == 'tversky':
            return TverskyLoss()
        elif loss_type == 'combined':
            return CombinedSegmentationLoss()
        else:
            return nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs['out']
                    loss = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                
                if self.config.get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('gradient_clip', 1.0)
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                loss = self.criterion(outputs, masks)
                
                loss.backward()
                
                if self.config.get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('gradient_clip', 1.0)
                    )
                
                self.optimizer.step()
            
            if self.scheduler and isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                pred_masks = outputs.argmax(dim=1)
                dice = self._calculate_dice(pred_masks, masks)
                total_dice += dice
            
            pbar.set_postfix({
                'loss': loss.item(),
                'dice': dice
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        
        return {'loss': avg_loss, 'dice': avg_dice}
    
    def validate(self, val_loader) -> Dict:
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                        if isinstance(outputs, dict):
                            outputs = outputs['out']
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs['out']
                    loss = self.criterion(outputs, masks)
                
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
    
    def _calculate_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_flat = pred.view(-1).float()
        target_flat = target.view(-1).float()
        
        intersection = (pred_flat * target_flat).sum()
        dice = 2 * intersection / (pred_flat.sum() + target_flat.sum() + 1e-8)
        
        return dice.item()
    
    def train(self, train_loader, val_loader, num_epochs: int):
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            
            for key, value in train_metrics.items():
                self.training_history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.training_history[f'val_{key}'].append(value)
            
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('dice', val_metrics.get('iou', 0)))
                elif not isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()
            
            score = val_metrics.get('dice', val_metrics.get('iou', 0))
            if score > self.best_score:
                self.best_score = score
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                logger.info(f"New best model! Score: {score:.4f}")
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Dice: {train_metrics['dice']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f}, Val IoU: {val_metrics['iou']:.4f}, "
                f"Val Dice: {val_metrics['dice']:.4f}"
            )
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return self.training_history
