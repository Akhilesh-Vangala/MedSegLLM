import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

class VisionTransformerSegmentation(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, num_classes: int = 2,
                 embed_dim: int = 768, num_heads: int = 12, num_layers: int = 12):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 4, num_classes, 4, 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = x[:, 1:]
        H = W = int(math.sqrt(x.size(1)))
        x = x.transpose(1, 2).view(B, self.embed_dim, H, W)
        
        x = self.decoder(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x


class SegFormer(nn.Module):
    def __init__(self, num_classes: int = 2, embed_dims: List[int] = [64, 128, 320, 512],
                 num_heads: List[int] = [1, 2, 5, 8], num_layers: List[int] = [2, 2, 2, 2]):
        super().__init__()
        self.encoder = SegFormerEncoder(embed_dims, num_heads, num_layers)
        self.decoder = SegFormerDecoder(embed_dims, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        output = self.decoder(features)
        return output


class SegFormerEncoder(nn.Module):
    def __init__(self, embed_dims: List[int], num_heads: List[int], num_layers: List[int]):
        super().__init__()
        self.stages = nn.ModuleList()
        
        for i, (embed_dim, num_head, num_layer) in enumerate(zip(embed_dims, num_heads, num_layers)):
            stage = SegFormerStage(embed_dim, num_head, num_layer, patch_size=4 if i == 0 else 2)
            self.stages.append(stage)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class SegFormerStage(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, patch_size: int = 2):
        super().__init__()
        self.patch_merge = nn.Conv2d(3 if embed_dim == 64 else embed_dim // 2, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([
            EfficientSelfAttention(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_merge(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        for block in self.blocks:
            x = block(x, H, W)
        
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x


class EfficientSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, sr_ratio: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sr_ratio = sr_ratio
        
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        if sr_ratio > 1:
            self.sr = nn.Conv2d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class SegFormerDecoder(nn.Module):
    def __init__(self, embed_dims: List[int], num_classes: int):
        super().__init__()
        self.linear_fuse = nn.Conv2d(sum(embed_dims), 256, kernel_size=1)
        self.linear_pred = nn.Conv2d(256, num_classes, kernel_size=1)
        self.bn = nn.BatchNorm2d(256)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        B, _, H, W = features[0].shape
        
        upsampled = []
        for feat in features:
            upsampled.append(F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False))
        
        fused = torch.cat(upsampled, dim=1)
        fused = self.linear_fuse(fused)
        fused = self.bn(fused)
        fused = F.relu(fused)
        
        output = self.linear_pred(fused)
        return output
