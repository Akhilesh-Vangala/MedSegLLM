import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int = 32, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, growth_rate, 3, padding=1, bias=False)
            ))
            channels += growth_rate
        
        self.transition = nn.Conv2d(channels, in_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        out = torch.cat(features, dim=1)
        out = self.transition(out)
        return out


class DenseUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        
        self.encoder1 = DenseBlock(in_channels, 32)
        self.encoder2 = DenseBlock(64, 64)
        self.encoder3 = DenseBlock(128, 128)
        self.encoder4 = DenseBlock(256, 256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.bottleneck = DenseBlock(512, 512)
        
        self.decoder4 = DenseBlock(512 + 256, 256)
        self.decoder3 = DenseBlock(256 + 128, 128)
        self.decoder2 = DenseBlock(128 + 64, 64)
        self.decoder1 = DenseBlock(64 + 32, 32)
        
        self.final = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e1_pool = self.pool(e1)
        
        e2 = self.encoder2(e1_pool)
        e2_pool = self.pool(e2)
        
        e3 = self.encoder3(e2_pool)
        e3_pool = self.pool(e3)
        
        e4 = self.encoder4(e3_pool)
        e4_pool = self.pool(e4)
        
        bottleneck = self.bottleneck(e4_pool)
        
        d4 = self.upsample(bottleneck)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upsample(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upsample(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upsample(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        output = self.final(d1)
        return output


class ResUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        
        self.encoder1 = ResidualBlock(in_channels, 64)
        self.encoder2 = ResidualBlock(64, 128)
        self.encoder3 = ResidualBlock(128, 256)
        self.encoder4 = ResidualBlock(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.bottleneck = ResidualBlock(512, 1024)
        
        self.decoder4 = ResidualBlock(1024 + 512, 512)
        self.decoder3 = ResidualBlock(512 + 256, 256)
        self.decoder2 = ResidualBlock(256 + 128, 128)
        self.decoder1 = ResidualBlock(128 + 64, 64)
        
        self.final = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e1_pool = self.pool(e1)
        
        e2 = self.encoder2(e1_pool)
        e2_pool = self.pool(e2)
        
        e3 = self.encoder3(e2_pool)
        e3_pool = self.pool(e3)
        
        e4 = self.encoder4(e3_pool)
        e4_pool = self.pool(e4)
        
        bottleneck = self.bottleneck(e4_pool)
        
        d4 = self.upsample(bottleneck)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upsample(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upsample(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upsample(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        output = self.final(d1)
        return output


class MultiScaleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(1024 + 512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Conv2d(64, num_classes, 1)
        
        self.aspp1 = nn.Conv2d(1024, 256, 1)
        self.aspp2 = nn.Conv2d(1024, 256, 3, padding=6, dilation=6)
        self.aspp3 = nn.Conv2d(1024, 256, 3, padding=12, dilation=12)
        self.aspp4 = nn.Conv2d(1024, 256, 3, padding=18, dilation=18)
        self.aspp_pool = nn.AdaptiveAvgPool2d(1)
        self.aspp_concat = nn.Conv2d(256 * 5, 1024, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        bottleneck = self.bottleneck(e4)
        
        aspp1 = self.aspp1(bottleneck)
        aspp2 = self.aspp2(bottleneck)
        aspp3 = self.aspp3(bottleneck)
        aspp4 = self.aspp4(bottleneck)
        aspp_pool = self.aspp_pool(bottleneck)
        aspp_pool = F.interpolate(aspp_pool, size=bottleneck.shape[2:], mode='bilinear', align_corners=False)
        
        aspp_concat = torch.cat([aspp1, aspp2, aspp3, aspp4, aspp_pool], dim=1)
        bottleneck = self.aspp_concat(aspp_concat)
        
        d4 = self.upsample(bottleneck)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upsample(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upsample(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upsample(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        output = self.final(d1)
        return output
