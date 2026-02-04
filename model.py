# This model.py file defines the custom model architecture.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# CBAM Block
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP (Multi-Layer Perceptron)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# Uses convolution on pooled features to generate a spatial attention map.
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# This Combines Channel and Spatial attention sequentially. Input Feature Map -> Channel Attention -> Spatial Attention -> Refined Output       
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x) # Refine Channels
        out = out * self.sa(out) # Refine Spatial
        return out

# Lite Transfer Connection Block -> Connects the Backbone to the Object Detection Module, adding CBAM
class TransferConnectionBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(TransferConnectionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1) 
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1) 
        self.cbam = CBAM(out_planes) # Integrating Attention Mechanism here

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.cbam(x)
        return x

# --- MAIN MODEL (Stable Stride 8/16) ---
class MobileNetRefineDetLiteCBAM(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetRefineDetLiteCBAM, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = 4  
        
        # Replaced VGG-16 with MobileNetV3-Large
        backbone = models.mobilenet_v3_large(weights='DEFAULT') 
        self.features = backbone.features
        
        # Instead of using 4 layers (standard RefineDet), we only use 2 layers to prevent issues with downsampling.
        # Layer 6  -> Stride 8  (40 channels)  
        # Layer 12 -> Stride 16 (112 channels) 
        # Deeper layers (Stride 32/64) were removed as they can't see small icons.
        self.arm_loc_layers = nn.ModuleList([
            nn.Conv2d(40, self.num_anchors * 4, 3, padding=1),
            nn.Conv2d(112, self.num_anchors * 4, 3, padding=1), 
        ])
        self.arm_conf_layers = nn.ModuleList([
            nn.Conv2d(40, self.num_anchors * 2, 3, padding=1),
            nn.Conv2d(112, self.num_anchors * 2, 3, padding=1),
        ])
        
        # Reduced from 4 blocks to 2 blocks.
        self.tcb_layers = nn.ModuleList([
            TransferConnectionBlock(40, 64),
            TransferConnectionBlock(112, 64),
        ])

        self.odm_loc_layers = nn.ModuleList([
            nn.Conv2d(64, self.num_anchors * 4, 3, padding=1),
            nn.Conv2d(64, self.num_anchors * 4, 3, padding=1),
        ])
        self.odm_conf_layers = nn.ModuleList([
            nn.Conv2d(64, self.num_anchors * num_classes, 3, padding=1),
            nn.Conv2d(64, self.num_anchors * num_classes, 3, padding=1),
        ])

    def forward(self, x):
        sources = []
        # Feature Extraction
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 6:  sources.append(x) # Stride 8
            if i == 12: sources.append(x) # Stride 16 as last stride

        # ARM
        arm_loc = [l(x).permute(0, 2, 3, 1).contiguous() for l, x in zip(self.arm_loc_layers, sources)]
        arm_conf = [l(x).permute(0, 2, 3, 1).contiguous() for l, x in zip(self.arm_conf_layers, sources)]
        
        arm_loc = torch.cat([o.view(o.size(0), -1, 4) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1, 2) for o in arm_conf], 1)

        # Feature fusion
        fpn_features = [tcb(x) for tcb, x in zip(self.tcb_layers, sources)]

        # Simple Top-Down merge. We avoid complex BiFPN connections to save speed.
        upper = fpn_features[1]
        lower = fpn_features[0]
        
        upper_upsampled = F.interpolate(upper, size=lower.shape[2:], mode='bilinear', align_corners=True)
        fpn_features[0] = lower + upper_upsampled 

        # ODM
        odm_loc = [l(x).permute(0, 2, 3, 1).contiguous() for l, x in zip(self.odm_loc_layers, fpn_features)]
        odm_conf = [l(x).permute(0, 2, 3, 1).contiguous() for l, x in zip(self.odm_conf_layers, fpn_features)]
        
        odm_loc = torch.cat([o.view(o.size(0), -1, 4) for o in odm_loc], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in odm_conf], 1)

        return arm_loc, arm_conf, odm_loc, odm_conf
