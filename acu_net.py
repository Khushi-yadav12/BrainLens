"""
Attention-based Convolutional U-Net (ACU-Net)
=============================================

Implementation of the ACU-Net model for brain tumor segmentation,
integrating Spatial and Channel Attention mechanisms and a combined
Dice + Cross-Entropy Loss function.

Paper Reference: "Enhancing Brain Tumor Segmentation using Attention-based Convolutional U-Net (ACU-Net)"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Modalities: Channel & Spatial Attention ──────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Channel Attention Module: Focuses on "what" features are important.
    Uses both Max-Pool and Avg-Pool across the spatial dimensions, 
    processes them via a shared MLP, and combines them.
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared Multi-Layer Perceptron (MLP)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module: Focuses on "where" the tumor is.
    Locates informative regions by pooling along the channel axis
    and applying a convolution layer.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pool across the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along the channel dimension (resulting in 2 channels)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # Convolve back to 1 channel and apply sigmoid activation
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class AttentionBlock(nn.Module):
    """
    Combined Attention Block for integration into skip-connections.
    Applies Channel Attention followed by Spatial Attention.
    """
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        # Multiply input by channel attention weight
        x = x * self.ca(x)
        # Multiply result by spatial attention weight
        x = x * self.sa(x)
        return x


# ── Backbone: Convolutional U-Net ─────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Standard U-Net Double Convolution Block."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ACUNet(nn.Module):
    """
    Attention-based Convolutional U-Net (ACU-Net) architecture.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(ACUNet, self).__init__()
        
        # Encoder (Downsampling)
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder (Upsampling) + Skip Connections with Attention Modules
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(512)
        self.decoder4 = DoubleConv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(256)
        self.decoder3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(128)
        self.decoder2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(64)
        self.decoder1 = DoubleConv(128, 64)
        
        # Final Point-wise Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # ── Encoder ──
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # ── Bottleneck ──
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # ── Decoder + Skip Connections (with Attention) ──
        dec4 = self.upconv4(bottleneck)
        att4 = self.att4(enc4)
        # Handle differing spatial sizes dynamically if input isn't pure power of 2
        diffY = att4.size()[2] - dec4.size()[2]
        diffX = att4.size()[3] - dec4.size()[3]
        dec4 = F.pad(dec4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec4 = torch.cat((att4, dec4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        att3 = self.att3(enc3)
        diffY = att3.size()[2] - dec3.size()[2]
        diffX = att3.size()[3] - dec3.size()[3]
        dec3 = F.pad(dec3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec3 = torch.cat((att3, dec3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        att2 = self.att2(enc2)
        diffY = att2.size()[2] - dec2.size()[2]
        diffX = att2.size()[3] - dec2.size()[3]
        dec2 = F.pad(dec2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec2 = torch.cat((att2, dec2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        att1 = self.att1(enc1)
        diffY = att1.size()[2] - dec1.size()[2]
        diffX = att1.size()[3] - dec1.size()[3]
        dec1 = F.pad(dec1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec1 = torch.cat((att1, dec1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # ── Output ──
        # Outputs logits. Use sigmoid alongside loss or predicting.
        return self.final_conv(dec1)


# ── Optimization: Combined Loss Function ──────────────────────────────────────

class DiceBCELoss(nn.Module):
    """
    Combined Loss Function: L = λ1 * DiceLoss + λ2 * BinaryCrossEntropyLoss
    Enhances overlap accuracy and mitigates class imbalance.
    """
    def __init__(self, weight=None, size_average=True, lambda_dice=1.0, lambda_bce=1.0):
        super(DiceBCELoss, self).__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce

    def forward(self, inputs, targets, smooth=1e-6):
        # Apply sigmoid to raw network output logits
        inputs = torch.sigmoid(inputs)       
        
        # Flatten label and prediction tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # Compute Dice Loss
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)  
        
        # Compute Binary Cross Entropy Loss
        bce_loss = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
        
        # Final combined loss equation
        return (self.lambda_dice * dice_loss) + (self.lambda_bce * bce_loss)
