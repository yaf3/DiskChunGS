#!/usr/bin/env python3
"""
Script to extract XFeat weights for use in C++ LibTorch implementation.
This script downloads the pretrained XFeat model and saves the weights
in a format that can be loaded by the C++ implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from pathlib import Path

# XFeat model definition (same as your reference)
class BasicLayer(nn.Module):
    """
    Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

class XFeatModel(nn.Module):
    """
    Implementation of architecture described in 
    "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    """
    def __init__(self, stride):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)

        ########### ⬇️ CNN Backbone & Heads ⬇️ ###########

        self.skip1 = nn.Sequential(
            nn.AvgPool2d(stride, stride=stride),
            nn.Conv2d(1, 24, 1, stride=1, padding=0)
        )

        if stride == 1 or stride == 2:
            self.block1 = nn.Sequential(
                BasicLayer(1, 4, stride=1),
                BasicLayer(4, 8, stride=stride),
                BasicLayer(8, 8, stride=1),
                BasicLayer(8, 24, stride=1),
            )
        elif stride == 4:
            self.block1 = nn.Sequential(
                BasicLayer(1, 4, stride=1),
                BasicLayer(4, 8, stride=2),
                BasicLayer(8, 8, stride=1),
                BasicLayer(8, 24, stride=2),
            )
        else:
            raise ValueError("Invalid stride value, must be 1, 2 or 4")

        self.block2 = nn.Sequential(
            BasicLayer(24, 24, stride=1),
            BasicLayer(24, 24, stride=1),
        )

        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, 1, padding=0),
        )
        
        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
        )

        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 64, 1, padding=0),
        )

        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            nn.Conv2d(64, 64, 1, padding=0)
        )

        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 65, 1),
        )

        ########### ⬇️ Fine Matcher MLP ⬇️ ###########

        self.fine_matcher = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
        )

    def forward(self, x):
        # Don't backprop through normalization
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)

        # Main backbone
        x1 = self.block1(x)
        xskip = self.skip1(x)
        x2 = self.block2(x1 + xskip)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        # Pyramid fusion
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        feats = self.block_fusion(x3 + x4 + x5)

        return feats

def download_pretrained_xfeat():
    """
    Download the pretrained XFeat model using torch.hub
    """
    print("Downloading pretrained XFeat model...")
    try:
        # Try to load from torch.hub
        xfeat_full = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
        print("✓ Successfully downloaded XFeat model from torch.hub")
        return xfeat_full
    except Exception as e:
        print(f"✗ Failed to download from torch.hub: {e}")
        print("Please install the model manually or check your internet connection.")
        return None

def extract_backbone_weights(xfeat_full_model):
    """
    Extract only the backbone weights that match our XFeatModel structure
    """
    print("Extracting backbone weights...")
    
    # Get the state dict from the full model
    full_state_dict = xfeat_full_model.state_dict()
    
    # Create our backbone model
    backbone = XFeatModel(stride=4)
    backbone_state_dict = backbone.state_dict()
    
    # Map weights from full model to backbone
    # The full XFeat model has a 'net' attribute that contains the backbone
    extracted_weights = {}
    
    for key in backbone_state_dict.keys():
        # Try to find corresponding key in full model
        full_key = f"net.{key}"
        if full_key in full_state_dict:
            extracted_weights[key] = full_state_dict[full_key]
            print(f"✓ Mapped {full_key} -> {key}")
        else:
            print(f"✗ Could not find {full_key} in pretrained model")
    
    print(f"Extracted {len(extracted_weights)}/{len(backbone_state_dict)} weights")
    return extracted_weights

def save_weights_for_cpp(weights_dict, output_path):
    """
    Save weights in a format that LibTorch can load
    """
    print(f"Saving weights to {output_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a model instance and load the weights
    model = XFeatModel(stride=4)
    
    # Load the extracted weights
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys}")
    
    # Save using torch.save (LibTorch compatible)
    torch.save(model.state_dict(), output_path)
    print(f"✓ Weights saved to {output_path}")
    
    # Also save the entire model for verification
    model_path = output_path.replace('.pt', '_full_model.pt')
    torch.save(model, model_path)
    print(f"✓ Full model saved to {model_path}")
    
    return model

def verify_weights(model, output_path):
    """
    Verify that the saved weights can be loaded correctly
    """
    print("Verifying saved weights...")
    
    # Test loading the weights
    test_model = XFeatModel(stride=4)
    test_model.load_state_dict(torch.load(output_path))
    test_model.eval()
    
    # Test inference
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = test_model(dummy_input)
    
    print(f"✓ Verification successful. Output shape: {output.shape}")
    return True

def main():
    print("XFeat Weight Extraction Script")
    print("=" * 40)
    
    # Set output path
    output_dir = "models"
    output_path = os.path.join(output_dir, "xfeat_weights.pt")
    
    # Download pretrained model
    xfeat_model = download_pretrained_xfeat()
    if xfeat_model is None:
        print("Failed to download model. Exiting.")
        return 1
    
    # Extract backbone weights
    backbone_weights = extract_backbone_weights(xfeat_model)
    if not backbone_weights:
        print("Failed to extract weights. Exiting.")
        return 1
    
    # Save weights for C++
    model = save_weights_for_cpp(backbone_weights, output_path)
    
    # Verify weights
    if verify_weights(model, output_path):
        print("\n" + "=" * 40)
        print("✓ SUCCESS! Weights are ready for C++")
        print(f"Weights saved to: {os.path.abspath(output_path)}")
        print("\nYou can now use these weights in your C++ implementation.")
        return 0
    else:
        print("✗ Verification failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)