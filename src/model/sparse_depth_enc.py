# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sparse Depth Encoder for optional LiDAR depth input."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Simple residual block with Conv -> GELU -> Conv -> Add."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.shortcut = nn.Identity() if in_channels == out_channels else \
                        nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.gelu(out)
        out = self.conv2(out)
        return out + identity


class SparseDepthEnc(nn.Module):
    """
    CNN encoder for sparse depth map to patch-wise feature map.

    Converts sparse depth input [B*S, 1, H, W] → [B*S, embed_dim, H/14, W/14].
    Uses PixelUnshuffle to fold spatial resolution into channels, then resnet blocks
    to learn depth feature representations.

    Architecture:
        Input: (B*S, 1, H, W)
        → PixelUnshuffle(14): (B*S, 196, H/14, W/14)
        → Conv2d(196 → 256): (B*S, 256, H/14, W/14)
        → ResidualBlock(256 → 512): (B*S, 512, H/14, W/14)
        → ResidualBlock(512 → embed_dim): (B*S, embed_dim, H/14, W/14)
        → Conv2d(embed_dim → embed_dim, 1x1): (B*S, embed_dim, H/14, W/14)
        → LayerNorm + reshape: (B*S, H/14, W/14, embed_dim)
        Output: (B*S, embed_dim, H/14, W/14) after reshape back
    """

    def __init__(self, embed_dim: int = 1024, patch_size: int = 14):
        """
        Args:
            embed_dim: Output feature dimension (should match ViT embed_dim)
            patch_size: ViT patch size (14 for DinoV2)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # PixelUnshuffle folds spatial resolution into channels
        # Input [B*S, 1, H, W] → Output [B*S, 1*patch_size^2, H/patch_size, W/patch_size]
        self.unshuffle = nn.PixelUnshuffle(patch_size)
        unshuffle_out_channels = 1 * (patch_size ** 2)

        # Initial conv projection
        self.initial_proj = nn.Conv2d(unshuffle_out_channels, 256, 3, padding=1)

        # Residual blocks for feature learning
        self.res_block1 = ResidualBlock(256, 512, kernel_size=3)
        self.res_block2 = ResidualBlock(512, embed_dim, kernel_size=3)

        # Final projection
        self.final_proj = nn.Conv2d(embed_dim, embed_dim, 1)

        # Normalization (applied before final output)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, sparse_depth: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sparse depth encoding.

        Args:
            sparse_depth: [B*S, 1, H, W] or [B, S, H, W] sparse depth map
                         Values > 0 are valid depths, 0 means invalid/unselected

        Returns:
            depth_features: [B*S, embed_dim, H/14, W/14] patch-wise feature map
        """
        # Handle input shape: if [B, S, H, W], reshape to [B*S, 1, H, W]
        if sparse_depth.dim() == 4 and sparse_depth.shape[1] > 1:
            # Assume [B, S, H, W]
            B, S, H, W = sparse_depth.shape
            sparse_depth = sparse_depth.reshape(B * S, 1, H, W)
        elif sparse_depth.dim() == 4:
            # Already [B*S, 1, H, W]
            pass
        else:
            raise ValueError(f"sparse_depth must be 4D tensor, got shape {sparse_depth.shape}")

        # Log normalization: log(depth + 1) to compress dynamic range
        # This is applied before encoding to normalize depth values
        sparse_depth = torch.log(torch.clamp(sparse_depth, min=0.0) + 1.0)

        # PixelUnshuffle: spatial resolution → channel dimension
        x = self.unshuffle(sparse_depth)  # [B*S, 196, H/14, W/14]

        # Initial projection
        x = self.initial_proj(x)  # [B*S, 256, H/14, W/14]

        # Residual blocks
        x = self.res_block1(x)  # [B*S, 512, H/14, W/14]
        x = self.res_block2(x)  # [B*S, embed_dim, H/14, W/14]

        # Final projection
        x = self.final_proj(x)  # [B*S, embed_dim, H/14, W/14]

        # Apply LayerNorm in CHW format (normalize over channel dimension)
        B_S, C, H_p, W_p = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B*S, H/14, W/14, embed_dim]
        x = self.norm(x)  # LayerNorm normalizes over last dimension
        x = x.permute(0, 3, 1, 2).contiguous()  # [B*S, embed_dim, H/14, W/14]

        return x
