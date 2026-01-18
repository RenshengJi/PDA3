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

"""Depth Anything 3 Network for depth estimation with camera pose."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from addict import Dict as AdDict

from .dinov2 import DinoV2
from .dualdpt import DualDPT
from .cam_dec import CameraDec
from .cam_enc import CameraEnc
from .utils.transform import pose_encoding_to_extri_intri
from .utils.geometry import affine_inverse, as_homogeneous
from .utils.ray_utils import get_extrinsic_from_camray


class DepthAnything3Net(nn.Module):
    """
    Depth Anything 3 network for depth estimation and camera pose estimation.

    This network consists of:
    - Backbone: DinoV2 feature extractor
    - Head: DualDPT for depth and ray prediction
    - Optional camera decoders for pose estimation
    """

    PATCH_SIZE = 14

    def __init__(
        self,
        encoder_name: str = "vitl",
        out_layers: List[int] = [11, 15, 19, 23],
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        alt_start: int = 8,
        qknorm_start: int = 8,
        rope_start: int = 8,
        cat_token: bool = True,
        predict_camera: bool = True,
        use_camera_enc: bool = False,
    ):
        """Initialize Depth Anything 3 network.

        Args:
            encoder_name: Name of DinoV2 encoder ("vits", "vitb", "vitl", "vitg")
            out_layers: Layer indices to extract features from
            features: Feature dimension for DPT head
            out_channels: Output channel dimensions for each layer
            alt_start: Layer index to start alternating attention
            qknorm_start: Layer index to start QK normalization
            rope_start: Layer index to start RoPE
            cat_token: Whether to concatenate tokens
            predict_camera: Whether to predict camera pose
            use_camera_enc: Whether to use camera encoder (requires GT camera poses)
        """
        super().__init__()

        self.encoder_name = encoder_name
        self.predict_camera = predict_camera
        self.use_camera_enc = use_camera_enc

        # Encoder dimensions
        encoder_dims = {
            "vits": 384,
            "vitb": 768,
            "vitl": 1024,
            "vitg": 1536,
        }
        self.embed_dim = encoder_dims[encoder_name]
        self.in_channels = self.embed_dim * 2 if cat_token else self.embed_dim

        # Initialize backbone (DinoV2)
        self.backbone = DinoV2(
            name=encoder_name,
            out_layers=out_layers,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
        )

        # Initialize DualDPT head for depth estimation
        self.head = DualDPT(
            dim_in=self.in_channels,
            features=features,
            out_channels=out_channels,
            output_dim=2,  # depth + conf
            pos_embed=True,
        )

        # Initialize camera decoder
        if predict_camera:
            self.cam_dec = CameraDec(dim_in=self.in_channels)
        else:
            self.cam_dec = None

        # Initialize camera encoder (optional, for when GT poses are provided)
        if use_camera_enc:
            self.cam_enc = CameraEnc(dim_out=self.embed_dim)
        else:
            self.cam_enc = None

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        export_feat_layers: List[int] = [],
        use_ray_pose: bool = False,
    ) -> AdDict:
        """Forward pass through the network.

        Args:
            x: Input images of shape [B, S, 3, H, W]
            extrinsics: Camera extrinsics [B, S, 4, 4] (optional, for camera encoder)
            intrinsics: Camera intrinsics [B, S, 3, 3] (optional, for camera encoder)
            export_feat_layers: List of layer indices to extract features from
            use_ray_pose: If True, use ray-based pose estimation instead of CameraDec

        Returns:
            Dictionary containing:
                - depth: Predicted depth maps [B, S, H, W]
                - depth_conf: Depth confidence [B, S, H, W]
                - ray: Ray map [B, S, H, W, 7] (if use_ray_pose or keep_ray)
                - ray_conf: Ray confidence [B, S, H, W] (if use_ray_pose or keep_ray)
                - extrinsics: Camera extrinsics [B, S, 4, 4] (if predict_camera)
                - intrinsics: Camera intrinsics [B, S, 3, 3] (if predict_camera)
        """
        # Handle camera token if camera encoder is available
        if extrinsics is not None and self.cam_enc is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        else:
            cam_token = None

        # Extract features using backbone
        feats, aux_feats = self.backbone(
            x, cam_token=cam_token, export_feat_layers=export_feat_layers
        )

        H, W = x.shape[-2], x.shape[-1]

        # Process features through heads
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self._process_depth_head(feats, H, W)
            # Choose between ray-based pose estimation and CameraDec
            if use_ray_pose:
                output = self._process_ray_pose_estimation(output, H, W)
            else:
                output = self._process_camera_estimation(feats, H, W, output)

        # Extract auxiliary features if requested
        output.aux = self._extract_auxiliary_features(aux_feats, export_feat_layers, H, W)

        return output

    def _process_depth_head(
        self, feats: List[torch.Tensor], H: int, W: int
    ) -> AdDict:
        """Process features through the depth prediction head."""
        return self.head(feats, H, W, patch_start_idx=0)

    def _process_camera_estimation(
        self, feats: List[torch.Tensor], H: int, W: int, output: AdDict
    ) -> AdDict:
        """Process camera pose estimation using CameraDec.

        Note: Ray output is preserved for potential use in loss computation.
        """
        if self.cam_dec is not None:
            # Camera tokens are the second element of the last feature
            pose_enc = self.cam_dec(feats[-1][1])

            # Note: We keep ray and ray_conf in output for loss computation
            # (previously they were deleted here)

            # Convert pose encoding to extrinsics and intrinsics
            c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
            # c2w is [B, S, 3, 4], convert to [B, S, 4, 4] before inverse
            c2w = as_homogeneous(c2w)
            output.extrinsics = affine_inverse(c2w)
            output.intrinsics = ixt

        return output

    def _process_ray_pose_estimation(
        self, output: AdDict, H: int, W: int
    ) -> AdDict:
        """Process camera pose estimation using ray map.

        Uses ray predictions to compute camera extrinsics and intrinsics.
        Note: get_extrinsic_from_camray returns c2w (camera-to-world) format,
        so we only need one affine_inverse to convert to w2c.
        """
        if "ray" in output and "ray_conf" in output:
            # Get extrinsics from ray map (returns c2w format)
            pred_c2w, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
                output.ray,
                output.ray_conf,
                output.ray.shape[-3],  # num_patches_y
                output.ray.shape[-2],  # num_patches_x
            )

            # Build intrinsic matrix
            B, S = pred_c2w.shape[:2]
            pred_intrinsic = torch.eye(3, 3, device=pred_c2w.device, dtype=pred_c2w.dtype)
            pred_intrinsic = pred_intrinsic[None, None].repeat(B, S, 1, 1).clone()
            pred_intrinsic[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * W
            pred_intrinsic[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * H
            pred_intrinsic[:, :, 0, 2] = pred_principal_points[:, :, 0] * W * 0.5
            pred_intrinsic[:, :, 1, 2] = pred_principal_points[:, :, 1] * H * 0.5

            # Convert c2w to w2c for output (to be consistent with CameraDec output)
            output.extrinsics = affine_inverse(pred_c2w)
            output.intrinsics = pred_intrinsic

        return output

    def _extract_auxiliary_features(
        self, feats: List[torch.Tensor], feat_layers: List[int], H: int, W: int
    ) -> AdDict:
        """Extract auxiliary features from specified layers."""
        aux_features = AdDict()
        if len(feats) != len(feat_layers):
            return aux_features

        for feat, feat_layer in zip(feats, feat_layers):
            feat_reshaped = feat.reshape(
                [
                    feat.shape[0],
                    feat.shape[1],
                    H // self.PATCH_SIZE,
                    W // self.PATCH_SIZE,
                    feat.shape[-1],
                ]
            )
            aux_features[f"feat_layer_{feat_layer}"] = feat_reshaped

        return aux_features

    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """Load pretrained weights.

        Args:
            checkpoint_path: Path to pretrained checkpoint (.pth, .pt, or .safetensors)
            strict: Whether to strictly enforce that the keys match
        """
        if checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                # From train.py save_checkpoint
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

        # Remove prefix if present (for checkpoints saved with DDP or wrapper)
        # Handle both 'module.' (DDP) and 'model.' prefixes
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            # Remove 'module.' prefix (from DDP)
            if new_key.startswith('module.'):
                new_key = new_key[7:]
            # Remove 'model.' prefix (from other wrappers)
            if new_key.startswith('model.'):
                new_key = new_key[6:]
            new_state_dict[new_key] = v
        state_dict = new_state_dict

        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        print(f"Loaded weights from {checkpoint_path}")
        if missing:
            print(f"Missing keys ({len(missing)}): {missing[:10]}...")
        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")

        return missing, unexpected


def create_da3_model(
    encoder_name: str = "vitl",
    pretrained: Optional[str] = None,
    **kwargs,
) -> DepthAnything3Net:
    """Create a Depth Anything 3 model.

    Args:
        encoder_name: Encoder name
        pretrained: Path to pretrained weights
        **kwargs: Additional arguments for model

    Returns:
        Initialized model
    """
    model = DepthAnything3Net(encoder_name=encoder_name, **kwargs)

    if pretrained:
        model.load_pretrained(pretrained)

    return model
