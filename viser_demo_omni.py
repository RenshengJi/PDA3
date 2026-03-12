"""
3D Visualization of OmniVGGT inference results using Viser.
Displays point clouds from predicted depth maps in an interactive 3D viewer.
Supports confidence-based filtering and per-frame viewing.
"""
import os
import sys
import argparse
import yaml
import time
from pathlib import Path

# Set environment variables before importing OmniVGGT
os.environ['TORCH_HUB_DIR'] = os.path.expanduser('~/.cache/torch/hub')

import torch
import numpy as np
import cv2
import viser
import viser.transforms as viser_tf
from safetensors.torch import load_file

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import WaymoDataset
from src.dataset.waymo import collate_fn

# Import OmniVGGT
sys.path.insert(0, '/home/ziqi.shi/OmniVGGT-official')

# Monkey-patch normalize_extrinsics to fix the shape bug
# The original code has a bug where extrinsics_homog[:, 0] returns [B, 5, 4] instead of [B, 4, 4]
# We patch the aggregator module before importing OmniVGGT
import omnivggt.models.omnivggt_aggregator as aggregator_module
from omnivggt.utils.geometry import closed_form_inverse_se3

_original_normalize_extrinsics = None

def _patched_normalize_extrinsics(self, extrinsics):
    """Fixed version of normalize_extrinsics that handles the shape correctly."""
    B, S, _, _ = extrinsics.shape
    device = extrinsics.device

    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # Fix: reshape to [B, 4, 4] before passing to closed_form_inverse_se3
    first_cam_extrinsic = extrinsics_homog[:, 0]  # [B, 5, 4]
    first_cam_extrinsic_4x4 = first_cam_extrinsic[:, :4, :4]  # [B, 4, 4]
    first_cam_extrinsic_inv = closed_form_inverse_se3(first_cam_extrinsic_4x4)

    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,S,4,4)

    if S > 1:
        cam_centers = new_extrinsics[:, :, :3, 3]  # (B, S, 3)
        ref_cam = cam_centers[:, 0:1, :]  # (B,1,3)
        rel_distances = torch.norm(cam_centers - ref_cam, dim=-1)[:,1:]  # (B, S-1)
        scale = rel_distances.mean(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1)
        new_extrinsics[:, :, :3, 3] /= scale.unsqueeze(-1)
    return new_extrinsics[:, :, :3]

# Apply the patch
aggregator_module.ZeroAggregator.normalize_extrinsics = _patched_normalize_extrinsics

from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.geometry import unproject_depth_map_to_point_map
from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri


def parse_args():
    parser = argparse.ArgumentParser(description='3D Visualization of OmniVGGT on Waymo')
    parser.add_argument('--config', type=str, default='config/train_waymo.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/OmniVGGT.safetensors',
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='Dataset split to use')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index to visualize')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port for viser server')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--max_points', type=int, default=100000,
                        help='Maximum number of points to display per view')
    parser.add_argument('--point_size', type=float, default=0.002,
                        help='Point size for visualization')
    parser.add_argument('--conf_threshold', type=float, default=50.0,
                        help='Initial confidence percentile threshold (0-100)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with debugpy')
    parser.add_argument('--use_sparse_depth', action='store_true',
                        help='Enable sparse depth input for visualization')
    parser.add_argument('--sparse_depth_ratio', type=float, default=0.1,
                        help='Ratio of depth pixels to keep for sparse depth (0-1)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def depth_to_points_with_conf(depth, conf, intrinsics, conf_threshold_percentile, max_depth, max_height=30, max_points=100000):
    """
    Convert depth map to 3D points with confidence filtering.

    Args:
        depth: (H, W) depth map
        conf: (H, W) confidence map
        intrinsics: (3, 3) camera intrinsics
        conf_threshold_percentile: confidence percentile threshold (0-100)
        max_depth: maximum depth value
        max_height: maximum height value (y coordinate)
        max_points: maximum number of points to return
    """
    H, W = depth.shape

    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)

    conf_flat = conf.flatten()
    valid_conf = conf_flat[conf_flat > 1e-5]
    if len(valid_conf) > 0:
        threshold_val = np.percentile(valid_conf, conf_threshold_percentile)
    else:
        threshold_val = 0

    valid_mask = (depth > 0) & (depth < max_depth) & (y > -max_height) & (conf >= threshold_val) & (conf > 1e-5)

    points = points[valid_mask]
    pixel_coords = np.stack([v[valid_mask], u[valid_mask]], axis=-1)

    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        pixel_coords = pixel_coords[indices]

    return points, pixel_coords


def transform_points(points, extrinsics, is_c2w=True):
    """Transform points from camera to world coordinate.

    Args:
        points: (N, 3) points in camera frame
        extrinsics: (4, 4) or (3, 4) camera pose matrix
        is_c2w: if True, extrinsics is camera-to-world; if False, it's world-to-camera

    Returns:
        (N, 3) points in world frame
    """
    # Handle both (3, 4) and (4, 4) extrinsics
    if extrinsics.shape[0] == 3:
        c2w = np.vstack([extrinsics, np.array([0, 0, 0, 1])])
    else:
        c2w = extrinsics

    if is_c2w:
        # extrinsics is c2w, so directly apply it
        c2w_mat = c2w
    else:
        # extrinsics is w2c (like GT), so invert it
        c2w_mat = np.linalg.inv(c2w)

    R = c2w_mat[:3, :3]
    t = c2w_mat[:3, 3]
    points_world = points @ R.T + t
    return points_world


class Visualizer:
    def __init__(self, config, checkpoint_path, device='cuda'):
        self.config = config
        self.device = torch.device(device)
        self.setup_model(checkpoint_path)

    def setup_model(self, checkpoint_path):
        """Setup model and load checkpoint."""
        print(f"Loading OmniVGGT model from {checkpoint_path}...")

        # Monkey patch torch.hub.load to skip downloading DINOv2
        # since we'll load weights from checkpoint anyway
        original_hub_load = torch.hub.load

        def patched_hub_load(*args, **kwargs):
            print(f"Skipping torch.hub.load for {args}")
            # Return a dummy module that we'll overwrite with checkpoint weights
            class DummyModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                def state_dict(self):
                    return {}
            return DummyModule()

        torch.hub.load = patched_hub_load

        try:
            self.model = OmniVGGT()
            state_dict = load_file(checkpoint_path)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device).eval()
        finally:
            # Restore original hub.load
            torch.hub.load = original_hub_load

        print(f"Model loaded successfully")

    def setup_dataloader(self, split='val', use_sparse_depth=False, sparse_depth_ratio=0.1):
        """Setup dataloader."""
        data_config = self.config.get('data', {})

        if split == 'val':
            root = data_config.get('val_root')
        else:
            root = data_config.get('train_root')

        sparse_depth_prob = 1.0 if use_sparse_depth else 0.0

        dataset = WaymoDataset(
            root=root,
            valid_camera_id_list=data_config.get('camera_ids', ["1", "2", "3"]),
            intervals=[2],
            num_views=data_config.get('num_views', 4),
            resolution=data_config.get('resolution', 518),
            split=split,
            sparse_depth_prob=sparse_depth_prob,
            sparse_depth_keep_ratio=sparse_depth_ratio,
        )

        return dataset

    @torch.no_grad()
    def infer_sample(self, sample):
        """Run inference on a single sample."""
        batch = collate_fn([sample])
        images = batch['images'].cpu().numpy()  # (B, S, 3, H, W), range [0, 1]
        extrinsics = batch['extrinsics'].cpu().numpy()  # (B, S, 3, 4) or (B, S, 4, 4)
        intrinsics = batch['intrinsics'].cpu().numpy()  # (B, S, 3, 3)

        B, S = images.shape[:2]

        # Convert extrinsics from (3, 4) to (4, 4) if needed
        if extrinsics.shape[-2:] == (3, 4):
            extrinsics_4x4 = np.zeros((B, S, 4, 4))
            extrinsics_4x4[:, :, :3, :] = extrinsics
            extrinsics_4x4[:, :, 3, 3] = 1
            extrinsics = extrinsics_4x4

        # Prepare inputs for OmniVGGT
        # OmniVGGT expects: images, extrinsics, intrinsics, depth (optional), mask (optional)
        inputs = {
            'images': batch['images'].to(self.device),  # (B, S, 3, H, W)
            'extrinsics': torch.from_numpy(extrinsics).float().to(self.device),  # (B, S, 4, 4)
            'intrinsics': batch['intrinsics'].to(self.device),  # (B, S, 3, 3)
        }

        # Add sparse depth if available, with metric scale conversion
        if 'sparse_depths' in batch and batch['sparse_depths'] is not None:
            sparse_depths = batch['sparse_depths']  # [B, S, H, W]
            depth_scale_factor = batch.get('depth_scale_factor', None)

            # Convert to metric scale (dataset normalizes depth by 1/avg_dist)
            if depth_scale_factor is not None:
                metric_scale = 1.0 / depth_scale_factor.item() if torch.is_tensor(depth_scale_factor) else 1.0 / depth_scale_factor
                sparse_depths = sparse_depths * metric_scale

            # Only pass depth if at least some frames have valid (non-zero) sparse depth
            if sparse_depths.max() > 0:
                # Add channel dimension to match expected shape [B, S, H, W, 1]
                inputs['depth'] = sparse_depths.unsqueeze(-1).to(self.device)

        # Add masks from batch if available, otherwise create dummy masks
        if 'masks' in batch:
            inputs['mask'] = batch['masks'].to(self.device)
        elif 'point_masks' in batch:
            inputs['mask'] = batch['point_masks'].to(self.device)
        else:
            # Create dummy masks (all ones) with correct shape [B, S, H, W]
            H, W = images.shape[-2:]
            inputs['mask'] = torch.ones(B, S, H, W, device=self.device)

        # Add indices of frames with valid depth and camera data
        # Only provide indices if we have sparse depth input
        if 'depth' in inputs:
            inputs['depth_gt_index'] = list(range(S))
            inputs['camera_gt_index'] = list(range(S))
        else:
            # No sparse depth, so don't use GT indices (model will still run inference)
            inputs['depth_gt_index'] = []
            inputs['camera_gt_index'] = []

        # Run inference
        print(f"Running OmniVGGT inference on {S} views...")
        outputs = self.model.inference(**inputs)

        # Extract and convert outputs
        pred_depths = outputs['depth'].cpu().numpy()  # (B, S, H, W, 1)
        if pred_depths.ndim == 5:
            pred_depths = pred_depths[:, :, :, :, 0]  # (B, S, H, W)

        depth_confs = outputs['depth_conf'].cpu().numpy() if 'depth_conf' in outputs else np.ones_like(pred_depths)
        if depth_confs.ndim == 5:
            depth_confs = depth_confs[:, :, :, :, 0]
        elif depth_confs.ndim == 4 and depth_confs.shape[1] == 1:
            depth_confs = depth_confs[:, 0]  # Remove extra dimension

        # Get extrinsics and intrinsics from output
        if 'extrinsic' in outputs:
            pred_extrinsics = outputs['extrinsic'].cpu().numpy()  # (B, S, 3, 4)
            # Convert (3, 4) to (4, 4) format
            if pred_extrinsics.shape[-2:] == (3, 4):
                B_out, S_out = pred_extrinsics.shape[:2]
                pred_extrinsics_4x4 = np.zeros((B_out, S_out, 4, 4))
                pred_extrinsics_4x4[:, :, :3, :] = pred_extrinsics
                pred_extrinsics_4x4[:, :, 3, 3] = 1
                pred_extrinsics = pred_extrinsics_4x4
        else:
            pred_extrinsics = extrinsics

        if 'intrinsic' in outputs:
            pred_intrinsics = outputs['intrinsic'].cpu().numpy()  # (B, S, 3, 3)
        else:
            pred_intrinsics = intrinsics

        # Get sparse depths if available
        sparse_depths_for_output = batch.get('sparse_depths', None)
        gt_depths_for_output = batch.get('depths', None)

        return {
            'pred_depth': pred_depths[0],  # (S, H, W)
            'depth_conf': depth_confs[0] if depth_confs.ndim > 3 else depth_confs,  # (S, H, W)
            'images': images[0],  # (S, 3, H, W)
            'extrinsics': extrinsics[0],  # (S, 4, 4) - w2c format (GT)
            'intrinsics': intrinsics[0],  # (S, 3, 3)
            'pred_extrinsics': pred_extrinsics[0],  # (S, 4, 4) - w2c format (OmniVGGT output)
            'pred_extrinsics_is_c2w': False,  # OmniVGGT outputs w2c format, NOT c2w
            'pred_intrinsics': pred_intrinsics[0],  # (S, 3, 3)
            'sparse_depths': sparse_depths_for_output.cpu().numpy()[0] if sparse_depths_for_output is not None else None,
            'gt_depths': gt_depths_for_output,  # GT depths for visualization
        }


def main():
    args = parse_args()

    # debug
    if args.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")

    config = load_config(args.config)

    visualizer = Visualizer(
        config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    dataset = visualizer.setup_dataloader(
        args.split,
        use_sparse_depth=args.use_sparse_depth,
        sparse_depth_ratio=args.sparse_depth_ratio,
    )
    print(f"Dataset size: {len(dataset)}")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"\nViser server started at http://localhost:{args.port}")
    print("Open this URL in your browser to view the 3D visualization\n")

    # State
    current_sample_idx = args.sample_idx
    cached_results = None
    num_frames = 4  # Will be updated after inference

    # Add GUI controls
    with server.gui.add_folder("Controls"):
        sample_slider = server.gui.add_slider(
            "Sample Index",
            min=0,
            max=len(dataset) - 1,
            step=1,
            initial_value=args.sample_idx,
        )
        # Frame selector dropdown - will be updated after inference
        frame_selector = server.gui.add_dropdown(
            "Show Frames",
            options=["All"] + [str(i) for i in range(num_frames)],
            initial_value="All",
        )
        show_cameras = server.gui.add_checkbox("Show Cameras", initial_value=True)
        point_size_slider = server.gui.add_slider(
            "Point Size",
            min=0.001,
            max=0.1,
            step=0.001,
            initial_value=args.point_size,
        )
        max_depth_slider = server.gui.add_slider(
            "Max Depth",
            min=10,
            max=200,
            step=5,
            initial_value=80,
        )
        max_height_slider = server.gui.add_slider(
            "Max Height",
            min=1,
            max=100,
            step=1,
            initial_value=30,
        )
        conf_threshold_slider = server.gui.add_slider(
            "Confidence Percent",
            min=0,
            max=100,
            step=1,
            initial_value=args.conf_threshold,
        )
        # Sparse depth controls
        sparse_depth_ratio_slider = server.gui.add_slider(
            "Sparse Depth Ratio",
            min=0.0,
            max=1.0,
            step=0.05,
            initial_value=args.sparse_depth_ratio,
        ) if args.use_sparse_depth else None
        show_sparse_depth = server.gui.add_checkbox(
            "Show Sparse Depth Points",
            initial_value=False,
        ) if args.use_sparse_depth else None
        show_gt_depth = server.gui.add_checkbox(
            "Show GT Depth Points",
            initial_value=False,
        )
        refresh_button = server.gui.add_button("Refresh")

    # Store camera frustum handles for visibility toggle
    camera_handles = []
    frame_handles = []

    def run_inference():
        """Run inference on current sample and cache results."""
        nonlocal cached_results, current_sample_idx, num_frames

        sample = dataset[current_sample_idx]
        print(f"Running inference on sample {current_sample_idx}...")

        # Update sparse depth ratio if sparse depth is enabled
        if args.use_sparse_depth:
            if sparse_depth_ratio_slider is not None and 'depths' in sample:
                # Regenerate sparse depth with current ratio
                sparse_ratio = sparse_depth_ratio_slider.value
                sample['sparse_depths'] = torch.stack([
                    dataset._sparsify_depth(d, sparse_ratio) for d in sample['depths']
                ], dim=0)
                sample['use_sparse_depth'] = sparse_ratio > 0
                print(f"Sparse depth ratio: {sparse_ratio}")

        cached_results = visualizer.infer_sample(sample)
        num_frames = cached_results['pred_depth'].shape[0]

        # Update frame selector options
        frame_selector.options = ["All"] + [str(i) for i in range(num_frames)]

        print(f"Inference completed for sample {current_sample_idx} ({num_frames} frames)")

    def update_visualization():
        """Update the 3D visualization using cached results."""
        nonlocal cached_results, camera_handles, frame_handles

        if cached_results is None:
            return

        # Clear previous objects
        server.scene.reset()
        camera_handles.clear()
        frame_handles.clear()

        # Add world frame
        server.scene.add_frame(
            "world",
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=np.array([0.0, 0.0, 0.0]),
            axes_length=1.0
        )

        pred_depth = cached_results['pred_depth']
        depth_conf = cached_results['depth_conf']
        images = cached_results['images']

        # Use predicted camera parameters (model output)
        pred_extrinsics = cached_results['pred_extrinsics']
        pred_intrinsics = cached_results['pred_intrinsics']
        pred_extrinsics_is_c2w = cached_results.get('pred_extrinsics_is_c2w', False)

        # Fallback to GT if prediction not available
        if pred_extrinsics is None:
            print("Warning: pred_extrinsics not available, using GT")
            pred_extrinsics = cached_results['extrinsics']
            pred_extrinsics_is_c2w = False  # GT is w2c
        if pred_intrinsics is None:
            print("Warning: pred_intrinsics not available, using GT")
            pred_intrinsics = cached_results['intrinsics']

        extrinsics = pred_extrinsics
        intrinsics = pred_intrinsics

        S = pred_depth.shape[0]
        max_depth = max_depth_slider.value
        max_height = max_height_slider.value
        conf_threshold = conf_threshold_slider.value

        # Determine which frames to show
        selected_frame = frame_selector.value
        if selected_frame == "All":
            frames_to_show = list(range(S))
        else:
            frames_to_show = [int(selected_frame)]

        view_colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
        ]

        for s in frames_to_show:
            view_color = view_colors[s % len(view_colors)]

            img = images[s].transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)

            H_d, W_d = pred_depth[s].shape
            H_i, W_i = img.shape[:2]

            scale_x = W_d / W_i
            scale_y = H_d / H_i

            intrinsics_scaled = intrinsics[s].copy()
            intrinsics_scaled[0, :] *= scale_x
            intrinsics_scaled[1, :] *= scale_y

            img_resized = cv2.resize(img, (W_d, H_d))

            if depth_conf is not None:
                conf = depth_conf[s] if depth_conf.ndim > 2 else depth_conf
            else:
                conf = np.ones_like(pred_depth[s])

            pred_points, pred_pixel_coords = depth_to_points_with_conf(
                pred_depth[s],
                conf,
                intrinsics_scaled,
                conf_threshold,
                max_depth,
                max_height,
                max_points=args.max_points
            )

            if len(pred_points) > 0:
                pred_points_world = transform_points(
                    pred_points,
                    extrinsics[s],
                    is_c2w=pred_extrinsics_is_c2w
                )
                pred_colors = img_resized[pred_pixel_coords[:, 0], pred_pixel_coords[:, 1]]

                server.scene.add_point_cloud(
                    f"pred_view_{s}",
                    points=pred_points_world.astype(np.float32),
                    colors=pred_colors.astype(np.uint8),
                    point_size=point_size_slider.value,
                    point_shape="circle",
                )

            # Add camera visualization using predicted extrinsics
            if show_cameras.value:
                if pred_extrinsics_is_c2w:
                    # extrinsics is c2w, use directly
                    c2w = extrinsics[s]
                else:
                    # extrinsics is w2c, invert it
                    c2w = np.linalg.inv(extrinsics[s])

                # Handle both (3, 4) and (4, 4) formats
                if c2w.shape[0] == 3:
                    c2w_3x4 = c2w
                else:
                    c2w_3x4 = c2w[:3, :]

                T_world_camera = viser_tf.SE3.from_matrix(c2w_3x4)

                # Add frame axis
                frame_handle = server.scene.add_frame(
                    f"camera_frame_{s}",
                    wxyz=T_world_camera.rotation().wxyz,
                    position=T_world_camera.translation(),
                    axes_length=0.3,
                    axes_radius=0.01,
                )
                frame_handles.append(frame_handle)

                # Add camera frustum with image
                h, w = img_resized.shape[:2]
                fy = intrinsics_scaled[1, 1]
                fov = float(2 * np.arctan2(h / 2, fy))

                frustum_handle = server.scene.add_camera_frustum(
                    f"camera_frustum_{s}",
                    fov=fov,
                    aspect=float(w / h),
                    scale=0.3,
                    image=img_resized,
                    wxyz=T_world_camera.rotation().wxyz,
                    position=T_world_camera.translation(),
                    color=view_color,
                )
                camera_handles.append(frustum_handle)

            # Visualize sparse depth if available and enabled
            if (show_sparse_depth is not None and show_sparse_depth.value and
                cached_results['sparse_depths'] is not None):
                sparse_depth = cached_results['sparse_depths'][s]
                if sparse_depth.shape[0] == 1:  # [1, H, W] -> [H, W]
                    sparse_depth = sparse_depth[0]

                # Convert sparse depth to 3D points
                if sparse_depth.max() > 0:
                    sparse_points, sparse_pixel_coords = depth_to_points_with_conf(
                        sparse_depth,
                        np.ones_like(sparse_depth),  # All valid pixels equally weighted
                        intrinsics_scaled,
                        conf_threshold_percentile=0,  # Show all sparse points
                        max_depth=max_depth,
                        max_height=max_height,
                        max_points=args.max_points
                    )

                    if len(sparse_points) > 0:
                        sparse_points_world = transform_points(sparse_points, extrinsics[s], is_c2w=pred_extrinsics_is_c2w)
                        sparse_colors = img_resized[sparse_pixel_coords[:, 0], sparse_pixel_coords[:, 1]]

                        server.scene.add_point_cloud(
                            f"sparse_depth_view_{s}",
                            points=sparse_points_world.astype(np.float32),
                            colors=sparse_colors.astype(np.uint8),
                            point_size=point_size_slider.value * 1.5,  # Slightly larger
                            point_shape="circle",
                        )

            # Visualize GT depth if available and enabled
            if show_gt_depth.value and cached_results.get('gt_depths') is not None:
                gt_depths = cached_results['gt_depths']
                gt_depths = gt_depths.cpu().numpy() if torch.is_tensor(gt_depths) else gt_depths

                # Handle different shapes
                if gt_depths.ndim == 5:  # (B, S, 1, H, W)
                    gt_depth = gt_depths[0, s, 0]
                elif gt_depths.ndim == 4:  # (B, S, H, W) or (S, 1, H, W)
                    if gt_depths.shape[0] == 1:  # (1, S, H, W)
                        gt_depth = gt_depths[0, s]
                    else:  # (B, S, H, W)
                        gt_depth = gt_depths[0, s]
                else:  # (S, H, W)
                    gt_depth = gt_depths[s]

                if gt_depth.max() > 0:
                    gt_points, gt_pixel_coords = depth_to_points_with_conf(
                        gt_depth,
                        np.ones_like(gt_depth),
                        intrinsics_scaled,
                        conf_threshold_percentile=0,
                        max_depth=max_depth,
                        max_height=max_height,
                        max_points=args.max_points
                    )

                    if len(gt_points) > 0:
                        # GT uses w2c format
                        gt_points_world = transform_points(gt_points, extrinsics[s], is_c2w=False)
                        # Color GT points with pure red (255, 0, 0)
                        gt_colors = np.tile(np.array([255, 0, 0], dtype=np.uint8), (len(gt_points), 1))

                        server.scene.add_point_cloud(
                            f"gt_depth_view_{s}",
                            points=gt_points_world.astype(np.float32),
                            colors=gt_colors.astype(np.uint8),
                            point_size=point_size_slider.value,
                            point_shape="circle",
                        )

        print(f"Visualization updated (frames: {selected_frame}, conf: {conf_threshold}%)")

    def update_camera_visibility():
        """Toggle camera visibility without full re-render."""
        for handle in camera_handles:
            handle.visible = show_cameras.value
        for handle in frame_handles:
            handle.visible = show_cameras.value

    # Initial inference and visualization
    run_inference()
    update_visualization()

    # Handle GUI events
    @sample_slider.on_update
    def _(_):
        nonlocal current_sample_idx
        current_sample_idx = int(sample_slider.value)
        run_inference()
        update_visualization()

    @refresh_button.on_click
    def _(_):
        run_inference()
        update_visualization()

    @frame_selector.on_update
    def _(_):
        update_visualization()

    @show_cameras.on_update
    def _(_):
        update_visualization()

    @point_size_slider.on_update
    def _(_):
        update_visualization()

    @max_depth_slider.on_update
    def _(_):
        update_visualization()

    @max_height_slider.on_update
    def _(_):
        update_visualization()

    @conf_threshold_slider.on_update
    def _(_):
        update_visualization()

    # Sparse depth controls
    if args.use_sparse_depth and sparse_depth_ratio_slider is not None:
        @sparse_depth_ratio_slider.on_update
        def _(_):
            run_inference()
            update_visualization()

    if args.use_sparse_depth and show_sparse_depth is not None:
        @show_sparse_depth.on_update
        def _(_):
            update_visualization()

    @show_gt_depth.on_update
    def _(_):
        update_visualization()

    # Keep server running
    print("Press Ctrl+C to stop the server")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
