"""
3D Visualization for comparing GT depth, teacher depth (no align), and teacher depth (aligned).
Uses real teacher model (DepthAnything3Net) for inference.
Uses Viser to display point clouds side-by-side for alignment quality inspection.
"""
import os
import sys
import argparse
import yaml
import time
from pathlib import Path

import torch
import numpy as np
import cv2
import viser
import viser.transforms as viser_tf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import DepthAnything3Net
from src.losses.loss import align_depth_least_squares
from src.dataset import WaymoDataset
from src.dataset.waymo import collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description='3D Visualization for Depth Alignment Comparison')
    parser.add_argument('--config', type=str, default='config/train_waymo.yaml',
                        help='Path to config file (teacher model config is defined here)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                        help='Dataset split to use')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index to visualize')
    parser.add_argument('--port', type=int, default=8081,
                        help='Port for viser server')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--max_points', type=int, default=50000,
                        help='Maximum number of points to display per view')
    parser.add_argument('--point_size', type=float, default=0.003,
                        help='Point size for visualization')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with debugpy')
    # Alignment parameters
    parser.add_argument('--with_shift', action='store_true', default=True,
                        help='Use affine alignment (scale+shift) instead of scale-only')
    parser.add_argument('--use_ransac', action='store_true', default=True,
                        help='Use RANSAC for robust alignment')
    parser.add_argument('--ransac_iters', type=int, default=100,
                        help='Number of RANSAC iterations')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def depth_to_points(depth, intrinsics, max_depth=80.0, max_points=50000, valid_mask=None):
    """
    Convert depth map to 3D points.

    Args:
        depth: [H, W] depth map
        intrinsics: [3, 3] camera intrinsics
        max_depth: Maximum depth threshold
        max_points: Maximum number of points to return
        valid_mask: Optional [H, W] mask for valid pixels

    Returns:
        points: [N, 3] 3D points in camera coordinates
        pixel_coords: [N, 2] corresponding pixel coordinates (v, u)
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

    # Create validity mask
    mask = (depth > 0.1) & (depth < max_depth) & np.isfinite(depth)
    if valid_mask is not None:
        mask = mask & valid_mask

    points = points[mask]
    pixel_coords = np.stack([v[mask], u[mask]], axis=-1)

    # Subsample if too many points
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        pixel_coords = pixel_coords[indices]

    return points, pixel_coords


def transform_points(points, extrinsics):
    """Transform points from camera to world coordinates."""
    c2w = np.linalg.inv(extrinsics)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    points_world = points @ R.T + t
    return points_world


class AlignmentVisualizer:
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = torch.device(device)
        self.setup_teacher_model()

    def setup_teacher_model(self):
        """Setup teacher model and load checkpoint from config."""
        teacher_config = self.config.get('teacher', {})
        model_config = self.config.get('model', {})

        # Create teacher model with teacher-specific architecture from config
        self.teacher_model = DepthAnything3Net(
            encoder_name=teacher_config.get('encoder_name', model_config.get('encoder_name', 'vitl')),
            out_layers=teacher_config.get('out_layers', model_config.get('out_layers', [11, 15, 19, 23])),
            features=teacher_config.get('features', model_config.get('features', 256)),
            out_channels=teacher_config.get('out_channels', model_config.get('out_channels', [256, 512, 1024, 1024])),
            alt_start=teacher_config.get('alt_start', model_config.get('alt_start', 8)),
            qknorm_start=teacher_config.get('qknorm_start', model_config.get('qknorm_start', 8)),
            rope_start=teacher_config.get('rope_start', model_config.get('rope_start', 8)),
            predict_camera=False,  # Teacher doesn't need camera prediction
            use_camera_enc=False,  # Teacher doesn't use camera encoding
        )

        # Load teacher weights from config
        teacher_checkpoint = teacher_config.get('checkpoint', model_config.get('pretrained', None))
        if teacher_checkpoint and os.path.exists(teacher_checkpoint):
            self.teacher_model.load_pretrained(teacher_checkpoint)
            print(f"Loaded teacher checkpoint from {teacher_checkpoint}")
        else:
            print(f"Warning: Teacher checkpoint not found at {teacher_checkpoint}")
            print("Using randomly initialized teacher model (for demo purposes)")

        self.teacher_model = self.teacher_model.to(self.device)
        self.teacher_model.eval()

        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def setup_dataloader(self, split='train'):
        """Setup dataloader."""
        data_config = self.config.get('data', {})

        if split == 'val':
            root = data_config.get('val_root')
        else:
            root = data_config.get('train_root')

        dataset = WaymoDataset(
            root=root,
            valid_camera_id_list=data_config.get('camera_ids', ["1", "2", "3"]),
            intervals=[2],
            num_views=data_config.get('num_views', 1),
            resolution=data_config.get('resolution', 518),
            split=split,
        )

        return dataset

    @torch.no_grad()
    def get_teacher_depth(self, images):
        """
        Get teacher model depth predictions.

        Args:
            images: [B, S, C, H, W] input images tensor

        Returns:
            teacher_depth: [B, S, H, W] teacher depth predictions
        """
        images = images.to(self.device)
        teacher_outputs = self.teacher_model(images)
        teacher_depth = teacher_outputs['depth']

        # Handle dimension: ensure [B, S, H, W]
        if teacher_depth.dim() == 5:
            teacher_depth = teacher_depth.squeeze(-1)

        return teacher_depth


def main():
    args = parse_args()

    # Debug mode
    if args.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")

    config = load_config(args.config)

    visualizer = AlignmentVisualizer(
        config,
        device=args.device
    )
    dataset = visualizer.setup_dataloader(args.split)
    print(f"Dataset size: {len(dataset)}")

    # Start viser server
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"\nViser server started at http://localhost:{args.port}")
    print("Open this URL in your browser to view the 3D visualization\n")

    # State
    current_sample_idx = args.sample_idx
    cached_data = None

    # GUI Controls
    with server.gui.add_folder("Sample Control"):
        sample_slider = server.gui.add_slider(
            "Sample Index",
            min=0,
            max=len(dataset) - 1,
            step=1,
            initial_value=args.sample_idx,
        )
        frame_selector = server.gui.add_dropdown(
            "Frame",
            options=["0", "1", "2", "3"],
            initial_value="0",
        )
        refresh_button = server.gui.add_button("Refresh")

    with server.gui.add_folder("Display Options"):
        show_gt = server.gui.add_checkbox("Show GT (Red)", initial_value=True)
        show_teacher_raw = server.gui.add_checkbox("Show Teacher Raw (Green)", initial_value=False)  # Default off since scale differs
        show_teacher_aligned = server.gui.add_checkbox("Show Teacher Aligned (Blue)", initial_value=True)
        overlay_mode = server.gui.add_checkbox("Overlay Mode (no offset)", initial_value=True)  # Overlay by default
        point_size_slider = server.gui.add_slider(
            "Point Size",
            min=0.001,
            max=0.02,
            step=0.001,
            initial_value=args.point_size,
        )
        max_depth_slider = server.gui.add_slider(
            "Max Depth",
            min=10,
            max=150,
            step=5,
            initial_value=80,
        )
        x_offset_slider = server.gui.add_slider(
            "X Offset (when not overlay)",
            min=0,
            max=50,
            step=1,
            initial_value=0,
        )

    with server.gui.add_folder("Alignment Parameters"):
        with_shift_checkbox = server.gui.add_checkbox("With Shift (Affine)", initial_value=args.with_shift)
        use_ransac_checkbox = server.gui.add_checkbox("Use RANSAC", initial_value=args.use_ransac)
        ransac_iters_slider = server.gui.add_slider(
            "RANSAC Iterations",
            min=10,
            max=500,
            step=10,
            initial_value=args.ransac_iters,
        )

    # Info display
    with server.gui.add_folder("Alignment Info"):
        scale_text = server.gui.add_text("Scale", initial_value="N/A")
        shift_text = server.gui.add_text("Shift", initial_value="N/A")
        gt_depth_range_text = server.gui.add_text("GT Depth Range", initial_value="N/A")
        teacher_depth_range_text = server.gui.add_text("Teacher Depth Range", initial_value="N/A")
        aligned_depth_range_text = server.gui.add_text("Aligned Depth Range", initial_value="N/A")

    def load_sample():
        """Load sample data and run teacher inference."""
        nonlocal cached_data

        sample = dataset[current_sample_idx]
        batch = collate_fn([sample])

        # Get GT data
        images = batch['images']  # [1, S, 3, H, W]
        gt_depths = batch['depths']  # [1, S, H, W]

        # Run teacher model inference
        print(f"Running teacher inference on sample {current_sample_idx}...")
        teacher_depth = visualizer.get_teacher_depth(images)  # [1, S, H, W]

        cached_data = {
            'images': images.numpy()[0],  # [S, 3, H, W]
            'depths': gt_depths.numpy()[0],  # [S, H, W]
            'teacher_depth': teacher_depth.cpu().numpy()[0],  # [S, H, W]
            'teacher_depth_tensor': teacher_depth,  # Keep tensor for alignment
            'gt_depth_tensor': gt_depths,  # Keep tensor for alignment
            'extrinsics': batch['extrinsics'].numpy()[0],  # [S, 4, 4]
            'intrinsics': batch['intrinsics'].numpy()[0],  # [S, 3, 3]
            'point_masks': batch['point_masks'].numpy()[0] if 'point_masks' in batch else None,
            'point_masks_tensor': batch['point_masks'] if 'point_masks' in batch else None,
        }

        num_frames = cached_data['depths'].shape[0]
        frame_selector.options = [str(i) for i in range(num_frames)]

        print(f"Loaded sample {current_sample_idx} with {num_frames} frames")
        print(f"  GT depth shape: {cached_data['depths'].shape}")
        print(f"  Teacher depth shape: {cached_data['teacher_depth'].shape}")

    def update_visualization():
        """Update the 3D visualization."""
        if cached_data is None:
            return

        server.scene.reset()

        frame_idx = int(frame_selector.value)
        max_depth = max_depth_slider.value
        x_offset = x_offset_slider.value

        # Get data for selected frame
        gt_depth = cached_data['depths'][frame_idx]  # [H, W]
        teacher_depth_raw = cached_data['teacher_depth'][frame_idx]  # [H, W]
        image = cached_data['images'][frame_idx].transpose(1, 2, 0)  # [H, W, 3]
        image = (image * 255).astype(np.uint8)
        extrinsics = cached_data['extrinsics'][frame_idx]  # [4, 4]
        intrinsics = cached_data['intrinsics'][frame_idx]  # [3, 3]
        point_mask = cached_data['point_masks'][frame_idx] if cached_data['point_masks'] is not None else None

        # Scale intrinsics to match depth resolution
        H_d, W_d = gt_depth.shape
        H_i, W_i = image.shape[:2]
        scale_x = W_d / W_i
        scale_y = H_d / H_i
        intrinsics_scaled = intrinsics.copy()
        intrinsics_scaled[0, :] *= scale_x
        intrinsics_scaled[1, :] *= scale_y

        # Resize image to match depth
        image_resized = cv2.resize(image, (W_d, H_d))

        # Align teacher depth to GT using least squares with RANSAC
        # Use tensors for alignment function - ensure all on same device (CPU for visualization)
        gt_depth_tensor = cached_data['gt_depth_tensor'][:, frame_idx:frame_idx+1].cpu()  # [1, 1, H, W]
        teacher_depth_tensor = cached_data['teacher_depth_tensor'][:, frame_idx:frame_idx+1].cpu()  # [1, 1, H, W]
        valid_mask_tensor = None
        if cached_data['point_masks_tensor'] is not None:
            valid_mask_tensor = cached_data['point_masks_tensor'][:, frame_idx:frame_idx+1].bool().cpu()

        aligned_depth_tensor, scale, shift = align_depth_least_squares(
            teacher_depth_tensor,
            gt_depth_tensor,
            valid_mask=valid_mask_tensor,
            with_shift=with_shift_checkbox.value,
            use_ransac=use_ransac_checkbox.value,
            ransac_iters=int(ransac_iters_slider.value),
        )

        teacher_depth_aligned = aligned_depth_tensor.squeeze().cpu().numpy()

        # Update info display
        scale_val = scale.item() if hasattr(scale, 'item') else float(scale)
        shift_val = shift.item() if hasattr(shift, 'item') else float(shift)
        scale_text.value = f"{scale_val:.4f}"
        shift_text.value = f"{shift_val:.4f}"

        # Compute depth ranges for valid regions
        valid_gt = gt_depth[point_mask] if point_mask is not None else gt_depth[gt_depth > 0.1]
        valid_teacher = teacher_depth_raw[point_mask] if point_mask is not None else teacher_depth_raw[teacher_depth_raw > 0.01]
        valid_aligned = teacher_depth_aligned[point_mask] if point_mask is not None else teacher_depth_aligned[teacher_depth_aligned > 0.1]

        gt_depth_range_text.value = f"[{valid_gt.min():.2f}, {valid_gt.max():.2f}]"
        teacher_depth_range_text.value = f"[{valid_teacher.min():.2f}, {valid_teacher.max():.2f}]"
        aligned_depth_range_text.value = f"[{valid_aligned.min():.2f}, {valid_aligned.max():.2f}]"

        print(f"Alignment: scale={scale_val:.4f}, shift={shift_val:.4f}")
        print(f"  GT depth range: [{valid_gt.min():.2f}, {valid_gt.max():.2f}]")
        print(f"  Teacher depth range: [{valid_teacher.min():.2f}, {valid_teacher.max():.2f}]")
        print(f"  Aligned depth range: [{valid_aligned.min():.2f}, {valid_aligned.max():.2f}]")

        # Add world frame
        server.scene.add_frame(
            "world",
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=np.array([0.0, 0.0, 0.0]),
            axes_length=2.0
        )

        # Add labels
        server.scene.add_label("label_gt", "GT Depth", position=np.array([-x_offset, 0, 5]))
        server.scene.add_label("label_raw", "Teacher (Raw)", position=np.array([0, 0, 5]))
        server.scene.add_label("label_aligned", "Teacher (Aligned)", position=np.array([x_offset, 0, 5]))

        point_size = point_size_slider.value

        # 1. GT depth point cloud (RED) - offset left
        if show_gt.value:
            gt_points, gt_pixel_coords = depth_to_points(
                gt_depth, intrinsics_scaled, max_depth, args.max_points, point_mask
            )
            if len(gt_points) > 0:
                gt_points_world = transform_points(gt_points, extrinsics)
                gt_points_world[:, 0] -= x_offset  # Offset left
                gt_colors = image_resized[gt_pixel_coords[:, 0], gt_pixel_coords[:, 1]]
                # Tint red
                gt_colors = gt_colors.astype(np.float32)
                gt_colors[:, 0] = np.minimum(gt_colors[:, 0] * 1.5, 255)
                gt_colors[:, 1] = gt_colors[:, 1] * 0.5
                gt_colors[:, 2] = gt_colors[:, 2] * 0.5
                gt_colors = gt_colors.astype(np.uint8)

                server.scene.add_point_cloud(
                    "gt_depth",
                    points=gt_points_world.astype(np.float32),
                    colors=gt_colors,
                    point_size=point_size,
                    point_shape="circle",
                )

        # 2. Teacher depth (raw, no alignment) point cloud (GREEN) - center
        if show_teacher_raw.value:
            # Use larger max_depth for raw teacher since it's not aligned
            raw_max_depth = max_depth * 3  # Teacher depth might have very different scale
            raw_points, raw_pixel_coords = depth_to_points(
                teacher_depth_raw, intrinsics_scaled, raw_max_depth, args.max_points, point_mask
            )
            if len(raw_points) > 0:
                raw_points_world = transform_points(raw_points, extrinsics)
                # No offset - center
                raw_colors = image_resized[raw_pixel_coords[:, 0], raw_pixel_coords[:, 1]]
                # Tint green
                raw_colors = raw_colors.astype(np.float32)
                raw_colors[:, 0] = raw_colors[:, 0] * 0.5
                raw_colors[:, 1] = np.minimum(raw_colors[:, 1] * 1.5, 255)
                raw_colors[:, 2] = raw_colors[:, 2] * 0.5
                raw_colors = raw_colors.astype(np.uint8)

                server.scene.add_point_cloud(
                    "teacher_raw",
                    points=raw_points_world.astype(np.float32),
                    colors=raw_colors,
                    point_size=point_size,
                    point_shape="circle",
                )

        # 3. Teacher depth (aligned) point cloud (BLUE) - offset right
        if show_teacher_aligned.value:
            aligned_points, aligned_pixel_coords = depth_to_points(
                teacher_depth_aligned, intrinsics_scaled, max_depth, args.max_points, point_mask
            )
            if len(aligned_points) > 0:
                aligned_points_world = transform_points(aligned_points, extrinsics)
                aligned_points_world[:, 0] += x_offset  # Offset right
                aligned_colors = image_resized[aligned_pixel_coords[:, 0], aligned_pixel_coords[:, 1]]
                # Tint blue
                aligned_colors = aligned_colors.astype(np.float32)
                aligned_colors[:, 0] = aligned_colors[:, 0] * 0.5
                aligned_colors[:, 1] = aligned_colors[:, 1] * 0.5
                aligned_colors[:, 2] = np.minimum(aligned_colors[:, 2] * 1.5, 255)
                aligned_colors = aligned_colors.astype(np.uint8)

                server.scene.add_point_cloud(
                    "teacher_aligned",
                    points=aligned_points_world.astype(np.float32),
                    colors=aligned_colors,
                    point_size=point_size,
                    point_shape="circle",
                )

        print(f"Visualization updated for frame {frame_idx}")

    # Initial load and visualization
    load_sample()
    update_visualization()

    # GUI event handlers
    @sample_slider.on_update
    def _(_):
        nonlocal current_sample_idx
        current_sample_idx = int(sample_slider.value)
        load_sample()
        update_visualization()

    @refresh_button.on_click
    def _(_):
        load_sample()
        update_visualization()

    @frame_selector.on_update
    def _(_):
        update_visualization()

    @show_gt.on_update
    def _(_):
        update_visualization()

    @show_teacher_raw.on_update
    def _(_):
        update_visualization()

    @show_teacher_aligned.on_update
    def _(_):
        update_visualization()

    @point_size_slider.on_update
    def _(_):
        update_visualization()

    @max_depth_slider.on_update
    def _(_):
        update_visualization()

    @x_offset_slider.on_update
    def _(_):
        update_visualization()

    @with_shift_checkbox.on_update
    def _(_):
        update_visualization()

    @use_ransac_checkbox.on_update
    def _(_):
        update_visualization()

    @ransac_iters_slider.on_update
    def _(_):
        update_visualization()

    # Keep server running
    print("\nControls:")
    print("  - GT Depth (Red): Ground truth depth, offset left")
    print("  - Teacher Raw (Green): Real teacher model depth (no alignment), center")
    print("  - Teacher Aligned (Blue): Aligned teacher depth, offset right")
    print("\nPress Ctrl+C to stop the server")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
