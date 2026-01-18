"""
3D Visualization of DA3 inference results using Viser.
Displays point clouds from predicted depth maps in an interactive 3D viewer.
Supports confidence-based filtering and per-frame viewing.
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
from src.dataset import WaymoDataset
from src.dataset.waymo import collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description='3D Visualization of DA3 on Waymo')
    parser.add_argument('--config', type=str, default='config/train_waymo.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model.safetensors',
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
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
    parser.add_argument('--use_ray_pose', action='store_true',
                        help='Use ray-based pose estimation instead of CameraDec')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def depth_to_points_with_conf(depth, conf, intrinsics, conf_threshold_percentile, max_depth, max_points=100000):
    """
    Convert depth map to 3D points with confidence filtering.
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

    valid_mask = (depth > 0) & (depth < max_depth) & (conf >= threshold_val) & (conf > 1e-5)

    points = points[valid_mask]
    pixel_coords = np.stack([v[valid_mask], u[valid_mask]], axis=-1)

    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        pixel_coords = pixel_coords[indices]

    return points, pixel_coords


def transform_points(points, extrinsics):
    """Transform points from camera to world coordinate."""
    c2w = np.linalg.inv(extrinsics)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    points_world = points @ R.T + t
    return points_world


class Visualizer:
    def __init__(self, config, checkpoint_path, device='cuda', use_ray_pose=False):
        self.config = config
        self.device = torch.device(device)
        self.use_ray_pose = use_ray_pose
        self.setup_model(checkpoint_path)

    def setup_model(self, checkpoint_path):
        """Setup model and load checkpoint."""
        model_config = self.config.get('model', {})

        self.model = DepthAnything3Net(
            encoder_name=model_config.get('encoder_name', 'vitl'),
            out_layers=model_config.get('out_layers', [11, 15, 19, 23]),
            features=model_config.get('features', 256),
            out_channels=model_config.get('out_channels', [256, 512, 1024, 1024]),
            alt_start=model_config.get('alt_start', 8),
            qknorm_start=model_config.get('qknorm_start', 8),
            rope_start=model_config.get('rope_start', 8),
            predict_camera=model_config.get('predict_camera', True),
            use_camera_enc=model_config.get('use_camera_enc', False),
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_pretrained(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

    def setup_dataloader(self, split='val'):
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
            num_views=data_config.get('num_views', 4),
            resolution=data_config.get('resolution', 518),
            split=split,
        )

        return dataset

    @torch.no_grad()
    def infer_sample(self, sample):
        """Run inference on a single sample."""
        batch = collate_fn([sample])
        images = batch['images'].to(self.device)

        outputs = self.model(images, use_ray_pose=self.use_ray_pose)

        return {
            'pred_depth': outputs['depth'].cpu().numpy()[0],
            'depth_conf': outputs['depth_conf'].cpu().numpy()[0],
            'images': batch['images'].cpu().numpy()[0],
            'extrinsics': batch['extrinsics'].cpu().numpy()[0],
            'intrinsics': batch['intrinsics'].cpu().numpy()[0],
            'pred_extrinsics': outputs['extrinsics'].cpu().numpy()[0] if 'extrinsics' in outputs else None,
            'pred_intrinsics': outputs['intrinsics'].cpu().numpy()[0] if 'intrinsics' in outputs else None,
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
        use_ray_pose=args.use_ray_pose,
    )

    dataset = visualizer.setup_dataloader(args.split)
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
        conf_threshold_slider = server.gui.add_slider(
            "Confidence Percent",
            min=0,
            max=100,
            step=1,
            initial_value=args.conf_threshold,
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

        # Fallback to GT if prediction not available
        if pred_extrinsics is None:
            print("Warning: pred_extrinsics not available, using GT")
            pred_extrinsics = cached_results['extrinsics']
        if pred_intrinsics is None:
            print("Warning: pred_intrinsics not available, using GT")
            pred_intrinsics = cached_results['intrinsics']

        extrinsics = pred_extrinsics
        intrinsics = pred_intrinsics

        S = pred_depth.shape[0]
        max_depth = max_depth_slider.value
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
                conf = depth_conf[s]
            else:
                conf = np.ones_like(pred_depth[s])

            pred_points, pred_pixel_coords = depth_to_points_with_conf(
                pred_depth[s],
                conf,
                intrinsics_scaled,
                conf_threshold,
                max_depth,
                max_points=args.max_points
            )

            if len(pred_points) > 0:
                pred_points_world = transform_points(pred_points, extrinsics[s])
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
                c2w = np.linalg.inv(extrinsics[s])
                R = c2w[:3, :3]
                t = c2w[:3, 3]

                # Convert to SE3 for viser
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

    @conf_threshold_slider.on_update
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
