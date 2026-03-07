"""
3D Visualization of MapAnything inference results using Viser.
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

from src.dataset import WaymoDataset
from src.dataset.waymo import collate_fn

# Import MapAnything
sys.path.insert(0, '/home/ziqi.shi/map-anything')
from mapanything.models import MapAnything
from mapanything.utils.image import load_images


def parse_args():
    parser = argparse.ArgumentParser(description='3D Visualization of MapAnything on Waymo')
    parser.add_argument('--config', type=str, default='config/train_waymo.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/mapanything.safetensors',
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


def transform_points(points, extrinsics, is_c2w=True):
    """Transform points from camera to world coordinate.

    Args:
        points: (N, 3) points in camera frame
        extrinsics: (4, 4) camera pose matrix
        is_c2w: if True, extrinsics is camera-to-world; if False, it's world-to-camera

    Returns:
        (N, 3) points in world frame
    """
    if is_c2w:
        # extrinsics is c2w, so directly apply it
        c2w = extrinsics
    else:
        # extrinsics is w2c (like GT), so invert it
        c2w = np.linalg.inv(extrinsics)

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    points_world = points @ R.T + t
    return points_world


class Visualizer:
    def __init__(self, config, checkpoint_path, device='cuda'):
        self.config = config
        self.device = torch.device(device)
        self.setup_model(checkpoint_path)

    def setup_model(self, checkpoint_path):
        """Setup model and load checkpoint."""
        # Load MapAnything model from safetensors checkpoint
        print(f"Loading MapAnything model from {checkpoint_path}...")

        from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local

        # Use the helper function to initialize from local checkpoint
        # Note: configs/train.yaml should be relative to map-anything repo
        mapanything_repo = '/home/ziqi.shi/map-anything'
        config_path = os.path.join(mapanything_repo, 'configs/train.yaml')

        local_config = {
            "path": config_path,
            "checkpoint_path": os.path.abspath(checkpoint_path),
            "config_overrides": ["model=mapanything"],
        }

        self.model = initialize_mapanything_local(local_config, device=self.device)
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
        extrinsics = batch['extrinsics'].cpu().numpy()  # (B, S, 4, 4)
        intrinsics = batch['intrinsics'].cpu().numpy()  # (B, S, 3, 3)
        sparse_depths_batch = batch.get('sparse_depths', None)  # (B, S, H, W)
        depth_scale_factor = batch.get('depth_scale_factor', None)

        B, S = images.shape[:2]

        # Import normalization utilities from MapAnything
        from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
        import torchvision.transforms.v2 as tvf

        # Get dinov2 normalization
        img_norm_config = IMAGE_NORMALIZATION_DICT['dinov2']
        img_normalize = tvf.Normalize(mean=img_norm_config.mean, std=img_norm_config.std)

        # Prepare views for MapAnything
        # MapAnything expects list of dict with 'img' (shape 1,3,H,W) and 'data_norm_type' (string)
        views = []
        for b in range(B):
            for s in range(S):
                # Convert from (3, H, W) with range [0, 1]
                img_np = images[b, s]  # (3, H, W) in range [0, 1]
                H, W = img_np.shape[1], img_np.shape[2]

                # Get original intrinsics before any cropping
                intr_original = intrinsics[b, s].copy()  # (3, 3)

                # Make sure dimensions are divisible by patch size (14)
                patch_size = 14
                H_new = (H // patch_size) * patch_size
                W_new = (W // patch_size) * patch_size

                # Crop if needed
                crop_top = (H - H_new) // 2
                crop_left = (W - W_new) // 2

                if H != H_new or W != W_new:
                    img_np = img_np[:, crop_top:crop_top+H_new, crop_left:crop_left+W_new]

                # Convert to tensor and normalize
                img_tensor = torch.from_numpy(img_np).float().to(self.device)  # (3, H, W)
                img_tensor = img_normalize(img_tensor)  # Apply normalization
                img_batched = img_tensor.unsqueeze(0)  # (1, 3, H, W)

                view_dict = {
                    'img': img_batched,
                    'data_norm_type': ['dinov2'],  # Must be a list!
                }

                # Add sparse depth if available and non-zero
                if sparse_depths_batch is not None:
                    sparse_depth = sparse_depths_batch[b, s]  # Keep as tensor (H, W)
                    if sparse_depth.dim() == 2:
                        sparse_depth = sparse_depth.unsqueeze(-1)  # (H, W) -> (H, W, 1)

                    # Crop sparse depth to match cropped image
                    if sparse_depth.shape[0] != H_new or sparse_depth.shape[1] != W_new:
                        sparse_depth = sparse_depth[crop_top:crop_top+H_new, crop_left:crop_left+W_new, :]

                    # Always add intrinsics when sparse depth is available (adjusted for crop)
                    intr_adjusted = intr_original.copy()
                    intr_adjusted[0, 2] -= crop_left  # principal point x
                    intr_adjusted[1, 2] -= crop_top   # principal point y
                    intr_tensor = torch.from_numpy(intr_adjusted).float().to(self.device).unsqueeze(0)  # (1, 3, 3)
                    view_dict['intrinsics'] = intr_tensor

                    # Only add depth_z and is_metric_scale if sparse depth has non-zero values
                    # Skip sparse depth input if all zeros to avoid issues in model (MapAnything not trained with all-zero depth input)
                    if sparse_depth.max() > 0:
                        # MapAnything internally uses depth_z which gets converted to depth_along_ray
                        # Shape should be (H, W) or (H, W, 1) for the API
                        sparse_depth_tensor = sparse_depth.squeeze(-1).float().to(self.device)  # (H, W)

                        # Convert to metric scale before passing to model
                        # depth_scale_factor is 1/avg_dist, so metric_scale = 1/depth_scale_factor
                        if depth_scale_factor is not None:
                            metric_scale = 1.0 / depth_scale_factor.item() if torch.is_tensor(depth_scale_factor) else 1.0 / depth_scale_factor
                            sparse_depth_tensor = sparse_depth_tensor * metric_scale

                        view_dict['depth_z'] = sparse_depth_tensor

                        # Mark as metric scale (from dataset normalization)
                        view_dict['is_metric_scale'] = torch.tensor([True], device=self.device)

                views.append(view_dict)

        # Run MapAnything inference
        print(f"Running MapAnything inference on {len(views)} views...")
        outputs = self.model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=False,
            mask_edges=False,
        )

        # Extract depths and confidences
        pred_depths = []
        depth_confs = []
        pred_intrinsics = []
        pred_extrinsics = []
        metric_scaling_factors = []

        for i, output in enumerate(outputs):
            # Get depth_z from output
            depth_z = output['depth_z'].cpu().numpy()  # (B, H, W, 1)
            if depth_z.ndim == 4:  # (B, H, W, 1)
                depth_z = depth_z[0, :, :, 0]  # (H, W)
            elif depth_z.ndim == 3:  # (H, W, 1)
                depth_z = depth_z[:, :, 0]  # (H, W)
            pred_depths.append(depth_z)

            # Get confidence from output
            conf = output['conf'].cpu().numpy()  # (B, H, W)
            if conf.ndim == 3:  # (B, H, W)
                conf = conf[0]  # (H, W)
            depth_confs.append(conf)

            # Get intrinsics
            intr = output['intrinsics'].cpu().numpy()  # (B, 3, 3) or (3, 3)
            if intr.ndim == 3:  # (B, 3, 3)
                intr = intr[0]  # (3, 3)
            pred_intrinsics.append(intr)

            # Get camera poses (c2w format from MapAnything)
            cam_pose = output['camera_poses'].cpu().numpy()  # (B, 4, 4) or (4, 4)
            if cam_pose.ndim == 3:  # (B, 4, 4)
                cam_pose = cam_pose[0]  # (4, 4)
            pred_extrinsics.append(cam_pose)

            # Get metric scaling factor if available
            if 'metric_scaling_factor' in output:
                scale_factor = output['metric_scaling_factor'].cpu().item()
            else:
                scale_factor = 1.0
            metric_scaling_factors.append(scale_factor)

        # Stack into batch format
        pred_depths = np.stack(pred_depths, axis=0)  # (S, H, W)
        depth_confs = np.stack(depth_confs, axis=0)  # (S, H, W)
        pred_intrinsics = np.stack(pred_intrinsics, axis=0)  # (S, 3, 3)
        pred_extrinsics = np.stack(pred_extrinsics, axis=0)  # (S, 4, 4) - c2w format
        metric_scaling_factors = np.array(metric_scaling_factors)  # (S,)

        # Convert GT depths to metric scale using depth_scale_factor
        gt_depths = batch.get('depths', None)
        sparse_depths_for_output = batch.get('sparse_depths', None)

        if depth_scale_factor is not None:
            metric_scale = 1.0 / depth_scale_factor.item() if torch.is_tensor(depth_scale_factor) else 1.0 / depth_scale_factor

            if gt_depths is not None:
                # depth_scale_factor is 1/avg_dist, so we need to invert it to recover metric scale
                gt_depths = gt_depths * metric_scale

            if sparse_depths_for_output is not None:
                # Also convert sparse depths to metric scale
                sparse_depths_for_output = sparse_depths_for_output * metric_scale

        return {
            'pred_depth': pred_depths,
            'depth_conf': depth_confs,
            'images': images[0],  # (S, 3, H, W)
            'extrinsics': extrinsics[0],  # (S, 4, 4) - w2c format (GT)
            'intrinsics': intrinsics[0],  # (S, 3, 3)
            'pred_extrinsics': pred_extrinsics,  # (S, 4, 4) - c2w format
            'pred_extrinsics_is_c2w': True,  # Mark as c2w
            'pred_intrinsics': pred_intrinsics,
            'metric_scaling_factors': metric_scaling_factors,  # (S,)
            'sparse_depths': sparse_depths_for_output.cpu().numpy()[0] if sparse_depths_for_output is not None else None,  # Converted to metric scale
            'gt_depths': gt_depths,  # GT depths converted to metric scale
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
        align_scale_to_gt = server.gui.add_checkbox(
            "Align Pred Scale to GT (using <30m points)",
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

    def compute_depth_scale_factor(pred_depths, gt_depths, max_depth_threshold=30.0):
        """
        Compute scale factor to align predicted depths to GT depths.
        Only uses points within max_depth_threshold (default 30m) for robust alignment.

        Args:
            pred_depths: (S, H, W) predicted depths
            gt_depths: (B, S, 1, H, W) or (S, H, W) ground truth depths
            max_depth_threshold: Only use points below this depth (meters)

        Returns:
            scale_factor: scalar to multiply pred_depths by to align with gt_depths
        """
        if gt_depths is None:
            return 1.0

        # Handle different gt_depths shapes
        gt_depths_np = gt_depths.cpu().numpy() if torch.is_tensor(gt_depths) else gt_depths
        if gt_depths_np.ndim == 5:  # (B, S, 1, H, W)
            gt_depths_np = gt_depths_np[0, :, 0, :, :]  # (S, H, W)
        elif gt_depths_np.ndim == 4:  # (B, S, H, W)
            gt_depths_np = gt_depths_np[0, :, :, :]  # (S, H, W)

        # Collect all valid points within depth threshold
        valid_pred = []
        valid_gt = []

        for s in range(pred_depths.shape[0]):
            pred = pred_depths[s]
            gt = gt_depths_np[s]

            # Find pixels where both pred and gt are valid and within threshold
            valid_mask = (gt > 0) & (gt < max_depth_threshold)

            if valid_mask.sum() > 0:
                valid_pred.append(pred[valid_mask])
                valid_gt.append(gt[valid_mask])

        if len(valid_pred) == 0:
            return 1.0

        # Concatenate all valid points
        all_pred = np.concatenate(valid_pred)
        all_gt = np.concatenate(valid_gt)

        # Compute median or mean scale factor
        # scale_factor = mean(gt) / mean(pred)
        mean_gt = np.mean(all_gt)
        mean_pred = np.mean(all_pred)

        if mean_pred > 0:
            scale_factor = mean_gt / mean_pred
        else:
            scale_factor = 1.0

        print(f"Scale alignment: pred_mean={mean_pred:.3f}, gt_mean={mean_gt:.3f}, scale_factor={scale_factor:.3f}")
        return scale_factor

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
        metric_scaling_factors = cached_results.get('metric_scaling_factors', np.ones(pred_depth.shape[0]))

        # Apply scale alignment if enabled
        if align_scale_to_gt.value and cached_results.get('gt_depths') is not None:
            scale_factor = compute_depth_scale_factor(pred_depth, cached_results['gt_depths'])
            pred_depth = pred_depth * scale_factor
            print(f"Applied scale alignment: pred_depth *= {scale_factor:.3f}")

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
            if (show_gt_depth.value and
                cached_results['gt_depths'] is not None):
                gt_depths = cached_results['gt_depths'].cpu().numpy() if torch.is_tensor(cached_results['gt_depths']) else cached_results['gt_depths']
                gt_depth = gt_depths[0, s]  # (B, S, 1, H, W) -> (H, W)
                if gt_depth.shape[0] == 1:
                    gt_depth = gt_depth[0]

                # Convert GT depth to 3D points (GT is already in metric scale)
                if gt_depth.max() > 0:
                    gt_points, gt_pixel_coords = depth_to_points_with_conf(
                        gt_depth,
                        np.ones_like(gt_depth),  # All valid pixels equally weighted
                        intrinsics_scaled,
                        conf_threshold_percentile=0,  # Show all GT points
                        max_depth=max_depth,
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

    @align_scale_to_gt.on_update
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
