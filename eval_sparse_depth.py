"""
Evaluation script for sparse depth input ablation study.
Evaluates model performance with varying amounts of sparse depth input (0%, 20%, ..., 100%).
Metrics: Depth Accuracy (AbsRel, RMSE, δ) and Pose Accuracy (rotation error, translation error).
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import DepthAnything3Net
from src.dataset import WaymoDataset
from src.dataset.waymo import collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate sparse depth ablation')
    parser.add_argument('--config', type=str, default='config/train_waymo.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model.safetensors',
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='data/waymo/val',
                        help='Path to validation data')
    parser.add_argument('--output_dir', type=str, default='outputs/eval_sparse_depth',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--eval_depth_only', action='store_true',
                        help='Only evaluate depth metrics, skip pose metrics')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with debugpy')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path, config, device, use_depth_enc=True):
    """Load model from checkpoint using config parameters."""
    model_config = config.get('model', {})

    model = DepthAnything3Net(
        encoder_name=model_config.get('encoder_name', 'vitl'),
        out_layers=model_config.get('out_layers', [11, 15, 19, 23]),
        features=model_config.get('features', 256),
        out_channels=model_config.get('out_channels', [256, 512, 1024, 1024]),
        alt_start=model_config.get('alt_start', 8),
        qknorm_start=model_config.get('qknorm_start', 8),
        rope_start=model_config.get('rope_start', 8),
        predict_camera=model_config.get('predict_camera', True),
        use_camera_enc=model_config.get('use_camera_enc', True),
        use_depth_enc=use_depth_enc,
    )

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract state dict from various checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        incompatible = model.load_state_dict(state_dict, strict=False)

        # Report any missing keys that are not from depth_enc
        if incompatible.missing_keys:
            missing_depth_enc = [k for k in incompatible.missing_keys if 'depth_enc' in k]
            missing_others = [k for k in incompatible.missing_keys if 'depth_enc' not in k]
            if missing_others:
                print(f"Warning: {len(missing_others)} non-depth_enc parameters missing:")
                for key in missing_others[:5]:
                    print(f"  - {key}")
                if len(missing_others) > 5:
                    print(f"  ... and {len(missing_others) - 5} more")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found, using random weights")

    model = model.to(device)
    model.eval()
    return model


class SparseDepthAblationDataset(Dataset):
    """Wrapper dataset that generates different sparsity levels."""

    def __init__(self, waymo_dataset, sparsity_ratios=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                 base_indices=None):
        """
        Args:
            waymo_dataset: Base WaymoDataset instance
            sparsity_ratios: List of sparsity ratios to test (fraction of pixels kept)
            base_indices: Optional list of specific base sample indices to use.
                         If None, uses all samples from waymo_dataset
        """
        self.dataset = waymo_dataset
        self.sparsity_ratios = sparsity_ratios
        self.num_ratios = len(sparsity_ratios)

        # Use provided base indices or all samples
        self.base_indices = base_indices if base_indices is not None else list(range(len(waymo_dataset)))
        self.num_base_samples = len(self.base_indices)

    def __len__(self):
        return self.num_base_samples * self.num_ratios

    def __getitem__(self, idx):
        # Map idx to (base_idx_in_subset, ratio_idx)
        base_idx_in_subset = idx // self.num_ratios
        ratio_idx = idx % self.num_ratios

        # Get actual sample index from base_indices
        actual_sample_idx = self.base_indices[base_idx_in_subset]
        keep_ratio = self.sparsity_ratios[ratio_idx]

        # Get base sample
        sample = self.dataset[actual_sample_idx]

        # Generate sparse depth with specified ratio
        # Use deterministic seed based on actual_sample_idx and ratio_idx for reproducibility
        torch.manual_seed(actual_sample_idx * 1000 + ratio_idx)
        np.random.seed(actual_sample_idx * 1000 + ratio_idx)

        sparse_depths = []
        for depth in sample['depths']:
            if keep_ratio >= 1.0:
                sparse_d = depth.clone()
            else:
                sparse_d = self.dataset._sparsify_depth(depth, keep_ratio)
            sparse_depths.append(sparse_d)

        sparse_depths = torch.stack(sparse_depths, dim=0)
        sample['sparse_depths'] = sparse_depths
        sample['sparse_ratio'] = keep_ratio
        sample['use_sparse_depth'] = keep_ratio > 0

        return sample


class DepthMetrics:
    """Compute depth estimation metrics."""

    @staticmethod
    def compute_metrics(pred_depth, gt_depth, valid_mask=None):
        """
        Compute depth metrics.

        Args:
            pred_depth: [H, W] predicted depth
            gt_depth: [H, W] ground truth depth
            valid_mask: [H, W] valid pixel mask

        Returns:
            dict of metrics
        """
        if valid_mask is None:
            valid_mask = (gt_depth > 0)

        # Clip predictions to avoid inf/nan
        pred_depth = torch.clamp(pred_depth, min=1e-3, max=100)
        gt_depth = torch.clamp(gt_depth, min=1e-3, max=100)

        # Select valid pixels
        pred = pred_depth[valid_mask]
        gt = gt_depth[valid_mask]

        if pred.numel() == 0:
            return {
                'abs_rel': 0.0,
                'sq_rel': 0.0,
                'rmse': 0.0,
                'rmse_log': 0.0,
                'delta_1': 0.0,
                'delta_2': 0.0,
                'delta_3': 0.0,
            }

        # Compute metrics
        abs_diff = torch.abs(pred - gt)
        abs_rel = (abs_diff / gt).mean().item()
        sq_rel = (torch.pow(abs_diff, 2) / gt).mean().item()
        rmse = torch.sqrt((torch.pow(abs_diff, 2)).mean()).item()
        rmse_log = torch.sqrt((torch.pow(torch.log(pred) - torch.log(gt), 2)).mean()).item()

        # Threshold metrics
        thresh = torch.max(pred / gt, gt / pred)
        delta_1 = (thresh < 1.25).float().mean().item()
        delta_2 = (thresh < 1.25 ** 2).float().mean().item()
        delta_3 = (thresh < 1.25 ** 3).float().mean().item()

        return {
            'abs_rel': abs_rel,
            'sq_rel': sq_rel,
            'rmse': rmse,
            'rmse_log': rmse_log,
            'delta_1': delta_1,
            'delta_2': delta_2,
            'delta_3': delta_3,
        }


class PoseMetrics:
    """Compute pose estimation metrics."""

    @staticmethod
    def compute_rotation_error(pred_rot, gt_rot):
        """Compute rotation error in degrees using angle-axis representation."""
        # Ensure orthogonality via SVD
        U, _, Vt = torch.svd(pred_rot)
        pred_rot = torch.matmul(U, Vt)

        # Ensure determinant is positive (proper rotation)
        if torch.det(pred_rot) < 0:
            Vt[-1] *= -1
            pred_rot = torch.matmul(U, Vt)

        U, _, Vt = torch.svd(gt_rot)
        gt_rot = torch.matmul(U, Vt)

        # Ensure determinant is positive (proper rotation)
        if torch.det(gt_rot) < 0:
            Vt[-1] *= -1
            gt_rot = torch.matmul(U, Vt)

        # Compute relative rotation: R_diff = gt_rot^T @ pred_rot
        R_diff = torch.matmul(gt_rot.T, pred_rot)

        # Clamp to avoid numerical issues
        trace = torch.trace(R_diff)
        cos_angle = torch.clamp((trace - 1) / 2, -1.0, 1.0)

        # Angle in radians
        angle_rad = torch.acos(cos_angle)
        angle_deg = torch.rad2deg(angle_rad)

        return angle_deg.item()

    @staticmethod
    def compute_translation_error(pred_trans, gt_trans):
        """Compute translation error in meters (absolute, not relative)."""
        # Use absolute L2 distance
        error = torch.norm(pred_trans - gt_trans).item()
        return error

    @staticmethod
    def compute_pose_metrics(pred_extrinsics, gt_extrinsics):
        """
        Compute pose metrics.

        Args:
            pred_extrinsics: [4, 4] predicted extrinsic matrix (w2c)
            gt_extrinsics: [4, 4] ground truth extrinsic matrix (w2c)

        Returns:
            dict of metrics
        """
        # Handle potential NaN/Inf
        if torch.isnan(pred_extrinsics).any() or torch.isinf(pred_extrinsics).any():
            return {
                'rotation_error_deg': float('nan'),
                'translation_error_m': float('nan'),
            }

        if torch.isnan(gt_extrinsics).any() or torch.isinf(gt_extrinsics).any():
            return {
                'rotation_error_deg': float('nan'),
                'translation_error_m': float('nan'),
            }

        pred_rot = pred_extrinsics[:3, :3]
        pred_trans = pred_extrinsics[:3, 3]
        gt_rot = gt_extrinsics[:3, :3]
        gt_trans = gt_extrinsics[:3, 3]

        rot_error = PoseMetrics.compute_rotation_error(pred_rot, gt_rot)
        trans_error = PoseMetrics.compute_translation_error(pred_trans, gt_trans)

        return {
            'rotation_error_deg': rot_error,
            'translation_error_m': trans_error,
        }


@torch.no_grad()
def evaluate(model, dataloader, device, sparsity_ratios, eval_depth_only=False):
    """
    Evaluate model on different sparsity levels.

    Args:
        eval_depth_only: If True, skip pose metric computation

    Returns:
        dict: {sparsity_ratio: {metric_name: [values]}}
    """
    results = defaultdict(lambda: defaultdict(list))

    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
        # Move to device
        images = batch['images'].to(device)
        sparse_depths = batch['sparse_depths'].to(device)

        if 'extrinsics' in batch and 'intrinsics' in batch:
            extrinsics = batch['extrinsics'].to(device)
            intrinsics = batch['intrinsics'].to(device)
        else:
            extrinsics = None
            intrinsics = None

        gt_depths = batch['depths'].to(device)
        point_masks = batch['point_masks'].to(device)
        use_sparse_depth_list = batch['use_sparse_depth']  # list of bools
        sparse_ratio_list = batch['sparse_ratio']  # list of floats

        # Forward pass - check if any sample has sparse depth
        use_sparse_depth_any = any(use_sparse_depth_list) if isinstance(use_sparse_depth_list, list) else use_sparse_depth_list.any()
        sparse_depth_input = sparse_depths if use_sparse_depth_any else None

        outputs = model(
            images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            sparse_depth=sparse_depth_input,
        )

        pred_depths = outputs['depth']

        # Compute metrics
        B, S = images.shape[:2]
        for b in range(B):
            for s in range(S):
                # Get sparse ratio for this sample
                if isinstance(sparse_ratio_list, (list, tuple)):
                    ratio = float(sparse_ratio_list[b])
                elif isinstance(sparse_ratio_list, torch.Tensor):
                    ratio = float(sparse_ratio_list[b])
                else:
                    ratio = float(sparse_ratio_list)

                pred_depth = pred_depths[b, s]
                gt_depth = gt_depths[b, s]
                valid_mask = point_masks[b, s]
                sparse_depth_sample = sparse_depths[b, s]

                # Depth metrics
                depth_metrics = DepthMetrics.compute_metrics(pred_depth, gt_depth, valid_mask)
                for metric_name, value in depth_metrics.items():
                    results[ratio][f'depth_{metric_name}'].append(value)

                # Pose metrics (if extrinsics available and eval_depth_only is False)
                if not eval_depth_only and extrinsics is not None and 'extrinsics' in outputs:
                    pred_ext = outputs['extrinsics'][b, s]
                    gt_ext = extrinsics[b, s]
                    pose_metrics = PoseMetrics.compute_pose_metrics(pred_ext, gt_ext)
                    for metric_name, value in pose_metrics.items():
                        results[ratio][f'pose_{metric_name}'].append(value)

    # Average metrics
    avg_results = {}
    for ratio in sparsity_ratios:
        avg_results[ratio] = {}
        if ratio in results:
            for metric_name, values in results[ratio].items():
                # Handle NaN values
                valid_values = [v for v in values if not np.isnan(v)]
                if valid_values:
                    avg_results[ratio][metric_name] = np.mean(valid_values)
                else:
                    avg_results[ratio][metric_name] = float('nan')

    return avg_results


def print_results(results, sparsity_ratios):
    """Pretty print evaluation results."""
    print("\n" + "="*100)
    print("Sparse Depth Ablation Study Results")
    print("="*100)

    # Depth metrics
    print("\n" + "-"*100)
    print("DEPTH METRICS")
    print("-"*100)
    print(f"{'Sparsity':<12} {'AbsRel':<12} {'RMSE':<12} {'δ₁.₂₅':<12} {'δ₁.₂₅²':<12} {'δ₁.₂₅³':<12}")
    print("-"*100)

    for ratio in sorted(sparsity_ratios):
        if ratio in results and results[ratio]:
            abs_rel = results[ratio].get('depth_abs_rel', 0)
            rmse = results[ratio].get('depth_rmse', 0)
            delta_1 = results[ratio].get('depth_delta_1', 0)
            delta_2 = results[ratio].get('depth_delta_2', 0)
            delta_3 = results[ratio].get('depth_delta_3', 0)

            ratio_str = f"{ratio*100:.0f}%"
            print(f"{ratio_str:<12} {abs_rel:<12.4f} {rmse:<12.4f} {delta_1:<12.4f} {delta_2:<12.4f} {delta_3:<12.4f}")

    # Pose metrics
    print("\n" + "-"*100)
    print("POSE METRICS (Warning: Only valid if model has pose estimation enabled)")
    print("-"*100)
    print(f"{'Sparsity':<12} {'Rotation Error (°)':<25} {'Translation Error (m)':<25}")
    print("-"*100)

    for ratio in sorted(sparsity_ratios):
        if ratio in results and results[ratio]:
            rot_err = results[ratio].get('pose_rotation_error_deg', None)
            trans_err = results[ratio].get('pose_translation_error_m', None)

            ratio_str = f"{ratio*100:.0f}%"
            if rot_err is not None and trans_err is not None:
                # Check if values are NaN
                if np.isnan(rot_err) or np.isnan(trans_err):
                    print(f"{ratio_str:<12} {'N/A (NaN)':<25} {'N/A (NaN)':<25}")
                else:
                    print(f"{ratio_str:<12} {rot_err:<25.4f} {trans_err:<25.6f}")
            else:
                print(f"{ratio_str:<12} {'N/A':<25} {'N/A':<25}")

    print("="*100)


def save_results(results, output_dir):
    """Save results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    import json
    results_file = os.path.join(output_dir, 'sparse_depth_ablation.json')
    with open(results_file, 'w') as f:
        json.dump(
            {str(k): v for k, v in results.items()},
            f,
            indent=2
        )
    print(f"\nResults saved to {results_file}")


def main():
    args = parse_args()

    # debug
    if args.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")

    device = torch.device(args.device)

    # Load config
    config = load_config(args.config)

    # Load model
    use_depth_enc = config.get('sparse_depth', {}).get('enabled', False)
    model = load_model(args.checkpoint, config, device, use_depth_enc=use_depth_enc)

    # Create dataset
    waymo_dataset = WaymoDataset(
        root=args.data_root,
        split='val',
        num_views=config['data']['num_views'],
        resolution=config['data']['resolution'],
        sparse_depth_prob=0.0,  # Will override per sample
    )

    # Create ablation dataset with multiple sparsity levels
    sparsity_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Select base samples for evaluation
    if args.num_samples > 0:
        num_base_samples = min(args.num_samples, len(waymo_dataset))
        base_indices = np.random.choice(len(waymo_dataset), num_base_samples, replace=False).tolist()
    else:
        base_indices = None  # Use all samples

    # Create ablation dataset with selected base indices
    ablation_dataset = SparseDepthAblationDataset(
        waymo_dataset,
        sparsity_ratios=sparsity_ratios,
        base_indices=base_indices
    )

    # Create dataloader with custom collate function
    def collate_fn_sparse(batch):
        """Custom collate function that preserves sparse_ratio as list."""
        result = {}
        for key in batch[0].keys():
            if key in ["scene_name", "sparse_ratio"]:
                # Keep these as lists
                result[key] = [b[key] for b in batch]
            elif key == "use_sparse_depth":
                # Keep as list of bools
                result[key] = [b[key] for b in batch]
            elif isinstance(batch[0][key], torch.Tensor):
                result[key] = torch.stack([b[key] for b in batch], dim=0)
            else:
                result[key] = [b[key] for b in batch]
        return result

    dataloader = DataLoader(
        ablation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_sparse,
    )

    # Evaluate
    results = evaluate(model, dataloader, device, sparsity_ratios, eval_depth_only=args.eval_depth_only)

    # Print and save results
    print_results(results, sparsity_ratios)
    save_results(results, args.output_dir)


if __name__ == '__main__':
    main()
