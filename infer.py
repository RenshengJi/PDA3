"""
Inference script for DA3 on Waymo dataset.
Outputs predicted depth maps and optionally camera poses.
"""
import os
import sys
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import DepthAnything3Net
from src.dataset import WaymoDataset
from src.dataset.waymo import collate_fn
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Inference DA3 on Waymo')
    parser.add_argument('--config', type=str, default='config/train_waymo.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model.safetensors',
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/inference',
                        help='Output directory for predictions')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='Dataset split to use')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='Number of samples to process (-1 for all)')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--save_depth', action='store_true', default=True,
                        help='Save depth maps as .npy files')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--use_ray_pose', action='store_true',
                        help='Use ray-based pose estimation instead of CameraDec')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def colorize_depth(depth, vmin=None, vmax=None, cmap='magma'):
    """
    Convert depth map to colored image for visualization.

    Args:
        depth: [H, W] depth map
        vmin: minimum depth value
        vmax: maximum depth value
        cmap: colormap name

    Returns:
        [H, W, 3] colored depth image (BGR format)
    """
    import matplotlib.pyplot as plt

    if vmin is None:
        vmin = np.percentile(depth[depth > 0], 2) if (depth > 0).any() else 0
    if vmax is None:
        vmax = np.percentile(depth[depth > 0], 98) if (depth > 0).any() else 1

    depth_normalized = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_normalized = np.clip(depth_normalized, 0, 1)

    colormap = plt.get_cmap(cmap)
    depth_colored = colormap(depth_normalized)[:, :, :3]  # Remove alpha
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)

    return depth_colored


def save_depth_comparison(pred_depth, gt_depth, rgb_image, save_path, mask=None):
    """
    Save side-by-side comparison of RGB, predicted depth, and GT depth.

    Args:
        pred_depth: [H, W] predicted depth
        gt_depth: [H, W] ground truth depth
        rgb_image: [H, W, 3] RGB image
        save_path: path to save the visualization
        mask: [H, W] valid mask
    """
    H, W = pred_depth.shape

    # Resize RGB to match depth if needed
    if rgb_image.shape[:2] != (H, W):
        rgb_image = cv2.resize(rgb_image, (W, H))

    # Determine depth range from GT
    if mask is not None:
        valid_gt = gt_depth[mask > 0]
    else:
        valid_gt = gt_depth[gt_depth > 0]

    if len(valid_gt) > 0:
        vmin = np.percentile(valid_gt, 2)
        vmax = np.percentile(valid_gt, 98)
    else:
        vmin, vmax = 0, 100

    # Colorize depth maps
    pred_colored = colorize_depth(pred_depth, vmin, vmax)
    gt_colored = colorize_depth(gt_depth, vmin, vmax)

    # Create comparison image
    comparison = np.concatenate([rgb_image, pred_colored, gt_colored], axis=1)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'RGB', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'Predicted', (W + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'Ground Truth', (2 * W + 10, 30), font, 1, (255, 255, 255), 2)

    cv2.imwrite(save_path, comparison)


def compute_metrics(pred_depth, gt_depth, mask):
    """
    Compute depth prediction metrics.

    Args:
        pred_depth: [H, W] predicted depth
        gt_depth: [H, W] ground truth depth
        mask: [H, W] valid mask

    Returns:
        dict: metrics dictionary
    """
    valid = mask > 0
    if not valid.any():
        return {}

    pred = pred_depth[valid]
    gt = gt_depth[valid]

    # Absolute Relative Error
    abs_rel = np.mean(np.abs(pred - gt) / (gt + 1e-8))

    # Squared Relative Error
    sq_rel = np.mean(((pred - gt) ** 2) / (gt + 1e-8))

    # RMSE
    rmse = np.sqrt(np.mean((pred - gt) ** 2))

    # RMSE log
    rmse_log = np.sqrt(np.mean((np.log(pred + 1e-8) - np.log(gt + 1e-8)) ** 2))

    # Threshold accuracy
    thresh = np.maximum(pred / (gt + 1e-8), gt / (pred + 1e-8))
    a1 = np.mean(thresh < 1.25)
    a2 = np.mean(thresh < 1.25 ** 2)
    a3 = np.mean(thresh < 1.25 ** 3)

    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3,
    }


class Inferencer:
    def __init__(self, config, checkpoint_path, device='cuda', use_ray_pose=False):
        self.config = config
        self.device = torch.device(device)
        self.use_ray_pose = use_ray_pose

        # Setup model
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

        # Load checkpoint
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
            intervals=[1],  # Use interval 1 for inference
            num_views=data_config.get('num_views', 4),
            resolution=data_config.get('resolution', 518),
            split=split,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        return dataloader

    @torch.no_grad()
    def infer_batch(self, batch):
        """
        Run inference on a batch.

        Args:
            batch: dict containing 'images' [B, S, C, H, W]

        Returns:
            dict: outputs containing 'depth', 'extrinsics', 'intrinsics'
        """
        images = batch['images'].to(self.device)
        outputs = self.model(images, use_ray_pose=self.use_ray_pose)

        return outputs

    def run(self, output_dir, split='val', num_samples=-1, save_vis=False, save_depth=True):
        """
        Run inference on dataset.

        Args:
            output_dir: directory to save outputs
            split: dataset split
            num_samples: number of samples to process (-1 for all)
            save_vis: whether to save visualization images
            save_depth: whether to save depth maps as .npy
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_vis:
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)

        if save_depth:
            depth_dir = output_dir / 'depth_predictions'
            depth_dir.mkdir(exist_ok=True)

        dataloader = self.setup_dataloader(split)

        all_metrics = []

        num_to_process = len(dataloader) if num_samples < 0 else min(num_samples, len(dataloader))

        pbar = tqdm(enumerate(dataloader), total=num_to_process, desc='Inference')

        for idx, batch in pbar:
            if idx >= num_to_process:
                break

            # Run inference
            outputs = self.infer_batch(batch)

            # Get predictions
            pred_depth = outputs['depth'].cpu().numpy()  # [B, S, H, W]

            # Get ground truth
            gt_depth = batch['depths'].cpu().numpy()  # [B, S, H, W]
            masks = batch['point_masks'].cpu().numpy()  # [B, S, H, W]
            images = batch['images'].cpu().numpy()  # [B, S, C, H, W]

            B, S = pred_depth.shape[:2]

            for b in range(B):
                for s in range(S):
                    sample_id = f"{idx:06d}_b{b}_s{s}"

                    pred = pred_depth[b, s]
                    gt = gt_depth[b, s]
                    mask = masks[b, s]

                    # Resize prediction to match GT if needed
                    if pred.shape != gt.shape:
                        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)

                    # Compute metrics
                    metrics = compute_metrics(pred, gt, mask)
                    if metrics:
                        all_metrics.append(metrics)

                    # Save depth map
                    if save_depth:
                        np.save(depth_dir / f"{sample_id}_pred.npy", pred)
                        np.save(depth_dir / f"{sample_id}_gt.npy", gt)

                    # Save visualization
                    if save_vis:
                        # Convert image from [C, H, W] to [H, W, C] BGR
                        img = images[b, s].transpose(1, 2, 0)
                        img = (img * 255).astype(np.uint8)
                        if img.shape[2] == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                        save_depth_comparison(
                            pred, gt, img,
                            str(vis_dir / f"{sample_id}_comparison.png"),
                            mask
                        )

            # Update progress bar with current metrics
            if all_metrics:
                avg_abs_rel = np.mean([m['abs_rel'] for m in all_metrics])
                avg_a1 = np.mean([m['a1'] for m in all_metrics])
                pbar.set_postfix({'abs_rel': f'{avg_abs_rel:.4f}', 'a1': f'{avg_a1:.4f}'})

        # Compute and save final metrics
        if all_metrics:
            final_metrics = {}
            for key in all_metrics[0].keys():
                final_metrics[key] = np.mean([m[key] for m in all_metrics])

            print("\n" + "=" * 50)
            print("Final Metrics:")
            print("=" * 50)
            for key, value in final_metrics.items():
                print(f"  {key}: {value:.4f}")

            # Save metrics to file
            with open(output_dir / 'metrics.txt', 'w') as f:
                f.write("Depth Prediction Metrics\n")
                f.write("=" * 30 + "\n")
                for key, value in final_metrics.items():
                    f.write(f"{key}: {value:.4f}\n")

            # Save as yaml
            with open(output_dir / 'metrics.yaml', 'w') as f:
                yaml.dump(final_metrics, f)

            print(f"\nResults saved to {output_dir}")

        return final_metrics if all_metrics else {}


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Create inferencer
    inferencer = Inferencer(
        config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_ray_pose=args.use_ray_pose,
    )

    # Run inference
    metrics = inferencer.run(
        output_dir=args.output_dir,
        split=args.split,
        num_samples=args.num_samples,
        save_vis=args.save_vis,
        save_depth=args.save_depth,
    )


if __name__ == '__main__':
    main()
