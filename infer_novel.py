"""
Novel view synthesis using per-frame 3D Gaussian rendering from Waymo dataset.

For each frame in the input sequence:
1. Use model to predict Gaussian parameters
2. Render 3 novel views: left (camera moved left), middle (original), right (camera moved right)
3. Combine into a single image with 3 columns
4. Generate output video with all frames

Usage:
    # Single sample (default: 0)
    python infer_novel.py --checkpoint checkpoints/model.safetensors

    # Multiple samples
    python infer_novel.py --checkpoint checkpoints/model.safetensors \
        --sample_start 0 --sample_end 5

    # Custom parameters
    python infer_novel.py --checkpoint checkpoints/model.safetensors \
        --sample_start 0 --sample_end 10 --camera_offset 0.3 --fps 30
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import List

import torch
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from src.model import DepthAnything3Net
from src.dataset import WaymoDataset
from src.dataset.waymo import collate_fn
from torch.utils.data import DataLoader
from gsplat.rendering import rasterization


def parse_args():
    parser = argparse.ArgumentParser(description='Novel view synthesis using per-frame Gaussians from Waymo')
    parser.add_argument('--config', type=str, default='config/train_gaussian_head_only.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='Dataset split to use')
    parser.add_argument('--sample_start', type=int, default=0,
                        help='Start sample index (inclusive)')
    parser.add_argument('--sample_end', type=int, default=None,
                        help='End sample index (exclusive), default is dataset size')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for videos')
    parser.add_argument('--camera_offset', type=float, default=0.2,
                        help='Camera translation offset magnitude (in world units)')
    parser.add_argument('--fps', type=int, default=20,
                        help='Output video FPS')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--no_middle', action='store_true',
                        help='Skip rendering middle view (only show left and right)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def render_novel_view(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmat_orig: torch.Tensor,
    K_orig: torch.Tensor,
    direction: str,
    offset_magnitude: float,
    image_height: int,
    image_width: int,
    sh_degree: int,
) -> torch.Tensor:
    """
    Render Gaussians from a novel camera viewpoint.

    Args:
        means, scales, rotations, opacities, colors: Gaussian parameters for one frame
        viewmat_orig: Original view matrix [1, 4, 4]
        K_orig: Original intrinsics [1, 3, 3]
        direction: 'left', 'middle', or 'right' camera offset direction
        offset_magnitude: Translation offset magnitude
        image_height: Image height
        image_width: Image width
        sh_degree: Spherical harmonics degree

    Returns:
        render_rgb: [H, W, 3] rendered RGB image
    """
    viewmat = viewmat_orig.clone()
    K = K_orig.clone()

    if direction == 'middle':
        # No offset for middle view
        pass
    elif direction == 'left':
        # Move camera to the left (negative X translation in world space)
        # w2c matrix, so we translate the camera position by modifying the translation part
        viewmat_left = viewmat.clone()
        viewmat_left[0, 0, 3] += offset_magnitude  # Translate camera position left
        viewmat = viewmat_left
    elif direction == 'right':
        # Move camera to the right (positive X translation in world space)
        viewmat_right = viewmat.clone()
        viewmat_right[0, 0, 3] -= offset_magnitude  # Translate camera position right
        viewmat = viewmat_right
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # Render
    render_color, _, _ = rasterization(
        means=means,
        quats=rotations,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=image_width,
        height=image_height,
        near_plane=0.0001,
        far_plane=1000.0,
        radius_clip=0,
        eps2d=0.3,
        sh_degree=sh_degree,
        render_mode="RGB",
    )

    # render_color: [1, H, W, 3]
    render_rgb = render_color[0]  # [H, W, 3]
    render_rgb = torch.clamp(render_rgb, min=0, max=1)

    return render_rgb


class NovelViewSynthesizer:
    def __init__(self, config, checkpoint_path, device='cuda'):
        self.config = config
        self.device = torch.device(device)
        self.setup_model(checkpoint_path)

    def setup_model(self, checkpoint_path):
        """Setup and load model."""
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
            predict_gaussians=model_config.get('predict_gaussians', True),
            gaussian_sh_degree=model_config.get('gaussian_sh_degree', 2),
            gaussian_output_dim=model_config.get('gaussian_output_dim', None),
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_pretrained(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

    def setup_dataloader(self, split='val'):
        """Setup dataloader for Waymo dataset."""
        data_config = self.config.get('data', {})

        if split == 'val':
            root = data_config.get('val_root')
        else:
            root = data_config.get('train_root')

        dataset = WaymoDataset(
            root=root,
            valid_camera_id_list=data_config.get('camera_ids', ["1"]),
            intervals=data_config.get('intervals', [2]),
            num_views=data_config.get('num_views', 8),
            resolution=data_config.get('resolution', 336),
            split=split,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        return dataloader

    @torch.no_grad()
    def process_sample(
        self,
        batch,
        camera_offset: float,
        sh_degree: int,
        no_middle: bool = False,
    ) -> List[np.ndarray]:
        """
        Process a single sample and render novel views.

        Args:
            batch: Batch from dataloader
            camera_offset: Camera translation offset
            sh_degree: Spherical harmonics degree
            no_middle: Skip middle view

        Returns:
            List of composite images
        """
        images = batch['images'].to(self.device)

        print(f"Processing sample with shape {images.shape}...")

        # Run inference to get predictions
        outputs = self.model(images)

        extrinsics = outputs.get('extrinsics')  # [B, S, 4, 4]
        intrinsics = outputs.get('intrinsics')  # [B, S, 3, 3]
        gaussians = outputs.get('gaussians')

        if gaussians is None:
            raise RuntimeError("Model did not produce Gaussian predictions")

        if extrinsics is None or intrinsics is None:
            raise RuntimeError("Model did not produce camera parameters")

        # Get image dimensions
        _, _, _, height, width = images.shape

        # Extract Gaussian parameters
        means_all = gaussians.means[0]  # [S*H*W, 3]
        scales_all = gaussians.scales[0]  # [S*H*W, 3]
        rotations_all = gaussians.rotations[0]  # [S*H*W, 4]
        opacities_all = gaussians.opacities[0]  # [S*H*W]
        harmonics_all = gaussians.harmonics[0]  # [S*H*W, 3, d_sh]

        # Transpose colors from [N, 3, d_sh] to [N, d_sh, 3]
        colors_all = harmonics_all.transpose(-1, -2)

        extrinsics_all = extrinsics[0]  # [S, 4, 4]
        intrinsics_all = intrinsics[0]  # [S, 3, 3]

        S = images.shape[1]
        composite_frames = []

        # Render each frame
        pbar = tqdm(range(S), desc='Rendering novel views')
        for frame_idx in pbar:
            # Extract Gaussians for this frame
            start_idx = frame_idx * height * width
            end_idx = (frame_idx + 1) * height * width

            means = means_all[start_idx:end_idx]
            scales = scales_all[start_idx:end_idx]
            rotations = rotations_all[start_idx:end_idx]
            opacities = opacities_all[start_idx:end_idx]
            colors = colors_all[start_idx:end_idx]

            viewmat = extrinsics_all[frame_idx:frame_idx+1]
            K = intrinsics_all[frame_idx:frame_idx+1]

            # Render three views
            if not no_middle:
                view_middle = render_novel_view(
                    means, scales, rotations, opacities, colors,
                    viewmat, K, 'middle', camera_offset,
                    height, width, sh_degree
                )

            view_left = render_novel_view(
                means, scales, rotations, opacities, colors,
                viewmat, K, 'left', camera_offset,
                height, width, sh_degree
            )

            view_right = render_novel_view(
                means, scales, rotations, opacities, colors,
                viewmat, K, 'right', camera_offset,
                height, width, sh_degree
            )

            # Composite into single image
            if no_middle:
                # [left | right]
                composite = torch.cat([view_left, view_right], dim=1)  # [H, 2*W, 3]
            else:
                # [left | middle | right]
                composite = torch.cat([view_left, view_middle, view_right], dim=1)  # [H, 3*W, 3]

            # Convert to numpy BGR for video writing
            composite_np = (composite.cpu().numpy() * 255).astype(np.uint8)
            composite_np = cv2.cvtColor(composite_np, cv2.COLOR_RGB2BGR)

            composite_frames.append(composite_np)

        return composite_frames

    def save_video(self, frames: List[np.ndarray], output_path: str, fps: int = 20):
        """Save frames as video."""
        if not frames:
            raise ValueError("No frames to save")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        H, W = frames[0].shape[:2]

        # Use H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        for frame in frames:
            writer.write(frame)

        writer.release()
        print(f"Saved video to {output_path}")


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Create synthesizer
    synthesizer = NovelViewSynthesizer(
        config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Setup dataloader
    dataloader = synthesizer.setup_dataloader(split=args.split)
    dataset_size = len(dataloader)

    # Determine sample range
    sample_start = args.sample_start
    sample_end = args.sample_end if args.sample_end is not None else dataset_size

    if sample_start < 0 or sample_start >= dataset_size:
        print(f"Error: sample_start {sample_start} out of range (0-{dataset_size-1})")
        return

    if sample_end > dataset_size:
        print(f"Warning: sample_end {sample_end} exceeds dataset size {dataset_size}, using {dataset_size}")
        sample_end = dataset_size

    if sample_start >= sample_end:
        print(f"Error: sample_start {sample_start} >= sample_end {sample_end}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process samples in range
    for sample_idx in range(sample_start, sample_end):
        print(f"\n{'='*50}")
        print(f"Processing sample {sample_idx}/{sample_end-1}")
        print(f"{'='*50}")

        # Get specific sample
        for idx, batch in enumerate(dataloader):
            if idx == sample_idx:
                break

        # Process sequence
        frames = synthesizer.process_sample(
            batch=batch,
            camera_offset=args.camera_offset,
            sh_degree=config['model'].get('gaussian_sh_degree', 2),
            no_middle=args.no_middle,
        )

        # Save video with sample index in filename
        video_name = f"sample_{sample_idx:06d}_novel_view.mp4"
        video_path = output_dir / video_name
        synthesizer.save_video(frames, str(video_path), fps=args.fps)

    print(f"\nAll {sample_end - sample_start} videos saved to {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
