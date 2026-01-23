#!/usr/bin/env python3
"""
Download Physical AI AV dataset for training.

This script downloads camera images, LiDAR point clouds, and calibration data
from NVIDIA's PhysicalAI-Autonomous-Vehicles dataset on Hugging Face.

Usage:
    python download_physical_ai_av.py --output_dir ./data --max_clips 100 --cameras all
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
from pathlib import Path
from physical_ai_av.dataset import PhysicalAIAVDatasetInterface


def parse_args():
    parser = argparse.ArgumentParser(description='Download Physical AI AV dataset')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for downloaded data')
    parser.add_argument('--token', type=str, default=None,
                        help='Hugging Face token (default: uses cached token)')
    parser.add_argument('--max_clips', type=int, default=None,
                        help='Maximum number of clips to download (default: all)')
    parser.add_argument('--cameras', type=str, default='all',
                        choices=['all', 'front', 'custom'],
                        help='Which cameras to download: all, front, or custom')
    parser.add_argument('--custom_cameras', type=str, nargs='+', default=None,
                        help='Custom camera list (only used if --cameras=custom)')
    parser.add_argument('--download_lidar', action='store_true', default=True,
                        help='Download LiDAR data')
    parser.add_argument('--start_clip', type=int, default=0,
                        help='Start downloading from this clip index')
    return parser.parse_args()


def get_camera_list(args):
    """Get list of cameras to download based on arguments."""
    if args.cameras == 'all':
        return [
            "camera_front_wide_120fov",
            "camera_front_tele_30fov",
            "camera_rear_tele_30fov",
            "camera_cross_left_120fov",
            "camera_cross_right_120fov",
            "camera_rear_left_70fov",
            "camera_rear_right_70fov",
        ]
    elif args.cameras == 'front':
        return [
            "camera_front_wide_120fov",
            "camera_front_tele_30fov",
        ]
    elif args.cameras == 'custom':
        if args.custom_cameras is None:
            raise ValueError("Must specify --custom_cameras when using --cameras=custom")
        return args.custom_cameras
    else:
        raise ValueError(f"Unknown camera option: {args.cameras}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Physical AI AV Dataset Downloader")
    print("=" * 80)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Max clips: {args.max_clips if args.max_clips else 'all'}")
    print(f"Start clip: {args.start_clip}")

    # Get camera list
    camera_list = get_camera_list(args)
    print(f"Cameras to download: {len(camera_list)}")
    for cam in camera_list:
        print(f"  - {cam}")
    print(f"Download LiDAR: {args.download_lidar}")
    print()

    # Initialize dataset interface
    print("Initializing dataset interface...")
    token = args.token if args.token else True  # True = use cached token
    ds = PhysicalAIAVDatasetInterface(
        token=token,
        local_dir=str(output_dir),
        cache_dir=None,
        confirm_download_threshold_gb=float('inf'),  # Disable confirmation prompts
    )

    # Get clip list
    all_clips = ds.clip_index.index.tolist()
    print(f"Total clips available: {len(all_clips)}")

    # Apply clip range
    if args.max_clips:
        end_clip = min(args.start_clip + args.max_clips, len(all_clips))
    else:
        end_clip = len(all_clips)

    clips_to_download = all_clips[args.start_clip:end_clip]
    print(f"Will download clips {args.start_clip} to {end_clip-1} ({len(clips_to_download)} clips)")
    print()

    # Collect all required chunks
    chunks_needed = set()
    for clip_id in clips_to_download:
        chunk_id = ds.get_clip_chunk(clip_id)
        chunks_needed.add(chunk_id)

    print(f"Chunks needed: {sorted(chunks_needed)}")
    print(f"Total chunks to download: {len(chunks_needed)}")
    print()

    # Build feature list
    features_to_download = camera_list + ["camera_intrinsics", "sensor_extrinsics"]
    if args.download_lidar:
        features_to_download.append("lidar_top_360fov")

    print("Features to download:")
    for feat in features_to_download:
        print(f"  - {feat}")
    print()

    # Download each chunk
    for i, chunk_id in enumerate(sorted(chunks_needed), 1):
        print("=" * 80)
        print(f"Downloading chunk {chunk_id} ({i}/{len(chunks_needed)})")
        print("=" * 80)

        try:
            ds.download_chunk_features(
                int(chunk_id),
                features=features_to_download,
            )
            print(f"Successfully downloaded chunk {chunk_id}")
        except Exception as e:
            print(f"Error downloading chunk {chunk_id}: {e}")
            print("Continuing with next chunk...")
            continue

        print()

    print("=" * 80)
    print("Download Complete!")
    print("=" * 80)
    print(f"Data saved to: {output_dir.absolute()}")
    print()
    print("You can now train with this data by setting:")
    print(f"  train_root: {output_dir.absolute()}")
    print("in your training config file.")


if __name__ == '__main__':
    main()
