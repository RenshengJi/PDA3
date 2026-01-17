"""
Simplified Waymo dataset for DA3 finetuning.
Only keeps essential functionality needed for depth and camera supervision.
"""
import os
import os.path as osp
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from .utils import imread_cv2, depthmap_to_camera_coordinates
import cv2
from PIL import Image


class WaymoDataset(Dataset):
    """
    Waymo outdoor street scenes dataset for DA3 finetuning.
    Loads multi-view images with camera parameters and depth.
    """

    def __init__(
        self,
        root,
        valid_camera_id_list=["1", "2", "3"],
        intervals=[1],
        num_views=4,
        resolution=(518, 518),
        split="train",
        transform=None,
        **kwargs
    ):
        """
        Initialize Waymo dataset.

        Args:
            root: Root directory of the dataset
            valid_camera_id_list: List of valid camera IDs to use
            intervals: Frame sampling intervals
            num_views: Number of context frames to load
            resolution: Target image resolution (H, W)
            split: Dataset split ("train", "val", "test")
            transform: Image transform to apply
        """
        self.root = root
        self.valid_camera_id_list = valid_camera_id_list
        self.num_views = num_views
        self.resolution = resolution if isinstance(resolution, tuple) else (resolution, resolution)
        self.split = split
        self.transform = transform

        if not isinstance(intervals, list):
            intervals = [intervals]
        self._intervals = intervals

        self._load_data()

    def _get_cache_filename(self):
        """Generate cache filename based on valid camera IDs."""
        camera_ids_str = "_".join(sorted(self.valid_camera_id_list))
        return f"waymo_scene_cache_{camera_ids_str}.json"

    def _load_data(self):
        """Load dataset metadata with caching support."""
        cache_path = osp.join(self.root, self._get_cache_filename())

        if osp.exists(cache_path):
            try:
                print(f"Loading scene metadata from cache: {cache_path}")
                with open(cache_path, 'r') as f:
                    self.scene_data = json.load(f)
                self.scene_names = sorted(list(self.scene_data.keys()))
                print(f"Loaded {len(self.scene_names)} scenes from cache")
                return
            except Exception as e:
                print(f"Warning: Failed to load cache file ({e}), performing full scan...")

        print("Performing full dataset scan...")
        scene_dirs = sorted([
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ])

        self.scene_data = {}

        for scene_name in scene_dirs:
            scene_dir = osp.join(self.root, scene_name)
            seq2frames = {}

            for f in os.listdir(scene_dir):
                if not f.endswith(".jpg"):
                    continue
                basename = f[:-4]
                frame_id = basename.split("_")[0]
                seq_id = basename.split("_")[1]

                if seq_id not in self.valid_camera_id_list:
                    continue

                if seq_id not in seq2frames:
                    seq2frames[seq_id] = []
                seq2frames[seq_id].append(frame_id)

            for seq_id in seq2frames:
                seq2frames[seq_id] = sorted(seq2frames[seq_id])

            if seq2frames:
                self.scene_data[scene_name] = seq2frames

        self.scene_names = sorted(list(self.scene_data.keys()))
        print(f"Loaded {len(self.scene_names)} scenes")

        # Save cache
        try:
            print(f"Saving scene metadata cache to: {cache_path}")
            with open(cache_path, 'w') as f:
                json.dump(self.scene_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache file ({e})")

    def __len__(self):
        return len(self.scene_names)

    def _crop_resize(self, image, depthmap, intrinsics, target_size):
        """Crop and resize image, depth and intrinsics to target size."""
        H, W = image.shape[:2]
        target_H, target_W = target_size

        # Calculate crop to match target aspect ratio
        target_ratio = target_W / target_H
        current_ratio = W / H

        if current_ratio > target_ratio:
            # Image is wider, crop width
            new_W = int(H * target_ratio)
            crop_left = (W - new_W) // 2
            crop_right = crop_left + new_W
            crop_top, crop_bottom = 0, H
        else:
            # Image is taller, crop height
            new_H = int(W / target_ratio)
            crop_top = (H - new_H) // 2
            crop_bottom = crop_top + new_H
            crop_left, crop_right = 0, W

        # Crop
        image = image[crop_top:crop_bottom, crop_left:crop_right]
        depthmap = depthmap[crop_top:crop_bottom, crop_left:crop_right]

        # Update intrinsics for crop
        intrinsics = intrinsics.copy()
        intrinsics[0, 2] -= crop_left
        intrinsics[1, 2] -= crop_top

        # Resize
        crop_H, crop_W = image.shape[:2]
        scale_x = target_W / crop_W
        scale_y = target_H / crop_H

        image = cv2.resize(image, (target_W, target_H), interpolation=cv2.INTER_LINEAR)
        depthmap = cv2.resize(depthmap, (target_W, target_H), interpolation=cv2.INTER_NEAREST)

        # Update intrinsics for resize
        intrinsics[0, 0] *= scale_x
        intrinsics[1, 1] *= scale_y
        intrinsics[0, 2] *= scale_x
        intrinsics[1, 2] *= scale_y

        return image, depthmap, intrinsics

    def _load_single_view(self, scene_dir, camera_id, frame_id, resolution):
        """Load a single view with all associated data."""
        impath = f"{frame_id}_{camera_id}"

        # Load image
        image = imread_cv2(osp.join(scene_dir, impath + ".jpg"))
        if image is None:
            image = np.array(Image.open(osp.join(scene_dir, impath + ".jpg")))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load depth
        depthmap = imread_cv2(osp.join(scene_dir, impath + ".exr"))
        if depthmap is not None and depthmap.ndim == 3:
            depthmap = depthmap[..., 0]

        # Load camera params
        camera_params = np.load(osp.join(scene_dir, impath + ".npz"))
        intrinsics = np.float32(camera_params["intrinsics"])
        camera_pose = np.float32(camera_params["cam2world"])

        # Crop and resize
        image, depthmap, intrinsics = self._crop_resize(
            image, depthmap, intrinsics, resolution
        )

        # Convert to tensors
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        depthmap_tensor = torch.from_numpy(depthmap).float()
        intrinsics_tensor = torch.from_numpy(intrinsics).float()
        camera_pose_tensor = torch.from_numpy(camera_pose).float()

        # Compute 3D points
        pts3d_cam, valid_mask = depthmap_to_camera_coordinates(
            depthmap=depthmap,
            camera_intrinsics=intrinsics
        )
        pts3d_tensor = torch.from_numpy(pts3d_cam).float()
        valid_mask_tensor = torch.from_numpy(valid_mask & np.isfinite(pts3d_cam).all(axis=-1)).bool()

        return {
            "img": img_tensor,
            "depthmap": depthmap_tensor,
            "camera_pose": camera_pose_tensor,
            "camera_intrinsics": intrinsics_tensor,
            "pts3d": pts3d_tensor,
            "valid_mask": valid_mask_tensor,
        }

    def __getitem__(self, idx):
        """Get a training sample."""
        rng = np.random.default_rng()

        scene_name = self.scene_names[idx]
        scene_dir = osp.join(self.root, scene_name)
        seq2frames = self.scene_data[scene_name]

        # Randomly select a camera
        available_cameras = sorted(list(seq2frames.keys()))
        camera_id = rng.choice(available_cameras)
        frame_ids = seq2frames[camera_id]

        # Random interval
        interval = rng.choice(self._intervals)

        # Calculate required frames
        required_frames = self.num_views * interval
        if len(frame_ids) < required_frames:
            interval = max(1, len(frame_ids) // self.num_views)
            required_frames = self.num_views * interval

        # Random start position
        max_start = len(frame_ids) - required_frames
        start_pos = rng.integers(0, max(1, max_start + 1))

        # Select context frame positions
        context_positions = [start_pos + i * interval for i in range(self.num_views)]

        # Load views
        views = []
        for pos in context_positions:
            if pos < len(frame_ids):
                frame_id = frame_ids[pos]
                view_dict = self._load_single_view(
                    scene_dir=scene_dir,
                    camera_id=camera_id,
                    frame_id=frame_id,
                    resolution=self.resolution,
                )
                views.append(view_dict)

        # Transform to reference frame (first frame)
        reference_cam_pose = views[0]['camera_pose']
        world_to_ref = torch.linalg.inv(reference_cam_pose)

        for view in views:
            pts3d_cam = view['pts3d']
            H, W, _ = pts3d_cam.shape
            cam_to_world = view['camera_pose']
            cam_to_ref = torch.matmul(world_to_ref, cam_to_world)

            pts3d_flat = pts3d_cam.reshape(-1, 3)
            pts3d_ref = torch.matmul(cam_to_ref[:3, :3], pts3d_flat.T).T + cam_to_ref[:3, 3]
            view['pts3d'] = pts3d_ref.reshape(H, W, 3)
            view['camera_pose'] = torch.linalg.inv(cam_to_ref)

        # Normalize by average depth
        all_pts = []
        for view in views:
            pts = view['pts3d']
            mask = view['valid_mask'].bool()
            all_pts.append(pts[mask])

        all_pts = torch.cat(all_pts, dim=0)
        if all_pts.numel() > 0:
            dist_avg = all_pts.norm(dim=-1).mean()
            depth_scale_factor = 1.0 / dist_avg
        else:
            depth_scale_factor = torch.tensor(1.0)

        for view in views:
            view['depthmap'] = view['depthmap'] * depth_scale_factor
            view['camera_pose'][:3, 3] = view['camera_pose'][:3, 3] * depth_scale_factor
            view['pts3d'] = view['pts3d'] * depth_scale_factor

        # Stack into batch format
        images = torch.stack([v['img'] for v in views], dim=0)  # [S, 3, H, W]
        depths = torch.stack([v['depthmap'] for v in views], dim=0)  # [S, H, W]
        extrinsics = torch.stack([v['camera_pose'] for v in views], dim=0)  # [S, 4, 4]
        intrinsics = torch.stack([v['camera_intrinsics'] for v in views], dim=0)  # [S, 3, 3]
        point_masks = torch.stack([v['valid_mask'] for v in views], dim=0)  # [S, H, W]
        world_points = torch.stack([v['pts3d'] for v in views], dim=0)  # [S, H, W, 3]

        return {
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "point_masks": point_masks,
            "world_points": world_points,
            "depth_scale_factor": depth_scale_factor,
            "scene_name": scene_name,
        }


def collate_fn(batch):
    """Custom collate function for Waymo dataset."""
    result = {}
    for key in batch[0].keys():
        if key in ["scene_name"]:
            result[key] = [b[key] for b in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch], dim=0)
        else:
            result[key] = [b[key] for b in batch]
    return result
