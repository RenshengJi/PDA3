"""
Physical AI AV dataset for DA3 finetuning.
Loads multi-view images from NVIDIA's PhysicalAI-Autonomous-Vehicles dataset
with camera parameters and depth generated from LiDAR point clouds.

NOTE: This dataset assumes data has been pre-downloaded using:
    python data_preprocess/download_physical_ai_av.py

This implementation reads local data directly WITHOUT using PhysicalAIAVDatasetInterface,
so NO HuggingFace authentication is required.
"""
import io
import zipfile
import numpy as np
import torch
import cv2
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from .utils import depthmap_to_camera_coordinates
from scipy.spatial.transform import Rotation
from scipy.ndimage import minimum_filter

# Reuse the official SeekVideoReader from physical_ai_av package
from physical_ai_av.video import SeekVideoReader


class LocalDataInterface:
    """
    Local data interface for Physical AI AV dataset.
    Reads data directly from local files WITHOUT HuggingFace authentication.
    """

    def __init__(self, data_dir: str):
        """
        Initialize local data interface.

        Args:
            data_dir: Root directory where data was downloaded
        """
        self.data_dir = Path(data_dir)

        # Load metadata
        self.clip_index = pd.read_parquet(self.data_dir / 'clip_index.parquet')
        self.features_csv = pd.read_csv(self.data_dir / 'features.csv')

        # Build feature lookup
        self._feature_info = {}
        for _, row in self.features_csv.iterrows():
            self._feature_info[row['feature']] = {
                'directory': row['directory'],
                'chunk_path': row['chunk_path'],
            }

        print(f"Loaded clip index with {len(self.clip_index)} clips")

    def get_clip_chunk(self, clip_id: str) -> int:
        """Get chunk ID for a clip."""
        return int(self.clip_index.at[clip_id, 'chunk'])

    def get_clip_feature(self, clip_id: str, feature: str):
        """
        Get feature data for a clip.

        Args:
            clip_id: UUID of the clip
            feature: Feature name (camera name, 'lidar_top_360fov', 'camera_intrinsics', etc.)

        Returns:
            - For cameras: LocalVideoReader object
            - For lidar: dict with DataFrame
            - For calibration: DataFrame row for the clip
        """
        chunk_id = self.get_clip_chunk(clip_id)

        # Handle calibration features (parquet files, not zipped)
        if feature in ['camera_intrinsics', 'sensor_extrinsics', 'vehicle_dimensions']:
            return self._get_calibration_feature(clip_id, feature, chunk_id)

        # Handle camera features
        if feature.startswith('camera_'):
            return self._get_camera_feature(clip_id, feature, chunk_id)

        # Handle lidar
        if feature == 'lidar_top_360fov':
            return self._get_lidar_feature(clip_id, feature, chunk_id)

        raise ValueError(f"Unknown feature: {feature}")

    def _get_calibration_feature(self, clip_id: str, feature: str, chunk_id: int) -> pd.Series:
        """Get calibration data for a clip."""
        # Calibration files are parquet, not zipped
        parquet_path = self.data_dir / 'calibration' / feature / f'{feature}.chunk_{chunk_id:04d}.parquet'

        if not parquet_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)

        # Filter by clip_id
        if 'clip_id' in df.columns:
            clip_data = df[df['clip_id'] == clip_id]
            if len(clip_data) == 0:
                raise ValueError(f"No calibration data found for clip {clip_id}")
            # Set sensor_name as index if available
            if 'sensor_name' in clip_data.columns:
                clip_data = clip_data.set_index('sensor_name')
            return clip_data
        else:
            # Assume clip_id is the index
            return df.loc[clip_id] if clip_id in df.index else df

    def _get_camera_feature(self, clip_id: str, camera_name: str, chunk_id: int) -> SeekVideoReader:
        """Get camera video reader for a clip."""
        zip_path = self.data_dir / 'camera' / camera_name / f'{camera_name}.chunk_{chunk_id:04d}.zip'

        if not zip_path.exists():
            raise FileNotFoundError(f"Camera ZIP not found: {zip_path}")

        # Extract video and timestamps from ZIP
        mp4_file = f"{clip_id}.{camera_name}.mp4"
        timestamps_file = f"{clip_id}.{camera_name}.timestamps.parquet"

        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Read timestamps
            timestamps_df = pd.read_parquet(io.BytesIO(zf.read(timestamps_file)))
            timestamps = timestamps_df['timestamp'].to_numpy()

            # Read video data into BytesIO
            video_data = io.BytesIO(zf.read(mp4_file))

        # Create SeekVideoReader (from physical_ai_av package)
        video_data.seek(0)
        return SeekVideoReader(video_data, timestamps=timestamps)

    def _get_lidar_feature(self, clip_id: str, feature: str, chunk_id: int) -> dict:
        """Get lidar data for a clip."""
        zip_path = self.data_dir / 'lidar' / feature / f'{feature}.chunk_{chunk_id:04d}.zip'

        if not zip_path.exists():
            raise FileNotFoundError(f"LiDAR ZIP not found: {zip_path}")

        # Read lidar parquet from ZIP
        parquet_file = f"{clip_id}.{feature}.parquet"

        with zipfile.ZipFile(zip_path, 'r') as zf:
            lidar_df = pd.read_parquet(io.BytesIO(zf.read(parquet_file)))

        # Return in same format as PhysicalAIAVDatasetInterface
        return {clip_id: lidar_df}


class PhysicalAIAVDataset(Dataset):
    """
    Physical AI Autonomous Vehicles dataset for DA3 finetuning.
    Loads multi-view images with camera parameters and LiDAR-derived depth.

    IMPORTANT: Data must be pre-downloaded using:
        python data_preprocess/download_physical_ai_av.py

    This class reads local data directly - NO HuggingFace authentication required.
    """

    def __init__(
        self,
        root,
        valid_camera_list=None,
        intervals=[1],
        num_views=4,
        resolution=(518, 518),
        split="train",
        transform=None,
        use_lidar_depth=True,
        time_window_ms=4,
        max_clips=None,
        **kwargs
    ):
        """
        Initialize Physical AI AV dataset.

        Args:
            root: Root directory where data was downloaded (same as download script's output_dir)
            valid_camera_list: List of camera names to use. Default uses front cameras:
                             ["camera_front_wide_120fov", "camera_front_tele_30fov"]
            intervals: Frame sampling intervals
            num_views: Number of context frames to load
            resolution: Target image resolution (H, W)
            split: Dataset split ("train", "val", "test")
            transform: Image transform to apply
            use_lidar_depth: Whether to generate depth from LiDAR points
            time_window_ms: Time window in milliseconds for LiDAR-camera association
            max_clips: Maximum number of clips to use. Must match downloaded clips count.
        """
        self.root = root
        self.num_views = num_views
        self.resolution = resolution if isinstance(resolution, tuple) else (resolution, resolution)
        self.split = split
        self.transform = transform
        self.use_lidar_depth = use_lidar_depth
        self.time_window_us = time_window_ms * 1000  # Convert to microseconds
        self.max_clips = max_clips

        if not isinstance(intervals, list):
            intervals = [intervals]
        self._intervals = intervals

        # Default camera list (front cameras for better depth coverage)
        if valid_camera_list is None:
            self.valid_camera_list = [
                "camera_front_wide_120fov",
                "camera_front_tele_30fov",
            ]
        else:
            self.valid_camera_list = valid_camera_list

        print(f"Initializing Physical AI AV dataset from {root}")
        print(f"Using cameras: {self.valid_camera_list}")

        # Initialize local data interface (NO HF authentication required)
        self.data_interface = LocalDataInterface(root)

        # Get clip list based on split
        self._load_clip_list()

        print(f"Loaded {len(self.clip_ids)} clips for split '{split}'")

    def _load_clip_list(self):
        """Load and filter clip list based on split."""
        # Get all clip IDs
        all_clips = self.data_interface.clip_index.index.tolist()

        # If max_clips is specified, only use first N clips
        if self.max_clips is not None and self.max_clips > 0:
            all_clips = all_clips[:self.max_clips]
            print(f"Using only first {self.max_clips} clips (max_clips={self.max_clips})")

        # Split: first 80% train, next 10% val, last 10% test
        n_clips = len(all_clips)
        train_end = int(0.8 * n_clips)
        val_end = int(0.9 * n_clips)

        if self.split == "train":
            self.clip_ids = all_clips[:train_end]
        elif self.split == "val":
            self.clip_ids = all_clips[train_end:val_end]
        elif self.split == "test":
            self.clip_ids = all_clips[val_end:]
        else:
            self.clip_ids = all_clips

    def __len__(self):
        return len(self.clip_ids)

    def _get_transform_matrix(self, ext):
        """Build 4x4 transformation matrix from extrinsics."""
        quat = [ext['qx'], ext['qy'], ext['qz'], ext['qw']]
        trans = [ext['x'], ext['y'], ext['z']]
        R = Rotation.from_quat(quat).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = trans
        return T

    def _decode_lidar_timestamps(self, mesh, reference_timestamp):
        """Decode per-point timestamps from DracoPy mesh colors."""
        colors = np.array(mesh.colors) if mesh.colors is not None else None
        if colors is None or len(colors) == 0:
            return np.full(len(mesh.points), reference_timestamp, dtype=np.float64)

        green = colors[:, 1].astype(np.uint16)
        blue = colors[:, 2].astype(np.uint16)
        encoded_uint16 = (blue << 8) | green
        encoded_int16 = encoded_uint16.view(np.int16)
        scale = 105000.0 / 32767.0
        relative_ts = encoded_int16.astype(np.float64) * scale
        absolute_ts = reference_timestamp + relative_ts - 105000.0
        return absolute_ts

    def _project_lidar_to_depth(self, points_lidar, T_cam_from_lidar,
                                 fw_poly, cx, cy, width, height,
                                 occlusion_filter_window=3, occlusion_margin=0.5):
        """
        Project LiDAR points to camera to create depth map.
        Uses f-theta camera model for projection.

        Args:
            points_lidar: Nx3 LiDAR points in LiDAR frame
            T_cam_from_lidar: 4x4 transformation matrix from LiDAR to camera
            fw_poly: f-theta polynomial coefficients
            cx, cy: Principal point
            width, height: Image dimensions
            occlusion_filter_window: Window size for min-depth envelope filter (0 to disable)
            occlusion_margin: Depth margin (meters) for occlusion test

        Returns:
            depth_map: HxW depth map with occlusion-filtered sparse depth
        """
        if len(points_lidar) == 0:
            return np.zeros((height, width), dtype=np.float32)

        N = points_lidar.shape[0]
        points_homo = np.hstack([points_lidar, np.ones((N, 1))])
        points_cam = (T_cam_from_lidar @ points_homo.T).T[:, :3]

        x_cam, y_cam, z_cam = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        valid_depth = z_cam > 0.1

        # F-theta projection model
        r_xy = np.sqrt(x_cam**2 + y_cam**2)
        theta = np.arctan2(r_xy, z_cam)

        # Polynomial evaluation (vectorized)
        r_pixel = np.zeros_like(theta)
        theta_power = np.ones_like(theta)
        for coef in fw_poly:
            r_pixel += coef * theta_power
            theta_power *= theta

        phi = np.arctan2(y_cam, x_cam)
        u = cx + r_pixel * np.cos(phi)
        v = cy + r_pixel * np.sin(phi)

        valid_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        valid_mask = valid_depth & valid_bounds

        # Extract valid projections
        u_valid = u[valid_mask].astype(np.int32)
        v_valid = v[valid_mask].astype(np.int32)
        depth_valid = z_cam[valid_mask]

        # Create depth map
        depth_map = np.zeros((height, width), dtype=np.float32)

        # Handle empty case
        if len(u_valid) == 0:
            return depth_map

        # Vectorized depth assignment: use sorting to keep minimum depth per pixel
        # Convert 2D pixel coords to 1D index for grouping
        pixel_indices = v_valid * width + u_valid

        # Sort by pixel index, then by depth (ascending)
        sort_order = np.lexsort((depth_valid, pixel_indices))
        pixel_indices_sorted = pixel_indices[sort_order]
        depth_sorted = depth_valid[sort_order]

        # Find first occurrence of each unique pixel (minimum depth due to sorting)
        unique_mask = np.concatenate(([True], pixel_indices_sorted[1:] != pixel_indices_sorted[:-1]))
        unique_pixel_indices = pixel_indices_sorted[unique_mask]
        unique_depths = depth_sorted[unique_mask]

        # Convert back to 2D and fill depth map
        v_unique = unique_pixel_indices // width
        u_unique = unique_pixel_indices % width
        depth_map[v_unique, u_unique] = unique_depths

        # Apply occlusion filtering via local min-depth envelope
        if occlusion_filter_window > 0:
            depth_map = self._filter_occluded_points(
                depth_map,
                window_size=occlusion_filter_window,
                margin=occlusion_margin
            )

        return depth_map

    def _filter_occluded_points(self, depth_map, window_size=5, margin=0.5):
        """
        Filter potentially occluded points using local min-depth envelope.

        Strategy: Front surfaces are often hit by neighboring LiDAR rays.
        We compute a local minimum depth envelope D_front(u,v) by taking
        the min depth in a window around each pixel. Points significantly
        behind this envelope are likely occluded and should be removed.

        Args:
            depth_map: HxW sparse depth map
            window_size: Window size for computing local minimum (e.g., 5 = 5x5 window)
            margin: Depth margin in meters. Points with depth > D_front + margin are removed

        Returns:
            Filtered depth map
        """
        if window_size <= 1 or margin <= 0:
            return depth_map

        # Replace zeros with inf so they don't interfere with minimum
        depth_for_filter = depth_map.copy()
        depth_for_filter[depth_for_filter == 0] = np.inf

        # Compute local minimum depth envelope
        D_front = minimum_filter(depth_for_filter, size=window_size, mode='constant', cval=np.inf)

        # Filter: keep only points within margin of front surface
        # Points with depth > D_front + margin are likely occluded
        valid_mask = (depth_map > 0) & (depth_map <= D_front + margin)

        filtered_depth = depth_map.copy()
        filtered_depth[~valid_mask] = 0

        return filtered_depth

    def _get_intrinsics_matrix(self, cam_intrinsic, fw_poly):
        """
        Convert f-theta camera model to approximate pinhole intrinsics.
        This is an approximation for the center region of the image.
        """
        # Use first-order approximation: r ≈ f * theta for small angles
        # fw_poly[0] is offset (usually 0), fw_poly[1] is focal length approximation
        focal_approx = fw_poly[1]

        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = focal_approx  # fx
        intrinsics[1, 1] = focal_approx  # fy
        intrinsics[0, 2] = cam_intrinsic['cx']  # cx
        intrinsics[1, 2] = cam_intrinsic['cy']  # cy

        return intrinsics

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

    def _load_single_view(self, clip_id, camera_name, frame_idx,
                          camera_data, lidar_points_dict):
        """Load a single view with all associated data."""
        import DracoPy

        # Get camera reader and parameters
        reader = camera_data[camera_name]['reader']
        cam_intrinsic = camera_data[camera_name]['intrinsic']
        T_cam_from_lidar = camera_data[camera_name]['T_cam_from_lidar']
        fw_poly = camera_data[camera_name]['fw_poly']
        width = camera_data[camera_name]['width']
        height = camera_data[camera_name]['height']

        # Decode image
        image = reader.decode_images_from_frame_indices(np.array([frame_idx]))[0]

        # Get frame timestamp
        frame_timestamp = reader.timestamps[frame_idx]

        # Generate depth from LiDAR if available
        if self.use_lidar_depth and lidar_points_dict is not None:
            # Find LiDAR points within time window
            t_min = frame_timestamp - self.time_window_us
            t_max = frame_timestamp + self.time_window_us

            # Lazy decode: only decode scans that overlap with our time window
            scan_list = lidar_points_dict.get('scan_list', [])
            points_in_window = []

            for scan_data in scan_list:
                ref_ts = scan_data['reference_timestamp']
                # reference_timestamp is the END time of the scan
                # Each scan covers approximately 105ms before the reference timestamp
                scan_t_min = ref_ts - 105000  # scan start time (105ms before end)
                scan_t_max = ref_ts  # scan end time

                # Check if this scan overlaps with our time window
                if scan_t_max < t_min or scan_t_min > t_max:
                    continue  # Skip scans outside our time window

                # Decode this scan on demand
                mesh = DracoPy.decode(scan_data['draco_bytes'])
                points = np.array(mesh.points)
                timestamps = self._decode_lidar_timestamps(mesh, ref_ts)

                # Filter points within time window
                time_mask = (timestamps >= t_min) & (timestamps <= t_max)
                if time_mask.sum() > 0:
                    points_in_window.append(points[time_mask])

            if points_in_window:
                points_in_window = np.vstack(points_in_window)
            else:
                points_in_window = np.empty((0, 3))

            # Project to depth map
            depthmap = self._project_lidar_to_depth(
                points_in_window, T_cam_from_lidar,
                fw_poly, cam_intrinsic['cx'], cam_intrinsic['cy'],
                width, height
            )
        else:
            # No depth available
            depthmap = np.zeros((height, width), dtype=np.float32)

        # Get approximate pinhole intrinsics
        intrinsics = self._get_intrinsics_matrix(cam_intrinsic, fw_poly)

        # Camera pose (cam to rig, we'll need to invert for world to cam)
        T_rig_from_cam = camera_data[camera_name]['T_rig_from_cam']
        camera_pose = np.linalg.inv(T_rig_from_cam).astype(np.float32)

        # Crop and resize
        image, depthmap, intrinsics = self._crop_resize(
            image, depthmap, intrinsics, self.resolution
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
        valid_mask_tensor = torch.from_numpy(
            valid_mask & np.isfinite(pts3d_cam).all(axis=-1)
        ).bool()

        return {
            "img": img_tensor,
            "depthmap": depthmap_tensor,
            "camera_pose": camera_pose_tensor,
            "camera_intrinsics": intrinsics_tensor,
            "pts3d": pts3d_tensor,
            "valid_mask": valid_mask_tensor,
        }

    def _load_clip_data(self, clip_id, camera_name):
        """
        Load camera and LiDAR data for a clip from local files.

        Args:
            clip_id: UUID of the clip
            camera_name: Name of the camera to load (only loads this one camera)

        NOTE: This assumes data has been pre-downloaded. Will raise an error
        if required files are not found.
        """
        import DracoPy

        # Load camera intrinsics and extrinsics
        camera_intrinsics = self.data_interface.get_clip_feature(clip_id, "camera_intrinsics")
        sensor_extrinsics = self.data_interface.get_clip_feature(clip_id, "sensor_extrinsics")

        # Prepare camera data - only load the specified camera
        camera_data = {}
        lidar_ext = sensor_extrinsics.loc["lidar_top_360fov"]
        T_rig_from_lidar = self._get_transform_matrix(lidar_ext)

        # Only load the selected camera (not all cameras in valid_camera_list)
        cam_intrinsic = camera_intrinsics.loc[camera_name]
        cam_ext = sensor_extrinsics.loc[camera_name]

        T_rig_from_cam = self._get_transform_matrix(cam_ext)
        T_cam_from_rig = np.linalg.inv(T_rig_from_cam)
        T_cam_from_lidar = T_cam_from_rig @ T_rig_from_lidar

        fw_poly = np.array([
            cam_intrinsic['fw_poly_0'],
            cam_intrinsic['fw_poly_1'],
            cam_intrinsic['fw_poly_2'],
            cam_intrinsic['fw_poly_3'],
            cam_intrinsic['fw_poly_4']
        ])

        # Get camera reader for the selected camera only
        camera_data[camera_name] = {
            'reader': self.data_interface.get_clip_feature(clip_id, camera_name),
            'intrinsic': cam_intrinsic,
            'width': int(cam_intrinsic['width']),
            'height': int(cam_intrinsic['height']),
            'T_cam_from_lidar': T_cam_from_lidar,
            'T_rig_from_cam': T_rig_from_cam,
            'fw_poly': fw_poly,
        }

        # Load LiDAR data if needed (lazy decoding - store raw data, decode on demand)
        lidar_points_dict = None
        if self.use_lidar_depth:
            try:
                lidar_data = self.data_interface.get_clip_feature(clip_id, "lidar_top_360fov")
                lidar_df = list(lidar_data.values())[0]

                # Store raw scan data for lazy decoding (avoid decoding all scans upfront)
                scan_list = []
                for idx in range(len(lidar_df)):
                    scan = lidar_df.iloc[idx]
                    scan_list.append({
                        'draco_bytes': scan['draco_encoded_pointcloud'],
                        'reference_timestamp': scan['reference_timestamp'],
                    })

                lidar_points_dict = {
                    'scan_list': scan_list,
                    'decoded': False,  # Mark as not decoded yet
                }
            except Exception as e:
                print(f"Warning: Failed to load LiDAR data for clip {clip_id}: {e}")
                lidar_points_dict = None

        return camera_data, lidar_points_dict

    def _try_load_sample(self, idx, rng):
        """
        Try to load a sample from the given index.

        Returns:
            dict or None: Sample dict if valid (has depth points), None otherwise
        """
        clip_id = self.clip_ids[idx]

        # Select camera first, before loading data (optimization: only load selected camera)
        camera_name = rng.choice(self.valid_camera_list)

        # Load clip data (from local files only, no downloading)
        try:
            import time
            t0 = time.time()
            camera_data, lidar_points_dict = self._load_clip_data(clip_id, camera_name)
            # print(f"[Timing] _load_clip_data took {time.time() - t0:.3f}s for clip {clip_id[:8]}...")
        except Exception as e:
            print(f"Warning: Failed to load clip {clip_id}: {e}")
            return None

        reader = camera_data[camera_name]['reader']
        num_frames = len(reader.timestamps)

        # Random interval
        interval = rng.choice(self._intervals)

        # Calculate required frames
        required_frames = self.num_views * interval
        if num_frames < required_frames:
            interval = max(1, num_frames // self.num_views)
            required_frames = self.num_views * interval

        # Random start position
        max_start = num_frames - required_frames
        start_pos = rng.integers(0, max(1, max_start + 1))

        # Select context frame positions
        context_positions = [start_pos + i * interval for i in range(self.num_views)]

        # Load views
        views = []
        for pos in context_positions:
            if pos < num_frames:
                view_dict = self._load_single_view(
                    clip_id=clip_id,
                    camera_name=camera_name,
                    frame_idx=pos,
                    camera_data=camera_data,
                    lidar_points_dict=lidar_points_dict,
                )
                views.append(view_dict)

        # Close camera readers to free resources
        for cam_name in camera_data:
            if hasattr(camera_data[cam_name]['reader'], 'close'):
                camera_data[cam_name]['reader'].close()

        # Check if we have any valid depth points
        total_valid_points = sum(v['valid_mask'].sum().item() for v in views)
        if total_valid_points == 0:
            # No valid depth points, skip this sample
            return None

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
        images = torch.stack([v['img'] for v in views], dim=0)
        depths = torch.stack([v['depthmap'] for v in views], dim=0)
        extrinsics = torch.stack([v['camera_pose'] for v in views], dim=0)
        intrinsics = torch.stack([v['camera_intrinsics'] for v in views], dim=0)
        point_masks = torch.stack([v['valid_mask'] for v in views], dim=0)
        world_points = torch.stack([v['pts3d'] for v in views], dim=0)

        return {
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "point_masks": point_masks,
            "world_points": world_points,
            "depth_scale_factor": depth_scale_factor,
            "scene_name": clip_id,
        }

    def __getitem__(self, idx):
        """
        Get a training sample.

        If the sample has no valid depth points (point_mask all empty),
        will try other random indices until finding a valid sample.
        """
        rng = np.random.default_rng()
        max_retries = 100  # Prevent infinite loop

        # Try the requested index first
        sample = self._try_load_sample(idx, rng)
        if sample is not None:
            return sample

        # If failed, try random other indices
        for _ in range(max_retries):
            new_idx = rng.integers(0, len(self.clip_ids))
            sample = self._try_load_sample(new_idx, rng)
            if sample is not None:
                return sample

        # If all retries failed, raise an error
        raise RuntimeError(
            f"Failed to find a valid sample after {max_retries} retries. "
            "All samples have empty point_masks. Check your LiDAR data and projection."
        )


def collate_fn(batch):
    """Custom collate function for Physical AI AV dataset."""
    result = {}
    for key in batch[0].keys():
        if key in ["scene_name"]:
            result[key] = [b[key] for b in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch], dim=0)
        else:
            result[key] = [b[key] for b in batch]
    return result
