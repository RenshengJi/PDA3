# Dataset utilities
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
from PIL import Image


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Read image using cv2 with support for various formats including EXR."""
    if path.endswith('.exr'):
        return cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    elif path.endswith('.png') or path.endswith('.jpg') or path.endswith('.jpeg'):
        img = cv2.imread(path, options)
        if img is None:
            # Fall back to PIL
            img = np.array(Image.open(path))
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    else:
        return cv2.imread(path, options)


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Convert a depth map to 3D camera coordinates.

    Args:
        depthmap: [H, W] or [H, W, 1] depth map
        camera_intrinsics: [3, 3] camera intrinsic matrix
        pseudo_focal: Optional pseudo focal length (not used if None)

    Returns:
        pts3d: [H, W, 3] 3D points in camera coordinates
        valid_mask: [H, W] valid depth mask
    """
    if depthmap.ndim == 3:
        depthmap = depthmap[..., 0]

    H, W = depthmap.shape
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    # Create pixel grid
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    # Convert to camera coordinates
    z = depthmap
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts3d = np.stack([x, y, z], axis=-1)

    # Valid mask: positive depth and finite values
    valid_mask = (z > 0) & np.isfinite(z)

    return pts3d, valid_mask


def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose):
    """
    Convert depth map to 3D points in world coordinates.

    Args:
        depthmap: [H, W] depth map
        camera_intrinsics: [3, 3] camera intrinsic matrix
        camera_pose: [4, 4] camera-to-world transformation matrix

    Returns:
        pts3d_world: [H, W, 3] 3D points in world coordinates
        valid_mask: [H, W] valid depth mask
    """
    pts3d_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    # Transform to world coordinates
    H, W = depthmap.shape
    pts3d_flat = pts3d_cam.reshape(-1, 3)

    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]

    pts3d_world = (R @ pts3d_flat.T).T + t
    pts3d_world = pts3d_world.reshape(H, W, 3)

    return pts3d_world, valid_mask
