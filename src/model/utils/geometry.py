# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Geometry utilities."""

import torch


def affine_inverse(A: torch.Tensor):
    """Compute the inverse of an affine transformation matrix.

    Args:
        A: Affine transformation matrix of shape (..., 4, 4)

    Returns:
        Inverse transformation matrix of shape (..., 4, 4)
    """
    R = A[..., :3, :3]  # ..., 3, 3
    T = A[..., :3, 3:]  # ..., 3, 1
    P = A[..., 3:, :]  # ..., 1, 4
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


def as_homogeneous(x: torch.Tensor) -> torch.Tensor:
    """Convert 3x4 matrix to 4x4 homogeneous matrix.

    Args:
        x: Matrix of shape (..., 3, 4) or (..., 4, 4)

    Returns:
        Homogeneous matrix of shape (..., 4, 4)
    """
    if x.shape[-2] == 4:
        return x
    # Add bottom row [0, 0, 0, 1]
    batch_shape = x.shape[:-2]
    bottom = torch.zeros(*batch_shape, 1, 4, device=x.device, dtype=x.dtype)
    bottom[..., 0, 3] = 1.0
    return torch.cat([x, bottom], dim=-2)


def homogenize_points(points: torch.Tensor) -> torch.Tensor:
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def inverse_intrinsic_matrix(ixts):
    """Compute the inverse of intrinsic matrix."""
    return torch.inverse(ixts)


def pixel_space_to_camera_space(pixel_space_points, depth, intrinsics):
    """
    Convert pixel space points to camera space points.

    Args:
        pixel_space_points (torch.Tensor): Pixel space points with shape (h, w, 2)
        depth (torch.Tensor): Depth map with shape (b, v, h, w, 1)
        intrinsics (torch.Tensor): Camera intrinsics with shape (b, v, 3, 3)

    Returns:
        torch.Tensor: Camera space points with shape (b, v, h, w, 3).
    """
    pixel_space_points = homogenize_points(pixel_space_points)
    camera_space_points = torch.einsum(
        "b v i j , h w j -> b v h w i", inverse_intrinsic_matrix(intrinsics), pixel_space_points
    )
    camera_space_points = camera_space_points * depth
    return camera_space_points


def camera_space_to_world_space(camera_space_points, c2w):
    """
    Convert camera space points to world space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """
    camera_space_points = homogenize_points(camera_space_points)
    world_space_points = torch.einsum("b v i j , b v h w j -> b v h w i", c2w, camera_space_points)
    return world_space_points[..., :3]


def unproject_depth(
    depth, intrinsics, c2w=None, ixt_normalized=False, num_patches_x=None, num_patches_y=None
):
    """
    Turn the depth map into a 3D point cloud in world space

    Args:
        depth: (b, v, h, w, 1)
        intrinsics: (b, v, 3, 3)
        c2w: (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """
    if c2w is None:
        c2w = torch.eye(4, device=depth.device, dtype=depth.dtype)
        c2w = c2w[None, None].repeat(depth.shape[0], depth.shape[1], 1, 1)

    if not ixt_normalized:
        # Compute indices of pixels
        h, w = depth.shape[-3], depth.shape[-2]
        x_grid, y_grid = torch.meshgrid(
            torch.arange(w, device=depth.device, dtype=depth.dtype),
            torch.arange(h, device=depth.device, dtype=depth.dtype),
            indexing="xy",
        )  # (h, w), (h, w)
    else:
        # ixt_normalized: h=w=2.0. cx, cy, fx, fy are normalized according to h=w=2.0
        assert num_patches_x is not None and num_patches_y is not None
        dx = 1 / num_patches_x
        dy = 1 / num_patches_y
        max_y = 1 - dy
        min_y = -max_y
        max_x = 1 - dx
        min_x = -max_x

        grid_shift = 1.0
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(
                min_y + grid_shift,
                max_y + grid_shift,
                num_patches_y,
                dtype=torch.float32,
                device=depth.device,
            ),
            torch.linspace(
                min_x + grid_shift,
                max_x + grid_shift,
                num_patches_x,
                dtype=torch.float32,
                device=depth.device,
            ),
            indexing="ij",
        )

    # Compute coordinates of pixels in camera space
    pixel_space_points = torch.stack((x_grid, y_grid), dim=-1)  # (..., h, w, 2)
    camera_points = pixel_space_to_camera_space(
        pixel_space_points, depth, intrinsics
    )  # (..., h, w, 3)

    # Convert points to world space
    world_points = camera_space_to_world_space(camera_points, c2w)  # (..., h, w, 3)

    return world_points
