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
