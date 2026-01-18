"""
Loss functions for DA3 training.
Based on the original DA3 paper: https://arxiv.org/abs/2511.10647

Loss components (from paper Section 3.3):
L = L_D (depth) + α * L_grad (gradient) + L_M (ray) + L_P (point) + β * L_C (camera)

Where:
- L_D: Confidence-weighted L1 depth loss with log regularization
- L_grad: Edge-aware gradient loss (α=1.0)
- L_M: Ray map prediction loss
- L_P: 3D point cloud loss
- L_C: Camera pose loss (β=1.0)

Teacher model usage:
- Teacher generates pseudo-depth labels for real-world data
- Supervision switches from GT depth to teacher labels at 120k steps (out of 200k)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor

from src.model.utils.transform import extri_intri_to_pose_encoding
from src.model.utils.ray_utils import get_extrinsic_from_camray


def _solve_least_squares_affine(pred_flat, gt_flat, with_shift=True, eps=1e-8):
    """
    Solve least squares for affine alignment: gt = scale * pred + shift

    Args:
        pred_flat: Flattened predicted depth values
        gt_flat: Flattened GT depth values
        with_shift: Whether to compute shift or just scale
        eps: Small epsilon for numerical stability

    Returns:
        scale, shift: Alignment parameters
    """
    device = pred_flat.device

    if with_shift:
        n = pred_flat.numel()
        sum_pred = pred_flat.sum()
        sum_gt = gt_flat.sum()
        sum_pred_sq = (pred_flat * pred_flat).sum()
        sum_pred_gt = (pred_flat * gt_flat).sum()

        det = sum_pred_sq * n - sum_pred * sum_pred
        if det.abs() < eps:
            scale = sum_pred_gt / sum_pred_sq.clamp_min(eps)
            shift = torch.tensor(0.0, device=device)
        else:
            scale = (sum_pred_gt * n - sum_pred * sum_gt) / det
            shift = (sum_pred_sq * sum_gt - sum_pred * sum_pred_gt) / det
    else:
        scale = (gt_flat * pred_flat).sum() / (pred_flat * pred_flat).sum().clamp_min(eps)
        shift = torch.tensor(0.0, device=device)

    return scale, shift


def align_depth_least_squares(
    pred_depth, gt_depth, valid_mask=None, with_shift=True,
    use_ransac=True, ransac_iters=100, ransac_sample_ratio=0.5,
    inlier_thresh=None, eps=1e-8
):
    """
    Align predicted depth to GT depth using least squares with RANSAC.

    Based on DA3 paper Section 4.2: Teaching Depth Anything 3
    For affine-invariant depth, we solve: gt = scale * pred + shift

    RANSAC is used to robustly estimate scale/shift by:
    1. Randomly sample a subset of points
    2. Fit scale/shift on the subset
    3. Count inliers based on residual threshold
    4. Keep the best model with most inliers
    5. Refit on all inliers

    Args:
        pred_depth: [B, S, H, W] or [B, S, H, W, 1] - Predicted/teacher depth
        gt_depth: [B, S, H, W] or [B, S, H, W, 1] - Ground truth depth
        valid_mask: [B, S, H, W] - Valid depth mask (optional)
        with_shift: Whether to compute shift (affine) or just scale
        use_ransac: Whether to use RANSAC for robust estimation
        ransac_iters: Number of RANSAC iterations
        ransac_sample_ratio: Ratio of points to sample in each iteration
        inlier_thresh: Inlier threshold (auto-computed if None)
        eps: Small epsilon for numerical stability

    Returns:
        aligned_depth: [B, S, H, W] - Aligned depth matching GT scale/shift
        scale: Scale factor used
        shift: Shift factor used (0 if with_shift=False)
    """
    # Handle dimension
    if pred_depth.dim() == 5:
        pred_depth = pred_depth.squeeze(-1)
    if gt_depth.dim() == 5:
        gt_depth = gt_depth.squeeze(-1)

    # Create valid mask if not provided
    if valid_mask is None:
        valid_mask = (gt_depth > eps) & (pred_depth > eps)
    else:
        valid_mask = valid_mask & (gt_depth > eps) & (pred_depth > eps)

    # Flatten for computation
    pred_flat = pred_depth[valid_mask]
    gt_flat = gt_depth[valid_mask]

    device = pred_depth.device
    n_valid = pred_flat.numel()

    if n_valid < 10:
        # Not enough valid points, return unaligned
        return pred_depth.clone(), torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)

    if not use_ransac or n_valid < 100:
        # Direct least squares without RANSAC
        scale, shift = _solve_least_squares_affine(pred_flat, gt_flat, with_shift, eps)
    else:
        # RANSAC-based robust estimation
        n_samples = max(10, int(n_valid * ransac_sample_ratio))

        # Auto-compute inlier threshold based on initial fit residuals
        if inlier_thresh is None:
            init_scale, init_shift = _solve_least_squares_affine(pred_flat, gt_flat, with_shift, eps)
            residuals = torch.abs(gt_flat - (init_scale * pred_flat + init_shift))
            inlier_thresh = torch.median(residuals) * 1.5  # 1.5x median residual

        best_scale = torch.tensor(1.0, device=device)
        best_shift = torch.tensor(0.0, device=device)
        best_inlier_count = 0
        best_inlier_mask = None

        for _ in range(ransac_iters):
            # Random sample
            idx = torch.randperm(n_valid, device=device)[:n_samples]
            sample_pred = pred_flat[idx]
            sample_gt = gt_flat[idx]

            # Fit on sample
            scale_cand, shift_cand = _solve_least_squares_affine(sample_pred, sample_gt, with_shift, eps)

            # Count inliers
            residuals = torch.abs(gt_flat - (scale_cand * pred_flat + shift_cand))
            inlier_mask = residuals < inlier_thresh
            inlier_count = inlier_mask.sum().item()

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_scale = scale_cand
                best_shift = shift_cand
                best_inlier_mask = inlier_mask

        # Refit on all inliers
        if best_inlier_mask is not None and best_inlier_count > 10:
            inlier_pred = pred_flat[best_inlier_mask]
            inlier_gt = gt_flat[best_inlier_mask]
            scale, shift = _solve_least_squares_affine(inlier_pred, inlier_gt, with_shift, eps)
        else:
            scale, shift = best_scale, best_shift

    # Clamp scale to reasonable range
    scale = scale.clamp(min=0.01, max=100.0)

    # Apply alignment
    aligned_depth = scale * pred_depth + shift

    # Ensure non-negative depth
    aligned_depth = aligned_depth.clamp(min=0.0)

    return aligned_depth, scale, shift


def check_and_fix_inf_nan(loss_tensor, loss_name, hard_max=100):
    """
    Check and fix inf/nan values in loss tensor.

    Args:
        loss_tensor: Loss tensor to check
        loss_name: Name of the loss for diagnostic prints
        hard_max: Maximum allowed value

    Returns:
        torch.Tensor: Fixed loss tensor with inf/nan replaced by 0
    """
    if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
        print(f"{loss_name} has inf or nan. Setting those values to 0.")
        loss_tensor = torch.where(
            torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
            torch.tensor(0.0, device=loss_tensor.device, requires_grad=True),
            loss_tensor
        )

    loss_tensor = torch.clamp(loss_tensor, min=-hard_max, max=hard_max)

    return loss_tensor


def torch_quantile(
    input: torch.Tensor,
    q: float,
    dim: int = None,
    keepdim: bool = False,
    interpolation: str = "nearest",
) -> torch.Tensor:
    """Efficient quantile computation using torch.kthvalue."""
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    if dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))
        dim_was_none = True
    else:
        dim_was_none = False

    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(f"Supported interpolations are {{'nearest', 'lower', 'higher'}}")

    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True)[0]

    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """Filter loss tensor by quantile threshold and clamp to maximum value."""
    if loss_tensor.numel() <= 1000:
        return loss_tensor

    if loss_tensor.numel() > 100000000:
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    loss_tensor = loss_tensor.clamp(max=hard_max)

    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def normalize_pointcloud(pts3d, valid_mask, eps=1e-3):
    """
    Normalize point cloud by average distance.

    Args:
        pts3d: [B, S, H, W, 3] - 3D points
        valid_mask: [B, S, H, W] - Valid point mask
        eps: Small epsilon for numerical stability

    Returns:
        pts3d: Normalized 3D points
        avg_scale: Average scale factor
    """
    dist = pts3d.norm(dim=-1)

    dist_sum = (dist * valid_mask).sum(dim=[1, 2, 3])
    valid_count = valid_mask.sum(dim=[1, 2, 3])

    avg_scale = (dist_sum / (valid_count + eps)).clamp(min=eps, max=1e3)

    pts3d = pts3d / avg_scale.view(-1, 1, 1, 1, 1)
    return pts3d, avg_scale


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss computing L1 difference between adjacent pixels.

    Based on DA3 paper Equation 3:
    L_grad = ||∇_x D̂ - ∇_x D||_1 + ||∇_y D̂ - ∇_y D||_1

    Args:
        prediction: [B, H, W, C] - Predicted values
        target: [B, H, W, C] - Ground truth values
        mask: [B, H, W] - Valid pixel mask
        conf: [B, H, W] - Confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization

    Returns:
        Gradient loss scalar
    """
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Horizontal gradient
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Vertical gradient
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Optional confidence weighting
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x.clamp(min=1e-6))
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y.clamp(min=1e-6))

    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return torch.tensor(0.0, device=prediction.device, requires_grad=True)
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def gradient_loss_multi_scale(prediction, target, mask, scales=4,
                              gradient_loss_fn=gradient_loss, conf=None):
    """
    Multi-scale gradient loss.

    Args:
        prediction: [B, H, W, C] - Predicted values
        target: [B, H, W, C] - Ground truth values
        mask: [B, H, W] - Valid mask
        scales: Number of scales
        gradient_loss_fn: Gradient loss function to use
        conf: Confidence (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)
        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Compute surface normals from 3D point map using cross products.

    Args:
        point_map: [B, H, W, 3] - 3D points in 2D grid layout
        mask: [B, H, W] - Valid pixel mask

    Returns:
        normals: [4, B, H, W, 3] - Normal vectors for 4 cross-product directions
        valids: [4, B, H, W] - Corresponding valid masks
    """
    with torch.cuda.amp.autocast(enabled=False):
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1, 1, 1, 1),
                    mode='constant', value=0).permute(0, 2, 3, 1)

        center = pts[:, 1:-1, 1:-1, :]
        up = pts[:, :-2,  1:-1, :]
        left = pts[:, 1:-1, :-2, :]
        down = pts[:, 2:,   1:-1, :]
        right = pts[:, 1:-1, 2:, :]

        up_dir = up - center
        left_dir = left - center
        down_dir = down - center
        right_dir = right - center

        n1 = torch.cross(up_dir,   left_dir,  dim=-1)
        n2 = torch.cross(left_dir, down_dir,  dim=-1)
        n3 = torch.cross(down_dir, right_dir, dim=-1)
        n4 = torch.cross(right_dir, up_dir,    dim=-1)

        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        normals = torch.stack([n1, n2, n3, n4], dim=0)
        valids = torch.stack([v1, v2, v3, v4], dim=0)

        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None):
    """
    Normal-based loss comparing angles between predicted and ground truth normals.

    Args:
        prediction: [B, H, W, 3] - Predicted 3D points
        target: [B, H, W, 3] - Ground truth 3D points
        mask: [B, H, W] - Valid pixel mask
        cos_eps: Epsilon for numerical stability
        conf: [B, H, W] - Confidence weights (optional)

    Returns:
        Scalar loss averaged over valid regions
    """
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals, gt_valids = point_map_to_normal(target, mask, eps=cos_eps)

    all_valid = pred_valids & gt_valids

    divisor = torch.sum(all_valid)
    if divisor < 10:
        return torch.tensor(0.0, device=prediction.device, requires_grad=True)

    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    dot = torch.sum(pred_normals * gt_normals, dim=-1)
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    loss = 1 - dot

    if loss.numel() < 10:
        return torch.tensor(0.0, device=prediction.device, requires_grad=True)
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            gamma = 1.0
            alpha = 0.2

            loss = gamma * loss * conf - alpha * torch.log(conf.clamp(min=1e-6))
            return loss.mean()
        else:
            return loss.mean()


def reg_loss(pts3d, gt_pts3d, valid_mask, gradient_loss_type=None):
    """
    Compute regression loss (L2 norm) with optional gradient loss.

    Separates first frame and other frames for loss computation.
    """
    first_frame_pts3d = pts3d[:, 0:1, ...]
    first_frame_gt_pts3d = gt_pts3d[:, 0:1, ...]
    first_frame_mask = valid_mask[:, 0:1, ...]

    other_frames_pts3d = pts3d[:, 1:, ...]
    other_frames_gt_pts3d = gt_pts3d[:, 1:, ...]
    other_frames_mask = valid_mask[:, 1:, ...]

    # L2 norm (Euclidean distance)
    loss_reg_first_frame = torch.norm(
        first_frame_gt_pts3d[first_frame_mask] - first_frame_pts3d[first_frame_mask], dim=-1)
    loss_reg_other_frames = torch.norm(
        other_frames_gt_pts3d[other_frames_mask] - other_frames_pts3d[other_frames_mask], dim=-1)

    # Optional gradient loss
    if gradient_loss_type == "grad":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(
            first_frame_pts3d.reshape(bb*ss, hh, ww, nc),
            first_frame_gt_pts3d.reshape(bb*ss, hh, ww, nc),
            first_frame_mask.reshape(bb*ss, hh, ww)
        )
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(
            other_frames_pts3d.reshape(bb*ss, hh, ww, nc),
            other_frames_gt_pts3d.reshape(bb*ss, hh, ww, nc),
            other_frames_mask.reshape(bb*ss, hh, ww)
        )
    elif gradient_loss_type == "normal":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(
            first_frame_pts3d.reshape(bb*ss, hh, ww, nc),
            first_frame_gt_pts3d.reshape(bb*ss, hh, ww, nc),
            first_frame_mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss, scales=3
        )
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(
            other_frames_pts3d.reshape(bb*ss, hh, ww, nc),
            other_frames_gt_pts3d.reshape(bb*ss, hh, ww, nc),
            other_frames_mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss, scales=3
        )
    else:
        loss_grad_first_frame = 0
        loss_grad_other_frames = 0

    loss_reg_first_frame = check_and_fix_inf_nan(loss_reg_first_frame, "loss_reg_first_frame")
    loss_reg_other_frames = check_and_fix_inf_nan(loss_reg_other_frames, "loss_reg_other_frames")

    return loss_reg_first_frame, loss_reg_other_frames, loss_grad_first_frame, loss_grad_other_frames


def conf_loss(pts3d, pts3d_conf, gt_pts3d, valid_mask, batch,
              normalize_gt=True, normalize_pred=True, gamma=1.0, alpha=0.2,
              gradient_loss_type=None, valid_range=-1, disable_conf=False,
              all_mean=False, postfix=""):
    """
    Confidence-weighted loss for depth/3D point predictions.

    Based on DA3 paper:
    L = (1/Z) Σ m_p * (D_c * |D̂ - D| - λ * log(D_c))

    Args:
        pts3d: Predicted depth/3D points
        pts3d_conf: Prediction confidence
        gt_pts3d: Ground truth
        valid_mask: Valid pixel mask
        batch: Batch dictionary
        normalize_gt: Whether to normalize ground truth
        normalize_pred: Whether to normalize predictions
        gamma: Confidence loss weight
        alpha: Confidence regularization weight (λ in paper)
        gradient_loss_type: Type of gradient loss ("grad" or "normal")
        valid_range: Quantile range for filtering
        disable_conf: Whether to disable confidence weighting
        all_mean: Whether to average all losses
        postfix: Suffix for loss names

    Returns:
        dict: Loss dictionary
    """
    if normalize_gt:
        gt_pts3d, gt_pts3d_scale = normalize_pointcloud(gt_pts3d, valid_mask)

    if normalize_pred:
        pts3d, pred_pts3d_scale = normalize_pointcloud(pts3d, valid_mask)

    loss_reg_first_frame, loss_reg_other_frames, loss_grad_first_frame, loss_grad_other_frames = reg_loss(
        pts3d, gt_pts3d, valid_mask, gradient_loss_type=gradient_loss_type)

    if disable_conf:
        conf_loss_first_frame = gamma * loss_reg_first_frame
        conf_loss_other_frames = gamma * loss_reg_other_frames
    else:
        first_frame_conf = pts3d_conf[:, 0:1, ...]
        other_frames_conf = pts3d_conf[:, 1:, ...]
        first_frame_mask = valid_mask[:, 0:1, ...]
        other_frames_mask = valid_mask[:, 1:, ...]

        # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
        conf_loss_first_frame = gamma * loss_reg_first_frame * \
            first_frame_conf[first_frame_mask] - alpha * \
            torch.log(first_frame_conf[first_frame_mask].clamp(min=1e-6))
        conf_loss_other_frames = gamma * loss_reg_other_frames * \
            other_frames_conf[other_frames_mask] - alpha * \
            torch.log(other_frames_conf[other_frames_mask].clamp(min=1e-6))

    if conf_loss_first_frame.numel() > 0 and conf_loss_other_frames.numel() > 0:
        if valid_range > 0:
            conf_loss_first_frame = filter_by_quantile(conf_loss_first_frame, valid_range)
            conf_loss_other_frames = filter_by_quantile(conf_loss_other_frames, valid_range)

        conf_loss_first_frame = check_and_fix_inf_nan(
            conf_loss_first_frame, f"conf_loss_first_frame{postfix}")
        conf_loss_other_frames = check_and_fix_inf_nan(
            conf_loss_other_frames, f"conf_loss_other_frames{postfix}")
    else:
        conf_loss_first_frame = pts3d.new_zeros(1, requires_grad=True).squeeze()
        conf_loss_other_frames = pts3d.new_zeros(1, requires_grad=True).squeeze()

    if all_mean and conf_loss_first_frame.numel() > 0 and conf_loss_other_frames.numel() > 0:
        all_conf_loss = torch.cat([conf_loss_first_frame, conf_loss_other_frames])
        conf_loss_val = all_conf_loss.mean() if all_conf_loss.numel() > 0 else 0

        conf_loss_first_frame = conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0
    else:
        conf_loss_first_frame = conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0
        conf_loss_val = conf_loss_first_frame + conf_loss_other_frames

    reg_loss_value = (loss_reg_first_frame.mean() if loss_reg_first_frame.numel() > 0 else 0) + \
               (loss_reg_other_frames.mean() if loss_reg_other_frames.numel() > 0 else 0)

    loss_dict = {
        f"loss_conf{postfix}": conf_loss_val,
        f"loss_reg{postfix}": reg_loss_value,
        f"loss_reg1{postfix}": loss_reg_first_frame.detach().mean() if isinstance(loss_reg_first_frame, torch.Tensor) and loss_reg_first_frame.numel() > 0 else 0,
        f"loss_reg2{postfix}": loss_reg_other_frames.detach().mean() if isinstance(loss_reg_other_frames, torch.Tensor) and loss_reg_other_frames.numel() > 0 else 0,
        f"loss_conf1{postfix}": conf_loss_first_frame,
        f"loss_conf2{postfix}": conf_loss_other_frames,
    }

    if gradient_loss_type is not None:
        loss_grad = loss_grad_first_frame + loss_grad_other_frames
        loss_dict[f"loss_grad1{postfix}"] = loss_grad_first_frame
        loss_dict[f"loss_grad2{postfix}"] = loss_grad_other_frames
        loss_dict[f"loss_grad{postfix}"] = loss_grad

    return loss_dict


def depth_loss(depth, depth_conf, batch, gamma=1.0, alpha=0.2,
               gradient_loss_type=None, valid_range=-1, disable_conf=False,
               all_mean=False, **kwargs):
    """
    Depth loss with confidence weighting and gradient regularization.

    Based on DA3 paper Section 3.3.

    Args:
        depth: [B, S, H, W, 1] - Predicted depth
        depth_conf: [B, S, H, W] - Depth confidence
        batch: Batch dictionary containing GT depths and masks
        gamma: Confidence loss weight
        alpha: Confidence regularization weight
        gradient_loss_type: Type of gradient loss
        valid_range: Quantile range for filtering outliers
        disable_conf: Whether to disable confidence weighting
        all_mean: Whether to average all losses

    Returns:
        dict: Loss dictionary
    """
    gt_depth = batch['depths'].clone()
    valid_mask = batch['point_masks']

    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth[..., None]

    conf_loss_dict = conf_loss(
        depth, depth_conf, gt_depth, valid_mask,
        batch, normalize_pred=False, normalize_gt=False,
        gamma=gamma, alpha=alpha, gradient_loss_type=gradient_loss_type,
        valid_range=valid_range, postfix="_depth",
        disable_conf=disable_conf, all_mean=all_mean
    )

    return conf_loss_dict


def point_loss(pts3d, pts3d_conf, batch, normalize_pred=True, gamma=1.0, alpha=0.2,
               gradient_loss_type=None, valid_range=-1, disable_conf=False,
               all_mean=False, **kwargs):
    """
    3D point cloud loss (L_P in paper).

    Based on DA3 paper: L_P(D̂⊙d + t, P)
    Projects predicted depth and rays into world coordinates and supervises
    against ground-truth 3D points.

    Args:
        pts3d: [B, S, H, W, 3] - Predicted 3D points
        pts3d_conf: [B, S, H, W] - Prediction confidence
        batch: Batch dictionary containing GT world_points
        normalize_pred: Whether to normalize predictions
        gamma: Confidence loss weight
        alpha: Confidence regularization weight
        gradient_loss_type: Type of gradient loss
        valid_range: Quantile range for filtering
        disable_conf: Whether to disable confidence weighting
        all_mean: Whether to average all losses

    Returns:
        dict: Loss dictionary
    """
    gt_pts3d = batch['world_points']
    valid_mask = batch['point_masks']
    gt_pts3d = check_and_fix_inf_nan(gt_pts3d, "gt_pts3d")

    conf_loss_dict = conf_loss(
        pts3d, pts3d_conf, gt_pts3d, valid_mask,
        batch, normalize_pred=normalize_pred, normalize_gt=True,
        gamma=gamma, alpha=alpha, gradient_loss_type=gradient_loss_type,
        valid_range=valid_range, disable_conf=disable_conf,
        all_mean=all_mean, postfix="_point"
    )

    return conf_loss_dict


def ray_loss(pred_ray, gt_ray, gamma=1.0):
    """
    Ray map prediction loss (L_M in paper).

    Each pixel ray r = (d, t) ∈ R^6, where:
    - d: ray direction (first 3 channels, unnormalized to preserve projection scale)
    - t: ray origin/camera center (last 3 channels)

    Note: No mask is needed for ray loss as it applies to all pixels.

    Args:
        pred_ray: [B, S, H, W, 6] predicted ray map
        gt_ray: [B, S, H, W, 6] ground truth ray map
        gamma: Loss weight

    Returns:
        dict: Loss dictionary with ray loss
    """
    if pred_ray is None or gt_ray is None:
        return {"loss_ray": torch.tensor(0.0, requires_grad=True)}

    # Ensure same spatial dimensions
    if pred_ray.shape[:-1] != gt_ray.shape[:-1]:
        B, S = pred_ray.shape[:2]
        pred_ray_flat = pred_ray.permute(0, 1, 4, 2, 3).reshape(B * S, 6, pred_ray.shape[2], pred_ray.shape[3])
        pred_ray_flat = F.interpolate(
            pred_ray_flat,
            size=gt_ray.shape[2:4],
            mode='bilinear',
            align_corners=True
        )
        pred_ray = pred_ray_flat.reshape(B, S, 6, gt_ray.shape[2], gt_ray.shape[3]).permute(0, 1, 3, 4, 2)

    # L1 loss on ray components (no mask needed)
    loss = torch.abs(pred_ray - gt_ray)

    # Average over all elements
    loss = loss.mean()

    loss = gamma * loss
    loss = check_and_fix_inf_nan(loss, "ray_loss")

    return {"loss_ray": loss}


def compute_ray_from_camera(extrinsics, intrinsics, H, W, device):
    """
    Compute ray map from camera parameters.

    Args:
        extrinsics: [B, S, 4, 4] camera extrinsics (world_to_cam)
        intrinsics: [B, S, 3, 3] camera intrinsics
        H, W: Image height and width
        device: Torch device

    Returns:
        ray_map: [B, S, H, W, 6] ray map (d, t) for each pixel
                 First 3 channels: direction (d)
                 Last 3 channels: origin/translation (t)
    """
    B, S = extrinsics.shape[:2]

    R = extrinsics[..., :3, :3]
    t = extrinsics[..., :3, 3]

    R_inv = R.transpose(-1, -2)
    cam_center = -torch.bmm(
        R_inv.reshape(B * S, 3, 3),
        t.reshape(B * S, 3, 1)
    ).reshape(B, S, 3)

    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]

    x_cam = (u[None, None] - cx[..., None, None]) / fx[..., None, None]
    y_cam = (v[None, None] - cy[..., None, None]) / fy[..., None, None]
    z_cam = torch.ones_like(x_cam)

    dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    dirs_cam_flat = dirs_cam.reshape(B * S, H * W, 3)

    dirs_world = torch.bmm(
        dirs_cam_flat,
        R_inv.reshape(B * S, 3, 3).transpose(-1, -2)
    ).reshape(B, S, H, W, 3)

    cam_center_expanded = cam_center[:, :, None, None, :].expand(B, S, H, W, 3)

    # Output format: [d, t] - direction first, then origin
    # This matches the model prediction format
    ray_map = torch.cat([dirs_world, cam_center_expanded], dim=-1)

    return ray_map


def depth_to_world_points(depth, extrinsics, intrinsics):
    """
    Convert depth map to 3D world points.

    Args:
        depth: [B, S, H, W] or [B, S, H, W, 1] - Depth map
        extrinsics: [B, S, 4, 4] - Camera extrinsics (world_to_cam)
        intrinsics: [B, S, 3, 3] - Camera intrinsics

    Returns:
        world_points: [B, S, H, W, 3] - 3D points in world coordinates
    """
    if depth.dim() == 5:
        depth = depth.squeeze(-1)

    B, S, H, W = depth.shape
    device = depth.device

    # Create pixel grid
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # Get camera intrinsics
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]

    # Pixel to camera coordinates
    x_cam = (u[None, None] - cx[..., None, None]) / fx[..., None, None] * depth
    y_cam = (v[None, None] - cy[..., None, None]) / fy[..., None, None] * depth
    z_cam = depth

    # Stack to get camera coordinates
    cam_points = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [B, S, H, W, 3]

    # Transform to world coordinates
    R = extrinsics[..., :3, :3]
    t = extrinsics[..., :3, 3]

    R_inv = R.transpose(-1, -2)

    # Reshape for batch matrix multiplication
    cam_points_flat = cam_points.reshape(B * S, H * W, 3)
    t_expanded = t.reshape(B * S, 1, 3).expand(-1, H * W, -1)

    # world = R^T @ (cam - t) for w2c, or cam = R @ world + t
    # So world = R^T @ cam - R^T @ t  FIXME: dirty
    world_points = torch.bmm(
        cam_points_flat - t_expanded,
        R_inv.reshape(B * S, 3, 3)
    ).reshape(B, S, H, W, 3)

    return world_points


def camera_loss(pred_pose_enc_list, batch, loss_type="l1", gamma=0.6,
                weight_T=1.0, weight_R=1.0, weight_fl=0.5):
    """
    Camera pose loss (L_C in paper) with multi-scale predictions.

    Args:
        pred_pose_enc_list: List of predicted pose encodings at different scales,
                           or single tensor [B, S, 9]
        batch: Batch dictionary containing GT camera parameters
        loss_type: Type of loss ("l1", "l2", or "huber")
        gamma: Weight decay factor for multi-scale losses
        weight_T: Weight for translation loss
        weight_R: Weight for rotation loss
        weight_fl: Weight for focal length loss

    Returns:
        dict: Loss dictionary
    """
    if not isinstance(pred_pose_enc_list, list):
        pred_pose_enc_list = [pred_pose_enc_list]

    mask_valid = batch['point_masks']
    batch_valid_mask = mask_valid[:, 0].sum(dim=[-1, -2]) > 100
    num_predictions = len(pred_pose_enc_list)

    gt_extrinsic = batch['extrinsics']
    gt_intrinsic = batch['intrinsics']
    image_size_hw = batch['images'].shape[-2:]

    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsic, gt_intrinsic, image_size_hw
    )

    loss_T = loss_R = loss_fl = 0

    for i in range(num_predictions):
        i_weight = gamma ** (num_predictions - i - 1)
        cur_pred_pose_enc = pred_pose_enc_list[i]

        if batch_valid_mask.sum() == 0:
            loss_T_i = (cur_pred_pose_enc * 0).mean()
            loss_R_i = (cur_pred_pose_enc * 0).mean()
            loss_fl_i = (cur_pred_pose_enc * 0).mean()
        else:
            loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(
                cur_pred_pose_enc[batch_valid_mask].clone(),
                gt_pose_encoding[batch_valid_mask].clone(),
                loss_type=loss_type
            )

        loss_T += loss_T_i * i_weight
        loss_R += loss_R_i * i_weight
        loss_fl += loss_fl_i * i_weight

    loss_T = loss_T / num_predictions
    loss_R = loss_R / num_predictions
    loss_fl = loss_fl / num_predictions
    loss_camera = loss_T * weight_T + loss_R * weight_R + loss_fl * weight_fl

    return {
        "loss_camera": loss_camera,
        "loss_T": loss_T,
        "loss_R": loss_R,
        "loss_fl": loss_fl,
    }


def camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="l1"):
    """Compute single-scale camera loss."""
    if loss_type == "l1":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).abs()
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).abs()
    elif loss_type == "l2":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(dim=-1)
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).norm(dim=-1)
    elif loss_type == "huber":
        loss_T = F.huber_loss(cur_pred_pose_enc[..., :3], gt_pose_encoding[..., :3], reduction='none')
        loss_R = F.huber_loss(cur_pred_pose_enc[..., 3:7], gt_pose_encoding[..., 3:7], reduction='none')
        loss_fl = F.huber_loss(cur_pred_pose_enc[..., 7:], gt_pose_encoding[..., 7:], reduction='none')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_fl = check_and_fix_inf_nan(loss_fl, "loss_fl")

    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_fl = loss_fl.mean()

    return loss_T, loss_R, loss_fl


class DA3Loss(nn.Module):
    """
    Complete DA3 loss module combining all loss components.

    Based on DA3 paper Section 3.3:
    L = L_D (depth) + α * L_grad (gradient) + L_M (ray) + L_P (point) + β * L_C (camera)

    Teacher model usage:
    - Teacher generates pseudo-depth labels to replace GT depth after switch_step
    - Supervision switches from GT to teacher labels at 120k steps (60% of 200k total)
    """

    def __init__(
        self,
        # Depth loss
        use_depth=True,
        depth_weight=1.0,
        depth_gamma=1.0,
        depth_alpha=0.2,
        depth_valid_range=0.95,
        disable_depth_conf=False,
        # Gradient loss
        use_gradient=True,
        gradient_weight=1.0,
        gradient_loss_type="grad",
        # Ray loss
        use_ray=True,
        ray_weight=1.0,
        # Point loss
        use_point=True,
        point_weight=1.0,
        point_gamma=1.0,
        point_alpha=0.2,
        point_valid_range=0.95,
        disable_point_conf=False,
        point_camera_source="ray",  # "ray" or "gt" - source of camera params for point loss
        # Camera loss
        use_camera=True,
        camera_weight=1.0,
        camera_loss_type="l1",
        camera_weight_T=1.0,
        camera_weight_R=1.0,
        camera_weight_fl=0.5,
        # Teacher settings
        use_teacher=False,
        switch_to_teacher_step=120000,
        teacher_align_with_shift=True,  # Whether to use affine (scale+shift) or scale-only alignment
        teacher_use_ransac=True,  # Whether to use RANSAC for robust alignment
        teacher_ransac_iters=100,  # Number of RANSAC iterations
    ):
        super().__init__()

        self.use_depth = use_depth
        self.depth_weight = depth_weight
        self.depth_gamma = depth_gamma
        self.depth_alpha = depth_alpha
        self.depth_valid_range = depth_valid_range
        self.disable_depth_conf = disable_depth_conf

        self.use_gradient = use_gradient
        self.gradient_weight = gradient_weight
        self.gradient_loss_type = gradient_loss_type

        self.use_ray = use_ray
        self.ray_weight = ray_weight

        self.use_point = use_point
        self.point_weight = point_weight
        self.point_gamma = point_gamma
        self.point_alpha = point_alpha
        self.point_valid_range = point_valid_range
        self.disable_point_conf = disable_point_conf
        self.point_camera_source = point_camera_source

        self.use_camera = use_camera
        self.camera_weight = camera_weight
        self.camera_loss_type = camera_loss_type
        self.camera_weight_T = camera_weight_T
        self.camera_weight_R = camera_weight_R
        self.camera_weight_fl = camera_weight_fl

        self.use_teacher = use_teacher
        self.switch_to_teacher_step = switch_to_teacher_step
        self.teacher_align_with_shift = teacher_align_with_shift
        self.teacher_use_ransac = teacher_use_ransac
        self.teacher_ransac_iters = teacher_ransac_iters

    def forward(
        self,
        outputs,
        batch,
        teacher_depth=None,
        current_step=0,
    ):
        """
        Compute all losses.

        Args:
            outputs: Model outputs dictionary containing:
                - depth: [B, S, H, W] or [B, S, H, W, 1] predicted depth
                - depth_conf: [B, S, H, W] depth confidence
                - ray: [B, S, H, W, 6] predicted ray map (optional)
                - extrinsics: [B, S, 4, 4] predicted extrinsics (optional)
                - intrinsics: [B, S, 3, 3] predicted intrinsics (optional)
            batch: Batch dictionary containing GT data
            teacher_depth: [B, S, H, W] teacher model predictions (optional)
            current_step: Current training step for teacher switching

        Returns:
            dict: Loss dictionary with all components and total loss
        """
        loss_dict = {}
        total_loss = 0

        # Get GT depth - use teacher labels if enabled and past switch step
        if self.use_teacher and teacher_depth is not None and current_step >= self.switch_to_teacher_step:
            # Align teacher depth to GT depth scale/shift before replacing
            # Teacher model outputs relative depth, need to align to GT metric depth
            original_gt_depth = batch['depths'].clone()
            valid_mask = batch.get('point_masks', None)

            # Use least_squares with RANSAC for robust alignment
            aligned_teacher_depth, scale, shift = align_depth_least_squares(
                teacher_depth, original_gt_depth, valid_mask,
                with_shift=self.teacher_align_with_shift,
                use_ransac=self.teacher_use_ransac,
                ransac_iters=self.teacher_ransac_iters
            )
            loss_dict['teacher_align_scale'] = scale
            loss_dict['teacher_align_shift'] = shift

            # Replace GT depth with aligned teacher pseudo-labels
            # Teacher outputs dense depth, so create a dense mask (all True where depth > 0)
            dense_mask = (aligned_teacher_depth > 0)

            # Recompute dense world_points from aligned teacher depth and GT camera params
            # This ensures the dense mask corresponds to dense point supervision
            dense_world_points = depth_to_world_points(
                aligned_teacher_depth,
                batch['extrinsics'],
                batch['intrinsics']
            )

            batch = {
                **batch,
                'depths': aligned_teacher_depth,
                'point_masks': dense_mask,
                'world_points': dense_world_points
            }

        # Get predicted depth
        pred_depth = outputs['depth']
        if pred_depth.dim() == 4:
            pred_depth = pred_depth.unsqueeze(-1)
        pred_depth_conf = outputs.get('depth_conf', torch.ones_like(pred_depth[..., 0]))

        # 1. Depth loss (L_D)
        if self.use_depth:
            depth_loss_dict = depth_loss(
                depth=pred_depth,
                depth_conf=pred_depth_conf,
                batch=batch,
                gamma=self.depth_gamma,
                alpha=self.depth_alpha,
                gradient_loss_type=self.gradient_loss_type if self.use_gradient else None,
                valid_range=self.depth_valid_range,
                disable_conf=self.disable_depth_conf,
                all_mean=True,
            )

            total_loss = total_loss + self.depth_weight * depth_loss_dict['loss_conf_depth']
            loss_dict.update(depth_loss_dict)

            # Add gradient loss separately if computed
            if self.use_gradient and 'loss_grad_depth' in depth_loss_dict:
                total_loss = total_loss + self.gradient_weight * depth_loss_dict['loss_grad_depth']

        # 2. Ray loss (L_M)
        # Ray format: [d, t] - first 3 channels are direction, last 3 are origin
        if self.use_ray and 'ray' in outputs:
            pred_ray = outputs['ray']

            H, W = batch['images'].shape[-2:]
            gt_ray = compute_ray_from_camera(
                batch['extrinsics'],
                batch['intrinsics'],
                H, W,
                device=pred_ray.device,
            )

            ray_loss_dict = ray_loss(
                pred_ray=pred_ray,
                gt_ray=gt_ray,
                gamma=self.ray_weight,
            )

            total_loss = total_loss + ray_loss_dict['loss_ray']
            loss_dict.update(ray_loss_dict)

        # 3. Point loss (L_P)
        # Per DA3 paper: L_P(D̂ ⊙ d + t, P)
        # world_point = depth * direction + origin (per-pixel computation)
        if self.use_point:
            if self.point_camera_source == "ray" and 'ray' in outputs:
                # Use ray map directly: world_points = depth * d + t
                # Ray format: [d, t] - first 3 channels are direction, last 3 are origin
                pred_ray = outputs['ray']

                # Get depth in correct shape
                depth = pred_depth.squeeze(-1) if pred_depth.dim() == 5 else pred_depth

                # Interpolate ray to match depth resolution if needed
                if pred_ray.shape[2:4] != depth.shape[2:4]:
                    B, S = pred_ray.shape[:2]
                    pred_ray_flat = pred_ray.permute(0, 1, 4, 2, 3).reshape(B * S, 6, pred_ray.shape[2], pred_ray.shape[3])
                    pred_ray_flat = F.interpolate(
                        pred_ray_flat,
                        size=depth.shape[2:4],
                        mode='bilinear',
                        align_corners=True
                    )
                    pred_ray = pred_ray_flat.reshape(B, S, 6, depth.shape[2], depth.shape[3]).permute(0, 1, 3, 4, 2)

                # Extract direction (d) and origin (t) from ray map
                ray_d = pred_ray[..., :3]  # [B, S, H, W, 3] - direction
                ray_t = pred_ray[..., 3:]  # [B, S, H, W, 3] - origin

                # Compute world points: P = D * d + t
                pred_world_points = depth[..., None] * ray_d + ray_t  # [B, S, H, W, 3]
            else:
                # Fall back to GT camera parameters
                pred_world_points = depth_to_world_points(
                    pred_depth.squeeze(-1) if pred_depth.dim() == 5 else pred_depth,
                    batch['extrinsics'],
                    batch['intrinsics'],
                )

            point_loss_dict = point_loss(
                pts3d=pred_world_points,
                pts3d_conf=pred_depth_conf,
                batch=batch,
                normalize_pred=False,
                gamma=self.point_gamma,
                alpha=self.point_alpha,
                gradient_loss_type=self.gradient_loss_type if self.use_gradient else None,
                valid_range=self.point_valid_range,
                disable_conf=self.disable_point_conf,
                all_mean=True,
            )

            total_loss = total_loss + self.point_weight * point_loss_dict['loss_conf_point']
            loss_dict.update(point_loss_dict)

        # 4. Camera loss (L_C)
        if self.use_camera and 'extrinsics' in outputs:
            # Both pred and GT extrinsics are in w2c format, use directly
            pred_pose_enc = extri_intri_to_pose_encoding(
                outputs['extrinsics'], outputs['intrinsics'], batch['images'].shape[-2:]
            )

            camera_loss_dict = camera_loss(
                pred_pose_enc,
                batch,
                loss_type=self.camera_loss_type,
                weight_T=self.camera_weight_T,
                weight_R=self.camera_weight_R,
                weight_fl=self.camera_weight_fl,
            )

            total_loss = total_loss + self.camera_weight * camera_loss_dict['loss_camera']
            loss_dict.update(camera_loss_dict)

        loss_dict['total_loss'] = total_loss

        return loss_dict
