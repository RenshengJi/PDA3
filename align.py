from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from src.alignment_misc import *
import torch.nn as nn


# 自定义带约束的线性回归估计器
class ConstrainedLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, min_coef=1e-8, max_bias=[0.0, 1.0]):
        self.min_coef = min_coef  # 系数的下界
        self.max_bias = max_bias
        self.model = LinearRegression()  # 基础线性回归模型

    def fit(self, X, y):
        # 先进行普通线性回归
        self.model.fit(X, y)
        # 强制系数大于min_coef
        if self.model.coef_[0] < self.min_coef:
            self.model.coef_[0] = self.min_coef  # 若系数太小，强制设为下界
        self.model.intercept_[0] = np.clip(self.model.intercept_[0], self.max_bias[0], self.max_bias[1])
        return self

    def predict(self, X):
        return self.model.predict(X)


class ARAPDepthAlignment(nn.Module):
    def __init__(self, mono_depth, guidance_depth, valid_mask, mono_depth_mask, K, w2c, points2d, device, depth_filter,
                 smoothing_kernel_size=3, lambda_arap=0.1, max_d=100.0, eps=1e-6):
        super(ARAPDepthAlignment, self).__init__()

        # 初始化使用最小二乘
        ransac = RANSACRegressor(
            min_samples=100,
            estimator=ConstrainedLinearRegression(min_coef=1e-4, max_bias=[-10.0, 10.0]),
            stop_probability=0.995,
            random_state=42
        )
        near_mask = valid_mask & (mono_depth <= depth_filter)
        far_mask = valid_mask & (mono_depth > depth_filter)

        self.device = device
        # Create smoothing kernel
        self.smoothing_kernel_size = smoothing_kernel_size
        self.smoothing_kernel = torch.ones((1, 1, smoothing_kernel_size, smoothing_kernel_size), device=device) / (smoothing_kernel_size ** 2)

        mono_inv_depth = 1.0 / mono_depth
        guidance_inv_depth = 1.0 / guidance_depth

        _ = ransac.fit(mono_inv_depth[near_mask].numpy().reshape(-1, 1), guidance_inv_depth[near_mask].numpy().reshape(-1, 1))
        k = ransac.estimator_.model.coef_[0][0]
        b = ransac.estimator_.model.intercept_[0]

        self.max_d = max_d
        self.mono_depth = mono_depth
        self.mono_depth_aligned = 1.0 / torch.clamp_min((1.0 / torch.clamp_min(mono_depth, eps)) * k + b, eps)
        self.mono_depth_aligned = torch.clamp_max(self.mono_depth_aligned, self.max_d)
        self.mono_depth_aligned[~mono_depth_mask] = 0

        if far_mask.float().mean() >= 0.05:
            _ = ransac.fit(mono_depth[far_mask].numpy().reshape(-1, 1), guidance_depth[far_mask].numpy().reshape(-1, 1))
            k2 = ransac.estimator_.model.coef_[0][0]
            b2 = ransac.estimator_.model.intercept_[0]

            far_aligned_depth = mono_depth * k2 + b2
            far_aligned_depth = torch.clip(far_aligned_depth, 0.05, self.max_d)
            far_aligned_depth[~mono_depth_mask] = 0
            combined_mask = (mono_depth > depth_filter) & mono_depth_mask & (far_aligned_depth > self.mono_depth_aligned)
            self.mono_depth_aligned[combined_mask] = far_aligned_depth[combined_mask]

        depth_range = self.mono_depth_aligned.unique()
        if len(depth_range) > 1:
            print(f"Depth Range after RANSAC: [{depth_range[1].item()}, {depth_range[-1].item()}]")
        self.aligned_depth = self.mono_depth_aligned.to(self.device)
        self.guidance_depth = guidance_depth.to(self.device)
        self.valid_mask = valid_mask.to(self.device)
        self.mono_depth_mask = mono_depth_mask.to(self.device)
        self.K = K.to(self.device)
        self.w2c = w2c.to(self.device)
        self.points2d = points2d.to(self.device)
        self.mono_depth = self.mono_depth.to(self.device)

        # Initialize scale map
        self.sc_map = nn.Parameter(torch.ones_like(self.aligned_depth).float(), requires_grad=True)
        self.lambda_arap = lambda_arap

        self.to(device)
        self.params = [p for p in self.parameters() if p.requires_grad]

    def depth2pcd(self, depth):
        points3d = self.w2c.inverse() @ point_padding((self.K.inverse() @ self.points2d.T).T * depth.reshape(-1, 1)).T
        points3d = points3d.T[:, :3]

        return points3d

    def return_depth(self):
        return torch.clip(self.aligned_depth * self.sc_map, 0, self.max_d)

    def fit(self, lr, niter):
        optimizer = torch.optim.Adam(self.params, lr=lr, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=niter, eta_min=1e-4)
        for i in range(niter):
            optimizer.zero_grad()

            aligned_points = self.depth2pcd(self.aligned_depth * self.sc_map)[self.valid_mask.reshape(-1)]
            guidance_points = self.depth2pcd(self.guidance_depth)[self.valid_mask.reshape(-1)]

            l1_loss = F.l1_loss(aligned_points, guidance_points, reduction="mean")

            # Apply smoothing filter to sc_map
            sc_map_reshaped = self.sc_map.unsqueeze(0).unsqueeze(0)
            sc_map_smoothed = F.conv2d(
                sc_map_reshaped,
                self.smoothing_kernel,
                padding=self.smoothing_kernel_size // 2
            ).squeeze(0).squeeze(0)

            # ARAP loss
            arap_loss = torch.abs(sc_map_smoothed - self.sc_map).mean()

            loss = l1_loss + arap_loss * self.lambda_arap
            loss.backward()

            optimizer.step()
            scheduler.step()

            if (i + 1) % 100 == 0:
                print(f"L1 Loss: {l1_loss.item():.4f},"
                      f" ARAP Loss: {arap_loss.item() * self.lambda_arap:.4f},"
                      f" Step:{i + 1}/{niter}")

        optimizer.zero_grad()




import einops
from src.alignment_misc import *
import utils3d

if __name__ == '__main__':
    width, height = 512, 512
    mono_depth = torch.randn((height, width), dtype=torch.float32)
    guided_depth = torch.randn((height, width), dtype=torch.float32)
    guided_depth_mask = torch.randint(0, 2, (height, width)).bool()
    mono_depth_mask = torch.randint(0, 2, (height, width)).bool()
    max_d = 50  # 根据全景图distance预设一个最大depth
    K = torch.eye(3)  # 内参
    w2c = torch.eye(4)  # 外参
    device = "cuda"

    # predefine 2d points
    x = torch.arange(width).float()
    y = torch.arange(height).float()
    points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
    points = einops.rearrange(points, 'w h c -> (h w) c')
    points = point_padding(points)

    mono_edge_mask = ~torch.from_numpy(utils3d.numpy.depth_edge(mono_depth.numpy(), rtol=0.05)).bool()

    # ARAP不能优化差距很大的2组depth，所以先要大致切分下近远景，也可以考虑后续用depth聚类去得到，目前8倍基本没问题
    depth_filter = torch.median(mono_depth[mono_depth > 0]) * 8

    valid_mask = guided_depth_mask & mono_depth_mask & mono_edge_mask
    align_optimizer = ARAPDepthAlignment(mono_depth, guided_depth, valid_mask, mono_depth_mask, K, w2c, points, device,
                                         depth_filter=depth_filter, smoothing_kernel_size=3, lambda_arap=0.1, max_d=max_d)
    align_optimizer.fit(lr=1e-3, niter=500)  # 感觉niter 300也可以
    aligned_depth = align_optimizer.return_depth().detach().cpu()
    aligned_edge_mask = ~torch.from_numpy(utils3d.numpy.depth_edge(aligned_depth.numpy(), rtol=0.1)).bool()
    mono_edge_mask = mono_edge_mask & aligned_edge_mask
    aligned_depth[~mono_edge_mask] = 0  # 这里depth=0表示这里depth不可用，相当于mask