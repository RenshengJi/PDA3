"""Metrics utilities for model evaluation."""

import torch
from typing import Optional

try:
    from lpips import LPIPS
    _lpips_model = None
    
    def get_lpips(device):
        """Get or create LPIPS model."""
        global _lpips_model
        if _lpips_model is None:
            _lpips_model = LPIPS(net='alex', version='0.1').to(device).eval()
        else:
            _lpips_model = _lpips_model.to(device)
        return _lpips_model
    
    def compute_lpips(ground_truth, predicted):
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity) loss.
        
        Args:
            ground_truth: [B, C, H, W] ground truth images
            predicted: [B, C, H, W] predicted images
            
        Returns:
            [B] LPIPS loss per batch
        """
        device = ground_truth.device
        lpips_model = get_lpips(device)
        
        with torch.no_grad():
            value = lpips_model.forward(ground_truth, predicted, normalize=True)
        
        return value[:, 0, 0, 0]

except ImportError:
    print("Warning: lpips package not found. compute_lpips will return zeros.")
    
    def compute_lpips(ground_truth, predicted):
        """Dummy LPIPS when lpips package is not available."""
        return torch.zeros(ground_truth.shape[0], device=ground_truth.device)
