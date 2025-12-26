#!/usr/bin/env python3
"""
HYPERWARP VAE v1.3 - SKIN-AWARE VERSION
Reduces plastic look, smooths pores, keeps detail where it matters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def wrap_diffusers_vae(pipe):
    
    class SkinAwareVAE(nn.Module):
        def __init__(self, original_vae):
            super().__init__()
            self.original_vae = original_vae
            
        @property
        def config(self):
            return self.original_vae.config
        
        @property
        def dtype(self):
            return self.original_vae.dtype
        
        @property
        def device(self):
            return self.original_vae.device
        
        def __getattr__(self, name):
            if name in ['original_vae']:
                return super().__getattr__(name)
            try:
                return getattr(self.original_vae, name)
            except AttributeError:
                raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
            
        def encode(self, x, return_dict=True):
            return self.original_vae.encode(x, return_dict=return_dict)
        
        def decode(self, z, return_dict=True, generator=None):
            # Subtle gaussian blur on latents to reduce micro-texture
            # This fights the "pore explosion" from HyperVAE
            
            smoothed = self._gentle_smooth(z, sigma=0.4)
            
            # Blend: 85% original, 15% smoothed
            # Keeps detail but reduces plastic pore look
            z_final = z * 0.85 + smoothed * 0.15
            
            return self.original_vae.decode(z_final, return_dict=return_dict)
        
        def _gentle_smooth(self, z, sigma=0.5):
            """Apply gentle gaussian smoothing to latents"""
            kernel_size = 3
            
            # Create gaussian kernel
            x = torch.arange(kernel_size, device=z.device, dtype=z.dtype) - kernel_size // 2
            kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            
            kernel_2d = kernel_1d.view(-1, 1) * kernel_1d.view(1, -1)
            kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
            
            # Apply per channel
            smoothed = []
            for c in range(z.shape[1]):
                ch = F.conv2d(
                    z[:, c:c+1], 
                    kernel_2d, 
                    padding=kernel_size//2
                )
                smoothed.append(ch)
            
            return torch.cat(smoothed, dim=1)
        
        def forward(self, x, sample_posterior=True):
            encoded = self.encode(x)
            if sample_posterior:
                z = encoded.latent_dist.sample()
            else:
                z = encoded.latent_dist.mode()
            return self.decode(z)
    
    return SkinAwareVAE(pipe.vae)
