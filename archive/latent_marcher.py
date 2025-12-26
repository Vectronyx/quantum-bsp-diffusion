#!/usr/bin/env python3
"""
LATENT SPACE RAYMARCHER v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Raymarch through 4D latent space as a signed distance field.
Generate depth/normal maps for ControlNet conditioning.

The latent space IS the geometry.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
from PIL import Image
import numpy as np


@dataclass 
class LatentMarcherConfig:
    # Marching
    max_steps: int = 64
    max_distance: float = 10.0
    base_epsilon: float = 0.01
    
    # Latent interpretation
    latent_scale: float = 0.18215      # SD latent scaling factor
    sdf_threshold: float = 0.0         # Isosurface level
    sdf_channel: int = 0               # Which latent channel = SDF (-1 for magnitude)
    
    # 4D navigation
    w_slice: float = 0.0               # Position in 4th dimension
    w_range: float = 2.0               # How far to explore in W
    
    # Output
    depth_near: float = 0.1
    depth_far: float = 10.0
    normal_strength: float = 1.0


class LatentSpaceSDF(nn.Module):
    """
    Interprets a latent tensor as a 4D signed distance field.
    
    The 4 latent channels become:
    - Channel 0: SDF value (or use magnitude of all 4)
    - Channels 1-3: Color/material properties
    
    Or interpret as:
    - 4D spatial coordinates where we slice through
    """
    
    def __init__(self, config: LatentMarcherConfig):
        super().__init__()
        self.config = config
        self.latent = None  # Set via set_latent()
        
    def set_latent(self, latent: torch.Tensor):
        """
        Set the latent tensor to raymarch through.
        Expected shape: (1, 4, H, W) or (4, H, W)
        """
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)
        self.latent = latent * self.config.latent_scale
        
    def sample_latent(self, position: torch.Tensor) -> torch.Tensor:
        """
        Sample latent at continuous 3D position using trilinear interpolation.
        Position: (..., 3) in range [-1, 1] for XY, any range for Z
        Returns: (..., 4) latent values
        """
        if self.latent is None:
            raise ValueError("Call set_latent() first")
        
        B, C, H, W = self.latent.shape
        
        # Normalize XY to grid sample range [-1, 1]
        xy = position[..., :2]  # (..., 2)
        
        # Z becomes interpolation weight between "slices"
        # We treat the 4 channels as depth slices in a pseudo-3D volume
        z = position[..., 2:3]  # (..., 1)
        z_norm = (z / self.config.w_range).clamp(-1, 1)
        
        # Reshape for grid_sample: (N, H, W, 2)
        orig_shape = xy.shape[:-1]
        xy_flat = xy.reshape(1, -1, 1, 2)  # (1, N, 1, 2)
        
        # Sample all 4 channels at XY position
        sampled = F.grid_sample(
            self.latent, 
            xy_flat,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # (1, 4, N, 1)
        
        sampled = sampled.squeeze(0).squeeze(-1).T  # (N, 4)
        sampled = sampled.reshape(*orig_shape, 4)
        
        # Interpolate along Z using channels as pseudo-depth
        # Channel weight based on Z position
        z_weight = (z_norm * 0.5 + 0.5) * 3  # Map to [0, 3] for 4 channels
        z_floor = z_weight.floor().long().clamp(0, 2)
        z_frac = z_weight - z_floor.float()
        
        # Gather and interpolate
        idx_lo = z_floor.expand_as(sampled[..., :1])
        idx_hi = (z_floor + 1).clamp(max=3).expand_as(sampled[..., :1])
        
        val_lo = sampled.gather(-1, idx_lo)
        val_hi = sampled.gather(-1, idx_hi)
        
        interpolated = val_lo * (1 - z_frac) + val_hi * z_frac
        
        return sampled, interpolated.squeeze(-1)
    
    def sdf(self, position: torch.Tensor) -> torch.Tensor:
        """
        Get SDF value at position.
        """
        full_sample, interp = self.sample_latent(position)
        
        if self.config.sdf_channel == -1:
            # Use magnitude of all channels
            sdf = full_sample.norm(dim=-1) - self.config.sdf_threshold
        else:
            # Use specific channel
            sdf = full_sample[..., self.config.sdf_channel] - self.config.sdf_threshold
        
        return sdf
    
    def color(self, position: torch.Tensor) -> torch.Tensor:
        """
        Get color/material at position (channels 1-3 as RGB).
        """
        full_sample, _ = self.sample_latent(position)
        
        # Channels 1-3 as color, normalized to [0, 1]
        color = full_sample[..., 1:4]
        color = (color - color.min()) / (color.max() - color.min() + 1e-8)
        
        return color


class LatentRayMarcher(nn.Module):
    """
    Raymarches through latent space, outputs depth/normal for ControlNet.
    """
    
    def __init__(self, config: Optional[LatentMarcherConfig] = None):
        super().__init__()
        self.config = config or LatentMarcherConfig()
        self.sdf_field = LatentSpaceSDF(self.config)
        
    def compute_normal(
        self, 
        position: torch.Tensor,
        epsilon: float = 0.01
    ) -> torch.Tensor:
        """Compute surface normal via central differences"""
        e = epsilon
        
        n = torch.stack([
            self.sdf_field.sdf(position + torch.tensor([e, 0, 0], device=position.device)) -
            self.sdf_field.sdf(position - torch.tensor([e, 0, 0], device=position.device)),
            
            self.sdf_field.sdf(position + torch.tensor([0, e, 0], device=position.device)) -
            self.sdf_field.sdf(position - torch.tensor([0, e, 0], device=position.device)),
            
            self.sdf_field.sdf(position + torch.tensor([0, 0, e], device=position.device)) -
            self.sdf_field.sdf(position - torch.tensor([0, 0, e], device=position.device)),
        ], dim=-1)
        
        return F.normalize(n, dim=-1)
    
    def generate_rays(
        self,
        width: int,
        height: int,
        fov: float = 60.0,
        device: str = 'cuda'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate camera rays"""
        aspect = width / height
        
        u = torch.linspace(-1, 1, width, device=device) * aspect * math.tan(math.radians(fov/2))
        v = torch.linspace(-1, 1, height, device=device) * math.tan(math.radians(fov/2))
        u, v = torch.meshgrid(u, v, indexing='xy')
        
        ray_dir = F.normalize(torch.stack([u, -v, torch.ones_like(u)], dim=-1), dim=-1)
        ray_origin = torch.zeros(height, width, 3, device=device)
        ray_origin[..., 2] = -2.0  # Camera position
        
        return ray_origin, ray_dir
    
    def march(
        self,
        latent: torch.Tensor,
        width: int = 512,
        height: int = 512,
        device: str = 'cuda'
    ) -> dict:
        """
        Raymarch through the latent and generate depth/normal maps.
        
        Returns dict with:
        - depth: (H, W) depth map [0, 1]
        - normal: (H, W, 3) normal map [-1, 1]
        - hit: (H, W) boolean hit mask
        - color: (H, W, 3) latent-derived color
        - depth_raw: (H, W) raw depth values
        """
        self.sdf_field.set_latent(latent.to(device))
        
        ray_origin, ray_dir = self.generate_rays(width, height, device=device)
        
        # Initialize
        position = ray_origin.clone()
        distance = torch.zeros(height, width, device=device)
        hit = torch.zeros(height, width, dtype=torch.bool, device=device)
        
        # March
        for step in range(self.config.max_steps):
            sdf = self.sdf_field.sdf(position)
            
            # Check hit
            new_hit = (sdf.abs() < self.config.base_epsilon) & ~hit
            hit = hit | new_hit
            
            # Advance
            position = position + ray_dir * sdf.unsqueeze(-1).clamp(min=self.config.base_epsilon)
            distance = distance + sdf.abs()
            
            # Early exit
            if hit.all() or (distance > self.config.max_distance).all():
                break
        
        # Compute normals at hit points
        normal = self.compute_normal(position)
        
        # Get color from latent
        color = self.sdf_field.color(position)
        
        # Normalize depth to [0, 1]
        depth_normalized = (distance - self.config.depth_near) / (self.config.depth_far - self.config.depth_near)
        depth_normalized = depth_normalized.clamp(0, 1)
        
        # Invert so closer = brighter (ControlNet convention)
        depth_normalized = 1.0 - depth_normalized
        
        # Set missed rays to 0
        depth_normalized = depth_normalized * hit.float()
        
        # Normal map: remap from [-1,1] to [0,1] for saving
        normal_normalized = normal * 0.5 + 0.5
        normal_normalized = normal_normalized * hit.unsqueeze(-1).float()
        
        return {
            'depth': depth_normalized,
            'normal': normal,
            'normal_image': normal_normalized,
            'hit': hit,
            'color': color,
            'depth_raw': distance,
            'position': position
        }


class ControlNetConditioner(nn.Module):
    """
    Generates ControlNet-ready depth and normal maps from latent raymarching.
    """
    
    def __init__(self, config: Optional[LatentMarcherConfig] = None):
        super().__init__()
        self.config = config or LatentMarcherConfig()
        self.marcher = LatentRayMarcher(self.config)
        
    def generate_depth_normal(
        self,
        latent: torch.Tensor,
        width: int = 512,
        height: int = 512
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Generate depth and normal maps ready for ControlNet.
        
        Returns: (depth_pil, normal_pil)
        """
        device = latent.device if hasattr(latent, 'device') else 'cuda'
        
        result = self.marcher.march(latent, width, height, device)
        
        # Convert to PIL
        depth_np = (result['depth'].cpu().numpy() * 255).astype(np.uint8)
        depth_pil = Image.fromarray(depth_np, mode='L')
        
        normal_np = (result['normal_image'].cpu().numpy() * 255).astype(np.uint8)
        normal_pil = Image.fromarray(normal_np, mode='RGB')
        
        return depth_pil, normal_pil
    
    def generate_from_noise(
        self,
        seed: int = 42,
        width: int = 512,
        height: int = 512,
        device: str = 'cuda'
    ) -> Tuple[Image.Image, Image.Image, torch.Tensor]:
        """
        Generate depth/normal from random latent noise.
        Good for creating abstract 3D-looking conditioning.
        
        Returns: (depth_pil, normal_pil, latent)
        """
        torch.manual_seed(seed)
        
        # Generate structured noise (not pure random)
        latent = torch.randn(1, 4, height // 8, width // 8, device=device)
        
        # Apply some structure via convolution
        kernel = torch.tensor([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=torch.float32, device=device).view(1, 1, 3, 3) / 16
        
        for c in range(4):
            latent[:, c:c+1] = F.conv2d(latent[:, c:c+1], kernel, padding=1)
        
        depth_pil, normal_pil = self.generate_depth_normal(latent, width, height)
        
        return depth_pil, normal_pil, latent


# ============== INTEGRATION WITH STABLE DIFFUSION ==============

class LatentGuidedGeneration:
    """
    Use raymarched latent depth/normals to guide SD generation via ControlNet.
    """
    
    def __init__(self, model_path: str, controlnet_path: Optional[str] = None):
        from diffusers import (
            StableDiffusionControlNetPipeline,
            StableDiffusionPipeline,
            ControlNetModel,
            DPMSolverMultistepScheduler
        )
        
        self.device = 'cuda'
        self.conditioner = ControlNetConditioner()
        
        if controlnet_path:
            # Load with ControlNet
            print("[*] Loading ControlNet...")
            controlnet = ControlNetModel.from_pretrained(
                controlnet_path,
                torch_dtype=torch.float16
            )
            
            print("[*] Loading SD + ControlNet pipeline...")
            self.pipe = StableDiffusionControlNetPipeline.from_single_file(
                model_path,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None
            )
        else:
            # Load standard SD (will just generate depth/normal without using them)
            print("[*] Loading SD pipeline (no ControlNet)...")
            self.pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                safety_checker=None
            )
        
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++"
        )
        
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()
        
        self.has_controlnet = controlnet_path is not None
        
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = -1,
        steps: int = 30,
        cfg: float = 7.0,
        width: int = 512,
        height: int = 512,
        use_depth: bool = True,
        controlnet_strength: float = 0.8
    ) -> dict:
        """
        Generate image with latent-raymarched depth/normal conditioning.
        """
        import random
        
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        # Generate depth/normal from random latent
        print("[*] Raymarching latent space for depth/normal...")
        depth_pil, normal_pil, source_latent = self.conditioner.generate_from_noise(
            seed=seed,
            width=width,
            height=height,
            device=self.device
        )
        
        generator = torch.Generator(self.device).manual_seed(seed)
        
        print("[*] Generating image...")
        
        if self.has_controlnet and use_depth:
            # Use depth as ControlNet conditioning
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=depth_pil,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
                controlnet_conditioning_scale=controlnet_strength
            ).images[0]
        else:
            # Standard generation
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator
            ).images[0]
        
        return {
            'image': image,
            'depth': depth_pil,
            'normal': normal_pil,
            'seed': seed,
            'source_latent': source_latent
        }


# ============== DEMO ==============

def demo_latent_march():
    """Demo: raymarch random latent, save depth/normal"""
    print("=" * 60)
    print("LATENT SPACE RAYMARCHER DEMO")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    conditioner = ControlNetConditioner()
    
    for seed in [42, 1337, 69420]:
        print(f"\n[*] Raymarching seed {seed}...")
        start = time.time()
        
        depth, normal, latent = conditioner.generate_from_noise(
            seed=seed,
            width=512,
            height=512,
            device=device
        )
        
        elapsed = time.time() - start
        print(f"    Done in {elapsed:.2f}s")
        
        # Save
        depth.save(os.path.expanduser(f'~/Desktop/latent_depth_{seed}.png'))
        normal.save(os.path.expanduser(f'~/Desktop/latent_normal_{seed}.png'))
        
        print(f"    Saved: ~/Desktop/latent_depth_{seed}.png")
        print(f"    Saved: ~/Desktop/latent_normal_{seed}.png")
    
    print("\n" + "=" * 60)
    print("Use these as ControlNet depth/normal conditioning!")
    print("=" * 60)


def demo_full_generation():
    """Demo: full pipeline with SD generation"""
    print("=" * 60)
    print("LATENT-GUIDED GENERATION DEMO")
    print("=" * 60)
    
    model_path = os.path.expanduser(
        "~/Desktop/realisticVisionV60B1_v51HyperVAE.safetensors"
    )
    
    # For full ControlNet, download depth controlnet:
    # controlnet_path = "lllyasviel/control_v11f1p_sd15_depth"
    controlnet_path = None  # Set to None if you don't have ControlNet
    
    gen = LatentGuidedGeneration(model_path, controlnet_path)
    
    result = gen.generate(
        prompt="beautiful alien landscape, crystalline structures, volumetric fog, cinematic lighting, 8k",
        negative_prompt="blurry, ugly, bad quality",
        seed=42069,
        steps=30,
        cfg=6.5
    )
    
    # Save everything
    result['image'].save(os.path.expanduser('~/Desktop/latent_guided_image.png'))
    result['depth'].save(os.path.expanduser('~/Desktop/latent_guided_depth.png'))
    result['normal'].save(os.path.expanduser('~/Desktop/latent_guided_normal.png'))
    
    print(f"\nSaved:")
    print(f"  ~/Desktop/latent_guided_image.png")
    print(f"  ~/Desktop/latent_guided_depth.png")
    print(f"  ~/Desktop/latent_guided_normal.png")
    print(f"  Seed: {result['seed']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        demo_full_generation()
    else:
        demo_latent_march()
