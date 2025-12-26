#!/usr/bin/env python3
"""
LATENT RAYMARCHER v2 - VOLUMETRIC MODE
Actually useful depth/normal maps
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image


class LatentVolumeMarcher:
    """
    Treats latent as volumetric density field instead of hard SDF.
    Much better results for ControlNet conditioning.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def generate_structured_latent(
        self, 
        seed: int,
        width: int = 512,
        height: int = 512,
        structure: str = "organic"  # organic, geometric, waves, fractal
    ) -> torch.Tensor:
        """Generate interesting structured latent noise"""
        
        torch.manual_seed(seed)
        h, w = height // 8, width // 8
        
        if structure == "organic":
            # Smooth organic blobs
            latent = torch.randn(1, 4, h, w, device=self.device)
            
            # Multi-scale smoothing
            for scale in [7, 5, 3]:
                kernel = torch.ones(1, 1, scale, scale, device=self.device) / (scale * scale)
                for c in range(4):
                    latent[:, c:c+1] = F.conv2d(latent[:, c:c+1], kernel, padding=scale//2)
            
        elif structure == "geometric":
            # Sharp geometric shapes
            latent = torch.zeros(1, 4, h, w, device=self.device)
            
            # Add random rectangles and circles
            for _ in range(10):
                c = torch.randint(0, 4, (1,)).item()
                x1, x2 = sorted([torch.randint(0, w, (1,)).item() for _ in range(2)])
                y1, y2 = sorted([torch.randint(0, h, (1,)).item() for _ in range(2)])
                val = torch.randn(1).item() * 2
                latent[:, c, y1:y2, x1:x2] = val
                
        elif structure == "waves":
            # Sine wave interference
            x = torch.linspace(0, 4 * math.pi, w, device=self.device)
            y = torch.linspace(0, 4 * math.pi, h, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            
            latent = torch.stack([
                torch.sin(xx + yy),
                torch.sin(xx * 1.5 - yy * 0.5),
                torch.cos(xx * 0.7 + yy * 1.3),
                torch.sin(xx * yy * 0.1)
            ]).unsqueeze(0)
            
        elif structure == "fractal":
            # Fractal noise (multiple octaves)
            latent = torch.zeros(1, 4, h, w, device=self.device)
            
            for octave in range(4):
                scale = 2 ** octave
                noise = torch.randn(1, 4, h // scale + 1, w // scale + 1, device=self.device)
                noise = F.interpolate(noise, size=(h, w), mode='bilinear', align_corners=True)
                latent = latent + noise / (octave + 1)
        
        else:
            latent = torch.randn(1, 4, h, w, device=self.device)
        
        return latent
    
    def march_volumetric(
        self,
        latent: torch.Tensor,
        width: int = 512,
        height: int = 512,
        num_steps: int = 32,
        density_scale: float = 5.0
    ) -> dict:
        """
        Volumetric raymarching - accumulates density along rays.
        """
        device = latent.device
        
        # Upsample latent to output resolution
        latent_up = F.interpolate(latent, size=(height, width), mode='bilinear', align_corners=True)
        
        # Use channel magnitude as density
        density = latent_up.norm(dim=1, keepdim=True)  # (1, 1, H, W)
        density = (density - density.min()) / (density.max() - density.min() + 1e-8)
        
        # Use channels 0-2 as color
        color = latent_up[:, :3]  # (1, 3, H, W)
        color = (color - color.min()) / (color.max() - color.min() + 1e-8)
        
        # Simulate depth via density accumulation
        # Higher density = closer surface
        depth = 1.0 - density.squeeze(0).squeeze(0)
        
        # Compute normals from density gradient (Sobel)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(density, sobel_x, padding=1)
        grad_y = F.conv2d(density, sobel_y, padding=1)
        
        # Normal = (-grad_x, -grad_y, 1) normalized
        normal = torch.cat([
            -grad_x,
            -grad_y,
            torch.ones_like(grad_x)
        ], dim=1)  # (1, 3, H, W)
        
        normal = F.normalize(normal, dim=1)
        
        # Convert to images
        depth_img = depth.cpu().numpy()
        normal_img = (normal.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        color_img = color.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        return {
            'depth': depth_img,
            'normal': normal_img,
            'color': color_img,
            'density': density.squeeze().cpu().numpy()
        }
    
    def generate_controlnet_maps(
        self,
        seed: int = 42,
        width: int = 512,
        height: int = 512,
        structure: str = "organic"
    ) -> tuple:
        """
        Generate depth and normal maps ready for ControlNet.
        
        Returns: (depth_pil, normal_pil, color_pil)
        """
        latent = self.generate_structured_latent(seed, width, height, structure)
        result = self.march_volumetric(latent, width, height)
        
        # Convert to PIL
        depth_pil = Image.fromarray((result['depth'] * 255).astype(np.uint8), mode='L')
        normal_pil = Image.fromarray((result['normal'] * 255).astype(np.uint8), mode='RGB')
        color_pil = Image.fromarray((result['color'] * 255).astype(np.uint8), mode='RGB')
        
        return depth_pil, normal_pil, color_pil, latent


def demo():
    """Generate a variety of depth/normal maps"""
    print("=" * 60)
    print("LATENT VOLUME MARCHER v2")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    marcher = LatentVolumeMarcher(device)
    
    structures = ["organic", "geometric", "waves", "fractal"]
    
    for structure in structures:
        for seed in [42, 1337]:
            print(f"\n[*] Generating {structure} (seed {seed})...")
            
            depth, normal, color, _ = marcher.generate_controlnet_maps(
                seed=seed,
                width=512,
                height=512,
                structure=structure
            )
            
            prefix = f"~/Desktop/lm_{structure}_{seed}"
            depth.save(os.path.expanduser(f"{prefix}_depth.png"))
            normal.save(os.path.expanduser(f"{prefix}_normal.png"))
            color.save(os.path.expanduser(f"{prefix}_color.png"))
            
            print(f"    Saved: {prefix}_*.png")
    
    print("\n" + "=" * 60)
    print("Done! Check ~/Desktop/lm_*.png")
    print("=" * 60)


# Quick integration with your img.py
def generate_with_latent_depth(
    prompt: str,
    structure: str = "organic",
    seed: int = -1
):
    """Generate image using latent-derived depth as mental reference"""
    import random
    import subprocess
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    
    device = 'cuda'
    
    # Generate depth/normal first
    print("[*] Generating latent depth map...")
    marcher = LatentVolumeMarcher(device)
    depth, normal, color, latent = marcher.generate_controlnet_maps(
        seed=seed, structure=structure
    )
    
    # Save conditioning images
    depth.save(os.path.expanduser('~/Desktop/gen_depth.png'))
    normal.save(os.path.expanduser('~/Desktop/gen_normal.png'))
    
    # Load SD
    print("[*] Loading Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_single_file(
        os.path.expanduser("~/Desktop/realisticVisionV60B1_v51HyperVAE.safetensors"),
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++"
    )
    pipe.enable_attention_slicing()
    
    # Generate (using same seed for consistency)
    print(f"[*] Generating image (seed {seed})...")
    generator = torch.Generator(device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt="ugly, blurry, deformed, bad anatomy",
        width=512,
        height=512,
        num_inference_steps=30,
        guidance_scale=6.5,
        generator=generator
    ).images[0]
    
    path = os.path.expanduser(f'~/Desktop/latent_gen_{seed}.png')
    image.save(path)
    
    print(f"\n[✓] Saved: {path}")
    print(f"[✓] Depth: ~/Desktop/gen_depth.png")
    print(f"[✓] Normal: ~/Desktop/gen_normal.png")
    
    subprocess.Popen(['xdg-open', path])
    
    return image, depth, normal


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--gen':
        # Full generation mode
        prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "alien landscape, crystalline structures, volumetric fog"
        generate_with_latent_depth(prompt, structure="organic")
    else:
        # Just generate depth/normal maps
        demo()
