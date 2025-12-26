#!/usr/bin/env python3
"""
LATENT RAYMARCHER v3 - TUNABLE DEPTH
Control the black/white balance and depth intensity
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import subprocess
import random


class TunableLatentMarcher:
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def generate(
        self,
        seed: int = -1,
        width: int = 512,
        height: int = 512,
        # TUNING PARAMS
        depth_contrast: float = 1.0,      # Higher = more contrast (0.5-3.0)
        depth_brightness: float = 0.5,    # 0=dark, 1=bright (0.0-1.0)
        depth_invert: bool = False,       # Flip black/white
        smoothness: int = 5,              # Blur kernel size (1-15)
        structure: str = "organic",       # organic, geometric, waves, fractal
    ) -> dict:
        
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        torch.manual_seed(seed)
        h, w = height // 8, width // 8
        
        # Generate base latent
        if structure == "organic":
            latent = torch.randn(1, 4, h, w, device=self.device)
            for scale in [7, 5, 3]:
                kernel = torch.ones(1, 1, scale, scale, device=self.device) / (scale**2)
                for c in range(4):
                    latent[:, c:c+1] = F.conv2d(latent[:, c:c+1], kernel, padding=scale//2)
                    
        elif structure == "geometric":
            latent = torch.randn(1, 4, h, w, device=self.device) * 0.3
            for _ in range(8):
                c = random.randint(0, 3)
                x1, x2 = sorted([random.randint(0, w-1) for _ in range(2)])
                y1, y2 = sorted([random.randint(0, h-1) for _ in range(2)])
                latent[:, c, y1:y2, x1:x2] += random.uniform(-2, 2)
                
        elif structure == "waves":
            x = torch.linspace(0, 6.28, w, device=self.device)
            y = torch.linspace(0, 6.28, h, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            latent = torch.stack([
                torch.sin(xx * 2 + yy),
                torch.cos(xx - yy * 1.5),
                torch.sin(xx * yy * 0.3),
                torch.cos(xx + yy * 2)
            ]).unsqueeze(0)
            
        elif structure == "fractal":
            latent = torch.zeros(1, 4, h, w, device=self.device)
            for octave in range(5):
                scale = 2 ** octave
                sh, sw = max(1, h // scale), max(1, w // scale)
                noise = torch.randn(1, 4, sh, sw, device=self.device)
                noise = F.interpolate(noise, size=(h, w), mode='bilinear', align_corners=True)
                latent += noise / (octave + 1)
        
        elif structure == "portrait":
            # Center-focused for portraits
            latent = torch.randn(1, 4, h, w, device=self.device)
            # Create center mask
            yc, xc = h // 2, w // 2
            y_grid = torch.arange(h, device=self.device).float()
            x_grid = torch.arange(w, device=self.device).float()
            yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
            dist = ((yy - yc)**2 + (xx - xc)**2).sqrt()
            mask = 1.0 - (dist / dist.max()).clamp(0, 1)
            mask = mask.unsqueeze(0).unsqueeze(0)
            latent = latent * mask + latent * 0.3 * (1 - mask)
            
        else:
            latent = torch.randn(1, 4, h, w, device=self.device)
        
        # Upsample
        latent_up = F.interpolate(latent, size=(height, width), mode='bilinear', align_corners=True)
        
        # Compute density (magnitude of latent)
        density = latent_up.norm(dim=1, keepdim=True)
        
        # Optional smoothing
        if smoothness > 1:
            kernel = torch.ones(1, 1, smoothness, smoothness, device=self.device) / (smoothness**2)
            density = F.conv2d(density, kernel, padding=smoothness//2)
        
        # Normalize to 0-1
        density = (density - density.min()) / (density.max() - density.min() + 1e-8)
        
        # Apply contrast
        density = (density - 0.5) * depth_contrast + 0.5
        density = density.clamp(0, 1)
        
        # Apply brightness shift
        density = density * (1 - depth_brightness) + depth_brightness * 0.5
        density = density.clamp(0, 1)
        
        # Invert if requested
        if depth_invert:
            density = 1.0 - density
        
        # Depth = inverted density (close = bright for ControlNet)
        depth = 1.0 - density.squeeze()
        
        # Compute normals via Sobel
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(density, sobel_x, padding=1)
        grad_y = F.conv2d(density, sobel_y, padding=1)
        
        normal = torch.cat([-grad_x, -grad_y, torch.ones_like(grad_x) * 0.5], dim=1)
        normal = F.normalize(normal, dim=1)
        normal = normal.squeeze(0).permute(1, 2, 0) * 0.5 + 0.5
        
        # Color from latent channels
        color = latent_up[:, :3].squeeze(0).permute(1, 2, 0)
        color = (color - color.min()) / (color.max() - color.min() + 1e-8)
        
        # To numpy
        depth_np = depth.cpu().numpy()
        normal_np = normal.cpu().numpy()
        color_np = color.cpu().numpy()
        
        # To PIL
        depth_pil = Image.fromarray((depth_np * 255).astype(np.uint8), mode='L')
        normal_pil = Image.fromarray((normal_np * 255).astype(np.uint8), mode='RGB')
        color_pil = Image.fromarray((color_np * 255).astype(np.uint8), mode='RGB')
        
        return {
            'depth': depth_pil,
            'normal': normal_pil,
            'color': color_pil,
            'seed': seed,
            'latent': latent
        }


def generate_sd_with_depth(
    prompt: str,
    negative: str = "ugly, blurry, deformed, bad anatomy, plastic skin",
    seed: int = -1,
    steps: int = 30,
    cfg: float = 6.0,
    # Depth tuning
    depth_contrast: float = 1.5,
    depth_brightness: float = 0.4,
    depth_invert: bool = False,
    smoothness: int = 7,
    structure: str = "portrait",
    # View results
    view: bool = True
):
    """Full generation with tunable latent depth"""
    
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    
    device = 'cuda'
    
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    
    # Generate depth/normal
    print(f"\n{'='*50}")
    print(f"LATENT DEPTH GENERATION")
    print(f"{'='*50}")
    print(f"Seed: {seed}")
    print(f"Structure: {structure}")
    print(f"Contrast: {depth_contrast}, Brightness: {depth_brightness}")
    
    marcher = TunableLatentMarcher(device)
    maps = marcher.generate(
        seed=seed,
        depth_contrast=depth_contrast,
        depth_brightness=depth_brightness,
        depth_invert=depth_invert,
        smoothness=smoothness,
        structure=structure
    )
    
    # Save depth/normal
    maps['depth'].save(os.path.expanduser('~/Desktop/gen_depth.png'))
    maps['normal'].save(os.path.expanduser('~/Desktop/gen_normal.png'))
    print(f"[✓] Depth: ~/Desktop/gen_depth.png")
    print(f"[✓] Normal: ~/Desktop/gen_normal.png")
    
    # Load SD
    print(f"\n[*] Loading RealisticVision...")
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
    
    # Generate
    print(f"\n[*] Generating: {prompt[:60]}...")
    generator = torch.Generator(device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=512,
        height=768,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
        clip_skip=2
    ).images[0]
    
    # Save
    path = os.path.expanduser(f'~/Desktop/latent_portrait_{seed}.png')
    image.save(path)
    
    print(f"\n[✓] Image: {path}")
    
    if view:
        subprocess.Popen(['xdg-open', path])
    
    return image, maps


# Presets for different looks
PRESETS = {
    'cinematic_dark': {
        'depth_contrast': 2.0,
        'depth_brightness': 0.2,
        'smoothness': 9,
        'structure': 'portrait',
        'cfg': 5.5
    },
    'soft_beauty': {
        'depth_contrast': 0.8,
        'depth_brightness': 0.6,
        'smoothness': 11,
        'structure': 'portrait',
        'cfg': 6.0
    },
    'dramatic': {
        'depth_contrast': 2.5,
        'depth_brightness': 0.3,
        'smoothness': 5,
        'structure': 'organic',
        'cfg': 7.0
    },
    'ethereal': {
        'depth_contrast': 1.2,
        'depth_brightness': 0.7,
        'depth_invert': True,
        'smoothness': 13,
        'structure': 'waves',
        'cfg': 5.0
    },
    'harsh': {
        'depth_contrast': 3.0,
        'depth_brightness': 0.1,
        'smoothness': 3,
        'structure': 'geometric',
        'cfg': 8.0
    }
}


if __name__ == "__main__":
    import sys
    
    # Parse args
    preset = 'cinematic_dark'
    prompt = "beautiful woman portrait, moody cinematic lighting, film grain, 85mm f1.4"
    
    if len(sys.argv) > 1:
        if sys.argv[1] in PRESETS:
            preset = sys.argv[1]
            if len(sys.argv) > 2:
                prompt = " ".join(sys.argv[2:])
        else:
            prompt = " ".join(sys.argv[1:])
    
    print(f"\nUsing preset: {preset}")
    print(f"Prompt: {prompt}\n")
    
    settings = PRESETS[preset]
    
    generate_sd_with_depth(
        prompt=prompt,
        **settings
    )
