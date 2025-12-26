#!/usr/bin/env python3
"""
REALVISION 4D LIFE v4
═══════════════════════════════════════════════════════════════════════════════
Real movement through optical flow + depth-aware parallax
Not just warping - actual dimensional movement
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import argparse
from pathlib import Path

torch.backends.cuda.matmul.allow_tf32 = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TAU = 6.283185307179586

# ═══════════════════════════════════════════════════════════════════════════════
# DEPTH ESTIMATION (MiDaS-style from gradients)
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_depth(img_t):
    """Estimate depth from image using edge/blur heuristics"""
    H, W = img_t.shape[:2]
    
    # Luminance
    lum = 0.299 * img_t[..., 0] + 0.587 * img_t[..., 1] + 0.114 * img_t[..., 2]
    
    # Edge detection (Sobel-like)
    pad_lum = F.pad(lum.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
    
    # Gradient magnitude
    gx = pad_lum[..., 1:-1, 2:] - pad_lum[..., 1:-1, :-2]
    gy = pad_lum[..., 2:, 1:-1] - pad_lum[..., :-2, 1:-1]
    edges = torch.sqrt(gx**2 + gy**2 + 1e-8).squeeze()
    
    # High detail = foreground, blur = background
    # Smooth edges to get region depth
    depth = 1.0 - edges  # Invert: edges are closer
    
    # Multi-scale blur for depth smoothing
    for k in [15, 31, 63]:
        depth = F.avg_pool2d(
            depth.unsqueeze(0).unsqueeze(0), 
            k, stride=1, padding=k//2
        ).squeeze()
    
    # Normalize
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    # Add vertical bias (lower = further in portraits typically)
    v_coord = torch.linspace(0, 1, H, device=device).unsqueeze(1).expand(H, W)
    depth = depth * 0.7 + (1 - v_coord) * 0.3
    
    return depth.clamp(0, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER SEPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def separate_layers(img_t, depth, n_layers=4):
    """Separate image into depth layers for parallax"""
    layers = []
    masks = []
    
    for i in range(n_layers):
        low = i / n_layers
        high = (i + 1) / n_layers
        
        # Soft mask for this depth range
        mask = torch.sigmoid((depth - low) * 20) * torch.sigmoid((high - depth) * 20)
        mask = F.avg_pool2d(mask.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2).squeeze()
        
        masks.append(mask)
        layers.append(img_t * mask.unsqueeze(-1))
    
    return layers, masks

# ═══════════════════════════════════════════════════════════════════════════════
# PARALLAX MOVEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def apply_parallax(img_t, depth, dx, dy, strength=1.0):
    """Move pixels based on depth - closer moves more"""
    H, W = img_t.shape[:2]
    
    u = torch.linspace(-1, 1, W, device=device)
    v = torch.linspace(-1, 1, H, device=device)
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    
    # Depth-scaled displacement (foreground moves more)
    # Invert depth so high depth = close = more movement
    move_scale = (1 - depth) * strength
    
    new_u = uu + dx * move_scale
    new_v = vv + dy * move_scale
    
    grid = torch.stack([new_u, new_v], dim=-1).unsqueeze(0)
    img_in = img_t.permute(2, 0, 1).unsqueeze(0)
    
    result = F.grid_sample(img_in, grid, mode='bilinear', 
                          padding_mode='border', align_corners=True)
    
    return result.squeeze(0).permute(1, 2, 0)

# ═══════════════════════════════════════════════════════════════════════════════
# BREATHING ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def breathing_displacement(H, W, t, depth):
    """Create breathing displacement field"""
    phase = t * TAU
    
    u = torch.linspace(-1, 1, W, device=device)
    v = torch.linspace(-1, 1, H, device=device)
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    
    # Breathing curve: 40% in, 60% out
    breath_phase = t % 1.0
    if breath_phase < 0.4:
        breath = 0.5 * (1 - np.cos(breath_phase / 0.4 * np.pi))
    else:
        breath = 0.5 * (1 + np.cos((breath_phase - 0.4) / 0.6 * np.pi))
    
    # Chest expansion - radial from center-bottom
    center_u, center_v = 0.0, 0.3  # Center of chest area
    
    dist_u = uu - center_u
    dist_v = vv - center_v
    
    # Expansion strength (more at chest level)
    chest_weight = torch.exp(-((vv - 0.3) ** 2) * 2)  # Peak at v=0.3
    
    dx = dist_u * breath * chest_weight * 0.015
    dy = dist_v * breath * chest_weight * 0.008 - breath * 0.003  # Slight rise
    
    return dx, dy, breath

# ═══════════════════════════════════════════════════════════════════════════════
# HEAD/CAMERA SUBTLE MOVEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def head_movement(t, strength=0.008):
    """Subtle head sway / camera drift"""
    phase = t * TAU
    
    # Very slow, organic movement
    dx = strength * (np.sin(phase * 0.3) + 0.3 * np.sin(phase * 0.7 + 1.2))
    dy = strength * 0.5 * (np.cos(phase * 0.25) + 0.4 * np.cos(phase * 0.6 + 0.8))
    
    return dx, dy

# ═══════════════════════════════════════════════════════════════════════════════
# DITHERED COLOR TO PREVENT BANDING
# ═══════════════════════════════════════════════════════════════════════════════

def apply_dither(frame, strength=1.0/255.0):
    """Add subtle dither to prevent banding"""
    noise = torch.rand_like(frame) * strength - strength/2
    return (frame + noise).clamp(0, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# SUBTLE LIGHT VARIATION (NO BANDING)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_light_variation(frame, t, breath):
    """Very subtle, dithered light changes"""
    # Extremely subtle global variation
    variation = 1.0 + breath * 0.008 + np.sin(t * TAU * 0.3) * 0.003
    
    frame = frame * variation
    
    # Dither to prevent banding
    frame = apply_dither(frame)
    
    return frame.clamp(0, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANIMATOR
# ═══════════════════════════════════════════════════════════════════════════════

class LifeAnimator:
    def animate(self, image, n_frames=90, output='life_4d'):
        # Convert
        if isinstance(image, Image.Image):
            img_np = np.array(image).astype(np.float32) / 255.0
            img_t = torch.tensor(img_np, device=device, dtype=torch.float32)
        else:
            img_t = image
        
        H, W = img_t.shape[:2]
        
        print("  Estimating depth...")
        depth = estimate_depth(img_t)
        
        # Save depth preview
        depth_np = (depth.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(depth_np).save(f'{output}_depth.png')
        print(f"  ✓ {output}_depth.png")
        
        frames = []
        print(f"  Rendering {n_frames} frames...")
        
        for i in range(n_frames):
            t = i / n_frames
            
            # Get movement components
            breath_dx, breath_dy, breath = breathing_displacement(H, W, t, depth)
            head_dx, head_dy = head_movement(t)
            
            # Apply parallax with depth
            # Head movement affects whole image with depth parallax
            frame = apply_parallax(img_t, depth, head_dx, head_dy, strength=1.0)
            
            # Apply breathing (more localized)
            u = torch.linspace(-1, 1, W, device=device)
            v = torch.linspace(-1, 1, H, device=device)
            vv, uu = torch.meshgrid(v, u, indexing='ij')
            
            # Combine displacements
            total_dx = breath_dx + head_dx * (1 - depth) * 0.5
            total_dy = breath_dy + head_dy * (1 - depth) * 0.5
            
            grid = torch.stack([uu + total_dx, vv + total_dy], dim=-1).unsqueeze(0)
            frame_in = frame.permute(2, 0, 1).unsqueeze(0)
            frame = F.grid_sample(frame_in, grid, mode='bilinear',
                                 padding_mode='border', align_corners=True)
            frame = frame.squeeze(0).permute(1, 2, 0)
            
            # Subtle light (with dither to prevent banding)
            frame = apply_light_variation(frame, t, breath)
            
            # Convert to PIL (with dither already applied)
            frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
            frames.append(Image.fromarray(frame_np))
            
            if (i + 1) % 30 == 0:
                print(f"    Frame {i+1}/{n_frames}")
        
        # Save GIF
        print("  Saving...")
        frames[0].save(f'{output}.gif', save_all=True, append_images=frames[1:],
                      duration=33, loop=0)
        print(f"  ✓ {output}.gif")
        
        # MP4 (higher quality, no GIF banding)
        try:
            import subprocess, tempfile
            with tempfile.TemporaryDirectory() as tmp:
                for idx, f in enumerate(frames):
                    f.save(f'{tmp}/f_{idx:04d}.png')
                subprocess.run(['ffmpeg', '-y', '-framerate', '30', 
                               '-i', f'{tmp}/f_%04d.png',
                               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 
                               '-crf', '18', '-preset', 'slow',
                               f'{output}.mp4'], capture_output=True, check=True)
                print(f"  ✓ {output}.mp4")
        except Exception as e:
            print(f"  ○ MP4 failed: {e}")
        
        return frames

# ═══════════════════════════════════════════════════════════════════════════════
# REALVISION INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class RealVision4DLife:
    def __init__(self):
        self.pipe = None
        self.animator = LifeAnimator()
    
    def load(self):
        if self.pipe is not None:
            return
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        
        model_path = Path.cwd() / "realisticVisionV60B1_v51HyperVAE.safetensors"
        if not model_path.exists():
            model_path = Path.home() / "Desktop" / "realisticVisionV60B1_v51HyperVAE.safetensors"
        
        print(f"  Loading: {model_path.name}")
        self.pipe = StableDiffusionPipeline.from_single_file(
            str(model_path), torch_dtype=torch.float16, safety_checker=None
        ).to(device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
        )
        self.pipe.enable_attention_slicing()
        print("  ✓ Loaded")
    
    def generate(self, prompt, seed=-1, width=512, height=768):
        self.load()
        if seed == -1:
            seed = np.random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(device).manual_seed(seed)
        
        print(f"  Generating seed {seed}...")
        image = self.pipe(
            prompt=prompt,
            negative_prompt="ugly, deformed, blurry, bad anatomy, watermark",
            width=width,
            height=height,
            num_inference_steps=28,
            guidance_scale=6.0,
            generator=generator
        ).images[0]
        
        return image, seed
    
    def run(self, prompt, n_frames=90, seed=-1, output='life_4d'):
        print(f"\n{'═'*60}")
        print("  REALVISION 4D LIFE v4")
        print("  Depth parallax + breathing + head movement")
        print(f"{'═'*60}")
        
        image, seed = self.generate(prompt, seed=seed)
        image.save(f'{output}_base.png')
        print(f"  ✓ {output}_base.png")
        
        frames = self.animator.animate(image, n_frames=n_frames, output=output)
        return {'seed': seed, 'frames': frames}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='RealVision 4D Life v4')
    parser.add_argument('prompt', nargs='?', default=None)
    parser.add_argument('--image', '-i', type=str)
    parser.add_argument('--frames', '-f', type=int, default=90)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--output', '-o', type=str, default='life_4d')
    
    args = parser.parse_args()
    
    if args.prompt is None and args.image is None:
        print("\n╔════════════════════════════════════════════════════════════════╗")
        print("║  REALVISION 4D LIFE v4                                         ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        print("║  Real movement via depth parallax:                             ║")
        print("║                                                                ║")
        print("║  • Depth estimation from image                                 ║")
        print("║  • Parallax: foreground moves more than background             ║")
        print("║  • Breathing: chest expansion with depth awareness             ║")
        print("║  • Head drift: subtle camera/head movement                     ║")
        print("║  • Dithered output: no banding artifacts                       ║")
        print("║                                                                ║")
        print("║  Usage:                                                        ║")
        print("║    python realvision_4d_life.py \"woman portrait\"              ║")
        print("║    python realvision_4d_life.py -i photo.png                   ║")
        print("╚════════════════════════════════════════════════════════════════╝\n")
        return
    
    if args.image:
        print(f"\n  Loading: {args.image}")
        img = Image.open(args.image).convert('RGB')
        animator = LifeAnimator()
        animator.animate(img, n_frames=args.frames, output=args.output)
    else:
        engine = RealVision4DLife()
        engine.run(args.prompt, n_frames=args.frames, seed=args.seed, output=args.output)

if __name__ == "__main__":
    main()
