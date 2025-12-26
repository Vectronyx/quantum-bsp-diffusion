#!/usr/bin/env python3
"""
Quantum SDF + RealVision - Generate woman with 4D quantum diffusion
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import subprocess
import os

device = 'cuda'

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM SDF FIELD
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumSDFField:
    def __init__(self, res=512):
        self.res = res
    
    def generate_field(self, t: float) -> torch.Tensor:
        u = torch.linspace(-1, 1, self.res, device=device)
        uu, vv = torch.meshgrid(u, u, indexing='xy')
        
        phase = t * 2 * np.pi
        field = torch.zeros(self.res, self.res, device=device)
        
        for i in range(4):
            freq = (i + 1) * 1.5
            wave_phase = i * np.pi / 4 + phase
            field += torch.sin(uu * freq * np.pi + wave_phase) * \
                     torch.cos(vv * freq * np.pi + wave_phase * 0.7) / (i + 1)
        
        r = torch.sqrt(uu**2 + vv**2)
        field += torch.sin(r * 6 * np.pi - phase * 2) * 0.25
        
        return field
    
    def apply_diffusion(self, img: torch.Tensor, strength: float, t: float) -> torch.Tensor:
        H, W = img.shape[:2]
        
        u = torch.linspace(-1, 1, W, device=device)
        v = torch.linspace(-1, 1, H, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        field = self.generate_field(t)
        
        dy = torch.roll(field, -1, 0) - torch.roll(field, 1, 0)
        dx = torch.roll(field, -1, 1) - torch.roll(field, 1, 1)
        
        mag = torch.sqrt(dx**2 + dy**2).clamp(min=1e-6)
        dx = dx / mag * strength * field
        dy = dy / mag * strength * field
        
        new_u = (uu + dx).clamp(-1, 1)
        new_v = (vv + dy).clamp(-1, 1)
        
        grid = torch.stack([new_u, new_v], dim=-1).unsqueeze(0)
        img_in = img.permute(2, 0, 1).unsqueeze(0)
        
        diffused = F.grid_sample(img_in, grid, mode='bilinear',
                                  padding_mode='border', align_corners=True)
        
        return diffused.squeeze(0).permute(1, 2, 0)
    
    def quantum_glow(self, img: torch.Tensor, t: float) -> torch.Tensor:
        H, W = img.shape[:2]
        
        u = torch.linspace(-1, 1, W, device=device)
        v = torch.linspace(-1, 1, H, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        phase = t * 2 * np.pi
        
        # Purple quantum glow
        glow_r = 0.6 + 0.2 * torch.sin(uu * 3 + phase)
        glow_g = 0.2 + 0.1 * torch.sin(vv * 3 + phase * 0.8)
        glow_b = 0.8 + 0.2 * torch.sin((uu + vv) * 2 + phase * 1.2)
        
        glow = torch.stack([glow_r, glow_g, glow_b], dim=-1)
        
        # Edge detection for glow placement
        gray = img.mean(dim=-1)
        edge_x = torch.abs(torch.roll(gray, 1, 1) - torch.roll(gray, -1, 1))
        edge_y = torch.abs(torch.roll(gray, 1, 0) - torch.roll(gray, -1, 0))
        edges = (edge_x + edge_y).unsqueeze(-1)
        
        blend = edges * 0.3
        return img * (1 - blend) + glow * blend
    
    def sdf_orbs(self, img: torch.Tensor, t: float) -> torch.Tensor:
        H, W = img.shape[:2]
        
        u = torch.linspace(-1, 1, W, device=device)
        v = torch.linspace(-1, 1, H, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        phase = t * 2 * np.pi
        
        # Multiple floating orbs
        orb_overlay = torch.zeros(H, W, 3, device=device)
        
        for i in range(3):
            cx = 0.4 * np.sin(phase + i * 2.1)
            cy = 0.4 * np.cos(phase * 0.7 + i * 1.7)
            r = 0.08 + 0.02 * np.sin(phase * 2 + i)
            
            d = torch.sqrt((uu - cx)**2 + (vv - cy)**2)
            
            glow = torch.exp(-d / r * 2)
            edge = torch.exp(-torch.abs(d - r) * 40)
            
            # Purple/pink orb
            orb_overlay[..., 0] += (glow * 0.5 + edge * 0.8) * 0.4
            orb_overlay[..., 1] += (glow * 0.2 + edge * 0.3) * 0.3
            orb_overlay[..., 2] += (glow * 0.7 + edge * 1.0) * 0.5
        
        return torch.clamp(img + orb_overlay * 0.6, 0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# REALVISION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def load_realvision():
    print("  Loading RealVision...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    
    return pipe


def generate_woman(pipe, seed=42):
    prompt = """beautiful woman, elegant, purple ambient lighting, 
    ethereal glow, soft skin, detailed eyes, flowing hair,
    cinematic lighting, high quality, 8k, photorealistic,
    mystical atmosphere, quantum energy aura"""
    
    negative = """ugly, deformed, bad anatomy, bad hands, 
    blurry, low quality, watermark, text"""
    
    generator = torch.Generator(device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=512,
        height=512,
        num_inference_steps=25,
        guidance_scale=7.0,
        generator=generator
    ).images[0]
    
    return image


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  QUANTUM SDF + REALVISION                                     ║")
    print("║  Woman with 4D quantum diffusion effects                      ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")
    
    # Load pipeline
    pipe = load_realvision()
    
    # Generate base image
    print("\n  Generating woman...")
    base_image = generate_woman(pipe, seed=2024)
    base_image.save('quantum_woman_base.png')
    print("  ✓ Base image saved")
    
    # Convert to tensor
    img_tensor = torch.tensor(
        np.array(base_image) / 255.0, 
        device=device, 
        dtype=torch.float32
    )
    
    # Apply quantum effects
    print("\n  Applying quantum SDF effects...")
    qsdf = QuantumSDFField(res=512)
    
    frames_dir = '/tmp/quantum_woman_frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    n_frames = 120
    
    for i in range(n_frames):
        t = i / n_frames
        
        if i % 20 == 0:
            print(f"    Frame {i+1:03d}/{n_frames}")
        
        # Subtle diffusion
        diffused = qsdf.apply_diffusion(img_tensor, strength=0.03, t=t)
        
        # Quantum glow on edges
        glowed = qsdf.quantum_glow(diffused, t)
        
        # Floating orbs
        final = qsdf.sdf_orbs(glowed, t)
        
        # Save frame
        final_np = (torch.clamp(final, 0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(final_np).save(f'{frames_dir}/frame_{i:04d}.png')
    
    # Encode MP4
    print("\n  Encoding MP4...")
    cmd = [
        'ffmpeg', '-y',
        '-framerate', '24',
        '-i', f'{frames_dir}/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        'quantum_woman.mp4'
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Also save GIF
    print("  Encoding GIF...")
    frames = []
    for i in range(0, n_frames, 2):  # Skip frames for smaller GIF
        frames.append(Image.open(f'{frames_dir}/frame_{i:04d}.png'))
    frames[0].save('quantum_woman.gif', save_all=True,
                   append_images=frames[1:], duration=80, loop=0)
    
    # Cleanup
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))
    os.rmdir(frames_dir)
    
    print(f"\n  ✓ Saved: quantum_woman.mp4")
    print(f"  ✓ Saved: quantum_woman.gif")
    print(f"  ✓ Saved: quantum_woman_base.png")


if __name__ == "__main__":
    main()
