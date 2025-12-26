#!/usr/bin/env python3
"""
Quantum SDF diffusion - MP4 output
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import subprocess
import os
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class QuantumImageDiffuser:
    def __init__(self, res=512):
        self.res = res
    
    def load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        img = img.resize((self.res, self.res), Image.LANCZOS)
        return torch.tensor(np.array(img) / 255.0, device=device, dtype=torch.float32)
    
    def quantum_sdf_field(self, t: float) -> torch.Tensor:
        u = torch.linspace(-1, 1, self.res, device=device)
        uu, vv = torch.meshgrid(u, u, indexing='xy')
        
        phase = t * 2 * np.pi
        field = torch.zeros(self.res, self.res, device=device)
        
        for i in range(4):
            freq = (i + 1) * 2
            wave_phase = i * np.pi / 4 + phase
            field += torch.sin(uu * freq * np.pi + wave_phase) * \
                     torch.cos(vv * freq * np.pi + wave_phase * 0.7) / (i + 1)
        
        r = torch.sqrt(uu**2 + vv**2)
        field += torch.sin(r * 8 * np.pi - phase * 2) * 0.3
        
        return field
    
    def diffuse(self, img: torch.Tensor, strength: float = 0.05, t: float = 0.0) -> torch.Tensor:
        H, W = img.shape[:2]
        
        u = torch.linspace(-1, 1, W, device=device)
        v = torch.linspace(-1, 1, H, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        field = self.quantum_sdf_field(t)
        
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
    
    def quantum_blend(self, img: torch.Tensor, t: float) -> torch.Tensor:
        H, W = img.shape[:2]
        
        u = torch.linspace(-1, 1, W, device=device)
        v = torch.linspace(-1, 1, H, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        phase = t * 2 * np.pi
        
        r_field = 0.5 + 0.3 * torch.sin(uu * 4 + phase)
        g_field = 0.5 + 0.3 * torch.sin(vv * 4 + phase * 0.7)
        b_field = 0.5 + 0.3 * torch.sin((uu + vv) * 3 + phase * 1.3)
        
        quantum_color = torch.stack([r_field, g_field, b_field], dim=-1)
        
        lum = img.mean(dim=-1, keepdim=True)
        blend = 0.15 * (1 - lum)
        
        return img * (1 - blend) + quantum_color * blend
    
    def render_sdf_overlay(self, img: torch.Tensor, t: float) -> torch.Tensor:
        H, W = img.shape[:2]
        
        u = torch.linspace(-1, 1, W, device=device)
        v = torch.linspace(-1, 1, H, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        phase = t * 2 * np.pi
        center_x = 0.3 * np.sin(phase)
        center_y = 0.3 * np.cos(phase * 0.7)
        radius = 0.2 + 0.05 * np.sin(phase * 2)
        
        p = torch.stack([uu, vv], dim=-1)
        center = torch.tensor([center_x, center_y], device=device)
        
        d = torch.norm(p - center, dim=-1) - radius
        
        edge = torch.exp(-torch.abs(d) * 30)
        glow = torch.exp(-d.clamp(min=0) * 5)
        
        sdf_color = torch.zeros(H, W, 3, device=device)
        sdf_color[..., 0] = edge * 0.7 + glow * 0.3
        sdf_color[..., 1] = edge * 0.2 + glow * 0.1
        sdf_color[..., 2] = edge * 0.9 + glow * 0.5
        
        return img * (1 - edge.unsqueeze(-1) * 0.7) + sdf_color * 0.7


def frames_to_mp4(frames_dir: str, output: str, fps: int = 24):
    """Convert frames to MP4 using ffmpeg"""
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', f'{frames_dir}/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output
    ]
    subprocess.run(cmd, check=True)


def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  QUANTUM SDF IMAGE DIFFUSION - MP4                            ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")
    
    diffuser = QuantumImageDiffuser(res=512)
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"  Loading: {img_path}")
        img = diffuser.load_image(img_path)
    else:
        print("  Generating test pattern...")
        u = torch.linspace(0, 1, 512, device=device)
        uu, vv = torch.meshgrid(u, u, indexing='xy')
        
        img = torch.zeros(512, 512, 3, device=device)
        img[..., 0] = 0.4 + 0.3 * torch.sin(uu * 10 * np.pi)
        img[..., 1] = 0.2 + 0.2 * torch.sin(vv * 10 * np.pi)
        img[..., 2] = 0.5 + 0.3 * torch.sin((uu + vv) * 8 * np.pi)
    
    print(f"  Device: {device}\n")
    
    # Create temp frames dir
    frames_dir = '/tmp/quantum_frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    n_frames = 120  # 5 seconds at 24fps
    
    for i in range(n_frames):
        t = i / n_frames
        
        if i % 20 == 0:
            print(f"  Frame {i+1:03d}/{n_frames}")
        
        diffused = diffuser.diffuse(img, strength=0.08, t=t)
        blended = diffuser.quantum_blend(diffused, t)
        final = diffuser.render_sdf_overlay(blended, t)
        
        final_np = (torch.clamp(final, 0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(final_np).save(f'{frames_dir}/frame_{i:04d}.png')
    
    print("\n  Encoding MP4...")
    frames_to_mp4(frames_dir, 'quantum_diffuse.mp4', fps=24)
    
    # Cleanup
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))
    os.rmdir(frames_dir)
    
    print(f"\n  ✓ Saved: quantum_diffuse.mp4")
    print("\n  Usage: python diffuse_quantum_mp4.py [image.png]")


if __name__ == "__main__":
    main()
