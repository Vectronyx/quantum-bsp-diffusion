#!/usr/bin/env python3
"""
Quantum SDF diffusion - apply 4D quantum partition to image
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class QuantumImageDiffuser:
    """Diffuse image through quantum SDF field"""
    
    def __init__(self, res=512):
        self.res = res
    
    def load_image(self, path: str) -> torch.Tensor:
        """Load and normalize image"""
        img = Image.open(path).convert('RGB')
        img = img.resize((self.res, self.res), Image.LANCZOS)
        return torch.tensor(np.array(img) / 255.0, device=device, dtype=torch.float32)
    
    def quantum_sdf_field(self, t: float) -> torch.Tensor:
        """Generate quantum SDF displacement field"""
        u = torch.linspace(-1, 1, self.res, device=device)
        uu, vv = torch.meshgrid(u, u, indexing='xy')
        
        # Quantum interference pattern
        phase = t * 2 * np.pi
        
        # Multiple wave superposition
        field = torch.zeros(self.res, self.res, device=device)
        for i in range(4):
            freq = (i + 1) * 2
            wave_phase = i * np.pi / 4 + phase
            field += torch.sin(uu * freq * np.pi + wave_phase) * \
                     torch.cos(vv * freq * np.pi + wave_phase * 0.7) / (i + 1)
        
        # Add radial component
        r = torch.sqrt(uu**2 + vv**2)
        field += torch.sin(r * 8 * np.pi - phase * 2) * 0.3
        
        return field
    
    def sdf_sphere(self, p: torch.Tensor, center: torch.Tensor, r: float) -> torch.Tensor:
        """Sphere SDF"""
        return torch.norm(p - center, dim=-1) - r
    
    def diffuse(self, img: torch.Tensor, strength: float = 0.05, 
                t: float = 0.0) -> torch.Tensor:
        """Apply quantum SDF diffusion to image"""
        H, W = img.shape[:2]
        
        # Create coordinate grid
        u = torch.linspace(-1, 1, W, device=device)
        v = torch.linspace(-1, 1, H, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        # Quantum field displacement
        field = self.quantum_sdf_field(t)
        
        # Gradient of field for displacement direction
        dy = torch.roll(field, -1, 0) - torch.roll(field, 1, 0)
        dx = torch.roll(field, -1, 1) - torch.roll(field, 1, 1)
        
        # Normalize displacement
        mag = torch.sqrt(dx**2 + dy**2).clamp(min=1e-6)
        dx = dx / mag * strength * field
        dy = dy / mag * strength * field
        
        # Apply displacement
        new_u = (uu + dx).clamp(-1, 1)
        new_v = (vv + dy).clamp(-1, 1)
        
        # Grid sample
        grid = torch.stack([new_u, new_v], dim=-1).unsqueeze(0)
        img_in = img.permute(2, 0, 1).unsqueeze(0)
        
        diffused = F.grid_sample(img_in, grid, mode='bilinear', 
                                  padding_mode='border', align_corners=True)
        
        return diffused.squeeze(0).permute(1, 2, 0)
    
    def quantum_blend(self, img: torch.Tensor, t: float) -> torch.Tensor:
        """Blend with quantum interference colors"""
        H, W = img.shape[:2]
        
        u = torch.linspace(-1, 1, W, device=device)
        v = torch.linspace(-1, 1, H, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        phase = t * 2 * np.pi
        
        # Quantum color field
        r_field = 0.5 + 0.3 * torch.sin(uu * 4 + phase)
        g_field = 0.5 + 0.3 * torch.sin(vv * 4 + phase * 0.7)
        b_field = 0.5 + 0.3 * torch.sin((uu + vv) * 3 + phase * 1.3)
        
        quantum_color = torch.stack([r_field, g_field, b_field], dim=-1)
        
        # Blend factor based on luminance
        lum = img.mean(dim=-1, keepdim=True)
        blend = 0.15 * (1 - lum)
        
        return img * (1 - blend) + quantum_color * blend
    
    def render_sdf_overlay(self, img: torch.Tensor, t: float) -> torch.Tensor:
        """Overlay SDF visualization"""
        H, W = img.shape[:2]
        
        u = torch.linspace(-1, 1, W, device=device)
        v = torch.linspace(-1, 1, H, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        # Pulsing sphere SDF
        phase = t * 2 * np.pi
        center_x = 0.3 * np.sin(phase)
        center_y = 0.3 * np.cos(phase * 0.7)
        radius = 0.2 + 0.05 * np.sin(phase * 2)
        
        p = torch.stack([uu, vv], dim=-1)
        center = torch.tensor([center_x, center_y], device=device)
        
        d = self.sdf_sphere(p, center, radius)
        
        # SDF visualization
        edge = torch.exp(-torch.abs(d) * 30)  # Sharp edge glow
        glow = torch.exp(-d.clamp(min=0) * 5)  # Outer glow
        
        # Purple tint for SDF
        sdf_color = torch.zeros(H, W, 3, device=device)
        sdf_color[..., 0] = edge * 0.7 + glow * 0.3  # R
        sdf_color[..., 1] = edge * 0.2 + glow * 0.1  # G
        sdf_color[..., 2] = edge * 0.9 + glow * 0.5  # B
        
        # Blend
        return img * (1 - edge.unsqueeze(-1) * 0.7) + sdf_color * 0.7


def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  QUANTUM SDF IMAGE DIFFUSION                                  ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")
    
    diffuser = QuantumImageDiffuser(res=512)
    
    # Check for input image
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"  Loading: {img_path}")
        img = diffuser.load_image(img_path)
    else:
        # Generate test pattern
        print("  No image provided, generating test pattern...")
        u = torch.linspace(0, 1, 512, device=device)
        uu, vv = torch.meshgrid(u, u, indexing='xy')
        
        img = torch.zeros(512, 512, 3, device=device)
        img[..., 0] = 0.4 + 0.3 * torch.sin(uu * 10 * np.pi)
        img[..., 1] = 0.2 + 0.2 * torch.sin(vv * 10 * np.pi)
        img[..., 2] = 0.5 + 0.3 * torch.sin((uu + vv) * 8 * np.pi)
    
    print(f"  Device: {device}\n")
    
    # Render animation
    frames = []
    n_frames = 48
    
    for i in range(n_frames):
        t = i / n_frames
        
        if i % 8 == 0:
            print(f"  Frame {i+1:02d}/{n_frames}")
        
        # Apply quantum diffusion
        diffused = diffuser.diffuse(img, strength=0.08, t=t)
        
        # Add quantum color blend
        blended = diffuser.quantum_blend(diffused, t)
        
        # Overlay SDF visualization
        final = diffuser.render_sdf_overlay(blended, t)
        
        # Convert to PIL
        final_np = (torch.clamp(final, 0, 1).cpu().numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(final_np))
    
    # Save
    frames[0].save('quantum_diffuse.gif', save_all=True,
                   append_images=frames[1:], duration=50, loop=0)
    frames[0].save('quantum_diffuse.png')
    
    print(f"\n  ✓ Saved: quantum_diffuse.gif, quantum_diffuse.png")
    print("\n  Usage: python diffuse_quantum.py [image.png]")


if __name__ == "__main__":
    main()
