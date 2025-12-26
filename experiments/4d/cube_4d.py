#!/usr/bin/env python3
"""
4D Cube with pulsing edges, spinning, purple environment
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TAU = 6.283185307179586

# ═══════════════════════════════════════════════════════════════════════════════
# 4D ROTATION
# ═══════════════════════════════════════════════════════════════════════════════

class Rot4D:
    @staticmethod
    def from_angles(xw=0., yw=0., zw=0., xy=0., xz=0., yz=0.):
        R = torch.eye(4, device=device, dtype=torch.float32)
        planes = ((0,1), (0,2), (0,3), (1,2), (1,3), (2,3))
        for idx, a in enumerate((xy, xz, xw, yz, yw, zw)):
            if abs(a) > 1e-8:
                c, s = np.cos(a), np.sin(a)
                i, j = planes[idx]
                Rp = torch.eye(4, device=device, dtype=torch.float32)
                Rp[i,i], Rp[j,j], Rp[i,j], Rp[j,i] = c, c, -s, s
                R = R @ Rp
        return R

# ═══════════════════════════════════════════════════════════════════════════════
# TESSERACT WITH PULSING EDGES
# ═══════════════════════════════════════════════════════════════════════════════

def sdf_tesseract_edges(p: torch.Tensor, size: float, edge_width: float, pulse: float) -> torch.Tensor:
    """
    Tesseract rendered as edges only - pulse controls edge thickness
    """
    # Box frame SDF - edges of a hypercube
    q = torch.abs(p) - size
    
    # Distance to faces (solid cube)
    d_box = torch.sqrt(torch.clamp(q, min=0.).pow(2).sum(dim=-1) + 1e-12) + \
            torch.clamp(q.max(dim=-1).values, max=0.)
    
    # Make it hollow - edge frame
    # Count how many dimensions are "inside" the box
    inside = (q < 0).float()
    n_inside = inside.sum(dim=-1)
    
    # Edge thickness varies with pulse
    w = edge_width * (0.6 + 0.4 * pulse)
    
    # Frame: keep only edges (where exactly 2 dimensions are inside for 3D, 3 for 4D)
    # For 4D tesseract edges: 3 coords inside, 1 outside
    edge_mask = (n_inside >= 3).float()
    
    # Distance to edges
    q_sorted = torch.sort(torch.abs(q), dim=-1).values
    # Two smallest absolute distances define edge proximity
    d_edge = torch.sqrt(q_sorted[..., 0]**2 + q_sorted[..., 1]**2 + 1e-12) - w
    
    # Combine: solid where on edge
    d_frame = torch.where(n_inside >= 3, d_edge, d_box - w * 2)
    
    return d_frame

def sdf_tesseract_wireframe(p: torch.Tensor, size: float, thickness: float) -> torch.Tensor:
    """
    True wireframe - just the edges as cylinders
    """
    # Clamp to box
    q = torch.clamp(p, -size, size)
    
    # Distance to nearest edge
    # Edges are where 2 coordinates are at ±size
    ax = torch.abs(p[..., 0]) - size
    ay = torch.abs(p[..., 1]) - size
    az = torch.abs(p[..., 2]) - size
    aw = torch.abs(p[..., 3]) - size
    
    # Sort to find which are closest to edges
    dists = torch.stack([ax, ay, az, aw], dim=-1)
    sorted_d = torch.sort(torch.abs(dists), dim=-1).values
    
    # Distance to edge = sqrt of two smallest
    d = torch.sqrt(sorted_d[..., 0]**2 + sorted_d[..., 1]**2 + 1e-12) - thickness
    
    # Only where we're near the box surface
    box_d = torch.sqrt(torch.clamp(dists, min=0.).pow(2).sum(dim=-1) + 1e-12) + \
            torch.clamp(dists.max(dim=-1).values, max=0.)
    
    return torch.maximum(d, box_d - thickness * 0.5)

# ═══════════════════════════════════════════════════════════════════════════════
# 4D→3D SLICER
# ═══════════════════════════════════════════════════════════════════════════════

class Slicer4D:
    def __init__(self, sdf_4d, w_slice: float, R_4d: torch.Tensor, shape: tuple):
        self.sdf_4d = sdf_4d
        self.w_slice = w_slice
        self.R_inv = torch.inverse(R_4d)
        self._buffer = torch.empty(*shape, 4, device=device, dtype=torch.float32)
    
    def __call__(self, p3: torch.Tensor) -> torch.Tensor:
        self._buffer[..., :3] = p3
        self._buffer[..., 3] = self.w_slice
        p4_rot = torch.einsum('ij,...j->...i', self.R_inv, self._buffer)
        return self.sdf_4d(p4_rot)

# ═══════════════════════════════════════════════════════════════════════════════
# RAYMARCHER
# ═══════════════════════════════════════════════════════════════════════════════

class Marcher:
    def __init__(self, res: int = 512):
        self.res = res
        self.max_steps = 80
        self.max_dist = 12.0
        
        fov = 1.0
        u = torch.linspace(-fov, fov, res, device=device)
        v = torch.linspace(-fov, fov, res, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        dirs = torch.stack([uu, -vv, torch.full_like(uu, 1.5)], dim=-1)
        self._dirs = F.normalize(dirs, dim=-1)
        self._origin = torch.tensor([0., 0., -3.5], device=device)
        
        self._t = torch.zeros(res, res, device=device)
        self._hit = torch.zeros(res, res, dtype=torch.bool, device=device)
        self._pos = torch.zeros(res, res, 3, device=device)
        self._n = torch.zeros(res, res, 3, device=device)
        self._glow = torch.zeros(res, res, device=device)
    
    def march(self, sdf_3d):
        self._t.zero_()
        self._hit.zero_()
        self._glow.zero_()
        
        active = torch.ones(self.res, self.res, dtype=torch.bool, device=device)
        
        for step in range(self.max_steps):
            if not active.any():
                break
            
            p = self._origin + self._t.unsqueeze(-1) * self._dirs
            d = sdf_3d(p)
            
            eps = 0.002 * (1.0 + self._t * 0.02)
            
            new_hit = active & (d < eps)
            self._hit |= new_hit
            self._pos = torch.where(new_hit.unsqueeze(-1), p, self._pos)
            
            # Accumulate glow near surface
            glow_add = torch.exp(-torch.abs(d) * 8.0) * active.float() * 0.04
            self._glow += glow_add
            
            active &= (~new_hit) & (self._t < self.max_dist)
            self._t = torch.where(active, self._t + d * 0.9, self._t)
        
        # Normals
        self._compute_normals(sdf_3d)
        
        return self._hit, self._t, self._n, self._glow, self._pos
    
    def _compute_normals(self, sdf_3d, eps=0.001):
        self._n.zero_()
        for i in range(3):
            pp = self._pos.clone()
            pn = self._pos.clone()
            pp[..., i] += eps
            pn[..., i] -= eps
            self._n[..., i] = sdf_3d(pp) - sdf_3d(pn)
        norm = torch.sqrt((self._n**2).sum(dim=-1, keepdim=True) + 1e-12)
        self._n /= norm
        self._n *= self._hit.unsqueeze(-1).float()

# ═══════════════════════════════════════════════════════════════════════════════
# PURPLE ENVIRONMENT RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class PurpleRenderer:
    def __init__(self, res: int = 512):
        self.res = res
        self.marcher = Marcher(res)
        self._light = F.normalize(torch.tensor([0.4, 0.8, -0.6], device=device), dim=0)
        self._img = torch.zeros(res, res, 3, device=device)
        
        # Purple gradient background
        u = torch.linspace(-1, 1, res, device=device).unsqueeze(0).expand(res, res)
        v = torch.linspace(-1, 1, res, device=device).unsqueeze(1).expand(res, res)
        
        self._bg_r = 0.08 + 0.06 * (1 - v) + 0.03 * torch.sin(u * 3 + v * 2)
        self._bg_g = 0.02 + 0.02 * (1 - v)
        self._bg_b = 0.18 + 0.12 * (1 - v) + 0.05 * torch.cos(u * 2 - v * 3)
    
    def render(self, sdf_4d, w_slice: float, R_4d: torch.Tensor, pulse: float) -> torch.Tensor:
        sdf_3d = Slicer4D(sdf_4d, w_slice, R_4d, (self.res, self.res))
        hit, depth, normal, glow, pos = self.marcher.march(sdf_3d)
        
        # Background with animated purple swirl
        phase = pulse * TAU
        swirl = 0.03 * torch.sin(self._bg_r * 20 + phase) * torch.cos(self._bg_b * 15 - phase * 0.7)
        
        self._img[..., 0] = self._bg_r + swirl + glow * 0.6
        self._img[..., 1] = self._bg_g + swirl * 0.3 + glow * 0.2
        self._img[..., 2] = self._bg_b + swirl * 0.5 + glow * 0.9
        
        if not hit.any():
            return self._img.clone()
        
        # Normalize depth for hit pixels
        hit_d = depth[hit]
        d_min, d_max = hit_d.min(), hit_d.max()
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
        
        # Lighting
        diffuse = (normal * self._light).sum(dim=-1).clamp(0, 1)
        
        # View-dependent fresnel
        fresnel = (1.0 - torch.abs(normal[..., 2])).pow(2) * 0.6
        
        # Specular
        view = torch.tensor([0., 0., 1.], device=device)
        half_v = F.normalize(self._light + view, dim=0)
        spec = (normal * half_v).sum(dim=-1).clamp(0, 1).pow(24)
        
        # Purple/magenta/cyan edge colors
        # Edges get brighter purple, faces get darker
        edge_intensity = 0.4 + 0.6 * diffuse
        
        r = 0.5 * edge_intensity + spec * 0.6 + fresnel * 0.4
        g = 0.1 * edge_intensity + spec * 0.3 + fresnel * 0.1
        b = 0.8 * edge_intensity + spec * 0.5 + fresnel * 0.7
        
        # Pulse brightness on edges
        pulse_bright = 0.8 + 0.2 * pulse
        r = (r * pulse_bright).clamp(0, 1)
        g = (g * pulse_bright).clamp(0, 1)
        b = (b * pulse_bright).clamp(0, 1)
        
        # Apply
        self._img[..., 0] = torch.where(hit, r, self._img[..., 0])
        self._img[..., 1] = torch.where(hit, g, self._img[..., 1])
        self._img[..., 2] = torch.where(hit, b, self._img[..., 2])
        
        return self._img.clone()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  4D CUBE - PULSING EDGES + SPINNING + PURPLE ENV               ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(f"  Device: {device}")
    
    renderer = PurpleRenderer(res=512)
    frames = []
    n_frames = 120
    
    print(f"\n  Rendering {n_frames} frames...")
    
    for i in range(n_frames):
        t = i / n_frames
        theta = t * TAU
        
        # Pulse: edges go in and out
        pulse = (np.sin(theta * 3) + 1) / 2  # 3 pulses per rotation
        edge_width = 0.06 + 0.04 * pulse  # 0.06 to 0.10
        
        # Scene: tesseract wireframe with pulsing edges
        def scene(p, ew=edge_width):
            return sdf_tesseract_wireframe(p, size=0.9, thickness=ew)
        
        # Spinning rotation - all 6 planes active
        xw = theta * 1.0
        yw = theta * 0.7
        zw = theta * 0.5
        xy = theta * 0.3
        xz = theta * 0.2
        yz = theta * 0.15
        
        R = Rot4D.from_angles(xw=xw, yw=yw, zw=zw, xy=xy, xz=xz, yz=yz)
        
        # W-slice oscillates
        w = np.sin(theta * 2) * 0.4
        
        img = renderer.render(scene, w, R, pulse)
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(img_np))
        
        if (i + 1) % 20 == 0:
            print(f"    Frame {i+1:03d}/{n_frames} │ pulse={pulse:.2f}")
    
    # Save
    frames[0].save('cube_4d.gif', save_all=True, append_images=frames[1:], duration=33, loop=0)
    frames[0].save('cube_4d_test.png')
    frames[n_frames//4].save('cube_4d_quarter.png')
    
    print(f"\n  ✓ cube_4d.gif (120 frames, 30fps)")
    print(f"  ✓ cube_4d_test.png")
    print(f"  ✓ cube_4d_quarter.png")

if __name__ == "__main__":
    main()
