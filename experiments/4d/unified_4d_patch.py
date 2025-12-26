#!/usr/bin/env python3
"""
unified_4d_patch.py - Fixed solid geometry, no artifacts
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Callable, Tuple
from dataclasses import dataclass

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TAU = 6.283185307179586

# ═══════════════════════════════════════════════════════════════════════════════
# CURVES (JIT)
# ═══════════════════════════════════════════════════════════════════════════════

class Curves:
    @staticmethod
    @torch.jit.script
    def smoothstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
        t = torch.clamp((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    
    @staticmethod
    @torch.jit.script
    def smootherstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
        t = torch.clamp((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    
    @staticmethod
    @torch.jit.script
    def bezier3(t: torch.Tensor, p0: float, p1: float, p2: float, p3: float) -> torch.Tensor:
        mt = 1.0 - t
        return mt**3*p0 + 3.0*mt**2*t*p1 + 3.0*mt*t**2*p2 + t**3*p3
    
    @staticmethod
    @torch.jit.script
    def exp_decay(x: torch.Tensor, rate: float) -> torch.Tensor:
        return torch.exp(-torch.abs(x) * rate)
    
    @staticmethod
    @torch.jit.script
    def smin(a: torch.Tensor, b: torch.Tensor, k: float) -> torch.Tensor:
        h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
        return b + (a - b) * h - k * h * (1.0 - h)

# ═══════════════════════════════════════════════════════════════════════════════
# 4D ROTATION
# ═══════════════════════════════════════════════════════════════════════════════

class Rot4D:
    PLANES = ((0,1), (0,2), (0,3), (1,2), (1,3), (2,3))
    
    @staticmethod
    def from_angles(xw=0., yw=0., zw=0., xy=0., xz=0., yz=0.) -> torch.Tensor:
        R = torch.eye(4, device=device, dtype=torch.float32)
        for idx, a in enumerate((xy, xz, xw, yz, yw, zw)):
            if abs(a) > 1e-8:
                c, s = float(np.cos(a)), float(np.sin(a))
                i, j = Rot4D.PLANES[idx]
                Rp = torch.eye(4, device=device, dtype=torch.float32)
                Rp[i,i], Rp[j,j], Rp[i,j], Rp[j,i] = c, c, -s, s
                R = R @ Rp
        return R

# ═══════════════════════════════════════════════════════════════════════════════
# 4D SDFs - SOLID PRIMITIVES (no holes)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.jit.script
def sdf_hypersphere(p: torch.Tensor, r: float) -> torch.Tensor:
    return torch.sqrt((p**2).sum(dim=-1) + 1e-12) - r

@torch.jit.script
def sdf_tesseract(p: torch.Tensor, size: float) -> torch.Tensor:
    q = torch.abs(p) - size
    outside = torch.sqrt(torch.clamp(q, min=0.).pow(2).sum(dim=-1) + 1e-12)
    inside = torch.clamp(q.max(dim=-1).values, max=0.)
    return outside + inside

@torch.jit.script
def sdf_tesseract_round(p: torch.Tensor, size: float, r: float) -> torch.Tensor:
    return sdf_tesseract(p, size - r) - r

@torch.jit.script
def sdf_hyperoctahedron(p: torch.Tensor, s: float) -> torch.Tensor:
    return torch.abs(p).sum(dim=-1) - s

@torch.jit.script
def sdf_duocylinder(p: torch.Tensor, r1: float, r2: float) -> torch.Tensor:
    d1 = torch.sqrt(p[...,0]**2 + p[...,1]**2 + 1e-12) - r1
    d2 = torch.sqrt(p[...,2]**2 + p[...,3]**2 + 1e-12) - r2
    outside = torch.sqrt(torch.clamp(d1,min=0.)**2 + torch.clamp(d2,min=0.)**2 + 1e-12)
    inside = torch.clamp(torch.maximum(d1, d2), max=0.)
    return outside + inside

@torch.jit.script
def sdf_hypertorus(p: torch.Tensor, R: float, r1: float, r2: float) -> torch.Tensor:
    dxy = torch.sqrt(p[...,0]**2 + p[...,1]**2 + 1e-12) - R
    dxyz = torch.sqrt(dxy**2 + p[...,2]**2 + 1e-12) - r1
    return torch.sqrt(dxyz**2 + p[...,3]**2 + 1e-12) - r2

@torch.jit.script
def sdf_tiger(p: torch.Tensor, R: float, r: float) -> torch.Tensor:
    d1 = torch.sqrt(p[...,0]**2 + p[...,2]**2 + 1e-12) - R
    d2 = torch.sqrt(p[...,1]**2 + p[...,3]**2 + 1e-12) - R
    return torch.sqrt(d1**2 + d2**2 + 1e-12) - r

# ═══════════════════════════════════════════════════════════════════════════════
# 4D→3D SLICER
# ═══════════════════════════════════════════════════════════════════════════════

class Slicer4D:
    def __init__(self, sdf_4d: Callable, w_slice: float, R_4d: torch.Tensor, shape: tuple):
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
# VOLUMETRIC MARCHER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarchResult:
    hit: torch.Tensor
    depth: torch.Tensor
    normal: torch.Tensor
    position: torch.Tensor
    density: torch.Tensor

class VolumetricMarcher3D:
    def __init__(self, res: int = 512, fov: float = 1.0, cam_dist: float = 4.0):
        self.res = res
        self.max_steps = 80
        self.max_dist = 16.0
        self.eps_base = fov / res * 0.5
        
        u = torch.linspace(-fov, fov, res, device=device)
        v = torch.linspace(-fov, fov, res, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        dirs = torch.stack([uu, -vv, torch.full_like(uu, 1.5)], dim=-1)
        self._dirs = F.normalize(dirs, dim=-1)
        self._origin = torch.tensor([0., 0., -cam_dist], device=device)
        
        self._t = torch.zeros(res, res, device=device)
        self._hit = torch.zeros(res, res, dtype=torch.bool, device=device)
        self._pos = torch.zeros(res, res, 3, device=device)
        self._active = torch.zeros(res, res, dtype=torch.bool, device=device)
        self._n = torch.zeros(res, res, 3, device=device)
        self._density = torch.zeros(res, res, device=device)
        self._pp = torch.zeros(res, res, 3, device=device)
        self._pn = torch.zeros(res, res, 3, device=device)
    
    def march(self, sdf_3d: Callable) -> MarchResult:
        self._t.zero_()
        self._hit.zero_()
        self._pos.zero_()
        self._density.zero_()
        
        for step in range(self.max_steps):
            torch.logical_not(self._hit, out=self._active)
            self._active &= (self._t < self.max_dist)
            
            if not self._active.any():
                break
            
            p = self._origin + self._t.unsqueeze(-1) * self._dirs
            d = sdf_3d(p)
            
            eps = self.eps_base * (1.0 + self._t * 0.02)
            
            new_hit = self._active & (d < eps)
            self._hit |= new_hit
            self._pos = torch.where(new_hit.unsqueeze(-1), p, self._pos)
            
            # Density accumulation
            rho = Curves.exp_decay(torch.clamp(d, min=0.), 4.0)
            self._density += rho * self._active.float() * 0.015
            
            # Step with relaxation
            t_norm = torch.tensor(step / self.max_steps, device=device)
            relax = Curves.bezier3(t_norm, 0.85, 0.9, 0.95, 1.0)
            step_mask = self._active & (~new_hit)
            self._t = torch.where(step_mask, self._t + d * relax.item(), self._t)
        
        self._compute_normals(sdf_3d)
        depth = self._normalize_depth()
        
        return MarchResult(hit=self._hit.clone(), depth=depth, normal=self._n.clone(),
                          position=self._pos.clone(), density=self._density.clone())
    
    def _compute_normals(self, sdf_3d: Callable, eps: float = 0.001):
        self._n.zero_()
        for i in range(3):
            self._pp.copy_(self._pos)
            self._pn.copy_(self._pos)
            self._pp[..., i] += eps
            self._pn[..., i] -= eps
            self._n[..., i] = sdf_3d(self._pp) - sdf_3d(self._pn)
        norm = torch.sqrt((self._n**2).sum(dim=-1, keepdim=True) + 1e-12)
        self._n /= norm
        self._n *= self._hit.unsqueeze(-1).float()
    
    def _normalize_depth(self) -> torch.Tensor:
        depth = self._t.clone()
        if self._hit.any():
            hit_d = depth[self._hit]
            depth = (depth - hit_d.min()) / (hit_d.max() - hit_d.min() + 1e-8)
        return torch.where(self._hit, depth, torch.ones_like(depth))

# ═══════════════════════════════════════════════════════════════════════════════
# RENDERER - FIXED SHADING + GRADIENT BACKGROUND
# ═══════════════════════════════════════════════════════════════════════════════

class VolumetricRenderer:
    def __init__(self, res: int = 512):
        self.res = res
        self.marcher = VolumetricMarcher3D(res=res)
        self._light = F.normalize(torch.tensor([0.5, 0.8, -0.5], device=device), dim=0)
        self._half = F.normalize(self._light + torch.tensor([0., 0., 1.], device=device), dim=0)
        self._img = torch.zeros(res, res, 3, device=device)
        
        # Gradient background
        v = torch.linspace(0, 1, res, device=device).unsqueeze(1).expand(res, res)
        self._bg_r = 0.05 + 0.08 * v
        self._bg_g = 0.02 + 0.06 * v
        self._bg_b = 0.12 + 0.15 * v
    
    def render(self, sdf_4d: Callable, w_slice: float, R_4d: torch.Tensor) -> dict:
        sdf_3d = Slicer4D(sdf_4d, w_slice, R_4d, (self.res, self.res))
        result = self.marcher.march(sdf_3d)
        self._shade(result)
        return {'image': self._img.clone(), 'depth': result.depth,
                'normal': result.normal, 'hit': result.hit, 'density': result.density}
    
    def _shade(self, result: MarchResult):
        hit = result.hit
        normal = result.normal
        depth = result.depth
        density = result.density
        
        # Background gradient + volumetric glow
        vol = Curves.smoothstep(0., 0.2, density)
        self._img[..., 0] = self._bg_r + vol * 0.15
        self._img[..., 1] = self._bg_g + vol * 0.08
        self._img[..., 2] = self._bg_b + vol * 0.20
        
        if not hit.any():
            return
        
        # Lighting
        diffuse = (normal * self._light).sum(dim=-1).clamp(0, 1)
        spec = (normal * self._half).sum(dim=-1).clamp(0, 1).pow(32)
        
        # Fresnel rim
        view_dot = torch.abs(normal[..., 2])
        fresnel = (1.0 - view_dot).pow(3) * 0.4
        
        # Color from depth - purple/cyan palette
        hue = Curves.smootherstep(0., 1., 1.0 - depth) * 0.35 + 0.55  # cyan to purple
        sat = 0.65 + 0.15 * (1.0 - depth)
        val = 0.25 + 0.65 * diffuse
        
        # HSV to RGB
        h6 = (hue * 6.0) % 6.0
        hi = h6.long() % 6
        f = h6 - h6.floor()
        p = val * (1 - sat)
        q = val * (1 - sat * f)
        t = val * (1 - sat * (1 - f))
        
        r = torch.where(hi==0, val, torch.where(hi==1, q, torch.where(hi==2, p, 
            torch.where(hi==3, p, torch.where(hi==4, t, val)))))
        g = torch.where(hi==0, t, torch.where(hi==1, val, torch.where(hi==2, val,
            torch.where(hi==3, q, torch.where(hi==4, p, p)))))
        b = torch.where(hi==0, p, torch.where(hi==1, p, torch.where(hi==2, t,
            torch.where(hi==3, val, torch.where(hi==4, val, q)))))
        
        # Add specular + fresnel
        r = (r + spec * 0.5 + fresnel * 0.3).clamp(0, 1)
        g = (g + spec * 0.5 + fresnel * 0.2).clamp(0, 1)
        b = (b + spec * 0.4 + fresnel * 0.5).clamp(0, 1)
        
        # Distance fog blend to background
        fog = Curves.exp_decay(depth * 3.0, 1.0)
        r = r * fog + self._bg_r * (1 - fog)
        g = g * fog + self._bg_g * (1 - fog)
        b = b * fog + self._bg_b * (1 - fog)
        
        self._img[..., 0] = torch.where(hit, r, self._img[..., 0])
        self._img[..., 1] = torch.where(hit, g, self._img[..., 1])
        self._img[..., 2] = torch.where(hit, b, self._img[..., 2])

# ═══════════════════════════════════════════════════════════════════════════════
# CONTROLNET EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_depth(depth: torch.Tensor, hit: torch.Tensor) -> Image.Image:
    d = ((1.0 - depth) * hit.float()).cpu().numpy()
    return Image.fromarray((d * 255).astype(np.uint8)).convert('RGB')

def export_normal(normal: torch.Tensor, hit: torch.Tensor) -> Image.Image:
    n = normal.cpu().numpy()
    n_rgb = (n + 1) * 0.5
    neutral = np.array([0.5, 0.5, 1.0])
    n_rgb = np.where(hit.cpu().numpy()[..., None], n_rgb, neutral)
    return Image.fromarray((np.clip(n_rgb, 0, 1) * 255).astype(np.uint8))

# ═══════════════════════════════════════════════════════════════════════════════
# SCENE PRESETS (solid, no holes)
# ═══════════════════════════════════════════════════════════════════════════════

def scene_morph_solid(p: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    """Morphing tesseract <-> hypersphere - SOLID"""
    d1 = sdf_tesseract_round(p, 0.9, 0.12)
    d2 = sdf_hypersphere(p, 1.0)
    return Curves.smin(d1, d2, 0.3)

def scene_duocylinder(p: torch.Tensor) -> torch.Tensor:
    """Duocylinder - unique 4D shape"""
    return sdf_duocylinder(p, 0.85, 0.85)

def scene_hypertorus(p: torch.Tensor) -> torch.Tensor:
    """Clifford torus"""
    return sdf_hypertorus(p, 1.0, 0.45, 0.18)

def scene_tiger(p: torch.Tensor) -> torch.Tensor:
    """Tiger - 4D toroid"""
    return sdf_tiger(p, 0.9, 0.35)

def scene_compound(p: torch.Tensor) -> torch.Tensor:
    """Compound: tesseract + octahedron"""
    d1 = sdf_tesseract_round(p, 0.75, 0.08)
    d2 = sdf_hyperoctahedron(p, 1.1)
    return Curves.smin(d1, d2, 0.15)

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  UNIFIED 4D PATCH - FIXED                                      ║")
    print(f"║  Device: {device:54s} ║")
    print("║  Solid geometry, gradient background, no artifacts             ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    renderer = VolumetricRenderer(res=512)
    
    # Test single frame
    print("\n  Single frame test...")
    R = Rot4D.from_angles(xw=0.5, yw=0.3)
    result = renderer.render(scene_morph_solid, 0.0, R)
    
    img = Image.fromarray((result['image'].cpu().numpy() * 255).astype(np.uint8))
    img.save('patch_test.png')
    export_depth(result['depth'], result['hit']).save('patch_depth.png')
    export_normal(result['normal'], result['hit']).save('patch_normal.png')
    print("  ✓ patch_test.png\n  ✓ patch_depth.png\n  ✓ patch_normal.png")
    
    # Animation
    print("\n  Generating animation...")
    frames = []
    n_frames = 60
    
    for i in range(n_frames):
        t = i / n_frames
        theta = t * TAU
        
        # Smooth rotation
        xw = theta * 1.0
        yw = theta * 0.618
        zw = theta * 0.382
        
        R = Rot4D.from_angles(xw=xw, yw=yw, zw=zw)
        w = np.sin(theta * 2) * 0.5  # Gentle w oscillation
        
        result = renderer.render(scene_morph_solid, w, R)
        img_np = (result['image'].cpu().numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(img_np))
        
        if (i + 1) % 15 == 0:
            print(f"    Frame {i+1:02d}/{n_frames}")
    
    frames[0].save('patch_4d.gif', save_all=True, append_images=frames[1:], duration=50, loop=0)
    print("  ✓ patch_4d.gif")
    
    # Different scenes
    print("\n  Rendering scene gallery...")
    scenes = [
        ('duocylinder', scene_duocylinder),
        ('hypertorus', scene_hypertorus),
        ('tiger', scene_tiger),
        ('compound', scene_compound),
    ]
    
    for name, scene_fn in scenes:
        R = Rot4D.from_angles(xw=0.4, yw=0.3, zw=0.2)
        result = renderer.render(scene_fn, 0.0, R)
        img = Image.fromarray((result['image'].cpu().numpy() * 255).astype(np.uint8))
        img.save(f'patch_{name}.png')
        print(f"  ✓ patch_{name}.png")
    
    print("\n  Done!")

if __name__ == "__main__":
    demo()
