#!/usr/bin/env python3
"""
4D VOLUMETRIC SDF RAYMARCHER - ULTIMATE
═══════════════════════════════════════════════════════════════════════════════
Full stack:
  • 4D SDFs (tesseract, hypersphere, duocylinder, tiger, gyroid)
  • 6-plane rotation (xw, yw, zw, xy, xz, yz)
  • Epsilon-adaptive marching
  • Volumetric density accumulation
  • Curve library (smoothstep, hermite, bezier)
  • ControlNet export (depth, normal)
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Callable, Tuple
import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TAU = 6.283185307179586

# ═══════════════════════════════════════════════════════════════════════════════
# PRECISION CURVES (JIT)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.jit.script
def smoothstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    t = torch.clamp((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

@torch.jit.script
def smootherstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    t = torch.clamp((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

@torch.jit.script
def exp_decay(x: torch.Tensor, rate: float) -> torch.Tensor:
    return torch.exp(-torch.abs(x) * rate)

@torch.jit.script
def smin(a: torch.Tensor, b: torch.Tensor, k: float) -> torch.Tensor:
    h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)

@torch.jit.script
def smax(a: torch.Tensor, b: torch.Tensor, k: float) -> torch.Tensor:
    return -smin(-a, -b, k)

# ═══════════════════════════════════════════════════════════════════════════════
# 4D ROTATION
# ═══════════════════════════════════════════════════════════════════════════════

class Rot4D:
    PLANES = ((0,1), (0,2), (0,3), (1,2), (1,3), (2,3))
    
    @staticmethod
    def matrix(plane_idx: int, theta: float) -> torch.Tensor:
        c, s = np.cos(theta), np.sin(theta)
        R = torch.eye(4, device=device, dtype=torch.float32)
        i, j = Rot4D.PLANES[plane_idx]
        R[i,i], R[j,j], R[i,j], R[j,i] = c, c, -s, s
        return R
    
    @staticmethod
    def from_angles(xy=0., xz=0., xw=0., yz=0., yw=0., zw=0.) -> torch.Tensor:
        R = torch.eye(4, device=device, dtype=torch.float32)
        for idx, angle in enumerate([xy, xz, xw, yz, yw, zw]):
            if abs(angle) > 1e-8:
                R = R @ Rot4D.matrix(idx, angle)
        return R

# ═══════════════════════════════════════════════════════════════════════════════
# 4D SDF PRIMITIVES (JIT)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.jit.script
def sdf_hypersphere(p: torch.Tensor, r: float) -> torch.Tensor:
    return torch.sqrt((p * p).sum(dim=-1) + 1e-12) - r

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

@torch.jit.script
def sdf_gyroid_4d(p: torch.Tensor, scale: float, thickness: float) -> torch.Tensor:
    ps = p * scale * 6.283185307179586
    g = (torch.sin(ps[...,0]) * torch.cos(ps[...,1]) +
         torch.sin(ps[...,1]) * torch.cos(ps[...,2]) +
         torch.sin(ps[...,2]) * torch.cos(ps[...,3]) +
         torch.sin(ps[...,3]) * torch.cos(ps[...,0]))
    return (torch.abs(g) - thickness) / scale

@torch.jit.script
def sdf_24cell(p: torch.Tensor, s: float) -> torch.Tensor:
    """24-cell: |x|+|y|+|z|+|w| and permutations of |a|+|b|"""
    ap = torch.abs(p)
    # 24-cell is intersection of tesseract and hyperoctahedron scaled
    d1 = ap.max(dim=-1).values - s
    d2 = (ap[...,0] + ap[...,1] + ap[...,2] + ap[...,3]) / 2.0 - s
    return torch.maximum(d1, d2)

# ═══════════════════════════════════════════════════════════════════════════════
# 4D→3D SLICER
# ═══════════════════════════════════════════════════════════════════════════════

class Slicer4D:
    """Slice 4D SDF at w=w₀ after rotation"""
    def __init__(self, sdf_4d: Callable, w_slice: float, R: torch.Tensor, shape: Tuple[int, int]):
        self.sdf_4d = sdf_4d
        self.w_slice = w_slice
        self.R_inv = torch.inverse(R)
        self._buf = torch.empty(*shape, 4, device=device, dtype=torch.float32)
    
    def __call__(self, p3: torch.Tensor) -> torch.Tensor:
        self._buf[..., :3] = p3
        self._buf[..., 3] = self.w_slice
        p4 = torch.einsum('ij,...j->...i', self.R_inv, self._buf)
        return self.sdf_4d(p4)

# ═══════════════════════════════════════════════════════════════════════════════
# VOLUMETRIC RAYMARCHER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarchResult:
    hit: torch.Tensor
    depth: torch.Tensor
    normal: torch.Tensor
    position: torch.Tensor
    density: torch.Tensor
    steps: torch.Tensor

class VolumetricMarcher:
    def __init__(self, res: int = 512, fov: float = 1.0, cam_dist: float = 4.0):
        self.res = res
        self.max_steps = 100
        self.max_dist = 20.0
        self.eps_base = fov / res * 0.4
        self.vol_samples = 48
        
        # Ray setup
        u = torch.linspace(-fov, fov, res, device=device)
        v = torch.linspace(-fov, fov, res, device=device)
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        dirs = torch.stack([uu, -vv, torch.full_like(uu, 1.5)], dim=-1)
        self._dirs = F.normalize(dirs, dim=-1)
        self._origin = torch.tensor([0., 0., -cam_dist], device=device)
        
        # Preallocate
        self._t = torch.zeros(res, res, device=device)
        self._hit = torch.zeros(res, res, dtype=torch.bool, device=device)
        self._pos = torch.zeros(res, res, 3, device=device)
        self._n = torch.zeros(res, res, 3, device=device)
        self._density = torch.zeros(res, res, device=device)
        self._steps = torch.zeros(res, res, dtype=torch.int32, device=device)
    
    def march(self, sdf: Callable, volumetric: bool = True) -> MarchResult:
        self._t.zero_()
        self._hit.zero_()
        self._density.zero_()
        self._steps.zero_()
        
        active = torch.ones(self.res, self.res, dtype=torch.bool, device=device)
        
        for step in range(self.max_steps):
            if not active.any():
                break
            
            p = self._origin + self._t.unsqueeze(-1) * self._dirs
            d = sdf(p)
            
            # Adaptive epsilon
            eps = self.eps_base * (1.0 + self._t * 0.015)
            
            # Hit
            new_hit = active & (d < eps)
            self._hit |= new_hit
            self._pos = torch.where(new_hit.unsqueeze(-1), p, self._pos)
            self._steps[new_hit] = step
            
            # Volumetric density
            if volumetric:
                rho = exp_decay(torch.clamp(d, min=0.), 5.0)
                self._density += rho * active.float() * 0.015
            
            # Step
            active &= ~new_hit & (self._t < self.max_dist)
            relax = 0.85 + 0.15 * (step / self.max_steps)
            self._t = torch.where(active, self._t + d * relax, self._t)
        
        # Normals
        self._compute_normals(sdf)
        
        # Normalize depth
        depth = self._t.clone()
        if self._hit.any():
            hit_d = depth[self._hit]
            depth = (depth - hit_d.min()) / (hit_d.max() - hit_d.min() + 1e-8)
        depth = torch.where(self._hit, depth, torch.ones_like(depth))
        
        return MarchResult(
            hit=self._hit.clone(), depth=depth, normal=self._n.clone(),
            position=self._pos.clone(), density=self._density.clone(),
            steps=self._steps.clone()
        )
    
    def _compute_normals(self, sdf: Callable, eps: float = 0.001):
        self._n.zero_()
        for i in range(3):
            pp = self._pos.clone()
            pn = self._pos.clone()
            pp[..., i] += eps
            pn[..., i] -= eps
            self._n[..., i] = sdf(pp) - sdf(pn)
        norm = torch.sqrt((self._n * self._n).sum(dim=-1, keepdim=True) + 1e-12)
        self._n = self._n / norm * self._hit.unsqueeze(-1).float()

# ═══════════════════════════════════════════════════════════════════════════════
# RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class Renderer4D:
    def __init__(self, res: int = 512):
        self.res = res
        self.marcher = VolumetricMarcher(res)
        self._light = F.normalize(torch.tensor([0.5, 0.8, -0.5], device=device), dim=0)
        self._half = F.normalize(self._light + torch.tensor([0., 0., 1.], device=device), dim=0)
        self._img = torch.zeros(res, res, 3, device=device)
        
        # Background gradient
        v = torch.linspace(0, 1, res, device=device).unsqueeze(1).expand(res, res)
        self._bg = torch.stack([
            0.03 + 0.05 * v,
            0.02 + 0.03 * v,
            0.08 + 0.10 * v
        ], dim=-1)
    
    def render(self, sdf_4d: Callable, w_slice: float, R: torch.Tensor) -> dict:
        sdf_3d = Slicer4D(sdf_4d, w_slice, R, (self.res, self.res))
        result = self.marcher.march(sdf_3d, volumetric=True)
        self._shade(result)
        
        return {
            'image': self._img.clone(),
            'depth': result.depth,
            'normal': result.normal,
            'hit': result.hit,
            'density': result.density
        }
    
    def _shade(self, result: MarchResult):
        hit = result.hit
        normal = result.normal
        depth = result.depth
        density = result.density
        
        # Background + volumetric glow
        vol_glow = smoothstep(0., 0.25, density)
        self._img[..., 0] = self._bg[..., 0] + vol_glow * 0.15
        self._img[..., 1] = self._bg[..., 1] + vol_glow * 0.08
        self._img[..., 2] = self._bg[..., 2] + vol_glow * 0.25
        
        if not hit.any():
            return
        
        # Diffuse
        diffuse = (normal * self._light).sum(dim=-1).clamp(0, 1)
        
        # Specular
        spec = (normal * self._half).sum(dim=-1).clamp(0, 1).pow(32)
        
        # Fresnel rim
        fresnel = (1.0 - torch.abs(normal[..., 2])).pow(3) * 0.4
        
        # Color from depth (purple/cyan)
        hue = smootherstep(0., 1., 1.0 - depth) * 0.4 + 0.5
        sat = 0.7
        val = 0.2 + 0.7 * diffuse
        
        # HSV→RGB vectorized
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
        
        # Fog
        fog = exp_decay(depth * 3.0, 0.7)
        r = r * fog + self._bg[..., 0] * (1 - fog)
        g = g * fog + self._bg[..., 1] * (1 - fog)
        b = b * fog + self._bg[..., 2] * (1 - fog)
        
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
# SCENE PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

def scene_tesseract(p: torch.Tensor) -> torch.Tensor:
    return sdf_tesseract_round(p, 0.9, 0.1)

def scene_hypersphere(p: torch.Tensor) -> torch.Tensor:
    return sdf_hypersphere(p, 1.0)

def scene_duocylinder(p: torch.Tensor) -> torch.Tensor:
    return sdf_duocylinder(p, 0.85, 0.85)

def scene_tiger(p: torch.Tensor) -> torch.Tensor:
    return sdf_tiger(p, 0.9, 0.35)

def scene_hypertorus(p: torch.Tensor) -> torch.Tensor:
    return sdf_hypertorus(p, 1.0, 0.45, 0.18)

def scene_24cell(p: torch.Tensor) -> torch.Tensor:
    return sdf_24cell(p, 0.9)

def scene_gyroid_shell(p: torch.Tensor) -> torch.Tensor:
    sphere = sdf_hypersphere(p, 1.1)
    gyroid = sdf_gyroid_4d(p, 2.5, 0.06)
    return smax(sphere, -gyroid, 0.05)

def scene_compound(p: torch.Tensor) -> torch.Tensor:
    t = sdf_tesseract_round(p, 0.75, 0.08)
    s = sdf_hypersphere(p, 1.0)
    return smin(t, s, 0.2)

def scene_carved(p: torch.Tensor) -> torch.Tensor:
    base = sdf_tesseract_round(p, 0.9, 0.1)
    hole1 = sdf_duocylinder(p, 0.5, 0.5)
    hole2 = sdf_hypersphere(p, 0.7)
    carved = smax(base, -hole1, 0.08)
    carved = smax(carved, -hole2, 0.08)
    return carved

SCENES = {
    'tesseract': scene_tesseract,
    'hypersphere': scene_hypersphere,
    'duocylinder': scene_duocylinder,
    'tiger': scene_tiger,
    'hypertorus': scene_hypertorus,
    '24cell': scene_24cell,
    'gyroid': scene_gyroid_shell,
    'compound': scene_compound,
    'carved': scene_carved,
}

# ═══════════════════════════════════════════════════════════════════════════════
# ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def animate(scene_name: str = 'compound', n_frames: int = 90, res: int = 512, output: str = '4d_vol'):
    print(f"\n{'═'*60}")
    print(f"  4D VOLUMETRIC RAYMARCHER")
    print(f"  Scene: {scene_name} | Frames: {n_frames} | Res: {res}")
    print(f"{'═'*60}")
    
    renderer = Renderer4D(res=res)
    scene = SCENES.get(scene_name, scene_compound)
    
    # Warmup JIT
    print("  JIT warmup...")
    R = Rot4D.from_angles(xw=0.5)
    _ = renderer.render(scene, 0.0, R)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    t0 = time.perf_counter()
    for _ in range(5):
        _ = renderer.render(scene, 0.0, R)
    if device == 'cuda':
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"  → {ms:.1f} ms/frame ({1000/ms:.1f} FPS)")
    
    # Render animation
    frames = []
    print(f"\n  Rendering...")
    
    for i in range(n_frames):
        t = i / n_frames
        theta = t * TAU
        
        # Smooth rotation through all 6 planes
        xw = theta * 1.0
        yw = theta * 0.618  # Golden ratio
        zw = theta * 0.382
        xy = theta * 0.15
        
        R = Rot4D.from_angles(xw=xw, yw=yw, zw=zw, xy=xy)
        
        # W-slice oscillation
        w = np.sin(theta * 1.5) * 0.6
        
        result = renderer.render(scene, w, R)
        
        img_np = (result['image'].cpu().numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(img_np))
        
        # Save ControlNet maps on first frame
        if i == 0:
            export_depth(result['depth'], result['hit']).save(f'{output}_depth.png')
            export_normal(result['normal'], result['hit']).save(f'{output}_normal.png')
        
        if (i + 1) % 30 == 0:
            print(f"    Frame {i+1}/{n_frames}")
    
    # Save
    print("  Saving...")
    frames[0].save(f'{output}.gif', save_all=True, append_images=frames[1:], duration=33, loop=0)
    frames[0].save(f'{output}_test.png')
    print(f"  ✓ {output}.gif")
    print(f"  ✓ {output}_depth.png")
    print(f"  ✓ {output}_normal.png")
    
    # MP4
    try:
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for idx, f in enumerate(frames):
                f.save(f'{tmp}/f_{idx:04d}.png')
            subprocess.run(['ffmpeg', '-y', '-framerate', '30', '-i', f'{tmp}/f_%04d.png',
                           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
                           f'{output}.mp4'], capture_output=True, check=True)
            print(f"  ✓ {output}.mp4")
    except:
        pass
    
    return frames

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='4D Volumetric Raymarcher')
    parser.add_argument('--scene', '-s', type=str, default='compound', choices=list(SCENES.keys()))
    parser.add_argument('--frames', '-f', type=int, default=90)
    parser.add_argument('--res', '-r', type=int, default=512)
    parser.add_argument('--output', '-o', type=str, default='4d_vol')
    parser.add_argument('--all', action='store_true', help='Render all scenes')
    
    args = parser.parse_args()
    
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║  4D VOLUMETRIC SDF RAYMARCHER                                  ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    print(f"║  Device: {device:54s} ║")
    print("║                                                                ║")
    print("║  Scenes: tesseract, hypersphere, duocylinder, tiger,           ║")
    print("║          hypertorus, 24cell, gyroid, compound, carved          ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    if args.all:
        for scene_name in SCENES.keys():
            animate(scene_name, n_frames=60, res=args.res, output=f'4d_{scene_name}')
    else:
        animate(args.scene, n_frames=args.frames, res=args.res, output=args.output)

if __name__ == "__main__":
    main()
