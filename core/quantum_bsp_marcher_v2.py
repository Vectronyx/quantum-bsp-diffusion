#!/usr/bin/env python3
"""
QUANTUM BSP VOLUMETRIC MARCHER V2
Extended with 4D support + Precision Curve Library
Integrates: 4d_volumetric_epsilon.py curves + quantum_bsp_marcher.py structure
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Tuple
import colorsys
import os

# ═══════════════════════════════════════════════════════════════════════════════
# PRECISION CURVE LIBRARY (from 4d_volumetric_epsilon.py - TORCH version)
# ═══════════════════════════════════════════════════════════════════════════════

class Curves:
    """1:1 mathematical curve alignment"""
    
    @staticmethod
    def smoothstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
        t = torch.clamp((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    
    @staticmethod
    def smootherstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
        t = torch.clamp((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    
    @staticmethod
    def hermite(t: torch.Tensor, p0: float, p1: float, m0: float, m1: float) -> torch.Tensor:
        t2, t3 = t * t, t * t * t
        h00 = 2*t3 - 3*t2 + 1
        h10 = t3 - 2*t2 + t
        h01 = -2*t3 + 3*t2
        h11 = t3 - t2
        return h00*p0 + h10*m0 + h01*p1 + h11*m1
    
    @staticmethod
    def bezier3(t: torch.Tensor, p0: float, p1: float, p2: float, p3: float) -> torch.Tensor:
        mt = 1.0 - t
        return mt**3*p0 + 3*mt**2*t*p1 + 3*mt*t**2*p2 + t**3*p3
    
    @staticmethod
    def sin(x: torch.Tensor, freq: float = 1.0, phase: float = 0.0) -> torch.Tensor:
        return torch.sin(x * freq * 2.0 * np.pi + phase)
    
    @staticmethod
    def cos(x: torch.Tensor, freq: float = 1.0, phase: float = 0.0) -> torch.Tensor:
        return torch.cos(x * freq * 2.0 * np.pi + phase)
    
    @staticmethod
    def exp_decay(x: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
        return torch.exp(-torch.abs(x) * rate)
    
    @staticmethod
    def smin(a: torch.Tensor, b: torch.Tensor, k: float = 0.2) -> torch.Tensor:
        h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
        return b * (1-h) + a * h - k * h * (1-h)
    
    @staticmethod
    def smax(a: torch.Tensor, b: torch.Tensor, k: float = 0.2) -> torch.Tensor:
        return -Curves.smin(-a, -b, k)
    
    @staticmethod
    def quantum_interference(x: torch.Tensor, n_waves: int = 4) -> torch.Tensor:
        result = torch.zeros_like(x)
        for i in range(n_waves):
            phase = i * np.pi / n_waves
            amp = 1.0 / (i + 1)
            result = result + amp * torch.sin(x * (i + 1) * np.pi + phase)
        return result / n_waves


# ═══════════════════════════════════════════════════════════════════════════════
# 4D ROTATION
# ═══════════════════════════════════════════════════════════════════════════════

class Rot4D:
    PLANES = {'xy':(0,1), 'xz':(0,2), 'xw':(0,3), 
              'yz':(1,2), 'yw':(1,3), 'zw':(2,3)}
    
    @staticmethod
    def matrix(plane: str, theta: float, device: str) -> torch.Tensor:
        c, s = np.cos(theta), np.sin(theta)
        R = torch.eye(4, dtype=torch.float32, device=device)
        i, j = Rot4D.PLANES[plane]
        R[i,i], R[j,j] = c, c
        R[i,j], R[j,i] = -s, s
        return R
    
    @staticmethod
    def from_6angles(xy=0, xz=0, xw=0, yz=0, yw=0, zw=0, device='cuda') -> torch.Tensor:
        R = torch.eye(4, dtype=torch.float32, device=device)
        for plane, angle in [('xy',xy),('xz',xz),('xw',xw),('yz',yz),('yw',yw),('zw',zw)]:
            if angle != 0:
                R = R @ Rot4D.matrix(plane, angle, device)
        return R


# ═══════════════════════════════════════════════════════════════════════════════
# 4D SDF PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

class SDF4D:
    @staticmethod
    def hypersphere(p: torch.Tensor, r: float = 1.0) -> torch.Tensor:
        return torch.norm(p, dim=-1) - r
    
    @staticmethod
    def tesseract(p: torch.Tensor, size: float = 1.0) -> torch.Tensor:
        q = torch.abs(p) - size
        return torch.norm(torch.clamp(q, min=0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0)
    
    @staticmethod
    def cell16(p: torch.Tensor, s: float = 1.0) -> torch.Tensor:
        return torch.sum(torch.abs(p), dim=-1) - s
    
    @staticmethod
    def hypertorus(p: torch.Tensor, R: float = 1.0, r1: float = 0.4, r2: float = 0.15) -> torch.Tensor:
        dxy = torch.sqrt(p[...,0]**2 + p[...,1]**2) - R
        dxyz = torch.sqrt(dxy**2 + p[...,2]**2) - r1
        return torch.sqrt(dxyz**2 + p[...,3]**2) - r2
    
    @staticmethod
    def gyroid_4d(p: torch.Tensor, scale: float = 1.0, thickness: float = 0.1) -> torch.Tensor:
        ps = p * scale
        g = (torch.sin(ps[...,0]*2*np.pi) * torch.cos(ps[...,1]*2*np.pi) +
             torch.sin(ps[...,1]*2*np.pi) * torch.cos(ps[...,2]*2*np.pi) +
             torch.sin(ps[...,2]*2*np.pi) * torch.cos(ps[...,3]*2*np.pi) +
             torch.sin(ps[...,3]*2*np.pi) * torch.cos(ps[...,0]*2*np.pi))
        return torch.abs(g) / scale - thickness


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM STATE 4D
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantumState4D:
    amplitudes: torch.Tensor    # Complex per-branch
    positions: torch.Tensor     # 4D positions per branch
    densities: torch.Tensor     # Accumulated density
    phase: torch.Tensor         # Phase per branch
    coherence: torch.Tensor     # Decoherence factor


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM BSP 4D MARCHER
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumBSP4DMarcher:
    """
    4D Quantum BSP Volumetric Marcher
    - Adaptive epsilon with curve modulation
    - Volumetric density integration
    - Quantum interference patterns
    - 6-plane 4D rotation
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        num_branches: int = 8,
        epsilon_base: float = 0.006,
        max_steps: int = 80,
        vol_steps: int = 40
    ):
        self.device = device
        self.num_branches = num_branches
        self.epsilon_base = epsilon_base
        self.max_steps = max_steps
        self.vol_steps = vol_steps
    
    def initialize_superposition(
        self,
        ray_origins: torch.Tensor,
        ray_dirs: torch.Tensor
    ) -> QuantumState4D:
        H, W = ray_origins.shape[:2]
        B = self.num_branches
        
        amplitudes = torch.ones(H, W, B, dtype=torch.complex64, device=self.device) / np.sqrt(B)
        positions = ray_origins.unsqueeze(2).expand(H, W, B, 4).clone()
        phase = torch.rand(H, W, B, device=self.device) * 2 * np.pi
        amplitudes = amplitudes * torch.exp(1j * phase)
        densities = torch.zeros(H, W, B, device=self.device)
        coherence = torch.ones(H, W, B, device=self.device)
        
        return QuantumState4D(amplitudes, positions, densities, phase, coherence)
    
    def adaptive_epsilon(self, t: torch.Tensor, step: int) -> torch.Tensor:
        """Curve-modulated adaptive epsilon"""
        dist_scale = 1.0 + t * 0.025
        step_t = torch.full_like(t, step / self.max_steps)
        step_factor = Curves.smoothstep(0, 1, step_t)
        return self.epsilon_base * dist_scale * (1.0 + step_factor * 0.5)
    
    def branch_sdf_4d(self, p: torch.Tensor, branch_idx: int, 
                      R_inv: torch.Tensor, time: float = 0.0) -> torch.Tensor:
        """4D SDF with branch offset"""
        shape = p.shape
        flat = p.reshape(-1, 4)
        p_rot = (R_inv @ flat.T).T.reshape(shape)
        
        offset = 0.12 * torch.tensor([
            np.sin(branch_idx * 0.7 + time),
            np.cos(branch_idx * 1.1 + time * 0.7),
            np.sin(branch_idx * 0.3 + time * 0.5),
            np.cos(branch_idx * 0.9 + time * 0.3)
        ], device=self.device, dtype=torch.float32)
        
        p_offset = p_rot - offset
        morph = (np.sin(time * 2 + branch_idx * 0.5) + 1) / 2
        
        d1 = SDF4D.tesseract(p_offset, 0.75)
        d2 = SDF4D.hypersphere(p_offset, 0.95)
        d3 = SDF4D.gyroid_4d(p_offset, 2.2, 0.1)
        
        base = d1 * (1 - morph) + d2 * morph
        return Curves.smax(base, -d3, 0.08)
    
    def collapse_wavefunction(self, state: QuantumState4D) -> Tuple[torch.Tensor, ...]:
        probs = (state.amplitudes.abs() ** 2).real * state.coherence
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        obs_weight = 1.0 - Curves.exp_decay(state.densities, rate=2.0)
        collapse_prob = probs * obs_weight
        collapse_prob = collapse_prob / (collapse_prob.sum(dim=-1, keepdim=True) + 1e-8)
        
        depth = (collapse_prob * state.positions[..., 2]).sum(dim=-1)
        w_coord = (collapse_prob * state.positions[..., 3]).sum(dim=-1)
        
        phase_diff = state.phase.unsqueeze(-1) - state.phase.unsqueeze(-2)
        interference = torch.cos(phase_diff).mean(dim=(-1, -2))
        q_field = Curves.quantum_interference(interference, n_waves=4)
        
        return depth, w_coord, interference, q_field
    
    def march(
        self,
        width: int = 512,
        height: int = 512,
        w_slice: float = 0.0,
        R: torch.Tensor = None,
        time: float = 0.0,
        seed: int = -1
    ) -> dict:
        if seed != -1:
            torch.manual_seed(seed)
        
        if R is None:
            R = torch.eye(4, device=self.device)
        R_inv = torch.inverse(R)
        
        aspect = width / height
        u = torch.linspace(-1.1 * aspect, 1.1 * aspect, width, device=self.device)
        v = torch.linspace(-1.1, 1.1, height, device=self.device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        ray_dirs = torch.zeros(height, width, 4, device=self.device)
        ray_dirs[..., 0] = uu
        ray_dirs[..., 1] = -vv
        ray_dirs[..., 2] = 1.5
        ray_dirs = F.normalize(ray_dirs, dim=-1)
        
        ray_origins = torch.zeros(height, width, 4, device=self.device)
        ray_origins[..., 2] = -3.5
        ray_origins[..., 3] = w_slice
        
        state = self.initialize_superposition(ray_origins, ray_dirs)
        
        t = torch.zeros(height, width, device=self.device)
        hit = torch.zeros(height, width, dtype=torch.bool, device=self.device)
        glow = torch.zeros(height, width, device=self.device)
        
        for step in range(self.max_steps):
            eps = self.adaptive_epsilon(t, step)
            
            for b in range(self.num_branches):
                sdf = self.branch_sdf_4d(state.positions[..., b, :], b, R_inv, time)
                
                density = Curves.exp_decay(sdf.abs(), rate=5.0)
                state.densities[..., b] += density * 0.1
                state.phase[..., b] += sdf * 0.5
                
                state.coherence[..., b] *= 1.0 - Curves.smoothstep(
                    0, self.max_steps, torch.full_like(sdf, step)) * 0.008
                
                new_hits = (sdf < eps) & ~hit
                hit |= new_hits
                
                # Glow accumulation
                glow += Curves.exp_decay(torch.clamp(sdf, min=0), rate=4) * 0.006 * (~hit).float()
                
                relax = Curves.bezier3(torch.tensor(step / self.max_steps), 0.85, 0.9, 0.95, 1.0)
                step_size = torch.clamp(sdf, min=eps) * relax
                
                state.positions[..., b, :] += ray_dirs * step_size.unsqueeze(-1)
                t += step_size / self.num_branches
                
                decay = Curves.exp_decay(step_size, rate=0.1)
                state.amplitudes[..., b] *= decay
        
        # Volumetric density pass
        vol_density = torch.zeros(height, width, device=self.device)
        for i in range(self.vol_steps):
            t_v = (i / self.vol_steps) * 8.0
            for b in range(min(self.num_branches, 4)):
                p = state.positions[..., b, :].clone()
                p[..., 2] = ray_origins[..., 2] + t_v
                d = self.branch_sdf_4d(p, b, R_inv, time)
                rho = Curves.exp_decay(torch.clamp(d, min=0), rate=3.0)
                w = Curves.smootherstep(0, 1, torch.tensor(i / self.vol_steps))
                vol_density += rho * w * state.coherence[..., b] / (self.vol_steps * 4)
        
        depth, w_coord, interference, q_field = self.collapse_wavefunction(state)
        
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        w_norm = Curves.smootherstep(-1.5, 1.5, w_coord)
        
        return {
            'depth': depth_norm,
            'w_coord': w_norm,
            'interference': interference,
            'q_field': q_field,
            'density': vol_density * 0.5,
            'hit': hit.float(),
            'glow': glow,
            'state': state
        }
    
    def render(self, result: dict, width: int, height: int, hue: float = 0.75) -> torch.Tensor:
        img = torch.zeros(height, width, 3, device=self.device)
        
        hit = result['hit']
        depth = result['depth']
        w_norm = result['w_coord']
        interference = result['interference']
        q_field = result['q_field']
        density = result['density']
        glow = result['glow']
        
        # HSV from w-coordinate
        h = (w_norm * 0.5 + hue + torch.abs(q_field) * 0.1) % 1.0
        s = 0.55 + 0.2 * Curves.sin(w_norm, freq=1.5)
        v = 0.12 + 0.65 * (1.0 - depth) * hit + interference * 0.08
        
        # Simple HSV→RGB
        c = v * s
        x = c * (1 - torch.abs((h * 6) % 2 - 1))
        m = v - c
        
        h6 = (h * 6).long() % 6
        
        r = torch.where(h6 == 0, c, torch.where(h6 == 1, x, 
            torch.where(h6 == 2, m, torch.where(h6 == 3, m,
            torch.where(h6 == 4, x, c))))) + m
        g = torch.where(h6 == 0, x, torch.where(h6 == 1, c,
            torch.where(h6 == 2, c, torch.where(h6 == 3, x,
            torch.where(h6 == 4, m, m))))) + m
        b = torch.where(h6 == 0, m, torch.where(h6 == 1, m,
            torch.where(h6 == 2, x, torch.where(h6 == 3, c,
            torch.where(h6 == 4, c, x))))) + m
        
        fog = Curves.exp_decay(depth * 1.8, rate=0.7)
        
        img[..., 0] = r * fog * hit
        img[..., 1] = g * fog * hit
        img[..., 2] = b * fog * hit
        
        # Background
        bg_mask = 1.0 - hit
        q_color = 0.5 + 0.5 * q_field
        
        img[..., 0] += (0.01 + glow * 0.08 + density * 0.06 * q_color) * bg_mask
        img[..., 1] += (0.005 + glow * 0.03 + density * 0.03) * bg_mask
        img[..., 2] += (0.02 + glow * 0.12 + density * 0.1 * (1 - q_color * 0.15)) * bg_mask
        
        return torch.clamp(img, 0, 1)
    
    def generate_maps(self, seed: int = -1, width: int = 512, height: int = 512,
                      w_slice: float = 0.0, R: torch.Tensor = None, time: float = 0.0):
        result = self.march(width, height, w_slice, R, time, seed)
        img = self.render(result, width, height)
        
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        depth_np = (result['depth'].cpu().numpy() * 255).astype(np.uint8)
        depth_pil = Image.fromarray(depth_np, mode='L')
        
        interf = result['interference'].cpu().numpy()
        interf_rgb = np.stack([
            (interf * 0.5 + 0.5) * 255,
            (np.sin(interf * np.pi) * 0.5 + 0.5) * 255,
            (np.cos(interf * np.pi) * 0.5 + 0.5) * 255
        ], axis=-1).astype(np.uint8)
        interf_pil = Image.fromarray(interf_rgb)
        
        return img_pil, depth_pil, interf_pil


def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  QUANTUM BSP 4D MARCHER V2                                    ║")
    print("║  Precision Curves + Volumetric Epsilon + 6-plane Rotation     ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}\n")
    
    marcher = QuantumBSP4DMarcher(
        device=device,
        num_branches=6,
        epsilon_base=0.005,
        max_steps=64,
        vol_steps=32
    )
    
    frames = []
    n_frames = 60
    res = 400
    
    for i in range(n_frames):
        t = i / n_frames
        theta = t * 2 * np.pi
        
        xw = float(Curves.hermite(torch.tensor(t), 0, 2*np.pi, 0.8, 0.8))
        yw = float(Curves.bezier3(torch.tensor(t), 0, np.pi*0.5, np.pi*1.3, 2*np.pi)) * 0.5
        zw = float(Curves.smootherstep(0, 1, torch.tensor(t))) * np.pi * 0.4
        
        R = Rot4D.from_6angles(xw=xw, yw=yw, zw=zw, xy=theta*0.15, device=device)
        w = float(Curves.sin(torch.tensor(t), freq=1.5)) * 0.6
        
        if i % 10 == 0:
            print(f"  ├─ Frame {i+1:02d}/{n_frames} │ w={w:+.3f}")
        
        result = marcher.march(res, res, w_slice=w, R=R, time=t * 4, seed=42)
        img = marcher.render(result, res, res)
        
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(img_np))
    
    frames[0].save('qbsp_4d_v2.gif', save_all=True,
                   append_images=frames[1:], duration=50, loop=0)
    frames[0].save('qbsp_4d_v2_test.png')
    
    print(f"\n  └─ ✓ Output: qbsp_4d_v2.gif, qbsp_4d_v2_test.png\n")
    
    print("  CURVES USED:")
    print("    ├─ smoothstep/smootherstep  │ epsilon modulation")
    print("    ├─ hermite/bezier3          │ rotation interpolation")
    print("    ├─ sin/cos                  │ w-slice oscillation")
    print("    ├─ exp_decay                │ fog + density falloff")
    print("    ├─ smin/smax                │ smooth CSG")
    print("    └─ quantum_interference     │ phase superposition")


if __name__ == "__main__":
    main()
