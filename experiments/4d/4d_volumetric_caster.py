#!/usr/bin/env python3
"""
4D VOLUMETRIC EPSILON CASTER
1:1 Curve Precision Alignment - smoothstep, sin, cos, tan, hermite, bezier
Extends quantum_bsp_marcher.py with full 4D rotation + adaptive epsilon
"""

import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import colorsys


# ═══════════════════════════════════════════════════════════════════════════════
# PRECISION CURVE LIBRARY - 1:1 MATHEMATICAL ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

class Curves:
    """Exact mathematical curves for epsilon/field modulation"""
    
    @staticmethod
    def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
        """C1 Hermite: t²(3-2t)"""
        t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    
    @staticmethod
    def smootherstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
        """C2 Perlin: t³(t(6t-15)+10)"""
        t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    
    @staticmethod
    def sin(x: np.ndarray, freq: float = 1.0, phase: float = 0.0) -> np.ndarray:
        """Phase-aligned sine: sin(2πfx + φ)"""
        return np.sin(x * freq * 2.0 * np.pi + phase)
    
    @staticmethod
    def cos(x: np.ndarray, freq: float = 1.0, phase: float = 0.0) -> np.ndarray:
        """Phase-aligned cosine: cos(2πfx + φ)"""
        return np.cos(x * freq * 2.0 * np.pi + phase)
    
    @staticmethod
    def tan_safe(x: np.ndarray, limit: float = 10.0) -> np.ndarray:
        """Asymptote-clamped tangent"""
        return np.clip(np.tan(x), -limit, limit)
    
    @staticmethod
    def hermite(t: np.ndarray, p0: float, p1: float, m0: float, m1: float) -> np.ndarray:
        """Cubic Hermite spline: h₀₀p₀ + h₁₀m₀ + h₀₁p₁ + h₁₁m₁"""
        t2, t3 = t * t, t * t * t
        h00 = 2*t3 - 3*t2 + 1
        h10 = t3 - 2*t2 + t
        h01 = -2*t3 + 3*t2
        h11 = t3 - t2
        return h00*p0 + h10*m0 + h01*p1 + h11*m1
    
    @staticmethod
    def bezier3(t: np.ndarray, p0: float, p1: float, p2: float, p3: float) -> np.ndarray:
        """Cubic Bezier: Bernstein basis B(t) = Σ bᵢ(t)Pᵢ"""
        mt = 1.0 - t
        return mt**3*p0 + 3*mt**2*t*p1 + 3*mt*t**2*p2 + t**3*p3
    
    @staticmethod
    def exp_decay(x: np.ndarray, rate: float = 1.0) -> np.ndarray:
        """Exponential falloff: e^(-|x|·k)"""
        return np.exp(-np.abs(x) * rate)
    
    @staticmethod
    def gaussian(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Gaussian: e^(-x²/2σ²)"""
        return np.exp(-(x * x) / (2.0 * sigma * sigma))
    
    @staticmethod
    def smin(a: np.ndarray, b: np.ndarray, k: float = 0.2) -> np.ndarray:
        """Polynomial smooth minimum for CSG"""
        h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
        return b * (1-h) + a * h - k * h * (1-h)
    
    @staticmethod
    def smax(a: np.ndarray, b: np.ndarray, k: float = 0.2) -> np.ndarray:
        """Polynomial smooth maximum"""
        return -Curves.smin(-a, -b, k)
    
    @staticmethod
    def quantum_interference(x: np.ndarray, n_waves: int = 4) -> np.ndarray:
        """Wave superposition: Σ(sin(x·i+φᵢ)/i)"""
        result = np.zeros_like(x)
        for i in range(1, n_waves + 1):
            phase = i * np.pi / n_waves
            result += np.sin(x * i + phase) / i
        return result / n_waves


# ═══════════════════════════════════════════════════════════════════════════════
# 4D ROTATION - ALL 6 PLANES
# ═══════════════════════════════════════════════════════════════════════════════

class Rot4D:
    """4D rotation through 6 independent planes"""
    
    PLANES = {'xy':(0,1), 'xz':(0,2), 'xw':(0,3),
              'yz':(1,2), 'yw':(1,3), 'zw':(2,3)}
    
    @staticmethod
    def matrix(plane: str, theta: float) -> np.ndarray:
        """Single-plane rotation matrix"""
        c, s = np.cos(theta), np.sin(theta)
        R = np.eye(4, dtype=np.float64)
        i, j = Rot4D.PLANES[plane]
        R[i,i], R[j,j] = c, c
        R[i,j], R[j,i] = -s, s
        return R
    
    @staticmethod
    def compose(*matrices) -> np.ndarray:
        """Compose rotations via matrix multiplication"""
        R = np.eye(4, dtype=np.float64)
        for M in matrices:
            R = R @ M
        return R
    
    @staticmethod
    def from_6angles(xy=0, xz=0, xw=0, yz=0, yw=0, zw=0) -> np.ndarray:
        """Build rotation from all 6 plane angles"""
        return Rot4D.compose(
            Rot4D.matrix('xy', xy), Rot4D.matrix('xz', xz),
            Rot4D.matrix('xw', xw), Rot4D.matrix('yz', yz),
            Rot4D.matrix('yw', yw), Rot4D.matrix('zw', zw)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4D SDF PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

class SDF4D:
    """4D Signed Distance Functions"""
    
    @staticmethod
    def hypersphere(p: np.ndarray, r: float = 1.0, 
                    center: np.ndarray = None) -> np.ndarray:
        """Glome / 3-sphere"""
        if center is not None:
            p = p - center
        return np.sqrt(np.sum(p**2, axis=-1)) - r
    
    @staticmethod
    def tesseract(p: np.ndarray, size: float = 1.0) -> np.ndarray:
        """8-cell hypercube"""
        q = np.abs(p) - size
        return (np.sqrt(np.sum(np.maximum(q, 0)**2, axis=-1)) + 
                np.minimum(np.max(q, axis=-1), 0))
    
    @staticmethod
    def tesseract_round(p: np.ndarray, size: float = 1.0, r: float = 0.1) -> np.ndarray:
        """Rounded tesseract"""
        return SDF4D.tesseract(p, size - r) - r
    
    @staticmethod
    def hyperoctahedron(p: np.ndarray, s: float = 1.0) -> np.ndarray:
        """16-cell cross-polytope"""
        return np.sum(np.abs(p), axis=-1) - s
    
    @staticmethod
    def duocylinder(p: np.ndarray, r1: float = 0.8, r2: float = 0.8) -> np.ndarray:
        """S¹×S¹ product - unique to 4D"""
        d1 = np.sqrt(p[...,0]**2 + p[...,1]**2) - r1
        d2 = np.sqrt(p[...,2]**2 + p[...,3]**2) - r2
        return (np.sqrt(np.maximum(d1,0)**2 + np.maximum(d2,0)**2) +
                np.minimum(np.maximum(d1, d2), 0))
    
    @staticmethod
    def hypertorus(p: np.ndarray, R: float = 1.0, r1: float = 0.4, r2: float = 0.15) -> np.ndarray:
        """Clifford torus - doubly nested"""
        dxy = np.sqrt(p[...,0]**2 + p[...,1]**2) - R
        dxyz = np.sqrt(dxy**2 + p[...,2]**2) - r1
        return np.sqrt(dxyz**2 + p[...,3]**2) - r2
    
    @staticmethod
    def gyroid_4d(p: np.ndarray, scale: float = 1.0, thickness: float = 0.1) -> np.ndarray:
        """4D triply-periodic minimal surface"""
        ps = p * scale
        g = (np.sin(ps[...,0]*2*np.pi) * np.cos(ps[...,1]*2*np.pi) +
             np.sin(ps[...,1]*2*np.pi) * np.cos(ps[...,2]*2*np.pi) +
             np.sin(ps[...,2]*2*np.pi) * np.cos(ps[...,3]*2*np.pi) +
             np.sin(ps[...,3]*2*np.pi) * np.cos(ps[...,0]*2*np.pi))
        return np.abs(g) / scale - thickness
    
    @staticmethod
    def tiger(p: np.ndarray, R: float = 1.0, r: float = 0.4) -> np.ndarray:
        """Tiger - unique 4D toroid"""
        d1 = np.sqrt(p[...,0]**2 + p[...,2]**2) - R
        d2 = np.sqrt(p[...,1]**2 + p[...,3]**2) - R
        return np.sqrt(d1**2 + d2**2) - r


# ═══════════════════════════════════════════════════════════════════════════════
# VOLUMETRIC EPSILON CASTER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CastResult:
    """Full raycast result with volumetric data"""
    hit: np.ndarray
    position: np.ndarray
    distance: np.ndarray
    normal: np.ndarray
    steps: np.ndarray
    density: np.ndarray
    quantum_field: np.ndarray
    w_coord: np.ndarray


class VolumetricEpsilonCaster:
    """
    4D Volumetric Raymarcher with:
    - Adaptive epsilon (distance + step scaled)
    - Curve-modulated step relaxation
    - Density accumulation
    - Quantum field integration
    """
    
    def __init__(self, res: int = 512):
        self.res = res
        self.max_steps = 100
        self.max_dist = 16.0
        self.epsilon_base = 0.0008
        self.vol_steps = 48
        self.density_scale = 0.6
    
    def _create_rays(self, w_slice: float, fov: float = 1.1,
                     cam_dist: float = 4.5) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 4D ray grid"""
        u = np.linspace(-fov, fov, self.res)
        v = np.linspace(-fov, fov, self.res)
        uu, vv = np.meshgrid(u, v)
        
        origins = np.zeros((self.res, self.res, 4), dtype=np.float64)
        origins[..., 2] = -cam_dist
        origins[..., 3] = w_slice
        
        dirs = np.zeros((self.res, self.res, 4), dtype=np.float64)
        dirs[..., 0] = uu
        dirs[..., 1] = -vv
        dirs[..., 2] = 1.5
        
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        return origins, dirs
    
    def _adaptive_epsilon(self, t: np.ndarray, step: int) -> np.ndarray:
        """
        Epsilon adapts to:
        - Distance traveled (further = larger epsilon)
        - Step count (convergence scaling)
        """
        dist_scale = 1.0 + t * 0.035
        step_factor = Curves.smoothstep(0, self.max_steps, 
                                        np.full_like(t, step))
        return self.epsilon_base * dist_scale * (1.0 + step_factor * 0.45)
    
    def _transform(self, p: np.ndarray, R_inv: np.ndarray) -> np.ndarray:
        """Apply inverse rotation for object-space SDF"""
        shape = p.shape
        flat = p.reshape(-1, 4)
        return (R_inv @ flat.T).T.reshape(shape)
    
    def _compute_normal(self, sdf: Callable, p: np.ndarray, 
                        R_inv: np.ndarray, eps: float = 0.001) -> np.ndarray:
        """4D gradient via central differences"""
        n = np.zeros_like(p)
        for i in range(4):
            pp, pn = p.copy(), p.copy()
            pp[...,i] += eps
            pn[...,i] -= eps
            n[...,i] = sdf(self._transform(pp, R_inv)) - sdf(self._transform(pn, R_inv))
        return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)
    
    def _volumetric_pass(self, sdf: Callable, origins: np.ndarray,
                         dirs: np.ndarray, R_inv: np.ndarray,
                         max_t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Accumulate density and quantum field along rays"""
        density = np.zeros((self.res, self.res), dtype=np.float64)
        q_field = np.zeros((self.res, self.res), dtype=np.float64)
        
        for i in range(self.vol_steps):
            t = (i / self.vol_steps) * max_t
            p = origins + np.full((self.res, self.res, 1), t) * dirs
            d = sdf(self._transform(p, R_inv))
            
            # Density from distance field
            rho = Curves.exp_decay(np.maximum(d, 0), rate=4.0)
            
            # Quantum interference
            q = Curves.quantum_interference(d, n_waves=4)
            
            # Curve-weighted accumulation
            w = Curves.smootherstep(0, 1, i / self.vol_steps)
            density += rho * w / self.vol_steps
            q_field += q * w / self.vol_steps
        
        return density * self.density_scale, q_field
    
    def cast(self, sdf: Callable, w_slice: float = 0.0,
             R: np.ndarray = None, volumetric: bool = True) -> CastResult:
        """Full volumetric epsilon cast through 4D space"""
        R_inv = np.linalg.inv(R) if R is not None else np.eye(4)
        origins, dirs = self._create_rays(w_slice)
        
        t = np.zeros((self.res, self.res), dtype=np.float64)
        active = np.ones((self.res, self.res), dtype=bool)
        hit = np.zeros((self.res, self.res), dtype=bool)
        hit_pos = np.zeros((self.res, self.res, 4), dtype=np.float64)
        steps = np.zeros((self.res, self.res), dtype=np.int32)
        
        for step in range(self.max_steps):
            if not active.any():
                break
            
            p = origins + t[..., np.newaxis] * dirs
            d = sdf(self._transform(p, R_inv))
            
            eps = self._adaptive_epsilon(t, step)
            
            new_hits = active & (d < eps)
            hit |= new_hits
            hit_pos[new_hits] = p[new_hits]
            steps[active] = step
            
            missed = active & (t > self.max_dist)
            active &= ~new_hits & ~missed
            
            # Bezier-relaxed stepping
            relax = Curves.bezier3(step / self.max_steps, 0.82, 0.88, 0.95, 1.0)
            t[active] += d[active] * relax
        
        # Normals
        normals = np.zeros((self.res, self.res, 4), dtype=np.float64)
        if hit.any():
            normals[hit] = self._compute_normal(sdf, hit_pos, R_inv)[hit]
        
        # Volumetric
        density = np.zeros((self.res, self.res))
        q_field = np.zeros((self.res, self.res))
        if volumetric:
            density, q_field = self._volumetric_pass(sdf, origins, dirs, R_inv, self.max_dist)
        
        return CastResult(
            hit=hit, position=hit_pos, distance=t, normal=normals,
            steps=steps, density=density, quantum_field=q_field,
            w_coord=hit_pos[..., 3]
        )


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class Renderer4D:
    """4D renderer with curve-based shading"""
    
    def __init__(self, res: int = 512):
        self.res = res
        self.caster = VolumetricEpsilonCaster(res)
        self.light = np.array([0.4, 0.7, -0.5, 0.3])
        self.light /= np.linalg.norm(self.light)
    
    def render(self, sdf: Callable, w_slice: float = 0.0,
               R: np.ndarray = None) -> np.ndarray:
        
        result = self.caster.cast(sdf, w_slice, R, volumetric=True)
        img = np.zeros((self.res, self.res, 3))
        
        # Lighting
        diffuse = np.clip(np.sum(result.normal * self.light, axis=-1), 0, 1)
        
        # Specular
        view = np.array([0, 0, 1, 0])
        half_v = self.light + view
        half_v /= np.linalg.norm(half_v)
        spec = np.clip(np.sum(result.normal * half_v, axis=-1), 0, 1) ** 24
        
        # W-depth color mapping
        w_norm = Curves.smootherstep(-1.5, 1.5, result.w_coord)
        q_color = Curves.smoothstep(0, 0.5, np.abs(result.quantum_field))
        
        for y in range(self.res):
            for x in range(self.res):
                if result.hit[y, x]:
                    h = (w_norm[y, x] * 0.6 + 0.55 + q_color[y, x] * 0.15) % 1.0
                    s = 0.6 + 0.25 * Curves.sin(np.array([w_norm[y, x]]), 1.5)[0]
                    v = 0.12 + 0.75 * diffuse[y, x]
                    
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    r = min(1, r + spec[y, x] * 0.5)
                    g = min(1, g + spec[y, x] * 0.5)
                    b = min(1, b + spec[y, x] * 0.4)
                    
                    fog = Curves.exp_decay(np.array([result.distance[y, x]]), 0.065)[0]
                    img[y, x] = [r * fog, g * fog, b * fog]
                else:
                    vol = result.density[y, x]
                    q = 0.5 + 0.5 * result.quantum_field[y, x]
                    img[y, x] = [0.015 + vol * 0.12 * q,
                                 0.008 + vol * 0.06,
                                 0.03 + vol * 0.18 * (1 - q * 0.25)]
        
        return img


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  4D VOLUMETRIC EPSILON CASTER                                 ║")
    print("║  1:1 Curve Precision: smoothstep, sin, cos, tan, hermite...   ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    
    renderer = Renderer4D(res=384)
    frames = []
    n_frames = 60
    
    for i in range(n_frames):
        t = i / n_frames
        theta = t * 2 * np.pi
        
        # Smooth curve-modulated rotations
        xw = Curves.hermite(t, 0, 2*np.pi, 0.8, 0.8)
        yw = Curves.bezier3(t, 0, np.pi*0.6, np.pi*1.4, 2*np.pi) * 0.55
        zw = Curves.smootherstep(0, 1, t) * np.pi * 0.45
        
        R = Rot4D.from_6angles(xw=xw, yw=yw, zw=zw, xy=theta*0.18)
        
        # W-slice with interference
        w = Curves.sin(np.array([t]), freq=1.8)[0] * 0.75
        
        # Morphing scene
        morph = (Curves.sin(np.array([t]), freq=1)[0] + 1) / 2
        
        def scene(p):
            d1 = SDF4D.tesseract_round(p, 0.85, 0.12)
            d2 = SDF4D.hypersphere(p, 1.05)
            d3 = SDF4D.gyroid_4d(p, 2.8, 0.12)
            
            base = d1 * (1 - morph) + d2 * morph
            return Curves.smax(base, -d3, 0.08)
        
        print(f"  ├─ Frame {i+1:02d}/{n_frames} │ w={w:+.3f} │ morph={morph:.2f}")
        
        img = renderer.render(scene, w_slice=w, R=R)
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        frames.append(Image.fromarray(img_u8))
    
    frames[0].save('4d_volumetric.gif', save_all=True,
                   append_images=frames[1:], duration=45, loop=0)
    frames[0].save('4d_volumetric_test.png')
    
    print("  └─ ✓ Output: 4d_volumetric.gif, 4d_volumetric_test.png")
    print()
    print("  CURVE LIBRARY (1:1 PRECISION):")
    print("    ├─ smoothstep(e0,e1,x)      → t²(3-2t)")
    print("    ├─ smootherstep(e0,e1,x)    → t³(t(6t-15)+10)")
    print("    ├─ hermite(t,p0,p1,m0,m1)   → cubic spline")
    print("    ├─ bezier3(t,p0,p1,p2,p3)   → cubic Bezier")
    print("    ├─ sin(x,freq,phase)        → sin(2πfx+φ)")
    print("    ├─ cos(x,freq,phase)        → cos(2πfx+φ)")
    print("    ├─ tan_safe(x,limit)        → clamped asymptotes")
    print("    ├─ exp_decay(x,rate)        → e^(-|x|·k)")
    print("    ├─ gaussian(x,sigma)        → e^(-x²/2σ²)")
    print("    ├─ smin/smax(a,b,k)         → smooth CSG")
    print("    └─ quantum_interference(x,n) → Σ(sin(x·i+φ)/i)")
    print()
    print("  EPSILON ADAPTATION:")
    print("    ├─ base: 0.0008")
    print("    ├─ dist_scale: 1.0 + t * 0.035")
    print("    └─ step_factor: smoothstep(0, max_steps, step)")


if __name__ == "__main__":
    main()
