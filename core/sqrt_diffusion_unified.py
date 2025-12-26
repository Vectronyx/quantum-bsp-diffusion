#!/usr/bin/env python3
"""
SQRT-OPTIMIZED 4D CASTER + DIFFUSION PROTOCOL
Unified system bridging 4D raymarching to SD latent conditioning
"""

import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Dict
import colorsys


# ═══════════════════════════════════════════════════════════════════════════════
# FAST MATH - SQRT ELIMINATION
# ═══════════════════════════════════════════════════════════════════════════════

class FastMath:
    """Sqrt-optimized operations"""
    
    @staticmethod
    def norm_sq(v: np.ndarray, axis: int = -1) -> np.ndarray:
        """Squared norm - zero sqrt"""
        return np.sum(v * v, axis=axis)
    
    @staticmethod
    def inv_sqrt(x: np.ndarray, iters: int = 2) -> np.ndarray:
        """Fast inverse sqrt via Newton-Raphson"""
        y = 1.0 / np.sqrt(np.maximum(x, 1e-12))
        for _ in range(iters):
            y = y * (1.5 - 0.5 * x * y * y)
        return y
    
    @staticmethod
    def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
        """Normalize using inv_sqrt"""
        sq = np.sum(v * v, axis=axis, keepdims=True)
        return v * FastMath.inv_sqrt(sq)
    
    @staticmethod
    def length_lt(v: np.ndarray, threshold: float) -> np.ndarray:
        """Check |v| < t without sqrt: |v|² < t²"""
        return FastMath.norm_sq(v) < threshold * threshold
    
    @staticmethod
    def safe_sqrt(x: np.ndarray) -> np.ndarray:
        """Sqrt with numerical safety"""
        return np.sqrt(np.maximum(x, 1e-12))


# ═══════════════════════════════════════════════════════════════════════════════
# CURVES - 1:1 PRECISION (NO SQRT NEEDED)
# ═══════════════════════════════════════════════════════════════════════════════

class Curves:
    """Exact mathematical curves"""
    
    @staticmethod
    def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
        t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    
    @staticmethod
    def smootherstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
        t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    
    @staticmethod
    def sin(x: np.ndarray, freq: float = 1.0, phase: float = 0.0) -> np.ndarray:
        return np.sin(x * freq * 2.0 * np.pi + phase)
    
    @staticmethod
    def cos(x: np.ndarray, freq: float = 1.0, phase: float = 0.0) -> np.ndarray:
        return np.cos(x * freq * 2.0 * np.pi + phase)
    
    @staticmethod
    def exp_decay(x: np.ndarray, rate: float = 1.0) -> np.ndarray:
        return np.exp(-np.abs(x) * rate)
    
    @staticmethod
    def hermite(t: np.ndarray, p0: float, p1: float, m0: float, m1: float) -> np.ndarray:
        t2, t3 = t * t, t * t * t
        return (2*t3 - 3*t2 + 1)*p0 + (t3 - 2*t2 + t)*m0 + (-2*t3 + 3*t2)*p1 + (t3 - t2)*m1
    
    @staticmethod
    def bezier3(t: np.ndarray, p0: float, p1: float, p2: float, p3: float) -> np.ndarray:
        mt = 1.0 - t
        return mt**3*p0 + 3*mt**2*t*p1 + 3*mt*t**2*p2 + t**3*p3
    
    @staticmethod
    def smin(a: np.ndarray, b: np.ndarray, k: float = 0.2) -> np.ndarray:
        h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
        return b * (1-h) + a * h - k * h * (1-h)


# ═══════════════════════════════════════════════════════════════════════════════
# 4D SDF - SQRT OPTIMIZED
# ═══════════════════════════════════════════════════════════════════════════════

class SDF4D:
    """4D SDFs with sqrt count annotations"""
    
    @staticmethod
    def hypersphere(p: np.ndarray, r: float = 1.0) -> np.ndarray:
        """[1 sqrt] - required for exact SDF"""
        return FastMath.safe_sqrt(FastMath.norm_sq(p)) - r
    
    @staticmethod
    def hypersphere_approx(p: np.ndarray, r: float = 1.0) -> np.ndarray:
        """[0 sqrt] - linearized near surface: (|p|² - r²) / 2r"""
        return (FastMath.norm_sq(p) - r*r) / (2*r)
    
    @staticmethod
    def tesseract(p: np.ndarray, size: float = 1.0) -> np.ndarray:
        """[1 sqrt] - exterior only"""
        q = np.abs(p) - size
        outside = FastMath.safe_sqrt(FastMath.norm_sq(np.maximum(q, 0)))
        inside = np.minimum(np.max(q, axis=-1), 0)
        return outside + inside
    
    @staticmethod
    def tesseract_approx(p: np.ndarray, size: float = 1.0) -> np.ndarray:
        """[0 sqrt] - Chebyshev approximation"""
        q = np.abs(p) - size
        return np.max(q, axis=-1)
    
    @staticmethod
    def hyperoctahedron(p: np.ndarray, s: float = 1.0) -> np.ndarray:
        """[0 sqrt] - L1 norm"""
        return np.sum(np.abs(p), axis=-1) - s
    
    @staticmethod
    def duocylinder(p: np.ndarray, r1: float = 0.8, r2: float = 0.8) -> np.ndarray:
        """[2 sqrt] - can reduce to 1 with approx"""
        sq1 = p[...,0]**2 + p[...,1]**2
        sq2 = p[...,2]**2 + p[...,3]**2
        d1 = FastMath.safe_sqrt(sq1) - r1
        d2 = FastMath.safe_sqrt(sq2) - r2
        return np.maximum(d1, d2)
    
    @staticmethod
    def duocylinder_fast(p: np.ndarray, r1: float = 0.8, r2: float = 0.8) -> np.ndarray:
        """[0 sqrt] - squared distance comparison"""
        sq1 = p[...,0]**2 + p[...,1]**2
        sq2 = p[...,2]**2 + p[...,3]**2
        # Approximate: use max of squared residuals
        d1_sq = sq1 - r1*r1
        d2_sq = sq2 - r2*r2
        # Linearize near surface
        d1_approx = d1_sq / (2*r1)
        d2_approx = d2_sq / (2*r2)
        return np.maximum(d1_approx, d2_approx)
    
    @staticmethod
    def hypertorus(p: np.ndarray, R: float = 1.0, r1: float = 0.4, r2: float = 0.15) -> np.ndarray:
        """[3 sqrt] - nested radii"""
        dxy = FastMath.safe_sqrt(p[...,0]**2 + p[...,1]**2) - R
        dxyz = FastMath.safe_sqrt(dxy**2 + p[...,2]**2) - r1
        return FastMath.safe_sqrt(dxyz**2 + p[...,3]**2) - r2
    
    @staticmethod
    def gyroid_4d(p: np.ndarray, scale: float = 1.0, thickness: float = 0.1) -> np.ndarray:
        """[0 sqrt] - pure trig"""
        ps = p * scale
        g = (np.sin(ps[...,0]*2*np.pi) * np.cos(ps[...,1]*2*np.pi) +
             np.sin(ps[...,1]*2*np.pi) * np.cos(ps[...,2]*2*np.pi) +
             np.sin(ps[...,2]*2*np.pi) * np.cos(ps[...,3]*2*np.pi) +
             np.sin(ps[...,3]*2*np.pi) * np.cos(ps[...,0]*2*np.pi))
        return np.abs(g) / scale - thickness


# ═══════════════════════════════════════════════════════════════════════════════
# 4D ROTATION (NO SQRT)
# ═══════════════════════════════════════════════════════════════════════════════

class Rot4D:
    PLANES = {'xy':(0,1), 'xz':(0,2), 'xw':(0,3), 'yz':(1,2), 'yw':(1,3), 'zw':(2,3)}
    
    @staticmethod
    def matrix(plane: str, theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        R = np.eye(4, dtype=np.float64)
        i, j = Rot4D.PLANES[plane]
        R[i,i], R[j,j] = c, c
        R[i,j], R[j,i] = -s, s
        return R
    
    @staticmethod
    def from_6angles(xy=0, xz=0, xw=0, yz=0, yw=0, zw=0) -> np.ndarray:
        R = np.eye(4, dtype=np.float64)
        for plane, angle in [('xy',xy), ('xz',xz), ('xw',xw), ('yz',yz), ('yw',yw), ('zw',zw)]:
            if angle != 0:
                R = R @ Rot4D.matrix(plane, angle)
        return R


# ═══════════════════════════════════════════════════════════════════════════════
# DIFFUSION PROTOCOL - SD LATENT BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════

class DiffusionBridge:
    """
    Bridge between 4D raymarcher and Stable Diffusion latent space
    Compatible with: ControlNet, IP-Adapter, latent injection
    """
    
    SD_LATENT_SCALE = 0.18215  # SD 1.5/2.1 VAE scaling
    SDXL_LATENT_SCALE = 0.13025  # SDXL VAE scaling
    
    @staticmethod
    def to_latent_res(arr: np.ndarray, target_h: int = 64, target_w: int = 64) -> np.ndarray:
        """Resize to latent resolution (image/8 for SD)"""
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        resized = img.resize((target_w, target_h), Image.BILINEAR)
        return np.array(resized).astype(np.float32) / 255.0
    
    @staticmethod
    def depth_to_controlnet(depth: np.ndarray, invert: bool = False) -> np.ndarray:
        """
        Format depth for ControlNet depth model
        Output: (H, W, 3) uint8
        """
        d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        if invert:
            d = 1.0 - d
        d_u8 = (d * 255).astype(np.uint8)
        return np.stack([d_u8, d_u8, d_u8], axis=-1)
    
    @staticmethod
    def normal_to_controlnet(normal: np.ndarray) -> np.ndarray:
        """
        Format normals for ControlNet normal model
        Input: (H, W, 4) normalized vectors
        Output: (H, W, 3) uint8 RGB
        """
        # Take xyz, ignore w
        n = normal[..., :3]
        # Map [-1, 1] to [0, 1]
        n_rgb = (n + 1.0) * 0.5
        return (np.clip(n_rgb, 0, 1) * 255).astype(np.uint8)
    
    @staticmethod
    def create_depth_latent(depth: np.ndarray, channels: int = 4) -> np.ndarray:
        """
        Create pseudo-latent from depth for direct injection
        Output: (1, channels, H, W) float32
        """
        d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        h, w = d_norm.shape
        
        latent = np.zeros((1, channels, h, w), dtype=np.float32)
        
        # Channel 0: depth
        latent[0, 0] = d_norm
        # Channel 1: depth gradient x
        latent[0, 1] = np.gradient(d_norm, axis=1)
        # Channel 2: depth gradient y
        latent[0, 2] = np.gradient(d_norm, axis=0)
        # Channel 3: laplacian (curvature)
        latent[0, 3] = np.gradient(latent[0, 1], axis=1) + np.gradient(latent[0, 2], axis=0)
        
        return latent * DiffusionBridge.SD_LATENT_SCALE
    
    @staticmethod
    def noise_schedule_cosine(t: float, s: float = 0.008) -> float:
        """Cosine schedule (improved DDPM)"""
        f_t = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        f_0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
        return f_t / f_0
    
    @staticmethod
    def noise_schedule_linear(t: float, beta_start: float = 0.0001, beta_end: float = 0.02) -> float:
        """Linear schedule (original DDPM)"""
        beta = beta_start + t * (beta_end - beta_start)
        return 1.0 - beta
    
    @staticmethod
    def add_noise(x: np.ndarray, t: float, schedule: str = 'cosine') -> Tuple[np.ndarray, np.ndarray]:
        """Forward diffusion: add noise at timestep t"""
        if schedule == 'cosine':
            alpha_bar = DiffusionBridge.noise_schedule_cosine(t)
        else:
            alpha_bar = DiffusionBridge.noise_schedule_linear(t)
        
        noise = np.random.randn(*x.shape).astype(np.float32)
        noisy = np.sqrt(alpha_bar) * x + np.sqrt(1 - alpha_bar) * noise
        return noisy, noise
    
    @staticmethod
    def scheduler_config() -> Dict:
        """Return config compatible with DPMSolverMultistepScheduler"""
        return {
            'num_train_timesteps': 1000,
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': 'scaled_linear',
            'prediction_type': 'epsilon',
            'thresholding': False,
            'algorithm_type': 'dpmsolver++',
            'solver_type': 'midpoint',
            'use_karras_sigmas': True,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZED CASTER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CastResult:
    hit: np.ndarray
    position: np.ndarray
    distance: np.ndarray
    normal: np.ndarray
    density: np.ndarray
    w_coord: np.ndarray
    # Diffusion-ready outputs
    depth_controlnet: np.ndarray = None
    normal_controlnet: np.ndarray = None
    depth_latent: np.ndarray = None


class OptimizedCaster4D:
    def __init__(self, res: int = 512, use_fast_sdf: bool = True):
        self.res = res
        self.max_steps = 96
        self.max_dist = 16.0
        self.epsilon_base = 0.001
        self.vol_steps = 32
        self.use_fast_sdf = use_fast_sdf
        
        # Pre-normalize rays (1 sqrt at init)
        self._init_rays()
    
    def _init_rays(self, fov: float = 1.1):
        u = np.linspace(-fov, fov, self.res)
        v = np.linspace(-fov, fov, self.res)
        uu, vv = np.meshgrid(u, v)
        
        dirs = np.zeros((self.res, self.res, 4), dtype=np.float64)
        dirs[..., 0] = uu
        dirs[..., 1] = -vv
        dirs[..., 2] = 1.5
        
        self._dirs = FastMath.normalize(dirs)
    
    def _create_rays(self, w_slice: float, cam_dist: float = 4.5) -> Tuple[np.ndarray, np.ndarray]:
        origins = np.zeros((self.res, self.res, 4), dtype=np.float64)
        origins[..., 2] = -cam_dist
        origins[..., 3] = w_slice
        return origins, self._dirs
    
    def _adaptive_epsilon(self, t: np.ndarray, step: int) -> np.ndarray:
        dist_scale = 1.0 + t * 0.04
        step_scale = Curves.smoothstep(0, self.max_steps, np.full_like(t, step))
        return self.epsilon_base * dist_scale * (1.0 + step_scale * 0.5)
    
    def _transform(self, p: np.ndarray, R_inv: np.ndarray) -> np.ndarray:
        shape = p.shape
        flat = p.reshape(-1, 4)
        return (R_inv @ flat.T).T.reshape(shape)
    
    def _compute_normal(self, sdf: Callable, p: np.ndarray, R_inv: np.ndarray, eps: float = 0.001) -> np.ndarray:
        n = np.zeros_like(p)
        for i in range(4):
            pp, pn = p.copy(), p.copy()
            pp[...,i] += eps
            pn[...,i] -= eps
            n[...,i] = sdf(self._transform(pp, R_inv)) - sdf(self._transform(pn, R_inv))
        return FastMath.normalize(n)
    
    def cast(self, sdf: Callable, w_slice: float = 0.0, R: np.ndarray = None) -> CastResult:
        R_inv = np.linalg.inv(R) if R is not None else np.eye(4)
        origins, dirs = self._create_rays(w_slice)
        
        t = np.zeros((self.res, self.res), dtype=np.float64)
        active = np.ones((self.res, self.res), dtype=bool)
        hit = np.zeros((self.res, self.res), dtype=bool)
        hit_pos = np.zeros((self.res, self.res, 4), dtype=np.float64)
        
        for step in range(self.max_steps):
            if not active.any():
                break
            
            p = origins + t[..., np.newaxis] * dirs
            d = sdf(self._transform(p, R_inv))
            
            eps = self._adaptive_epsilon(t, step)
            
            new_hits = active & (d < eps)
            hit |= new_hits
            hit_pos[new_hits] = p[new_hits]
            
            missed = active & (t > self.max_dist)
            active &= ~new_hits & ~missed
            
            relax = Curves.bezier3(step / self.max_steps, 0.8, 0.88, 0.95, 1.0)
            t[active] += d[active] * relax
        
        # Normals
        normals = np.zeros((self.res, self.res, 4), dtype=np.float64)
        if hit.any():
            normals[hit] = self._compute_normal(sdf, hit_pos, R_inv)[hit]
        
        # Volumetric density
        density = np.zeros((self.res, self.res))
        for i in range(self.vol_steps):
            ti = (i / self.vol_steps) * self.max_dist
            p = origins + ti * dirs
            d = sdf(self._transform(p, R_inv))
            density += Curves.exp_decay(np.maximum(d, 0), rate=4.0) / self.vol_steps
        
        # Depth normalization
        depth = (t - t.min()) / (t.max() - t.min() + 1e-8)
        
        # Generate diffusion-ready outputs
        depth_cn = DiffusionBridge.depth_to_controlnet(depth)
        normal_cn = DiffusionBridge.normal_to_controlnet(normals)
        depth_lat = DiffusionBridge.create_depth_latent(
            DiffusionBridge.to_latent_res(depth, 64, 64)
        )
        
        return CastResult(
            hit=hit, position=hit_pos, distance=t, normal=normals,
            density=density, w_coord=hit_pos[..., 3],
            depth_controlnet=depth_cn, normal_controlnet=normal_cn,
            depth_latent=depth_lat
        )


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class Renderer4D:
    def __init__(self, res: int = 512):
        self.res = res
        self.caster = OptimizedCaster4D(res)
        self.light = FastMath.normalize(np.array([0.4, 0.7, -0.5, 0.3]))
    
    def render(self, sdf: Callable, w_slice: float = 0.0, R: np.ndarray = None) -> Tuple[np.ndarray, CastResult]:
        result = self.caster.cast(sdf, w_slice, R)
        img = np.zeros((self.res, self.res, 3))
        
        diffuse = np.clip(np.sum(result.normal * self.light, axis=-1), 0, 1)
        
        view = np.array([0, 0, 1, 0])
        half_v = FastMath.normalize(self.light + view)
        spec = np.clip(np.sum(result.normal * half_v, axis=-1), 0, 1) ** 24
        
        w_norm = Curves.smootherstep(-1.5, 1.5, result.w_coord)
        depth_norm = (result.distance - result.distance.min()) / (result.distance.max() - result.distance.min() + 1e-8)
        
        for y in range(self.res):
            for x in range(self.res):
                if result.hit[y, x]:
                    h = (w_norm[y, x] * 0.6 + 0.55) % 1.0
                    s = 0.65
                    v = 0.12 + 0.75 * diffuse[y, x]
                    
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    r = min(1, r + spec[y, x] * 0.5)
                    g = min(1, g + spec[y, x] * 0.5)
                    b = min(1, b + spec[y, x] * 0.4)
                    
                    fog = Curves.exp_decay(np.array([depth_norm[y, x]]), 0.8)[0]
                    img[y, x] = [r * fog, g * fog, b * fog]
                else:
                    vol = result.density[y, x]
                    img[y, x] = [0.015 + vol * 0.1, 0.008 + vol * 0.05, 0.03 + vol * 0.15]
        
        return img, result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  SQRT-OPTIMIZED 4D + DIFFUSION BRIDGE                         ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    
    renderer = Renderer4D(res=384)
    frames = []
    n_frames = 48
    
    for i in range(n_frames):
        t = i / n_frames
        theta = t * 2 * np.pi
        
        xw = Curves.hermite(t, 0, 2*np.pi, 0.8, 0.8)
        yw = Curves.bezier3(t, 0, np.pi*0.6, np.pi*1.4, 2*np.pi) * 0.55
        zw = Curves.smootherstep(0, 1, t) * np.pi * 0.45
        
        R = Rot4D.from_6angles(xw=xw, yw=yw, zw=zw, xy=theta*0.18)
        w = Curves.sin(np.array([t]), freq=1.8)[0] * 0.75
        morph = (Curves.sin(np.array([t]), freq=1)[0] + 1) / 2
        
        def scene(p):
            # Use optimized SDFs where possible
            d1 = SDF4D.tesseract(p, 0.85) - 0.12  # rounded
            d2 = SDF4D.hypersphere(p, 1.05)
            d3 = SDF4D.gyroid_4d(p, 2.8, 0.12)  # 0 sqrt
            base = d1 * (1 - morph) + d2 * morph
            return Curves.smin(-d3, base, 0.08)
        
        print(f"  ├─ Frame {i+1:02d}/{n_frames} │ w={w:+.3f}")
        
        img, result = renderer.render(scene, w_slice=w, R=R)
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        frames.append(Image.fromarray(img_u8))
        
        # Save conditioning maps for frame 0
        if i == 0:
            Image.fromarray(result.depth_controlnet).save('depth_controlnet.png')
            Image.fromarray(result.normal_controlnet).save('normal_controlnet.png')
            np.save('depth_latent.npy', result.depth_latent)
            print(f"      └─ Saved: depth_controlnet.png, normal_controlnet.png, depth_latent.npy")
    
    frames[0].save('4d_diffusion.gif', save_all=True, append_images=frames[1:], duration=50, loop=0)
    frames[0].save('4d_diffusion_test.png')
    
    print("  └─ ✓ Output: 4d_diffusion.gif")
    print()
    print("  SQRT OPTIMIZATION:")
    sdf_sqrt_counts = [
        ("hypersphere", 1), ("hypersphere_approx", 0),
        ("tesseract", 1), ("tesseract_approx", 0),
        ("hyperoctahedron", 0), ("gyroid_4d", 0),
        ("duocylinder", 2), ("duocylinder_fast", 0),
        ("hypertorus", 3),
    ]
    for name, count in sdf_sqrt_counts:
        status = "✓" if count == 0 else f"{count}"
        print(f"    ├─ {name:22s} │ sqrt: {status}")
    
    print()
    print("  DIFFUSION BRIDGE:")
    print("    ├─ depth_to_controlnet()  → (H,W,3) uint8")
    print("    ├─ normal_to_controlnet() → (H,W,3) uint8")
    print("    ├─ create_depth_latent()  → (1,4,H,W) float32")
    print("    ├─ noise_schedule_cosine  → improved DDPM")
    print("    └─ scheduler_config()     → DPMSolver++ compatible")


if __name__ == "__main__":
    main()
