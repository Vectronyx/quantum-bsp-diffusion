#!/usr/bin/env python3
"""
SD 4D PIPELINE - End-to-end 4D raymarching → Stable Diffusion
Connects sqrt-optimized caster to ControlNet/latent injection
"""

import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import colorsys

# Import from our sqrt-optimized caster
from sqrt_diffusion_unified import (
    FastMath, Curves, SDF4D, Rot4D, 
    DiffusionBridge, OptimizedCaster4D, CastResult
)


# ═══════════════════════════════════════════════════════════════════════════════
# TORCH-ACCELERATED 4D CASTER
# ═══════════════════════════════════════════════════════════════════════════════

class TorchCaster4D:
    """GPU-accelerated 4D raymarcher"""
    
    def __init__(self, res: int = 512, device: str = 'cuda'):
        self.res = res
        self.device = device
        self.max_steps = 80
        self.max_dist = 16.0
        self.epsilon_base = 0.001
        self.vol_steps = 24
        
        self._init_rays()
    
    def _init_rays(self, fov: float = 1.1):
        u = torch.linspace(-fov, fov, self.res, device=self.device)
        v = torch.linspace(-fov, fov, self.res, device=self.device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        dirs = torch.zeros(self.res, self.res, 4, device=self.device)
        dirs[..., 0] = uu
        dirs[..., 1] = -vv
        dirs[..., 2] = 1.5
        
        self._dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    
    def cast(self, sdf_fn: Callable, w_slice: float = 0.0, 
             R: torch.Tensor = None) -> dict:
        
        R_inv = torch.inverse(R) if R is not None else torch.eye(4, device=self.device)
        
        origins = torch.zeros(self.res, self.res, 4, device=self.device)
        origins[..., 2] = -4.5
        origins[..., 3] = w_slice
        
        t = torch.zeros(self.res, self.res, device=self.device)
        hit = torch.zeros(self.res, self.res, dtype=torch.bool, device=self.device)
        hit_pos = torch.zeros(self.res, self.res, 4, device=self.device)
        
        for step in range(self.max_steps):
            active = ~hit & (t < self.max_dist)
            if not active.any():
                break
            
            p = origins + t.unsqueeze(-1) * self._dirs
            p_rot = torch.einsum('ij,...j->...i', R_inv, p)
            d = sdf_fn(p_rot)
            
            # Adaptive epsilon
            eps = self.epsilon_base * (1.0 + t * 0.04)
            
            new_hits = active & (d < eps)
            hit |= new_hits
            hit_pos[new_hits] = p[new_hits]
            
            # Bezier relaxation
            relax = 0.8 + 0.2 * (step / self.max_steps)
            t = torch.where(active, t + d * relax, t)
        
        # Normals via gradient
        normals = self._compute_normals(sdf_fn, hit_pos, R_inv)
        
        # Depth normalize
        depth = (t - t.min()) / (t.max() - t.min() + 1e-8)
        
        return {
            'hit': hit,
            'position': hit_pos,
            'distance': t,
            'depth': depth,
            'normal': normals,
            'w_coord': hit_pos[..., 3]
        }
    
    def _compute_normals(self, sdf_fn: Callable, p: torch.Tensor, 
                         R_inv: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
        n = torch.zeros_like(p)
        for i in range(4):
            pp, pn = p.clone(), p.clone()
            pp[..., i] += eps
            pn[..., i] -= eps
            pp_rot = torch.einsum('ij,...j->...i', R_inv, pp)
            pn_rot = torch.einsum('ij,...j->...i', R_inv, pn)
            n[..., i] = sdf_fn(pp_rot) - sdf_fn(pn_rot)
        return n / (torch.norm(n, dim=-1, keepdim=True) + 1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# TORCH 4D SDFs (SQRT-OPTIMIZED)
# ═══════════════════════════════════════════════════════════════════════════════

class TorchSDF4D:
    """GPU 4D SDFs with sqrt annotations"""
    
    @staticmethod
    def hypersphere(p: torch.Tensor, r: float = 1.0) -> torch.Tensor:
        """[1 sqrt]"""
        return torch.sqrt((p ** 2).sum(dim=-1) + 1e-8) - r
    
    @staticmethod
    def hypersphere_approx(p: torch.Tensor, r: float = 1.0) -> torch.Tensor:
        """[0 sqrt] - linearized"""
        return ((p ** 2).sum(dim=-1) - r*r) / (2*r)
    
    @staticmethod
    def tesseract(p: torch.Tensor, size: float = 1.0) -> torch.Tensor:
        """[1 sqrt]"""
        q = torch.abs(p) - size
        outside = torch.sqrt((torch.clamp(q, min=0) ** 2).sum(dim=-1) + 1e-8)
        inside = torch.clamp(q.max(dim=-1).values, max=0)
        return outside + inside
    
    @staticmethod
    def tesseract_approx(p: torch.Tensor, size: float = 1.0) -> torch.Tensor:
        """[0 sqrt] - Chebyshev"""
        return (torch.abs(p) - size).max(dim=-1).values
    
    @staticmethod
    def hyperoctahedron(p: torch.Tensor, s: float = 1.0) -> torch.Tensor:
        """[0 sqrt] - L1"""
        return torch.abs(p).sum(dim=-1) - s
    
    @staticmethod
    def gyroid_4d(p: torch.Tensor, scale: float = 1.0, thick: float = 0.1) -> torch.Tensor:
        """[0 sqrt] - pure trig"""
        ps = p * scale
        g = (torch.sin(ps[...,0]*2*np.pi) * torch.cos(ps[...,1]*2*np.pi) +
             torch.sin(ps[...,1]*2*np.pi) * torch.cos(ps[...,2]*2*np.pi) +
             torch.sin(ps[...,2]*2*np.pi) * torch.cos(ps[...,3]*2*np.pi) +
             torch.sin(ps[...,3]*2*np.pi) * torch.cos(ps[...,0]*2*np.pi))
        return torch.abs(g) / scale - thick
    
    @staticmethod
    def duocylinder(p: torch.Tensor, r1: float = 0.8, r2: float = 0.8) -> torch.Tensor:
        """[2 sqrt]"""
        d1 = torch.sqrt(p[...,0]**2 + p[...,1]**2 + 1e-8) - r1
        d2 = torch.sqrt(p[...,2]**2 + p[...,3]**2 + 1e-8) - r2
        return torch.maximum(d1, d2)
    
    @staticmethod
    def hypertorus(p: torch.Tensor, R: float = 1.0, r1: float = 0.4, r2: float = 0.15) -> torch.Tensor:
        """[3 sqrt]"""
        dxy = torch.sqrt(p[...,0]**2 + p[...,1]**2 + 1e-8) - R
        dxyz = torch.sqrt(dxy**2 + p[...,2]**2 + 1e-8) - r1
        return torch.sqrt(dxyz**2 + p[...,3]**2 + 1e-8) - r2
    
    @staticmethod
    def smin(a: torch.Tensor, b: torch.Tensor, k: float = 0.2) -> torch.Tensor:
        h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0, 1)
        return b * (1-h) + a * h - k * h * (1-h)


# ═══════════════════════════════════════════════════════════════════════════════
# SD PIPELINE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class SD4DPipeline:
    """
    Stable Diffusion pipeline with 4D raymarcher conditioning
    Supports: ControlNet depth/normal, latent injection, IP-Adapter
    """
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5",
                 controlnet_id: str = None, device: str = 'cuda'):
        self.device = device
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.pipe = None
        self.caster = TorchCaster4D(res=512, device=device)
    
    def load(self):
        """Load SD pipeline with optional ControlNet"""
        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionControlNetPipeline,
            ControlNetModel,
            DPMSolverMultistepScheduler
        )
        
        if self.controlnet_id:
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_id, torch_dtype=torch.float16
            )
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id, controlnet=controlnet, torch_dtype=torch.float16
            )
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.float16
            )
        
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++"
        )
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()
        
        return self
    
    def generate_conditioning(self, sdf_fn: Callable, w_slice: float = 0.0,
                              R: torch.Tensor = None) -> dict:
        """Generate 4D depth/normal maps for SD conditioning"""
        
        result = self.caster.cast(sdf_fn, w_slice, R)
        
        # Convert to numpy for PIL
        depth_np = result['depth'].cpu().numpy()
        normal_np = result['normal'].cpu().numpy()
        
        # Format for ControlNet
        depth_cn = DiffusionBridge.depth_to_controlnet(depth_np)
        normal_cn = DiffusionBridge.normal_to_controlnet(normal_np)
        
        # Create latent
        depth_64 = DiffusionBridge.to_latent_res(depth_np, 64, 64)
        depth_latent = DiffusionBridge.create_depth_latent(depth_64)
        
        return {
            'depth_image': Image.fromarray(depth_cn),
            'normal_image': Image.fromarray(normal_cn),
            'depth_latent': torch.from_numpy(depth_latent).to(self.device),
            'raw_result': result
        }
    
    def generate(self, prompt: str, sdf_fn: Callable, 
                 w_slice: float = 0.0, R: torch.Tensor = None,
                 steps: int = 20, guidance: float = 7.5,
                 seed: int = -1) -> Image.Image:
        """Generate image with 4D conditioning"""
        
        if self.pipe is None:
            raise RuntimeError("Call load() first")
        
        # Get conditioning
        cond = self.generate_conditioning(sdf_fn, w_slice, R)
        
        # Set seed
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Generate
        if self.controlnet_id:
            image = self.pipe(
                prompt=prompt,
                image=cond['depth_image'],
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            ).images[0]
        else:
            # Latent injection mode
            with torch.no_grad():
                # Encode depth as initial latent bias
                latent = torch.randn(1, 4, 64, 64, device=self.device, dtype=torch.float16)
                depth_bias = cond['depth_latent'].to(torch.float16)
                latent = latent + depth_bias * 0.3
                
                image = self.pipe(
                    prompt=prompt,
                    latents=latent,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator
                ).images[0]
        
        return image


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO: STANDALONE 4D RENDER (NO SD REQUIRED)
# ═══════════════════════════════════════════════════════════════════════════════

def render_4d_standalone(res: int = 384, n_frames: int = 24) -> None:
    """Quick 4D render without SD pipeline"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"╔═══════════════════════════════════════════════════════════════╗")
    print(f"║  4D TORCH CASTER - Device: {device:40s} ║")
    print(f"╚═══════════════════════════════════════════════════════════════╝")
    
    caster = TorchCaster4D(res=res, device=device)
    frames = []
    
    for i in range(n_frames):
        t = i / n_frames
        theta = t * 2 * np.pi
        
        # Rotation
        xw = t * 2 * np.pi
        yw = t * np.pi * 0.6
        zw = t * np.pi * 0.4
        
        R = torch.tensor(Rot4D.from_6angles(xw=xw, yw=yw, zw=zw, xy=theta*0.2),
                        device=device, dtype=torch.float32)
        
        w = np.sin(t * 2 * np.pi * 1.5) * 0.7
        morph = (np.sin(t * 2 * np.pi) + 1) / 2
        
        def scene(p):
            d1 = TorchSDF4D.tesseract(p, 0.85) - 0.1
            d2 = TorchSDF4D.hypersphere(p, 1.05)
            d3 = TorchSDF4D.gyroid_4d(p, 2.8, 0.12)
            base = d1 * (1 - morph) + d2 * morph
            return TorchSDF4D.smin(-d3, base, 0.1)
        
        print(f"  ├─ Frame {i+1:02d}/{n_frames} │ w={w:+.3f}")
        
        result = caster.cast(scene, w_slice=w, R=R)
        
        # Simple render
        depth = result['depth'].cpu().numpy()
        hit = result['hit'].cpu().numpy()
        w_coord = result['w_coord'].cpu().numpy()
        
        img = np.zeros((res, res, 3))
        
        # Vectorized rendering
        w_norm = (w_coord + 1.5) / 3.0
        w_norm = np.clip(w_norm, 0, 1)
        
        for y in range(res):
            for x in range(res):
                if hit[y, x]:
                    h = (w_norm[y, x] * 0.6 + 0.5) % 1.0
                    v = 0.3 + 0.6 * (1 - depth[y, x])
                    r, g, b = colorsys.hsv_to_rgb(h, 0.7, v)
                    img[y, x] = [r, g, b]
                else:
                    img[y, x] = [0.02, 0.01, 0.04]
        
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        frames.append(Image.fromarray(img_u8))
    
    frames[0].save('4d_torch.gif', save_all=True, append_images=frames[1:], duration=50, loop=0)
    frames[0].save('4d_torch_test.png')
    
    print("  └─ ✓ Output: 4d_torch.gif, 4d_torch_test.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--sd':
        # Full SD pipeline
        print("Loading SD + ControlNet pipeline...")
        pipeline = SD4DPipeline(
            model_id="runwayml/stable-diffusion-v1-5",
            controlnet_id="lllyasviel/sd-controlnet-depth"
        ).load()
        
        def scene(p):
            return TorchSDF4D.tesseract(p, 0.9) - 0.1
        
        R = torch.tensor(Rot4D.from_6angles(xw=0.5, yw=0.3), 
                        device='cuda', dtype=torch.float32)
        
        image = pipeline.generate(
            prompt="crystalline 4D hypercube, volumetric lighting, ethereal",
            sdf_fn=scene,
            w_slice=0.0,
            R=R,
            steps=25,
            seed=42
        )
        image.save('4d_sd_output.png')
        print("Saved: 4d_sd_output.png")
    else:
        # Standalone 4D render
        render_4d_standalone(res=384, n_frames=36)
