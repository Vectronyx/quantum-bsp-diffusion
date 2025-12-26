#!/usr/bin/env python3
"""
ULTIMATE SDF + QUANTUM + LATENT MARCHER
A + B + C + D + E combined
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
from pathlib import Path
import time as time_module

# Optional imports
try:
    import imageio
    HAS_IMAGEIO = True
except:
    HAS_IMAGEIO = False
    print("pip install imageio[ffmpeg] for MP4 export")

try:
    from diffusers import StableDiffusionImg2ImgPipeline
    HAS_SD = True
except:
    HAS_SD = False
    print("pip install diffusers for SD texture")

EPSILON = 1e-4
MAX_STEPS = 128
MAX_DIST = 100.0

# ============================================
# SDF PRIMITIVES
# ============================================

def sdf_sphere(p, r=1.0):
    return torch.norm(p, dim=-1) - r

def sdf_torus(p, R=1.0, r=0.25):
    q = torch.stack([torch.norm(p[..., :2], dim=-1) - R, p[..., 2]], dim=-1)
    return torch.norm(q, dim=-1) - r

def sdf_box(p, b):
    q = torch.abs(p) - b
    return torch.norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0.0)

def sdf_octahedron(p, s=1.0):
    p = torch.abs(p)
    return (p[..., 0] + p[..., 1] + p[..., 2] - s) * 0.57735

def sdf_gyroid(p, scale=1.0, thickness=0.03):
    p = p * scale
    d = torch.sin(p[..., 0]) * torch.cos(p[..., 1]) + \
        torch.sin(p[..., 1]) * torch.cos(p[..., 2]) + \
        torch.sin(p[..., 2]) * torch.cos(p[..., 0])
    return (torch.abs(d) - thickness) / scale

# ============================================
# SMOOTH OPERATIONS
# ============================================

def op_smooth_union(d1, d2, k=0.1):
    h = torch.clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return torch.lerp(d2, d1, h) - k * h * (1.0 - h)

def op_smooth_subtract(d1, d2, k=0.1):
    h = torch.clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0)
    return torch.lerp(d1, -d2, h) + k * h * (1.0 - h)

# ============================================
# DOMAIN WARPS
# ============================================

def op_twist(p, k):
    c, s = torch.cos(k * p[..., 1]), torch.sin(k * p[..., 1])
    return torch.stack([c*p[...,0] - s*p[...,2], p[...,1], s*p[...,0] + c*p[...,2]], dim=-1)

def op_bend(p, k):
    c, s = torch.cos(k * p[..., 0]), torch.sin(k * p[..., 0])
    return torch.stack([c*p[...,0] - s*p[...,1], s*p[...,0] + c*p[...,1], p[...,2]], dim=-1)

def op_repeat(p, s):
    return torch.remainder(p + s*0.5, s) - s*0.5

def op_displace_sin(p, amp=0.2, freq=4.0):
    return amp * (torch.sin(freq*p[...,0]) * torch.sin(freq*p[...,1]) * torch.sin(freq*p[...,2]))

# ============================================
# C: QUANTUM INTERFERENCE
# ============================================

def quantum_wave(p, centers, phases, freq=2.0):
    """Multiple wave sources creating interference"""
    wave = torch.zeros(p.shape[0], device=p.device)
    for i, (center, phase) in enumerate(zip(centers, phases)):
        dist = torch.norm(p - center, dim=-1)
        wave += torch.sin(dist * freq - phase) / (1.0 + dist * 0.5)
    return wave / len(centers)

def quantum_probability_field(p, t, num_sources=5):
    """Quantum probability density visualization"""
    device = p.device
    
    # Orbiting wave sources
    centers = []
    phases = []
    for i in range(num_sources):
        angle = t * (0.5 + i * 0.1) + i * 2 * math.pi / num_sources
        r = 1.5 + math.sin(t * 0.3 + i) * 0.5
        centers.append(torch.tensor([
            math.cos(angle) * r,
            math.sin(t * 0.7 + i) * 0.5,
            math.sin(angle) * r
        ], device=device))
        phases.append(t * 2.0 + i * 0.5)
    
    wave = quantum_wave(p, centers, phases, freq=3.0)
    
    # Convert to probability (|ψ|²)
    prob = wave ** 2
    
    # Create SDF from probability field
    threshold = 0.1 + 0.05 * math.sin(t)
    return prob - threshold

# ============================================
# UTILITIES
# ============================================

def smoothstep(a, b, x):
    t = torch.clamp((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def calc_normal(sdf, p):
    e = torch.tensor([EPSILON, 0., 0.], device=p.device)
    return F.normalize(torch.stack([
        sdf(p + e.roll(0)) - sdf(p - e.roll(0)),
        sdf(p + e.roll(1)) - sdf(p - e.roll(1)),
        sdf(p + e.roll(2)) - sdf(p - e.roll(2))
    ], dim=-1), dim=-1)

def calc_ao(sdf, p, n, steps=5):
    ao = torch.zeros(p.shape[0], device=p.device)
    for i in range(steps):
        d = 0.05 * (i + 1)
        ao += (d - sdf(p + n * d)) / (2 ** i)
    return torch.clamp(1.0 - ao * 2.0, 0.0, 1.0)

def ray_march(ro, rd, sdf, max_steps=MAX_STEPS):
    t = torch.zeros(ro.shape[0], device=ro.device)
    d = torch.zeros_like(t)
    for _ in range(max_steps):
        p = ro + t.unsqueeze(-1) * rd
        d = sdf(p)
        t = t + d * 0.9
        if (d < EPSILON).all() or (t > MAX_DIST).all():
            break
    return t, d < EPSILON, d

# ============================================
# B: NORMAL/DEPTH COLORING
# ============================================

def normal_to_rgb(n):
    """Convert normals to RGB (standard normal map coloring)"""
    return (n + 1.0) * 0.5

def depth_to_color(t, max_d=10.0):
    """Depth-based coloring with gradient"""
    d = torch.clamp(t / max_d, 0.0, 1.0)
    
    # Cool depth gradient (blue near, purple far)
    r = d * 0.5 + 0.2
    g = (1.0 - d) * 0.3
    b = 0.8 - d * 0.3
    
    return torch.stack([r, g, b], dim=-1)

def fresnel(n, rd, power=2.0):
    """Fresnel effect for rim lighting"""
    return torch.pow(1.0 - torch.abs((n * -rd).sum(-1)), power)

# ============================================
# A: LATENT SPACE MARCHER INTEGRATION
# ============================================

class LatentSDFBridge:
    """Bridge between SDF and latent space"""
    
    def __init__(self, latent_dim=4, device='cuda'):
        self.device = device
        self.latent_dim = latent_dim
    
    def sdf_to_latent_coords(self, p, t):
        """
        Map 3D SDF coordinates to latent space coordinates
        This creates structured regions in latent space
        """
        # Use SDF position + time to create latent coordinates
        # Normalize to typical latent space range
        x = torch.tanh(p[..., 0] * 0.5 + math.sin(t) * 0.3)
        y = torch.tanh(p[..., 1] * 0.5 + math.cos(t * 0.7) * 0.3)
        z = torch.tanh(p[..., 2] * 0.5)
        w = torch.tanh(torch.norm(p, dim=-1) * 0.3)
        
        return torch.stack([x, y, z, w], dim=-1)
    
    def latent_influence_field(self, p, t, seed=42):
        """
        Create influence weights from latent space structure
        Use to guide diffusion sampling
        """
        torch.manual_seed(seed)
        
        latent_coords = self.sdf_to_latent_coords(p, t)
        
        # Create attractor points in latent space
        num_attractors = 6
        attractors = torch.randn(num_attractors, self.latent_dim, device=self.device)
        weights = torch.rand(num_attractors, device=self.device)
        
        # Calculate influence from each attractor
        influence = torch.zeros(p.shape[0], device=self.device)
        for i in range(num_attractors):
            dist = torch.norm(latent_coords - attractors[i], dim=-1)
            influence += weights[i] * torch.exp(-dist * 2.0)
        
        return influence / num_attractors

# ============================================
# MAIN SCENE
# ============================================

class QuantumLatentScene:
    def __init__(self, device='cuda', mode='quantum'):
        self.device = device
        self.time = 0.0
        self.mode = mode  # 'sdf', 'quantum', 'latent', 'hybrid'
        self.latent_bridge = LatentSDFBridge(device=device)
        self.seed = 42
    
    def scene_sdf(self, p):
        """Pure SDF scene"""
        t = self.time
        
        # Orbiting metaballs
        spheres = []
        for i in range(4):
            phase = i * math.pi * 0.5
            speed = 1.0 + i * 0.2
            pos = torch.tensor([
                math.sin(t * speed + phase) * 1.8,
                math.cos(t * 0.6 + phase) * 0.6,
                math.cos(t * speed + phase) * 1.8
            ], device=p.device)
            spheres.append(sdf_sphere(p - pos, 0.5 + i * 0.1))
        
        # Twisted torus
        p_twist = op_twist(p, math.sin(t * 0.3) * 1.5)
        torus = sdf_torus(p_twist, 1.0, 0.2)
        
        # Gyroid core
        gyroid = sdf_gyroid(p, scale=3.0, thickness=0.05)
        core = op_smooth_subtract(sdf_sphere(p, 0.8), gyroid, 0.1)
        
        # Blend
        k = 0.4 + 0.15 * math.sin(t * 0.5)
        scene = spheres[0]
        for s in spheres[1:]:
            scene = op_smooth_union(scene, s, k)
        scene = op_smooth_union(scene, torus, k * 0.8)
        scene = op_smooth_union(scene, core, k * 0.6)
        
        # Displacement
        scene = scene + op_displace_sin(p, amp=0.03, freq=8.0)
        
        return scene
    
    def scene_quantum(self, p):
        """Quantum interference field"""
        t = self.time
        
        # Quantum probability SDF
        quantum = quantum_probability_field(p, t, num_sources=7)
        
        # Combine with base geometry
        base = sdf_sphere(p, 2.0)
        
        return op_smooth_union(quantum * 0.5, base, 0.3)
    
    def scene_hybrid(self, p):
        """SDF + Quantum + Latent hybrid"""
        t = self.time
        
        # Base SDF
        sdf = self.scene_sdf(p)
        
        # Quantum modulation
        quantum = quantum_probability_field(p, t, num_sources=5)
        
        # Latent influence
        latent_influence = self.latent_bridge.latent_influence_field(p, t, self.seed)
        
        # Combine: SDF modulated by quantum, weighted by latent
        modulated = sdf + quantum * 0.1 * latent_influence.unsqueeze(-1) if len(quantum.shape) < len(sdf.shape) else sdf + quantum * 0.1
        
        return sdf + quantum * 0.1
    
    def scene(self, p):
        if self.mode == 'sdf':
            return self.scene_sdf(p)
        elif self.mode == 'quantum':
            return self.scene_quantum(p)
        elif self.mode == 'hybrid':
            return self.scene_hybrid(p)
        else:
            return self.scene_sdf(p)
    
    def render(self, w=512, h=512, color_mode='full'):
        """
        color_mode: 'normal', 'depth', 'ao', 'full', 'psychedelic'
        """
        device = self.device
        t = self.time
        
        # Camera orbit
        cam_angle = t * 0.1
        cam_dist = 6.0 + math.sin(t * 0.2) * 0.5
        cam_pos = torch.tensor([
            math.sin(cam_angle) * cam_dist,
            math.sin(t * 0.15) * 1.5 + 1.0,
            math.cos(cam_angle) * cam_dist
        ], device=device)
        
        # Look at origin
        cam_target = torch.tensor([0., 0., 0.], device=device)
        cam_fwd = F.normalize(cam_target - cam_pos, dim=0)
        cam_right = F.normalize(torch.cross(torch.tensor([0., 1., 0.], device=device), cam_fwd), dim=0)
        cam_up = torch.cross(cam_fwd, cam_right)
        
        # Ray setup
        y, x = torch.meshgrid(
            torch.linspace(1, -1, h, device=device),
            torch.linspace(-1, 1, w, device=device), indexing='ij')
        
        ro = cam_pos.expand(h*w, 3)
        rd = F.normalize(
            x.flatten().unsqueeze(-1) * cam_right + 
            y.flatten().unsqueeze(-1) * cam_up + 
            cam_fwd * 1.5, dim=-1)
        
        # March
        dist, hit, min_d = ray_march(ro, rd, self.scene)
        p = ro + dist.unsqueeze(-1) * rd
        n = calc_normal(self.scene, p)
        
        # === B: COLOR MODES ===
        
        if color_mode == 'normal':
            rgb = normal_to_rgb(n)
            rgb = rgb * hit.float().unsqueeze(-1)
        
        elif color_mode == 'depth':
            rgb = depth_to_color(dist)
            rgb = rgb * hit.float().unsqueeze(-1)
        
        elif color_mode == 'ao':
            ao = calc_ao(self.scene, p, n)
            rgb = ao.unsqueeze(-1).expand(-1, 3) * hit.float().unsqueeze(-1)
        
        elif color_mode == 'psychedelic':
            # Normals as hue base
            hue = (torch.atan2(n[..., 0], n[..., 2]) / (2 * math.pi) + 0.5)
            hue = hue + t * 0.1  # Animate hue
            
            # Quantum influence on saturation
            quantum = quantum_probability_field(p, t, 3)
            sat = 0.7 + 0.3 * quantum
            
            # Fresnel for brightness
            fres = fresnel(n, rd, 3.0)
            val = 0.5 + 0.5 * fres
            
            # HSV to RGB (simplified)
            h6 = (hue % 1.0) * 6.0
            c = val * sat
            x_c = c * (1.0 - torch.abs(h6 % 2.0 - 1.0))
            m = val - c
            
            # RGB based on hue sector
            r = torch.where(h6 < 1, c, torch.where(h6 < 2, x_c, torch.where(h6 < 4, torch.zeros_like(c), torch.where(h6 < 5, x_c, c)))) + m
            g = torch.where(h6 < 1, x_c, torch.where(h6 < 3, c, torch.where(h6 < 4, x_c, torch.zeros_like(c)))) + m
            b = torch.where(h6 < 2, torch.zeros_like(c), torch.where(h6 < 3, x_c, torch.where(h6 < 5, c, x_c))) + m
            
            rgb = torch.stack([r, g, b], dim=-1) * hit.float().unsqueeze(-1)
        
        else:  # 'full'
            # Multi-light setup
            lights = [
                (torch.tensor([1., 1., -1.], device=device), torch.tensor([1.0, 0.9, 0.8], device=device)),
                (torch.tensor([-1., 0.5, -0.5], device=device), torch.tensor([0.3, 0.4, 0.8], device=device)),
                (torch.tensor([0., -1., 0.], device=device), torch.tensor([0.2, 0.1, 0.1], device=device)),
            ]
            
            diffuse = torch.zeros(h*w, 3, device=device)
            for light_dir, light_col in lights:
                l = F.normalize(light_dir, dim=0)
                ndotl = torch.clamp((n * l).sum(-1), 0, 1)
                diffuse += ndotl.unsqueeze(-1) * light_col
            
            # Specular
            view = -rd
            half_vec = F.normalize(F.normalize(lights[0][0], dim=0) + view, dim=-1)
            spec = torch.pow(torch.clamp((n * half_vec).sum(-1), 0, 1), 32.0)
            
            # Fresnel rim
            fres = fresnel(n, rd, 2.5)
            rim = fres.unsqueeze(-1) * torch.tensor([0.2, 0.3, 0.5], device=device)
            
            # AO
            ao = calc_ao(self.scene, p, n)
            
            # Combine
            rgb = (diffuse * 0.7 + spec.unsqueeze(-1) * 0.3 + rim) * ao.unsqueeze(-1)
            rgb = rgb * hit.float().unsqueeze(-1)
            
            # Glow for misses
            glow_intensity = 0.15 / (1.0 + min_d * 0.5)
            glow = glow_intensity.unsqueeze(-1) * torch.tensor([0.1, 0.2, 0.4], device=device)
            rgb = rgb + glow * (~hit).float().unsqueeze(-1)
        
        # Background gradient
        bg_t = (y.flatten() + 1.0) * 0.5
        bg = torch.stack([
            0.02 + bg_t * 0.05,
            0.02 + bg_t * 0.08,
            0.05 + bg_t * 0.15
        ], dim=-1)
        
        rgb = torch.where(hit.unsqueeze(-1), rgb, bg)
        
        return rgb.reshape(h, w, 3)

# ============================================
# E: STABLE DIFFUSION TEXTURE
# ============================================

class SDTextureEnhancer:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        if not HAS_SD:
            raise ImportError("Install diffusers: pip install diffusers transformers accelerate")
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to("cuda")
        self.pipe.enable_attention_slicing()
    
    def enhance(self, img_np, prompt, strength=0.5, seed=42):
        """Apply SD img2img to SDF render"""
        generator = torch.Generator("cuda").manual_seed(seed)
        
        img_pil = Image.fromarray(img_np)
        
        result = self.pipe(
            prompt=prompt,
            image=img_pil,
            strength=strength,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        
        return np.array(result)

# ============================================
# D: VIDEO EXPORT
# ============================================

def export_video(frames, output_path, fps=30):
    """Export frames as MP4"""
    if not HAS_IMAGEIO:
        print("Saving as GIF instead (install imageio[ffmpeg] for MP4)")
        frames[0].save(output_path.replace('.mp4', '.gif'), 
                       save_all=True, append_images=frames[1:], 
                       duration=int(1000/fps), loop=0)
        return
    
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()
    print(f"Saved: {output_path}")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create output dir
    Path("output").mkdir(exist_ok=True)
    
    # Initialize scene
    scene = QuantumLatentScene(device=device, mode='hybrid')
    
    # === Render all color modes ===
    print("\n=== Rendering color modes ===")
    modes = ['normal', 'depth', 'ao', 'full', 'psychedelic']
    
    for mode in modes:
        scene.time = 1.5
        img = scene.render(512, 512, color_mode=mode)
        img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_np, 'RGB').save(f'output/sdf_{mode}.png')
        print(f"  Saved: output/sdf_{mode}.png")
    
    # === Render all scene modes ===
    print("\n=== Rendering scene modes ===")
    for smode in ['sdf', 'quantum', 'hybrid']:
        scene.mode = smode
        scene.time = 2.0
        img = scene.render(512, 512, color_mode='full')
        img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_np, 'RGB').save(f'output/scene_{smode}.png')
        print(f"  Saved: output/scene_{smode}.png")
    
    # === D: Video Export ===
    print("\n=== Rendering video (180 frames) ===")
    scene.mode = 'hybrid'
    frames = []
    for i in range(180):
        scene.time = i * 0.05
        
        # Alternate color modes for variety
        if i < 60:
            cmode = 'full'
        elif i < 120:
            cmode = 'psychedelic'
        else:
            cmode = 'full'
        
        img = scene.render(512, 512, color_mode=cmode)
        img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        frames.append(Image.fromarray(img_np, 'RGB'))
        
        if i % 30 == 0:
            print(f"  Frame {i}/180")
    
    export_video(frames, 'output/sdf_quantum_latent.mp4', fps=30)
    
    # === E: SD Texture Enhancement (optional) ===
    if HAS_SD:
        print("\n=== SD Texture Enhancement ===")
        enhancer = SDTextureEnhancer()
        
        scene.time = 2.5
        base_img = scene.render(512, 512, color_mode='full')
        base_np = (base_img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        prompts = [
            "crystalline alien structure, iridescent, sci-fi, 8k",
            "organic bioluminescent creature, deep sea, ethereal",
            "cyberpunk neon architecture, rain, reflections",
        ]
        
        for i, prompt in enumerate(prompts):
            enhanced = enhancer.enhance(base_np, prompt, strength=0.6, seed=42+i)
            Image.fromarray(enhanced).save(f'output/sd_enhanced_{i}.png')
            print(f"  Saved: output/sd_enhanced_{i}.png")
    
    print("\n✅ DONE! Check ./output/ folder")
    print("\nView results:")
    print("  xdg-open output/")
