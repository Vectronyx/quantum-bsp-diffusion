#!/usr/bin/env python3
"""
SDF Water Rendering
Waves, caustics, reflections, refraction
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image

EPSILON = 1e-4
MAX_STEPS = 100
MAX_DIST = 50.0

# ============================================
# WAVE FUNCTIONS
# ============================================

def wave_sin(p, t, freq=1.0, amp=0.1, speed=1.0):
    """Simple sine wave"""
    return amp * torch.sin(p[..., 0] * freq + t * speed)

def wave_gerstner(p, t, dir_x=1.0, dir_z=0.0, freq=1.0, amp=0.1, speed=1.0, steepness=0.5):
    """Gerstner wave - more realistic ocean waves"""
    d = torch.tensor([dir_x, dir_z], device=p.device)
    d = d / torch.norm(d)
    
    dot = p[..., 0] * d[0] + p[..., 2] * d[1]
    phase = dot * freq - t * speed
    
    # Height
    h = amp * torch.sin(phase)
    
    return h

def wave_fbm(p, t, octaves=4, lacunarity=2.0, gain=0.5):
    """Fractal Brownian Motion waves - natural looking"""
    h = torch.zeros(p.shape[0], device=p.device)
    amp = 0.3
    freq = 0.5
    
    for i in range(octaves):
        # Multiple wave directions
        h += amp * torch.sin(p[..., 0] * freq + t * (1.0 + i * 0.2))
        h += amp * 0.7 * torch.sin(p[..., 2] * freq * 1.3 + t * (0.8 + i * 0.15))
        h += amp * 0.5 * torch.sin((p[..., 0] + p[..., 2]) * freq * 0.7 + t * 1.2)
        
        freq *= lacunarity
        amp *= gain
    
    return h

def wave_combined(p, t):
    """Combine multiple wave types"""
    h = torch.zeros(p.shape[0], device=p.device)
    
    # Large swells
    h += wave_gerstner(p, t, 1.0, 0.3, freq=0.3, amp=0.4, speed=0.8, steepness=0.6)
    h += wave_gerstner(p, t, 0.7, 0.7, freq=0.4, amp=0.3, speed=1.0, steepness=0.5)
    
    # Medium waves
    h += wave_gerstner(p, t, -0.5, 0.8, freq=0.8, amp=0.15, speed=1.3, steepness=0.4)
    h += wave_gerstner(p, t, 0.9, -0.4, freq=1.0, amp=0.1, speed=1.5, steepness=0.3)
    
    # Small ripples
    h += 0.05 * torch.sin(p[..., 0] * 3.0 + p[..., 2] * 2.5 + t * 2.0)
    h += 0.03 * torch.sin(p[..., 0] * 5.0 - p[..., 2] * 4.0 + t * 3.0)
    h += 0.02 * torch.sin(p[..., 0] * 8.0 + p[..., 2] * 7.0 + t * 4.0)
    
    return h

# ============================================
# SDF PRIMITIVES
# ============================================

def sdf_water(p, t, wave_func=wave_combined):
    """Water surface as SDF"""
    water_level = 0.0
    wave_height = wave_func(p, t)
    return p[..., 1] - (water_level + wave_height)

def sdf_sphere(p, r=1.0):
    return torch.norm(p, dim=-1) - r

def sdf_box(p, b):
    q = torch.abs(p) - b
    return torch.norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0.0)

def sdf_floor(p, h=-3.0):
    return p[..., 1] - h

def op_smooth_union(d1, d2, k=0.2):
    h = torch.clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return torch.lerp(d2, d1, h) - k * h * (1.0 - h)

# ============================================
# SCENE
# ============================================

class WaterScene:
    def __init__(self, device='cuda'):
        self.device = device
        self.time = 0.0
    
    def scene_solid(self, p):
        """Non-water geometry for reflections"""
        # Ocean floor
        floor = sdf_floor(p, -4.0)
        
        # Rocks
        rock1 = sdf_sphere(p - torch.tensor([3., -0.5, 2.], device=p.device), 1.2)
        rock2 = sdf_sphere(p - torch.tensor([-4., -1., 3.], device=p.device), 1.8)
        rock3 = sdf_sphere(p - torch.tensor([0., -0.8, -5.], device=p.device), 1.5)
        
        # Floating box
        bob = math.sin(self.time * 0.8) * 0.2
        box = sdf_box(p - torch.tensor([-2., bob + 0.3, 0.], device=p.device), 
                      torch.tensor([0.5, 0.3, 0.5], device=p.device))
        
        scene = floor
        scene = op_smooth_union(scene, rock1, 0.3)
        scene = op_smooth_union(scene, rock2, 0.3)
        scene = op_smooth_union(scene, rock3, 0.3)
        scene = op_smooth_union(scene, box, 0.1)
        
        return scene
    
    def scene_water(self, p):
        """Water surface"""
        return sdf_water(p, self.time)
    
    def scene_combined(self, p):
        """Everything"""
        solid = self.scene_solid(p)
        water = self.scene_water(p)
        return torch.min(solid, water), solid, water

# ============================================
# RENDERING
# ============================================

def calc_normal(sdf, p, eps=EPSILON):
    e = torch.tensor([eps, 0., 0.], device=p.device)
    return F.normalize(torch.stack([
        sdf(p + e.roll(0)) - sdf(p - e.roll(0)),
        sdf(p + e.roll(1)) - sdf(p - e.roll(1)),
        sdf(p + e.roll(2)) - sdf(p - e.roll(2))
    ], dim=-1), dim=-1)

def ray_march(ro, rd, sdf, max_steps=MAX_STEPS):
    t = torch.zeros(ro.shape[0], device=ro.device)
    for _ in range(max_steps):
        d = sdf(ro + t.unsqueeze(-1) * rd)
        t = t + d * 0.7  # Conservative step for waves
        if (d < EPSILON).all() or (t > MAX_DIST).all():
            break
    return t, d < EPSILON

def fresnel_schlick(cos_theta, f0=0.02):
    """Fresnel reflectance"""
    return f0 + (1.0 - f0) * torch.pow(1.0 - cos_theta, 5.0)

def reflect(rd, n):
    """Reflect direction"""
    return rd - 2.0 * (rd * n).sum(-1, keepdim=True) * n

def refract(rd, n, eta=0.75):
    """Refract direction (eta = n1/n2, water ~1.33, so air->water = 1/1.33)"""
    cos_i = -(rd * n).sum(-1, keepdim=True)
    sin_t2 = eta * eta * (1.0 - cos_i * cos_i)
    
    # Total internal reflection check
    valid = sin_t2 < 1.0
    
    cos_t = torch.sqrt(torch.clamp(1.0 - sin_t2, min=0.0))
    refracted = eta * rd + (eta * cos_i - cos_t) * n
    
    return F.normalize(refracted, dim=-1), valid.squeeze(-1)

def render_water(scene, width=640, height=480):
    device = scene.device
    t = scene.time
    
    # Camera - orbiting view
    cam_angle = t * 0.1
    cam_dist = 12.0
    cam_height = 3.0 + math.sin(t * 0.2) * 0.5
    
    cam_pos = torch.tensor([
        math.sin(cam_angle) * cam_dist,
        cam_height,
        math.cos(cam_angle) * cam_dist
    ], device=device)
    
    cam_target = torch.tensor([0., -0.5, 0.], device=device)
    cam_fwd = F.normalize(cam_target - cam_pos, dim=0)
    cam_right = F.normalize(torch.linalg.cross(torch.tensor([0., 1., 0.], device=device), cam_fwd), dim=0)
    cam_up = torch.linalg.cross(cam_fwd, cam_right)
    
    # Rays
    aspect = width / height
    y, x = torch.meshgrid(
        torch.linspace(1, -1, height, device=device),
        torch.linspace(-aspect, aspect, width, device=device), indexing='ij')
    
    n_pixels = height * width
    ro = cam_pos.unsqueeze(0).expand(n_pixels, 3)
    rd = F.normalize(x.flatten().unsqueeze(-1) * cam_right + 
                     y.flatten().unsqueeze(-1) * cam_up + 
                     cam_fwd * 1.8, dim=-1)
    
    # === FIRST PASS: Find water and solid hits ===
    
    # March to water
    t_water, hit_water = ray_march(ro, rd, scene.scene_water)
    p_water = ro + t_water.unsqueeze(-1) * rd
    n_water = calc_normal(scene.scene_water, p_water)
    
    # March to solid
    t_solid, hit_solid = ray_march(ro, rd, scene.scene_solid)
    p_solid = ro + t_solid.unsqueeze(-1) * rd
    n_solid = calc_normal(scene.scene_solid, p_solid)
    
    # === WATER SHADING ===
    
    # View direction
    view = -rd
    
    # Fresnel
    cos_theta = torch.clamp((n_water * view).sum(-1), 0.0, 1.0)
    fres = fresnel_schlick(cos_theta, f0=0.02)
    
    # Reflection direction
    refl_dir = reflect(rd, n_water)
    
    # Refraction direction
    refr_dir, refr_valid = refract(rd, n_water, eta=0.75)
    
    # === REFLECTION COLOR ===
    # March reflected ray to find what it hits
    refl_ro = p_water + n_water * 0.01  # Offset to avoid self-hit
    t_refl, hit_refl = ray_march(refl_ro, refl_dir, scene.scene_solid, max_steps=50)
    p_refl = refl_ro + t_refl.unsqueeze(-1) * refl_dir
    n_refl = calc_normal(scene.scene_solid, p_refl)
    
    # Sky color for missed reflections
    sky_blend = (refl_dir[..., 1] + 1.0) * 0.5
    sky_color = torch.stack([
        0.4 + sky_blend * 0.3,
        0.6 + sky_blend * 0.3,
        0.9 + sky_blend * 0.1
    ], dim=-1)
    
    # Sun reflection (specular highlight)
    sun_dir = F.normalize(torch.tensor([0.5, 0.8, -0.3], device=device), dim=0)
    sun_spec = torch.pow(torch.clamp((refl_dir * sun_dir).sum(-1), 0.0, 1.0), 128.0)
    sun_color = torch.tensor([1.0, 0.95, 0.8], device=device)
    
    # Reflected solid color
    light = F.normalize(torch.tensor([0.5, 1., -0.3], device=device), dim=0)
    refl_diff = torch.clamp((n_refl * light).sum(-1), 0.1, 1.0)
    refl_solid_color = refl_diff.unsqueeze(-1) * torch.tensor([0.5, 0.4, 0.35], device=device)
    
    # Combine reflection
    refl_color = torch.where(hit_refl.unsqueeze(-1), refl_solid_color, sky_color)
    refl_color = refl_color + sun_spec.unsqueeze(-1) * sun_color * 2.0
    
    # === REFRACTION / UNDERWATER COLOR ===
    # March refracted ray underwater
    refr_ro = p_water - n_water * 0.01  # Go slightly under surface
    t_refr, hit_refr = ray_march(refr_ro, refr_dir, scene.scene_solid, max_steps=50)
    
    # Underwater depth absorption
    underwater_depth = torch.clamp(t_refr * 0.3, 0.0, 5.0)
    absorption = torch.exp(-underwater_depth.unsqueeze(-1) * torch.tensor([0.2, 0.1, 0.05], device=device))
    
    # Underwater solid color
    p_under = refr_ro + t_refr.unsqueeze(-1) * refr_dir
    n_under = calc_normal(scene.scene_solid, p_under)
    under_diff = torch.clamp((n_under * light).sum(-1), 0.1, 1.0)
    under_color = under_diff.unsqueeze(-1) * torch.tensor([0.6, 0.5, 0.4], device=device) * absorption
    
    # Deep water color for misses
    deep_color = torch.tensor([0.0, 0.1, 0.2], device=device).expand(n_pixels, 3)
    
    refr_color = torch.where(hit_refr.unsqueeze(-1), under_color, deep_color)
    
    # === CAUSTICS ===
    # Fake caustics based on wave normal distortion
    caustic_pattern = torch.sin(p_water[..., 0] * 3.0 + t * 2.0) * torch.sin(p_water[..., 2] * 3.5 + t * 1.5)
    caustic_pattern = caustic_pattern * torch.sin(p_water[..., 0] * 7.0 - t * 3.0)
    caustics = torch.clamp(caustic_pattern * 0.5 + 0.5, 0.0, 1.0)
    caustics = caustics * 0.3 * torch.exp(-t_water * 0.1)  # Fade with depth
    
    refr_color = refr_color + caustics.unsqueeze(-1) * torch.tensor([0.2, 0.3, 0.4], device=device)
    
    # === COMBINE WATER ===
    water_color = fres.unsqueeze(-1) * refl_color + (1.0 - fres.unsqueeze(-1)) * refr_color
    
    # Water tint
    water_tint = torch.tensor([0.1, 0.3, 0.4], device=device)
    water_color = water_color * 0.85 + water_tint * 0.15
    
    # === SOLID (above water) SHADING ===
    solid_diff = torch.clamp((n_solid * light).sum(-1), 0.1, 1.0)
    solid_color = solid_diff.unsqueeze(-1) * torch.tensor([0.6, 0.5, 0.4], device=device)
    
    # === FINAL COMPOSITE ===
    # Water in front of solid?
    water_in_front = (t_water < t_solid) & hit_water
    solid_visible = hit_solid & ~water_in_front
    
    # Sky
    sky_y = (y.flatten() + 1.0) * 0.5
    sky_bg = torch.stack([
        0.5 + sky_y * 0.2,
        0.7 + sky_y * 0.2,
        0.9 + sky_y * 0.1
    ], dim=-1)
    
    # Composite
    rgb = sky_bg
    rgb = torch.where(solid_visible.unsqueeze(-1), solid_color, rgb)
    rgb = torch.where(water_in_front.unsqueeze(-1), water_color, rgb)
    
    # Fog
    min_t = torch.min(t_water, t_solid)
    fog = torch.exp(-min_t * 0.02)
    fog_color = torch.tensor([0.6, 0.7, 0.8], device=device)
    rgb = rgb * fog.unsqueeze(-1) + fog_color * (1.0 - fog.unsqueeze(-1))
    
    return rgb.reshape(height, width, 3)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    scene = WaterScene(device=device)
    
    # Single frame
    print("Rendering water frame...")
    scene.time = 2.0
    img = render_water(scene, 640, 480)
    img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np, 'RGB').save('water_test.png')
    print("Saved: water_test.png")
    
    # Animation
    print("\nRendering water animation (120 frames)...")
    frames = []
    for i in range(120):
        scene.time = i * 0.1
        img = render_water(scene, 512, 384)
        img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        frames.append(Image.fromarray(img_np, 'RGB'))
        if i % 20 == 0:
            print(f"  Frame {i}/120")
    
    frames[0].save('water_anim.gif', save_all=True, append_images=frames[1:], duration=50, loop=0)
    print("Saved: water_anim.gif")
    
    # MP4 if available
    try:
        import imageio
        print("\nExporting MP4...")
        writer = imageio.get_writer('water.mp4', fps=24, codec='libx264', quality=8)
        for f in frames:
            writer.append_data(np.array(f))
        writer.close()
        print("Saved: water.mp4")
    except:
        print("(Install imageio[ffmpeg] for MP4 export)")
    
    print("\n✅ Done!")
    print("xdg-open water_test.png")
    print("xdg-open water_anim.gif")
