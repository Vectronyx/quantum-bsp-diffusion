#!/usr/bin/env python3
"""
FIXED: Water Physics - iteration only affects style, not wave motion
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List

EPSILON = 1e-4
MAX_STEPS = 120
MAX_DIST = 60.0

@dataclass
class RigidBody:
    pos: torch.Tensor
    vel: torch.Tensor
    ang_vel: torch.Tensor
    rotation: float
    mass: float
    buoyancy: float
    drag: float
    shape: str
    size: torch.Tensor
    color: torch.Tensor

class PhysicsWorld:
    def __init__(self, device='cuda'):
        self.device = device
        self.gravity = torch.tensor([0., -9.8, 0.], device=device)
        self.water_level = 0.0
        self.bodies: List[RigidBody] = []
        self.time = 0.0
    
    def add_body(self, pos, vel=None, shape='sphere', size=None, mass=1.0, 
                 buoyancy=0.6, drag=0.5, color=None):
        if vel is None:
            vel = torch.zeros(3, device=self.device)
        if size is None:
            size = torch.tensor([0.5], device=self.device)
        if color is None:
            color = torch.rand(3, device=self.device) * 0.5 + 0.3
        
        body = RigidBody(
            pos=torch.tensor(pos, device=self.device, dtype=torch.float32),
            vel=torch.tensor(vel, device=self.device, dtype=torch.float32) if isinstance(vel, list) else vel.clone(),
            ang_vel=torch.zeros(3, device=self.device),
            rotation=0.0,
            mass=mass,
            buoyancy=buoyancy,
            drag=drag,
            shape=shape,
            size=torch.tensor(size, device=self.device, dtype=torch.float32) if isinstance(size, list) else size,
            color=torch.tensor(color, device=self.device, dtype=torch.float32) if isinstance(color, list) else color
        )
        self.bodies.append(body)
        return body
    
    def get_water_height(self, x, z, t):
        h = 0.0
        # CONSISTENT waves - no random seeds
        h += 0.5 * math.sin(x * 0.3 + t * 0.8)
        h += 0.4 * math.sin(z * 0.4 + t * 1.0)
        h += 0.3 * math.sin((x + z) * 0.25 + t * 0.7)
        h += 0.2 * math.sin(x * 0.8 + z * 0.5 + t * 1.5)
        h += 0.15 * math.sin(x * 1.2 - z * 0.8 + t * 1.8)
        h += 0.05 * math.sin(x * 3.0 + t * 3.0)
        h += 0.03 * math.sin(z * 4.0 + t * 4.0)
        return h
    
    def step(self, dt=0.016):
        self.time += dt
        t = self.time
        
        for body in self.bodies:
            water_h = self.get_water_height(body.pos[0].item(), body.pos[2].item(), t)
            submerged = water_h - body.pos[1].item()
            submerged_ratio = max(0.0, min(1.0, submerged / (body.size[0].item() * 2)))
            
            force = self.gravity * body.mass
            
            if submerged > 0:
                buoyancy_force = torch.tensor([0., 9.8 * body.buoyancy * submerged_ratio * body.mass * 2.5, 0.], device=self.device)
                force = force + buoyancy_force
                drag_force = -body.vel * body.drag * submerged_ratio * 5.0
                force = force + drag_force
                wave_dx = 0.3 * math.cos(body.pos[0].item() * 0.3 + t * 0.8)
                wave_dz = 0.3 * math.cos(body.pos[2].item() * 0.4 + t * 1.0)
                wave_force = torch.tensor([wave_dx, 0., wave_dz], device=self.device) * submerged_ratio
                force = force + wave_force
            
            force = force - body.vel * 0.1
            
            acc = force / body.mass
            body.vel = body.vel + acc * dt
            body.pos = body.pos + body.vel * dt
            
            if submerged > 0:
                body.ang_vel[1] = body.ang_vel[1] * 0.95 + wave_dx * 0.5
                body.rotation += body.ang_vel[1].item() * dt
            
            body.pos[0] = torch.clamp(body.pos[0], -15., 15.)
            body.pos[2] = torch.clamp(body.pos[2], -15., 15.)
            body.pos[1] = torch.clamp(body.pos[1], -5., 10.)

# ============================================
# WAVES - CONSISTENT, TIME-BASED ONLY
# ============================================

def wave_consistent(p, t):
    """
    FIXED: Waves depend ONLY on position and time
    No iteration, no random seeds per frame
    """
    h = torch.zeros(p.shape[0], device=p.device)
    
    # Large swells
    h += 0.5 * torch.sin(p[..., 0] * 0.3 + t * 0.8)
    h += 0.4 * torch.sin(p[..., 2] * 0.4 + t * 1.0)
    h += 0.35 * torch.sin((p[..., 0] + p[..., 2]) * 0.25 + t * 0.7)
    h += 0.25 * torch.sin((p[..., 0] - p[..., 2]) * 0.35 + t * 0.9)
    
    # Medium waves
    h += 0.2 * torch.sin(p[..., 0] * 0.8 + p[..., 2] * 0.5 + t * 1.5)
    h += 0.15 * torch.sin(p[..., 0] * 1.2 - p[..., 2] * 0.8 + t * 1.8)
    h += 0.12 * torch.sin(-p[..., 0] * 0.9 + p[..., 2] * 1.1 + t * 1.3)
    
    # Small ripples
    h += 0.06 * torch.sin(p[..., 0] * 3.0 + t * 3.0)
    h += 0.04 * torch.sin(p[..., 2] * 4.0 + t * 4.0)
    h += 0.03 * torch.sin(p[..., 0] * 5.0 + p[..., 2] * 4.5 + t * 5.0)
    h += 0.02 * torch.sin(p[..., 0] * 7.0 - p[..., 2] * 6.0 + t * 6.0)
    
    return h

# ============================================
# SDF
# ============================================

def sdf_sphere(p, r=1.0):
    return torch.norm(p, dim=-1) - r

def sdf_box(p, b):
    q = torch.abs(p) - b
    return torch.norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0.0)

def sdf_capsule(p, h=1.0, r=0.3):
    p_y = torch.clamp(p[..., 1], -h, h)
    return torch.norm(p - torch.stack([torch.zeros_like(p_y), p_y, torch.zeros_like(p_y)], dim=-1), dim=-1) - r

def sdf_floor(p, h=-5.0):
    return p[..., 1] - h

def op_smooth_union(d1, d2, k=0.2):
    h = torch.clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return torch.lerp(d2, d1, h) - k * h * (1.0 - h)

def op_twist(p, k):
    c, s = torch.cos(k * p[..., 1]), torch.sin(k * p[..., 1])
    return torch.stack([c*p[...,0] - s*p[...,2], p[...,1], s*p[...,0] + c*p[...,2]], dim=-1)

def op_rotate_y(p, angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.stack([c * p[..., 0] + s * p[..., 2], p[..., 1], -s * p[..., 0] + c * p[..., 2]], dim=-1)

# ============================================
# SCENE
# ============================================

class WaterScene:
    def __init__(self, device='cuda'):
        self.device = device
        self.time = 0.0
        self.physics = PhysicsWorld(device=device)
        self._spawn_bodies()
    
    def _spawn_bodies(self):
        # Crates
        self.physics.add_body([2., 1., 0.], shape='box', size=[0.6, 0.4, 0.6], mass=1.0, buoyancy=0.7, color=[0.6, 0.4, 0.2])
        self.physics.add_body([-3., 2., 2.], vel=[0.5, 0., -0.3], shape='box', size=[0.4, 0.3, 0.4], mass=0.5, buoyancy=0.8, color=[0.5, 0.35, 0.2])
        
        # Balls
        self.physics.add_body([0., 1.5, -2.], shape='sphere', size=[0.5], mass=0.2, buoyancy=0.95, drag=0.3, color=[1.0, 0.3, 0.3])
        self.physics.add_body([-1., 2., 3.], vel=[0.3, 0., 0.], shape='sphere', size=[0.4], mass=0.15, buoyancy=0.9, drag=0.3, color=[0.3, 1.0, 0.3])
        self.physics.add_body([4., 1., -3.], shape='sphere', size=[0.35], mass=0.1, buoyancy=0.92, drag=0.25, color=[0.3, 0.3, 1.0])
        
        # Heavy sinker
        self.physics.add_body([5., 3., 0.], vel=[0., -1., 0.], shape='sphere', size=[0.7], mass=5.0, buoyancy=0.3, drag=0.8, color=[0.3, 0.3, 0.35])
        
        # Buoy
        self.physics.add_body([-5., 0.5, -1.], shape='capsule', size=[0.8, 0.25], mass=0.8, buoyancy=0.85, color=[1.0, 0.6, 0.1])
    
    def scene_water(self, p):
        h = wave_consistent(p, self.time)
        return p[..., 1] - h
    
    def scene_bodies(self, p):
        if not self.physics.bodies:
            return torch.full((p.shape[0],), 999.0, device=self.device)
        
        scene = torch.full((p.shape[0],), 999.0, device=self.device)
        for body in self.physics.bodies:
            p_local = op_rotate_y(p - body.pos, -body.rotation)
            if body.shape == 'sphere':
                d = sdf_sphere(p_local, body.size[0].item())
            elif body.shape == 'box':
                d = sdf_box(p_local, body.size)
            elif body.shape == 'capsule':
                d = sdf_capsule(p_local, body.size[0].item(), body.size[1].item() if len(body.size) > 1 else 0.25)
            else:
                d = sdf_sphere(p_local, 0.5)
            scene = op_smooth_union(scene, d, 0.05)
        return scene
    
    def scene_environment(self, p):
        floor_h = -4.0 + 0.5 * torch.sin(p[..., 0] * 0.3) * torch.sin(p[..., 2] * 0.4)
        floor = p[..., 1] - floor_h
        rock1 = sdf_sphere(p - torch.tensor([8., -2., 5.], device=p.device), 2.5)
        rock2 = sdf_sphere(p - torch.tensor([-7., -1.5, -6.], device=p.device), 2.0)
        rock3 = sdf_sphere(p - torch.tensor([0., -3., 10.], device=p.device), 3.0)
        p_coral = op_twist(p - torch.tensor([-4., -3., 2.], device=p.device), 0.5)
        coral = sdf_box(p_coral, torch.tensor([0.3, 1.5, 0.3], device=p.device))
        scene = floor
        scene = op_smooth_union(scene, rock1, 0.5)
        scene = op_smooth_union(scene, rock2, 0.5)
        scene = op_smooth_union(scene, rock3, 0.5)
        scene = op_smooth_union(scene, coral, 0.2)
        return scene
    
    def scene_solid(self, p):
        return op_smooth_union(self.scene_environment(p), self.scene_bodies(p), 0.1)
    
    def get_body_color(self, p):
        colors = torch.full((p.shape[0], 3), 0.5, device=self.device)
        min_dist = torch.full((p.shape[0],), 999.0, device=self.device)
        for body in self.physics.bodies:
            p_local = op_rotate_y(p - body.pos, -body.rotation)
            if body.shape == 'sphere':
                d = sdf_sphere(p_local, body.size[0].item())
            elif body.shape == 'box':
                d = sdf_box(p_local, body.size)
            elif body.shape == 'capsule':
                d = sdf_capsule(p_local, body.size[0].item(), body.size[1].item() if len(body.size) > 1 else 0.25)
            else:
                d = sdf_sphere(p_local, 0.5)
            mask = d < min_dist
            min_dist = torch.where(mask, d, min_dist)
            colors = torch.where(mask.unsqueeze(-1), body.color.expand(p.shape[0], 3), colors)
        return colors

# ============================================
# RENDER
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
        t = t + d * 0.6
        if (d < EPSILON).all() or (t > MAX_DIST).all():
            break
    return t, d < EPSILON

def fresnel(cos_theta, f0=0.02):
    return f0 + (1.0 - f0) * torch.pow(torch.clamp(1.0 - cos_theta, 0.0, 1.0), 5.0)

def reflect(rd, n):
    return rd - 2.0 * (rd * n).sum(-1, keepdim=True) * n

def render(scene, width=640, height=480):
    device = scene.device
    t = scene.time
    
    # FIXED: Camera orbit is smooth, time-based
    cam_angle = t * 0.15
    cam_dist = 14.0 + math.sin(t * 0.2) * 2.0
    cam_height = 4.0 + math.sin(t * 0.15) * 1.5
    
    cam_pos = torch.tensor([math.sin(cam_angle) * cam_dist, cam_height, math.cos(cam_angle) * cam_dist], device=device)
    cam_target = torch.tensor([0., -0.5, 0.], device=device)
    cam_fwd = F.normalize(cam_target - cam_pos, dim=0)
    cam_right = F.normalize(torch.linalg.cross(torch.tensor([0., 1., 0.], device=device), cam_fwd), dim=0)
    cam_up = torch.linalg.cross(cam_fwd, cam_right)
    
    aspect = width / height
    y, x = torch.meshgrid(
        torch.linspace(1, -1, height, device=device),
        torch.linspace(-aspect, aspect, width, device=device), indexing='ij')
    
    n_pix = height * width
    ro = cam_pos.unsqueeze(0).expand(n_pix, 3)
    rd = F.normalize(x.flatten().unsqueeze(-1) * cam_right + y.flatten().unsqueeze(-1) * cam_up + cam_fwd * 1.8, dim=-1)
    
    # March
    t_water, hit_water = ray_march(ro, rd, scene.scene_water)
    p_water = ro + t_water.unsqueeze(-1) * rd
    n_water = calc_normal(scene.scene_water, p_water)
    
    t_solid, hit_solid = ray_march(ro, rd, scene.scene_solid)
    p_solid = ro + t_solid.unsqueeze(-1) * rd
    n_solid = calc_normal(scene.scene_solid, p_solid)
    
    # Water fresnel
    view = -rd
    cos_theta = torch.clamp((n_water * view).sum(-1), 0.0, 1.0)
    fres = fresnel(cos_theta)
    
    # Reflection
    refl_dir = reflect(rd, n_water)
    refl_ro = p_water + n_water * 0.02
    t_refl, hit_refl = ray_march(refl_ro, refl_dir, scene.scene_solid, max_steps=60)
    
    # FIXED: Sky color is CONSTANT - no iteration variance!
    sky_blend = (refl_dir[..., 1] + 1.0) * 0.5
    sky_color = torch.stack([
        0.35 + sky_blend * 0.35,
        0.55 + sky_blend * 0.35,
        0.85 + sky_blend * 0.15
    ], dim=-1)
    
    # Sun
    sun_dir = F.normalize(torch.tensor([0.4, 0.7, -0.5], device=device), dim=0)
    sun_spec = torch.pow(torch.clamp((refl_dir * sun_dir).sum(-1), 0.0, 1.0), 96.0)
    sun_color = torch.tensor([1.0, 0.95, 0.85], device=device)
    
    # Reflected solid
    light = F.normalize(torch.tensor([0.4, 0.9, -0.3], device=device), dim=0)
    p_refl = refl_ro + t_refl.unsqueeze(-1) * refl_dir
    n_refl = calc_normal(scene.scene_solid, p_refl)
    refl_diff = torch.clamp((n_refl * light).sum(-1), 0.15, 1.0)
    
    body_colors = scene.get_body_color(p_refl)
    env_color = torch.tensor([0.5, 0.45, 0.4], device=device)
    body_dist = scene.scene_bodies(p_refl)
    env_dist = scene.scene_environment(p_refl)
    is_body = body_dist < env_dist
    refl_base = torch.where(is_body.unsqueeze(-1), body_colors, env_color.expand(n_pix, 3))
    refl_solid_color = refl_diff.unsqueeze(-1) * refl_base
    
    refl_color = torch.where(hit_refl.unsqueeze(-1), refl_solid_color, sky_color)
    refl_color = refl_color + sun_spec.unsqueeze(-1) * sun_color * 3.0
    
    # Refraction / underwater
    water_tint = torch.tensor([0.1, 0.25, 0.35], device=device)
    underwater_depth = torch.clamp(t_water * 0.25, 0.0, 4.0)
    absorption = torch.exp(-underwater_depth.unsqueeze(-1) * torch.tensor([0.25, 0.12, 0.06], device=device))
    
    refr_dir = F.normalize(rd + n_water * 0.2, dim=-1)
    refr_ro = p_water - n_water * 0.02
    t_refr, hit_refr = ray_march(refr_ro, refr_dir, scene.scene_solid, max_steps=50)
    
    p_under = refr_ro + t_refr.unsqueeze(-1) * refr_dir
    n_under = calc_normal(scene.scene_solid, p_under)
    under_diff = torch.clamp((n_under * light).sum(-1), 0.1, 1.0)
    
    body_under = scene.get_body_color(p_under)
    body_dist_u = scene.scene_bodies(p_under)
    env_dist_u = scene.scene_environment(p_under)
    is_body_u = body_dist_u < env_dist_u
    under_base = torch.where(is_body_u.unsqueeze(-1), body_under, env_color.expand(n_pix, 3))
    under_color = under_diff.unsqueeze(-1) * under_base * absorption
    
    deep = water_tint * 0.3
    refr_color = torch.where(hit_refr.unsqueeze(-1), under_color, deep.expand(n_pix, 3))
    
    # Caustics - FIXED: consistent pattern, just animated by time
    c1 = torch.sin(p_water[..., 0] * 2.5 + t * 1.5)
    c2 = torch.sin(p_water[..., 2] * 3.0 + t * 1.2)
    c3 = torch.sin((p_water[..., 0] + p_water[..., 2]) * 2.0 - t * 1.8)
    caustics = torch.clamp((c1 * c2 + c3) * 0.3 + 0.5, 0.0, 1.0)
    caustics = caustics ** 2 * 0.35 * torch.exp(-t_water * 0.15)
    refr_color = refr_color + caustics.unsqueeze(-1) * torch.tensor([0.2, 0.35, 0.45], device=device)
    
    # Foam
    wave_h = -scene.scene_water(p_water)
    foam = torch.clamp((wave_h - 0.4) * 4.0, 0.0, 1.0)
    foam_color = torch.tensor([0.9, 0.95, 1.0], device=device)
    
    # Combine water
    water_color = fres.unsqueeze(-1) * refl_color + (1.0 - fres.unsqueeze(-1)) * refr_color
    water_color = water_color * (1.0 - foam.unsqueeze(-1)) + foam_color * foam.unsqueeze(-1)
    water_color = water_color * 0.9 + water_tint * 0.1
    
    # Solid shading
    solid_diff = torch.clamp((n_solid * light).sum(-1), 0.15, 1.0)
    body_solid = scene.get_body_color(p_solid)
    body_dist_s = scene.scene_bodies(p_solid)
    env_dist_s = scene.scene_environment(p_solid)
    is_body_s = body_dist_s < env_dist_s
    solid_base = torch.where(is_body_s.unsqueeze(-1), body_solid, env_color.expand(n_pix, 3))
    solid_color = solid_diff.unsqueeze(-1) * solid_base
    
    half_vec = F.normalize(light + view, dim=-1)
    solid_spec = torch.pow(torch.clamp((n_solid * half_vec).sum(-1), 0.0, 1.0), 32.0)
    solid_color = solid_color + solid_spec.unsqueeze(-1) * 0.3
    
    # Composite
    water_front = (t_water < t_solid) & hit_water
    solid_vis = hit_solid & ~water_front
    
    # FIXED: Sky background - CONSTANT colors
    sky_y = (y.flatten() + 1.0) * 0.5
    sky_bg = torch.stack([
        0.4 + sky_y * 0.25,
        0.55 + sky_y * 0.35,
        0.85 + sky_y * 0.15
    ], dim=-1)
    
    # Sun disk
    sun_disk = torch.clamp(1.0 - torch.norm(rd - sun_dir, dim=-1) * 8.0, 0.0, 1.0)
    sun_disk = sun_disk ** 0.5
    sky_bg = sky_bg + sun_disk.unsqueeze(-1) * sun_color * 2.0
    
    rgb = sky_bg
    rgb = torch.where(solid_vis.unsqueeze(-1), solid_color, rgb)
    rgb = torch.where(water_front.unsqueeze(-1), water_color, rgb)
    
    # Fog
    min_t = torch.where(water_front, t_water, torch.where(hit_solid, t_solid, torch.full_like(t_water, MAX_DIST)))
    fog = torch.exp(-min_t * 0.015)
    fog_color = torch.tensor([0.6, 0.7, 0.85], device=device)
    rgb = rgb * fog.unsqueeze(-1) + fog_color * (1.0 - fog.unsqueeze(-1))
    
    # Tone map
    rgb = rgb / (rgb + 1.0)
    
    return torch.clamp(rgb.reshape(height, width, 3), 0.0, 1.0)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    scene = WaterScene(device=device)
    
    # Test frame
    print("Test frame...")
    scene.time = 2.0
    img = render(scene, 800, 600)
    img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np, 'RGB').save('water_fixed_test.png')
    print("Saved: water_fixed_test.png")
    
    # Animation
    print("\nRendering 240 frames...")
    frames = []
    dt = 1.0 / 30.0
    
    for i in range(240):
        scene.physics.step(dt)
        scene.time = i * dt
        
        img = render(scene, 640, 480)
        img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        frames.append(Image.fromarray(img_np, 'RGB'))
        
        if i % 40 == 0:
            print(f"  Frame {i}/240")
    
    # GIF
    print("Saving GIF...")
    frames[0].save('water_fixed.gif', save_all=True, append_images=frames[1:], duration=33, loop=0)
    print("Saved: water_fixed.gif")
    
    # MP4
    try:
        import imageio
        print("Saving MP4...")
        writer = imageio.get_writer('water_fixed.mp4', fps=30, codec='libx264', quality=8)
        for f in frames:
            writer.append_data(np.array(f))
        writer.close()
        print("Saved: water_fixed.mp4")
    except:
        print("(imageio not available)")
    
    print("\n✅ Done!")
