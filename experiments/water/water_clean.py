#!/usr/bin/env python3
"""
Clean water with proper waves and shading
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image

EPSILON = 1e-4
MAX_STEPS = 80
MAX_DIST = 50.0

class Body:
    def __init__(self, pos, size, color, device='cuda'):
        self.device = device
        self.pos = torch.tensor(pos, device=device, dtype=torch.float32)
        self.size = size
        self.color = torch.tensor(color, device=device, dtype=torch.float32)
        self.start_x = pos[0]
        self.start_z = pos[2]
    
    def update(self, t):
        wave_h = 0.3 * math.sin(self.start_x * 0.3 + t * 0.7)
        wave_h += 0.25 * math.sin(self.start_z * 0.35 + t * 0.8)
        wave_h += 0.15 * math.sin((self.start_x + self.start_z) * 0.2 + t * 0.5)
        self.pos[1] = wave_h + self.size * 0.3

class Scene:
    def __init__(self, device='cuda'):
        self.device = device
        self.time = 0.0
        self.bodies = []
        self._spawn()
    
    def _spawn(self):
        self.bodies.append(Body([0., 0., 0.], 0.5, [0.8, 0.2, 0.2], self.device))
        self.bodies.append(Body([3., 0., 2.], 0.4, [0.2, 0.6, 0.8], self.device))
        self.bodies.append(Body([-2., 0., -2.], 0.45, [0.3, 0.8, 0.3], self.device))
        self.bodies.append(Body([4., 0., -3.], 0.35, [0.8, 0.6, 0.1], self.device))
        self.bodies.append(Body([-3., 0., 3.], 0.5, [0.6, 0.3, 0.7], self.device))
    
    def update(self, dt):
        self.time += dt
        for body in self.bodies:
            body.update(self.time)
    
    def wave_height(self, p):
        t = self.time
        h = torch.zeros(p.shape[0], device=self.device)
        h += 0.3 * torch.sin(p[..., 0] * 0.3 + t * 0.7)
        h += 0.25 * torch.sin(p[..., 2] * 0.35 + t * 0.8)
        h += 0.15 * torch.sin((p[..., 0] + p[..., 2]) * 0.2 + t * 0.5)
        h += 0.1 * torch.sin(p[..., 0] * 0.8 - p[..., 2] * 0.6 + t * 1.0)
        h += 0.05 * torch.sin(p[..., 0] * 1.5 + t * 1.5)
        return h
    
    def scene_water(self, p):
        return p[..., 1] - self.wave_height(p)
    
    def scene_bodies(self, p):
        result = torch.full((p.shape[0],), 999.0, device=self.device)
        for body in self.bodies:
            d = torch.norm(p - body.pos, dim=-1) - body.size
            result = torch.min(result, d)
        return result
    
    def scene_floor(self, p):
        return p[..., 1] + 5.0
    
    def scene_solid(self, p):
        return torch.min(self.scene_floor(p), self.scene_bodies(p))
    
    def get_body_color(self, p):
        colors = torch.full((p.shape[0], 3), 0.3, device=self.device)
        min_dist = torch.full((p.shape[0],), 999.0, device=self.device)
        for body in self.bodies:
            d = torch.norm(p - body.pos, dim=-1) - body.size
            mask = d < min_dist
            min_dist = torch.where(mask, d, min_dist)
            colors = torch.where(mask.unsqueeze(-1), body.color.unsqueeze(0).expand(p.shape[0], 3), colors)
        return colors

def calc_normal(scene, sdf_func, p):
    e = EPSILON
    d = scene.device
    return F.normalize(torch.stack([
        sdf_func(p + torch.tensor([e, 0, 0], device=d)) - sdf_func(p - torch.tensor([e, 0, 0], device=d)),
        sdf_func(p + torch.tensor([0, e, 0], device=d)) - sdf_func(p - torch.tensor([0, e, 0], device=d)),
        sdf_func(p + torch.tensor([0, 0, e], device=d)) - sdf_func(p - torch.tensor([0, 0, e], device=d))
    ], dim=-1), dim=-1)

def ray_march(ro, rd, sdf_func):
    t = torch.zeros(ro.shape[0], device=ro.device)
    hit = torch.zeros(ro.shape[0], dtype=torch.bool, device=ro.device)
    for _ in range(MAX_STEPS):
        p = ro + t.unsqueeze(-1) * rd
        d = sdf_func(p)
        hit = hit | (d < EPSILON)
        t = t + d * 0.7
        if hit.all() or (t > MAX_DIST).all():
            break
    return t, d < EPSILON

def render(scene, W=640, H=480):
    dev = scene.device
    t = scene.time
    
    # Camera
    angle = t * 0.1
    cam_pos = torch.tensor([
        math.sin(angle) * 14.0,
        6.0,
        math.cos(angle) * 14.0
    ], device=dev)
    
    look_at = torch.tensor([0., 0., 0.], device=dev)
    fwd = F.normalize(look_at - cam_pos, dim=0)
    right = F.normalize(torch.linalg.cross(torch.tensor([0., 1., 0.], device=dev), fwd), dim=0)
    up = torch.linalg.cross(fwd, right)
    
    asp = W / H
    yy, xx = torch.meshgrid(
        torch.linspace(1, -1, H, device=dev),
        torch.linspace(-asp, asp, W, device=dev),
        indexing='ij'
    )
    
    N = H * W
    ro = cam_pos.unsqueeze(0).expand(N, 3)
    rd = F.normalize(
        xx.reshape(-1, 1) * right +
        yy.reshape(-1, 1) * up +
        fwd * 2.0,
        dim=-1
    )
    
    # Ray march water
    t_water, hit_water = ray_march(ro, rd, scene.scene_water)
    p_water = ro + t_water.unsqueeze(-1) * rd
    n_water = calc_normal(scene, scene.scene_water, p_water)
    
    # Ray march solids
    t_solid, hit_solid = ray_march(ro, rd, scene.scene_solid)
    p_solid = ro + t_solid.unsqueeze(-1) * rd
    n_solid = calc_normal(scene, scene.scene_solid, p_solid)
    
    # Lighting
    light_dir = F.normalize(torch.tensor([0.4, 0.8, -0.3], device=dev), dim=0)
    view_dir = -rd
    
    # === WATER SHADING ===
    # Fresnel - more reflection at grazing angles
    n_dot_v = torch.clamp((n_water * view_dir).sum(-1), 0.0, 1.0)
    fresnel = 0.02 + 0.98 * torch.pow(1.0 - n_dot_v, 5.0)
    
    # Reflection
    refl_dir = rd - 2.0 * (rd * n_water).sum(-1, keepdim=True) * n_water
    
    # Sky color for reflection
    refl_y = torch.clamp(refl_dir[..., 1], -1, 1)
    sky_r = torch.full((N,), 0.5, device=dev) + refl_y * 0.15
    sky_g = torch.full((N,), 0.65, device=dev) + refl_y * 0.2
    sky_b = torch.full((N,), 0.9, device=dev) + refl_y * 0.1
    sky_refl = torch.stack([sky_r, sky_g, sky_b], dim=-1)
    
    # Water diffuse lighting (shows wave shape)
    water_diffuse = torch.clamp((n_water * light_dir).sum(-1), 0.0, 1.0)
    
    # Water base color
    water_deep = torch.tensor([0.05, 0.15, 0.25], device=dev)
    water_shallow = torch.tensor([0.1, 0.3, 0.4], device=dev)
    
    # Mix based on fresnel and diffuse
    water_base = water_deep + water_diffuse.unsqueeze(-1) * 0.2
    water_color = fresnel.unsqueeze(-1) * sky_refl + (1.0 - fresnel.unsqueeze(-1)) * water_base
    
    # Add specular highlight (small, tight)
    half_vec = F.normalize(light_dir + view_dir, dim=-1)
    spec = torch.pow(torch.clamp((n_water * half_vec).sum(-1), 0.0, 1.0), 128.0)
    water_color = water_color + spec.unsqueeze(-1) * 0.5
    
    # === SOLID SHADING ===
    solid_diffuse = torch.clamp((n_solid * light_dir).sum(-1), 0.15, 1.0)
    solid_base = scene.get_body_color(p_solid)
    solid_color = solid_diffuse.unsqueeze(-1) * solid_base
    
    # Solid specular
    half_s = F.normalize(light_dir + view_dir, dim=-1)
    spec_s = torch.pow(torch.clamp((n_solid * half_s).sum(-1), 0.0, 1.0), 32.0)
    solid_color = solid_color + spec_s.unsqueeze(-1) * 0.2
    
    # === SKY BACKGROUND ===
    sky_t = (yy.reshape(-1) + 1.0) * 0.5
    sky_bg = torch.stack([
        0.5 + sky_t * 0.2,
        0.6 + sky_t * 0.25,
        0.85 + sky_t * 0.1
    ], dim=-1)
    
    # === COMPOSITE ===
    water_in_front = hit_water & (t_water < t_solid)
    solid_visible = hit_solid & (~hit_water | (t_solid < t_water))
    
    rgb = sky_bg.clone()
    rgb[solid_visible] = solid_color[solid_visible]
    rgb[water_in_front] = water_color[water_in_front]
    
    return torch.clamp(rgb.reshape(H, W, 3), 0.0, 1.0)

if __name__ == "__main__":
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {dev}")
    
    scene = Scene(dev)
    
    # Test frame
    print("Test frame...")
    for _ in range(30):
        scene.update(1/30)
    img = render(scene, 800, 600)
    Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8), 'RGB').save('water_test.png')
    print("Saved: water_test.png")
    print("Check: xdg-open water_test.png")
    
    # Ask before rendering full animation
    input("\nPress Enter to render animation (or Ctrl+C to stop)...")
    
    scene = Scene(dev)
    print("Rendering 180 frames...")
    frames = []
    
    for i in range(180):
        scene.update(1/30)
        img = render(scene, 640, 480)
        frames.append(Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8), 'RGB'))
        if i % 30 == 0:
            print(f"  Frame {i}/180")
    
    frames[0].save('water.gif', save_all=True, append_images=frames[1:], duration=33, loop=0)
    print("Saved: water.gif")
    
    try:
        import imageio
        w = imageio.get_writer('water.mp4', fps=30, codec='libx264', quality=8)
        for f in frames:
            w.append_data(np.array(f))
        w.close()
        print("Saved: water.mp4")
    except:
        pass
    
    print("Done!")
