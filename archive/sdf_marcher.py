#!/usr/bin/env python3
"""
SDF Ray Marcher for Latent Space
"""

import torch
import torch.nn.functional as F
import math

EPSILON = 1e-4
MAX_STEPS = 128
MAX_DIST = 100.0

# SDF PRIMITIVES
def sdf_sphere(p, radius=1.0):
    return torch.norm(p, dim=-1) - radius

def sdf_box(p, b):
    q = torch.abs(p) - b
    return torch.norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(torch.max(q, dim=-1).values, max=0.0)

def sdf_torus(p, major=1.0, minor=0.25):
    q = torch.stack([torch.norm(p[..., :2], dim=-1) - major, p[..., 2]], dim=-1)
    return torch.norm(q, dim=-1) - minor

# BOOLEAN OPS
def op_union(d1, d2):
    return torch.min(d1, d2)

def op_subtract(d1, d2):
    return torch.max(d1, -d2)

def op_intersect(d1, d2):
    return torch.max(d1, d2)

# SMOOTH BOOLEAN
def op_smooth_union(d1, d2, k=0.1):
    h = torch.clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return torch.lerp(d2, d1, h) - k * h * (1.0 - h)

def op_smooth_subtract(d1, d2, k=0.1):
    h = torch.clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0)
    return torch.lerp(d1, -d2, h) + k * h * (1.0 - h)

# DOMAIN OPS
def op_twist(p, k):
    c = torch.cos(k * p[..., 1])
    s = torch.sin(k * p[..., 1])
    return torch.stack([c * p[..., 0] - s * p[..., 2], p[..., 1], s * p[..., 0] + c * p[..., 2]], dim=-1)

def op_repeat(p, spacing):
    return torch.remainder(p + spacing * 0.5, spacing) - spacing * 0.5

# UTILITIES
def smoothstep(edge0, edge1, x):
    t = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def calc_normal(sdf_func, p, eps=EPSILON):
    e = torch.tensor([eps, 0.0, 0.0], device=p.device)
    return F.normalize(torch.stack([
        sdf_func(p + e.roll(0)) - sdf_func(p - e.roll(0)),
        sdf_func(p + e.roll(1)) - sdf_func(p - e.roll(1)),
        sdf_func(p + e.roll(2)) - sdf_func(p - e.roll(2))
    ], dim=-1), dim=-1)

def ray_march(ro, rd, sdf_func, max_steps=MAX_STEPS):
    t = torch.zeros(ro.shape[0], device=ro.device)
    d = torch.zeros_like(t)
    for _ in range(max_steps):
        p = ro + t.unsqueeze(-1) * rd
        d = sdf_func(p)
        t = t + d
        if (d < EPSILON).all() or (t > MAX_DIST).all():
            break
    return t, d < EPSILON

class SDFScene:
    def __init__(self, device='cuda'):
        self.device = device
        self.time = 0.0
    
    def set_time(self, t):
        self.time = t
        return self
    
    def scene(self, p):
        t = self.time
        s1 = torch.tensor([math.sin(t)*1.5, math.cos(t*0.7)*0.5, math.cos(t)*1.5], device=p.device)
        s2 = torch.tensor([math.sin(t+2.1)*1.5, math.cos(t*0.7+1)*0.5, math.cos(t+2.1)*1.5], device=p.device)
        d1 = sdf_sphere(p - s1, 0.8)
        d2 = sdf_sphere(p - s2, 0.6)
        d3 = sdf_torus(op_twist(p, math.sin(t*0.3)*0.5), 1.0, 0.3)
        k = 0.5 + 0.3 * math.sin(t * 0.5)
        return op_smooth_union(op_smooth_union(d1, d2, k), d3, k*0.5)
    
    def render(self, width=512, height=512):
        y, x = torch.meshgrid(
            torch.linspace(1, -1, height, device=self.device),
            torch.linspace(-1, 1, width, device=self.device), indexing='ij')
        ro = torch.tensor([0., 0., -5.], device=self.device).expand(height*width, 3)
        rd = F.normalize(torch.stack([x.flatten(), y.flatten(), torch.ones(height*width, device=self.device)*2], dim=-1), dim=-1)
        t, hit = ray_march(ro, rd, self.scene)
        p = ro + t.unsqueeze(-1) * rd
        n = calc_normal(self.scene, p)
        light = F.normalize(torch.tensor([1., 1., -1.], device=self.device), dim=0)
        diff = torch.clamp((n * light).sum(-1), 0, 1)
        color = (diff + 0.1) * hit.float()
        glow = 0.3 * (~hit).float() / (1 + t * 0.1)
        return (color + glow).reshape(height, width)

if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    scene = SDFScene(device)
    img = scene.set_time(0).render(512, 512)
    Image.fromarray((img.cpu().numpy()*255).clip(0,255).astype(np.uint8), 'L').save('sdf_test.png')
    print("Saved: sdf_test.png")
    frames = [Image.fromarray((scene.set_time(i*0.1).render(256,256).cpu().numpy()*255).clip(0,255).astype(np.uint8), 'L') for i in range(60)]
    frames[0].save('sdf_anim.gif', save_all=True, append_images=frames[1:], duration=50, loop=0)
    print("Saved: sdf_anim.gif")
