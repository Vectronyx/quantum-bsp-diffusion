#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import math
from PIL import Image
import numpy as np

EPSILON = 1e-4
MAX_STEPS = 128
MAX_DIST = 100.0

def sdf_sphere(p, r=1.0):
    return torch.norm(p, dim=-1) - r

def sdf_torus(p, R=1.0, r=0.25):
    q = torch.stack([torch.norm(p[..., :2], dim=-1) - R, p[..., 2]], dim=-1)
    return torch.norm(q, dim=-1) - r

def sdf_box(p, b):
    q = torch.abs(p) - b
    return torch.norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0.0)

def op_smooth_union(d1, d2, k=0.1):
    h = torch.clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return torch.lerp(d2, d1, h) - k * h * (1.0 - h)

def op_twist(p, k):
    c, s = torch.cos(k * p[..., 1]), torch.sin(k * p[..., 1])
    return torch.stack([c*p[...,0] - s*p[...,2], p[...,1], s*p[...,0] + c*p[...,2]], dim=-1)

def op_repeat(p, s):
    return torch.remainder(p + s*0.5, s) - s*0.5

def calc_normal(sdf, p):
    e = torch.tensor([EPSILON, 0., 0.], device=p.device)
    return F.normalize(torch.stack([
        sdf(p + e.roll(0)) - sdf(p - e.roll(0)),
        sdf(p + e.roll(1)) - sdf(p - e.roll(1)),
        sdf(p + e.roll(2)) - sdf(p - e.roll(2))
    ], dim=-1), dim=-1)

def ray_march(ro, rd, sdf):
    t = torch.zeros(ro.shape[0], device=ro.device)
    for _ in range(MAX_STEPS):
        d = sdf(ro + t.unsqueeze(-1) * rd)
        t = t + d
        if (d < EPSILON).all() or (t > MAX_DIST).all():
            break
    return t, d < EPSILON

class ChaosScene:
    def __init__(self, device='cuda'):
        self.device = device
        self.time = 0.0
    
    def scene(self, p):
        t = self.time
        
        # 5 orbiting spheres at different speeds/phases
        spheres = []
        for i in range(5):
            phase = i * 1.256
            speed = 1.0 + i * 0.3
            radius = 0.3 + i * 0.1
            orbit = 1.5 + math.sin(t * 0.5 + i) * 0.5
            
            pos = torch.tensor([
                math.sin(t * speed + phase) * orbit,
                math.cos(t * 0.7 + phase) * 0.8,
                math.cos(t * speed + phase) * orbit
            ], device=p.device)
            
            spheres.append(sdf_sphere(p - pos, radius))
        
        # Twisted torus
        twist_amt = math.sin(t * 0.2) * 2.0
        p_twist = op_twist(p, twist_amt)
        torus = sdf_torus(p_twist, 1.2, 0.15)
        
        # Inner cube (rotating)
        rot = t * 0.5
        c, s = math.cos(rot), math.sin(rot)
        p_rot = p.clone()
        p_rot[..., 0] = c * p[..., 0] - s * p[..., 2]
        p_rot[..., 2] = s * p[..., 0] + c * p[..., 2]
        cube = sdf_box(p_rot, torch.tensor([0.4, 0.4, 0.4], device=p.device))
        
        # Blend everything
        k = 0.4 + 0.2 * math.sin(t)
        
        scene = spheres[0]
        for s in spheres[1:]:
            scene = op_smooth_union(scene, s, k)
        scene = op_smooth_union(scene, torus, k * 0.7)
        scene = op_smooth_union(scene, cube, k * 0.5)
        
        return scene
    
    def render(self, w=512, h=512):
        y, x = torch.meshgrid(
            torch.linspace(1, -1, h, device=self.device),
            torch.linspace(-1, 1, w, device=self.device), indexing='ij')
        
        ro = torch.tensor([0., 0., -6.], device=self.device).expand(h*w, 3)
        rd = F.normalize(torch.stack([x.flatten(), y.flatten(), torch.ones(h*w, device=self.device)*2], dim=-1), dim=-1)
        
        t, hit = ray_march(ro, rd, self.scene)
        p = ro + t.unsqueeze(-1) * rd
        n = calc_normal(self.scene, p)
        
        # Two-tone lighting
        l1 = F.normalize(torch.tensor([1., 1., -1.], device=self.device), dim=0)
        l2 = F.normalize(torch.tensor([-1., 0.5, -0.5], device=self.device), dim=0)
        
        d1 = torch.clamp((n * l1).sum(-1), 0, 1)
        d2 = torch.clamp((n * l2).sum(-1), 0, 1) * 0.3
        
        # RGB channels
        r = (d1 * 0.9 + d2 * 0.1 + 0.1) * hit.float()
        g = (d1 * 0.7 + d2 * 0.3 + 0.1) * hit.float()
        b = (d1 * 0.5 + d2 * 0.5 + 0.15) * hit.float()
        
        # Glow
        glow = 0.2 / (1 + t * 0.1) * (~hit).float()
        r, g, b = r + glow * 0.3, g + glow * 0.5, b + glow
        
        return torch.stack([r, g, b], dim=-1).reshape(h, w, 3)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    scene = ChaosScene(device)
    
    # Static
    scene.time = 0.0
    img = scene.render(512, 512)
    img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np, 'RGB').save('sdf_chaos.png')
    print("Saved: sdf_chaos.png")
    
    # Animation
    print("Rendering 120 frames...")
    frames = []
    for i in range(120):
        scene.time = i * 0.08
        img = scene.render(384, 384)
        img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        frames.append(Image.fromarray(img_np, 'RGB'))
        if i % 20 == 0:
            print(f"  Frame {i}/120")
    
    frames[0].save('sdf_chaos.gif', save_all=True, append_images=frames[1:], duration=40, loop=0)
    print("Saved: sdf_chaos.gif")
