#!/usr/bin/env python3
import torch
import numpy as np
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def smin(a, b, k=0.08):
    h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1-h) + a * h - k * h * (1-h)

def stellated_mace(p, s=0.5):
    """Clean mace-head stellated shape"""
    # Core sphere
    d = torch.norm(p, dim=-1) - s * 0.22
    
    # 8 main spikes (±x, ±y, ±z, ±w)
    for axis in range(4):
        for sign in [-1.0, 1.0]:
            dir_v = torch.zeros(4, device=device)
            dir_v[axis] = sign
            
            proj = (p * dir_v).sum(dim=-1)
            perp = torch.sqrt(torch.clamp((p**2).sum(dim=-1) - proj**2, min=1e-8))
            
            # Cone
            spike_len = s * 0.65
            cone = perp - 0.06 * s * (1 - proj / spike_len)
            cone = torch.where((proj > 0) & (proj < spike_len), cone, torch.full_like(cone, 1e10))
            
            d = smin(d, cone, 0.03)
    
    return d

# Render
res = 512
R = torch.eye(4, device=device)
theta = 0.5
c, s = np.cos(theta), np.sin(theta)
R[0,0], R[3,3] = c, c
R[0,3], R[3,0] = -s, s

fov = 1.0
u = torch.linspace(-fov, fov, res, device=device)
uu, vv = torch.meshgrid(u, u, indexing='xy')

origins = torch.zeros(res, res, 4, device=device)
origins[..., 2] = -2.5

dirs = torch.stack([uu, -vv, torch.full_like(uu, 1.8), torch.zeros_like(uu)], dim=-1)
dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

t = torch.zeros(res, res, device=device)
hit = torch.zeros(res, res, dtype=torch.bool, device=device)

for step in range(80):
    p = origins + t.unsqueeze(-1) * dirs
    p_rot = (p.view(-1, 4) @ R.T).view(p.shape)
    d = stellated_mace(p_rot)
    
    hit |= ~hit & (d < 0.001)
    t[~hit] += d[~hit].clamp(min=0.001) * 0.8
    t = torch.clamp(t, max=10.0)

# Simple purple shading
img = torch.zeros(res, res, 3, device=device)
img[hit, 0] = 0.35  # R
img[hit, 1] = 0.15  # G  
img[hit, 2] = 0.45  # B

# Depth shading
depth_shade = 1.0 - (t[hit] - t[hit].min()) / (t[hit].max() - t[hit].min() + 1e-8) * 0.5
img[hit, 0] *= depth_shade
img[hit, 1] *= depth_shade
img[hit, 2] *= depth_shade

img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(img_np).save('stellated_test.png')

print(f"Hits: {hit.sum().item()} / {res*res}")
print("Saved: stellated_test.png")
