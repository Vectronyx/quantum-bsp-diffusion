#!/usr/bin/env python3
import torch
import numpy as np
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Simple 4D tesseract test
def tesseract_sdf(p, s=0.6):
    q = torch.abs(p) - s
    return torch.norm(torch.clamp(q, min=0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0)

# Setup
res = 256
fov, cam_z = 1.0, 2.5

u = torch.linspace(-fov, fov, res, device=device)
v = torch.linspace(-fov, fov, res, device=device)
uu, vv = torch.meshgrid(u, v, indexing='xy')

origins = torch.zeros(res, res, 4, device=device)
origins[..., 2] = -cam_z
origins[..., 3] = 0.0  # w-slice

dirs = torch.stack([uu, -vv, torch.full_like(uu, 1.8), torch.zeros_like(uu)], dim=-1)
dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

# Simple 4D rotation (XW plane)
theta = 0.5
c, s = np.cos(theta), np.sin(theta)
R = torch.eye(4, device=device)
R[0,0], R[3,3] = c, c
R[0,3], R[3,0] = -s, s

# March
t = torch.zeros(res, res, device=device)
hit = torch.zeros(res, res, dtype=torch.bool, device=device)

for step in range(80):
    p = origins + t.unsqueeze(-1) * dirs
    # Apply rotation
    p_rot = (p.view(-1, 4) @ R.T).view(p.shape)
    d = tesseract_sdf(p_rot)
    
    new_hit = ~hit & (d < 0.001)
    hit |= new_hit
    
    t[~hit] += d[~hit] * 0.8
    t = torch.clamp(t, max=15.0)

# Render
img = torch.zeros(res, res, 3, device=device)
img[hit, 0] = 0.3
img[hit, 1] = 0.25
img[hit, 2] = 0.5

# Add simple shading based on distance
shade = 1.0 - t[hit] / 5.0
img[hit, 0] *= shade
img[hit, 1] *= shade
img[hit, 2] *= shade

img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(img_np).save('test_4d_quick.png')

hit_count = hit.sum().item()
print(f"Hits: {hit_count} / {res*res} ({100*hit_count/(res*res):.1f}%)")
print("Saved: test_4d_quick.png")
