#!/usr/bin/env python3
"""
showcase_4d.py - 4D Polytope Showcase Grid
All regular 4-polytopes + exotic tori in one image
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4D ROTATION
# ═══════════════════════════════════════════════════════════════════════════════

def rot4d(xw=0, yw=0, zw=0, xy=0, xz=0, yz=0):
    R = torch.eye(4, device=device)
    for (i, j, a) in [(0,3,xw), (1,3,yw), (2,3,zw), (0,1,xy), (0,2,xz), (1,2,yz)]:
        if a != 0:
            c, s = np.cos(a), np.sin(a)
            r = torch.eye(4, device=device)
            r[i,i], r[j,j] = c, c
            r[i,j], r[j,i] = -s, s
            R = R @ r
    return R

# ═══════════════════════════════════════════════════════════════════════════════
# 4D SDFs
# ═══════════════════════════════════════════════════════════════════════════════

def smin(a, b, k=0.1):
    h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1-h) + a * h - k * h * (1-h)

class SDF:
    @staticmethod
    def sphere(p, r=1.0):
        return torch.norm(p, dim=-1) - r
    
    @staticmethod
    def tesseract(p, s=0.6):
        q = torch.abs(p) - s
        return torch.norm(torch.clamp(q, min=0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0)
    
    @staticmethod
    def cell16(p, s=0.75):
        return torch.sum(torch.abs(p), dim=-1) - s
    
    @staticmethod
    def cell24(p, s=0.65):
        q = torch.abs(p)
        d_inf = q.max(dim=-1).values
        sq = torch.sort(q, dim=-1, descending=True).values
        d_12 = (sq[..., 0] + sq[..., 1]) * 0.5
        return torch.maximum(d_inf, d_12) - s * 0.707
    
    @staticmethod
    def cell5(p, s=0.7):
        """5-cell / Pentachoron / 4-simplex"""
        # Vertices of regular 5-cell
        sq5 = np.sqrt(5)
        vertices = torch.tensor([
            [1, 1, 1, -1/sq5],
            [1, -1, -1, -1/sq5],
            [-1, 1, -1, -1/sq5],
            [-1, -1, 1, -1/sq5],
            [0, 0, 0, sq5 - 1/sq5]
        ], device=device, dtype=torch.float32) * 0.5
        
        d = torch.full(p.shape[:-1], -1e10, device=device)
        for v in vertices:
            plane_d = (p * v).sum(dim=-1) - s * 0.4
            d = torch.maximum(d, plane_d)
        return d
    
    @staticmethod
    def clifford(p, R=0.55, r=0.35):
        d1 = torch.sqrt(p[...,0]**2 + p[...,1]**2 + 1e-8) - R
        d2 = torch.sqrt(p[...,2]**2 + p[...,3]**2 + 1e-8) - r
        return torch.sqrt(d1**2 + d2**2 + 1e-8) - 0.1
    
    @staticmethod
    def tiger(p, R=0.45, r=0.12):
        d1 = torch.sqrt(p[...,0]**2 + p[...,2]**2 + 1e-8) - R
        d2 = torch.sqrt(p[...,1]**2 + p[...,3]**2 + 1e-8) - R
        return torch.sqrt(d1**2 + d2**2 + 1e-8) - r
    
    @staticmethod
    def duocylinder(p, r=0.5):
        d1 = torch.sqrt(p[...,0]**2 + p[...,1]**2 + 1e-8) - r
        d2 = torch.sqrt(p[...,2]**2 + p[...,3]**2 + 1e-8) - r
        return torch.maximum(d1, d2)
    
    @staticmethod
    def ditorus(p, R1=0.45, R2=0.18, r=0.06):
        dxy = torch.sqrt(p[...,0]**2 + p[...,1]**2 + 1e-8) - R1
        d1 = torch.sqrt(dxy**2 + p[...,2]**2 + 1e-8) - R2
        return torch.sqrt(d1**2 + p[...,3]**2 + 1e-8) - r

# ═══════════════════════════════════════════════════════════════════════════════
# RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def render_shape(sdf_fn, res=200, w_slice=0.0, R=None, hue_base=0.6):
    if R is None:
        R = rot4d(xw=0.4, yw=0.25, zw=0.15)
    R_inv = R.T
    
    fov, cam = 0.9, 2.2
    u = torch.linspace(-fov, fov, res, device=device)
    v = torch.linspace(-fov, fov, res, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')
    
    origins = torch.zeros(res, res, 4, device=device)
    origins[..., 2] = -cam
    origins[..., 3] = w_slice
    
    dirs = torch.stack([uu, -vv, torch.full_like(uu, 1.6), torch.zeros_like(uu)], dim=-1)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    
    t = torch.zeros(res, res, device=device)
    hit = torch.zeros(res, res, dtype=torch.bool, device=device)
    hit_pos = torch.zeros(res, res, 4, device=device)
    glow = torch.zeros(res, res, device=device)
    
    for step in range(80):
        p = origins + t.unsqueeze(-1) * dirs
        p_rot = (p.view(-1, 4) @ R_inv).view(p.shape)
        d = sdf_fn(p_rot)
        
        new_hit = ~hit & (d < 0.0008)
        hit |= new_hit
        hit_pos[new_hit] = p[new_hit]
        
        glow += torch.exp(-torch.clamp(d, min=0) * 8) * 0.006 * (~hit).float()
        t[~hit] += d[~hit] * 0.75
        t = torch.clamp(t, max=12.0)
    
    # Normals
    normals = torch.zeros_like(hit_pos)
    eps = 0.001
    for i in range(4):
        pp, pn = hit_pos.clone(), hit_pos.clone()
        pp[..., i] += eps
        pn[..., i] -= eps
        normals[..., i] = sdf_fn((pp.view(-1,4) @ R_inv).view(pp.shape)) - \
                          sdf_fn((pn.view(-1,4) @ R_inv).view(pn.shape))
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-10)
    
    # Render
    L = torch.tensor([0.5, 0.7, -0.4, 0.15], device=device)
    L = L / torch.norm(L)
    diff = torch.clamp((normals * L).sum(dim=-1), 0.05, 1.0)
    
    w_n = (hit_pos[..., 3] + 0.6) / 1.2
    w_n = torch.clamp(w_n, 0, 1)
    
    fog = torch.exp(-t * 0.1)
    
    img = torch.zeros(res, res, 3, device=device)
    
    # Background
    glow_cpu = glow.cpu().numpy()
    
    hit_np = hit.cpu().numpy()
    diff_np = diff.cpu().numpy()
    w_np = w_n.cpu().numpy()
    fog_np = fog.cpu().numpy()
    
    img_np = np.zeros((res, res, 3))
    
    for y in range(res):
        for x in range(res):
            if hit_np[y, x]:
                hue = (hue_base + w_np[y, x] * 0.15) % 1.0
                sat = 0.5 + w_np[y, x] * 0.2
                val = 0.15 + 0.6 * diff_np[y, x]
                r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
                img_np[y, x] = [r * fog_np[y, x], g * fog_np[y, x], b * fog_np[y, x]]
            else:
                g = glow_cpu[y, x]
                img_np[y, x] = [0.005 + g * 0.1, 0.006 + g * 0.08, 0.015 + g * 0.18]
    
    return np.clip(img_np, 0, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Creating 4D Polytope Showcase...")
    
    shapes = [
        ("5-cell", SDF.cell5, 0.55, {'xw': 0.3, 'yw': 0.2, 'zw': 0.1}),
        ("Tesseract", SDF.tesseract, 0.58, {'xw': 0.4, 'yw': 0.25, 'zw': 0.15}),
        ("16-cell", SDF.cell16, 0.52, {'xw': 0.5, 'yw': 0.3, 'zw': 0.2}),
        ("24-cell", SDF.cell24, 0.48, {'xw': 0.35, 'yw': 0.4, 'zw': 0.1}),
        ("Clifford Torus", SDF.clifford, 0.62, {'xw': 0.5, 'yw': 0.7, 'zw': 0.2}),
        ("Tiger", SDF.tiger, 0.45, {'xw': 0.4, 'yw': 0.3, 'zw': 0.5}),
        ("Duocylinder", SDF.duocylinder, 0.7, {'xw': 0.3, 'yw': 0.5, 'zw': 0.15}),
        ("Ditorus", SDF.ditorus, 0.5, {'xw': 0.4, 'yw': 0.2, 'zw': 0.3}),
    ]
    
    cell_size = 240
    cols = 4
    rows = 2
    padding = 4
    label_height = 24
    
    grid_w = cols * cell_size + (cols + 1) * padding
    grid_h = rows * (cell_size + label_height) + (rows + 1) * padding
    
    grid = np.zeros((grid_h, grid_w, 3))
    grid[:] = [0.02, 0.02, 0.03]  # Dark background
    
    for idx, (name, sdf_fn, hue, rot_params) in enumerate(shapes):
        row, col = idx // cols, idx % cols
        print(f"  [{idx+1}/8] {name}...")
        
        R = rot4d(**rot_params)
        img = render_shape(sdf_fn, res=cell_size, w_slice=0.0, R=R, hue_base=hue)
        
        x = padding + col * (cell_size + padding)
        y = padding + row * (cell_size + label_height + padding)
        
        grid[y:y+cell_size, x:x+cell_size] = img
    
    # Convert to PIL and add labels
    grid_img = Image.fromarray((grid * 255).astype(np.uint8))
    draw = ImageDraw.Draw(grid_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    for idx, (name, _, _, _) in enumerate(shapes):
        row, col = idx // cols, idx % cols
        x = padding + col * (cell_size + padding) + cell_size // 2
        y = padding + row * (cell_size + label_height + padding) + cell_size + 4
        
        # Center text
        bbox = draw.textbbox((0, 0), name, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw//2, y), name, fill=(180, 180, 200), font=font)
    
    grid_img.save('4d_showcase.png')
    print(f"\n  ✓ Saved: 4d_showcase.png ({grid_w}x{grid_h})")
    
    # Also save individual high-res versions
    print("\n  Rendering high-res individuals...")
    for name, sdf_fn, hue, rot_params in shapes[:4]:  # First 4 only for speed
        fname = name.lower().replace(' ', '_').replace('-', '')
        R = rot4d(**rot_params)
        img = render_shape(sdf_fn, res=400, w_slice=0.0, R=R, hue_base=hue)
        Image.fromarray((img * 255).astype(np.uint8)).save(f'4d_hires_{fname}.png')
        print(f"    4d_hires_{fname}.png")


if __name__ == "__main__":
    main()
