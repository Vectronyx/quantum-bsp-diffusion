#!/usr/bin/env python3
"""
stellated_4d.py - Stellated 4D Polytopes (Fixed)
Simple working star polytopes
"""

import torch
import numpy as np
from PIL import Image
import colorsys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

def rot4d(xw=0, yw=0, zw=0, xy=0):
    R = torch.eye(4, device=device)
    for (i, j, a) in [(0,3,xw), (1,3,yw), (2,3,zw), (0,1,xy)]:
        if a != 0:
            c, s = np.cos(a), np.sin(a)
            r = torch.eye(4, device=device)
            r[i,i], r[j,j] = c, c
            r[i,j], r[j,i] = -s, s
            R = R @ r
    return R

def smin(a, b, k=0.08):
    h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1-h) + a * h - k * h * (1-h)

# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE WORKING SDFs
# ═══════════════════════════════════════════════════════════════════════════════

def sphere(p, r=1.0):
    return torch.norm(p, dim=-1) - r

def tesseract(p, s=1.0):
    q = torch.abs(p) - s
    return torch.norm(torch.clamp(q, min=0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0)

def cell16(p, s=1.0):
    return torch.sum(torch.abs(p), dim=-1) - s

def octahedron_3d(p, s=1.0):
    """3D octahedron for spike tips"""
    return (torch.abs(p[..., 0]) + torch.abs(p[..., 1]) + torch.abs(p[..., 2])) * 0.577 - s

# ═══════════════════════════════════════════════════════════════════════════════
# STELLATED SHAPES - SIMPLE VERSIONS THAT WORK
# ═══════════════════════════════════════════════════════════════════════════════

def stellated_tesseract(p, s=0.5):
    """Tesseract with pyramidal spikes on each face"""
    # Base tesseract (smaller)
    d = tesseract(p, s * 0.6)
    
    # Add 8 spikes (one for each cubic cell)
    # Spikes along each axis direction
    for axis in range(4):
        for sign in [-1.0, 1.0]:
            # Spike center
            center = torch.zeros(4, device=device)
            center[axis] = sign * s * 0.8
            
            # Cone pointing outward
            p_off = p - center
            
            # Direction of spike
            spike_dir = torch.zeros(4, device=device)
            spike_dir[axis] = sign
            
            # Project onto spike direction
            proj = (p_off * spike_dir).sum(dim=-1)
            
            # Perpendicular distance
            perp = torch.sqrt(torch.sum(p_off**2, dim=-1) - proj**2 + 1e-8)
            
            # Cone: narrows as we go outward
            cone = perp - (s * 0.5 - proj * sign) * 0.5
            cone = torch.where(proj * sign > 0, cone, torch.full_like(cone, 1e10))
            
            d = torch.minimum(d, cone)
    
    return d

def stellated_16cell(p, s=0.6):
    """16-cell with extended vertices (spiky star)"""
    # Inner 16-cell
    d_inner = cell16(p, s * 0.5)
    
    # Outer spikes - extend along each axis
    d_spikes = torch.full(p.shape[:-1], 1e10, device=device)
    
    for axis in range(4):
        for sign in [-1.0, 1.0]:
            # Thin spike along axis
            spike_center = torch.zeros(4, device=device)
            spike_center[axis] = sign * s * 0.3
            
            p_off = p - spike_center
            
            # Length along spike
            length = p_off[..., axis] * sign
            
            # Radius perpendicular to spike
            mask = torch.ones(4, device=device, dtype=torch.bool)
            mask[axis] = False
            perp_sq = (p_off[..., mask] ** 2).sum(dim=-1)
            perp = torch.sqrt(perp_sq + 1e-8)
            
            # Tapered spike (cone)
            taper = 0.15 * (1.0 - length / (s * 1.2))
            taper = torch.clamp(taper, min=0.02)
            
            spike = perp - taper
            spike = torch.where(length > 0, spike, torch.full_like(spike, 1e10))
            spike = torch.where(length < s * 1.2, spike, torch.full_like(spike, 1e10))
            
            d_spikes = torch.minimum(d_spikes, spike)
    
    return torch.minimum(d_inner, d_spikes)

def stellated_24cell(p, s=0.5):
    """24-cell with stellations at vertices"""
    q = torch.abs(p)
    
    # Base 24-cell
    d_inf = q.max(dim=-1).values
    sq = torch.sort(q, dim=-1, descending=True).values
    d_12 = (sq[..., 0] + sq[..., 1]) * 0.5
    d_base = torch.maximum(d_inf, d_12) - s * 0.6
    
    # Add spikes at 24 vertices (permutations of ±1,±1,0,0)
    d_spikes = torch.full(p.shape[:-1], 1e10, device=device)
    
    for i in range(4):
        for j in range(i+1, 4):
            for si in [-1.0, 1.0]:
                for sj in [-1.0, 1.0]:
                    # Vertex position
                    v = torch.zeros(4, device=device)
                    v[i] = si * s * 0.7
                    v[j] = sj * s * 0.7
                    
                    # Small sphere at vertex (creates bumpy appearance)
                    d_v = torch.norm(p - v, dim=-1) - s * 0.2
                    d_spikes = torch.minimum(d_spikes, d_v)
    
    return smin(d_base, d_spikes, 0.05)

def compound_star(p, s=0.5):
    """Compound of tesseract and 16-cell - simple star"""
    d1 = tesseract(p, s * 0.7)
    d2 = cell16(p, s * 0.9)
    
    # Carve out center for hollow look
    d3 = sphere(p, s * 0.35)
    
    d = smin(d1, d2, 0.06)
    d = torch.maximum(d, -d3)
    
    return d

def spiky_ball(p, s=0.5, n_spikes=20):
    """Simple spiky sphere - guaranteed to work"""
    # Base sphere
    d = sphere(p, s * 0.4)
    
    # Add spikes using golden spiral distribution
    phi = (1 + np.sqrt(5)) / 2
    
    for i in range(n_spikes):
        # Golden spiral point on sphere
        y = 1 - (i / (n_spikes - 1)) * 2
        r = np.sqrt(1 - y*y)
        theta = 2 * np.pi * i / phi
        
        # 4D direction (extend 3D point to 4D)
        dir_3d = np.array([r * np.cos(theta), y, r * np.sin(theta)])
        
        # Rotate into 4D
        w_angle = i * 0.3
        dir_4d = torch.tensor([
            dir_3d[0] * np.cos(w_angle),
            dir_3d[1],
            dir_3d[2],
            dir_3d[0] * np.sin(w_angle)
        ], device=device, dtype=torch.float32)
        dir_4d = dir_4d / torch.norm(dir_4d)
        
        # Spike center
        center = dir_4d * s * 0.5
        
        # Cone spike
        p_off = p - center
        proj = (p_off * dir_4d).sum(dim=-1)
        perp = torch.sqrt((p_off**2).sum(dim=-1) - proj**2 + 1e-8)
        
        # Cone shape
        cone = perp - (s * 0.4 - proj) * 0.3
        cone = torch.where(proj > -s * 0.2, cone, torch.full_like(cone, 1e10))
        
        d = torch.minimum(d, cone)
    
    return d

# ═══════════════════════════════════════════════════════════════════════════════
# RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def render(sdf_fn, res=400, w_slice=0.0, R=None, hue=0.8):
    if R is None:
        R = rot4d(xw=0.4, yw=0.2, zw=0.15)
    R_inv = R.T
    
    fov, cam = 1.0, 2.5
    u = torch.linspace(-fov, fov, res, device=device)
    v = torch.linspace(-fov, fov, res, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')
    
    origins = torch.zeros(res, res, 4, device=device)
    origins[..., 2] = -cam
    origins[..., 3] = w_slice
    
    dirs = torch.stack([uu, -vv, torch.full_like(uu, 1.8), torch.zeros_like(uu)], dim=-1)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    
    t = torch.zeros(res, res, device=device)
    hit = torch.zeros(res, res, dtype=torch.bool, device=device)
    hit_pos = torch.zeros(res, res, 4, device=device)
    glow = torch.zeros(res, res, device=device)
    
    for step in range(100):
        if hit.all():
            break
            
        p = origins + t.unsqueeze(-1) * dirs
        p_rot = (p.view(-1, 4) @ R_inv).view(p.shape)
        d = sdf_fn(p_rot)
        
        new_hit = ~hit & (d < 0.001)
        hit |= new_hit
        hit_pos[new_hit] = p[new_hit]
        
        glow += torch.exp(-torch.clamp(d, min=0) * 5) * 0.01 * (~hit).float()
        t[~hit] += d[~hit].clamp(min=0.001) * 0.8
        t = torch.clamp(t, max=15.0)
    
    print(f"    Hits: {hit.sum().item()}")
    
    # Normals
    normals = torch.zeros_like(hit_pos)
    eps = 0.002
    for i in range(4):
        pp, pn = hit_pos.clone(), hit_pos.clone()
        pp[..., i] += eps
        pn[..., i] -= eps
        normals[..., i] = sdf_fn((pp.view(-1,4) @ R_inv).view(pp.shape)) - \
                          sdf_fn((pn.view(-1,4) @ R_inv).view(pn.shape))
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-10)
    
    # Lighting
    L = torch.tensor([0.5, 0.7, -0.4, 0.2], device=device)
    L = L / torch.norm(L)
    diff = torch.clamp((normals * L).sum(dim=-1), 0.1, 1.0)
    
    V = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)
    H = L + V
    H = H / torch.norm(H)
    spec = torch.clamp((normals * H).sum(dim=-1), 0, 1) ** 32
    
    w_n = torch.clamp((hit_pos[..., 3] + 0.8) / 1.6, 0, 1)
    fog = torch.exp(-t * 0.1)
    
    # To numpy
    hit_np = hit.cpu().numpy()
    diff_np = diff.cpu().numpy()
    spec_np = spec.cpu().numpy()
    w_np = w_n.cpu().numpy()
    fog_np = fog.cpu().numpy()
    glow_np = glow.cpu().numpy()
    
    img = np.zeros((res, res, 3))
    
    for y in range(res):
        for x in range(res):
            if hit_np[y, x]:
                # Purple material
                h = (hue + w_np[y, x] * 0.1) % 1.0
                s = 0.55 + w_np[y, x] * 0.15
                v = 0.15 + 0.55 * diff_np[y, x]
                
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                
                # Specular
                r += spec_np[y, x] * 0.5
                g += spec_np[y, x] * 0.4
                b += spec_np[y, x] * 0.6
                
                # Fog
                r = r * fog_np[y, x] + 0.01 * (1 - fog_np[y, x])
                g = g * fog_np[y, x] + 0.005 * (1 - fog_np[y, x])
                b = b * fog_np[y, x] + 0.02 * (1 - fog_np[y, x])
                
                img[y, x] = [r, g, b]
            else:
                g = glow_np[y, x]
                img[y, x] = [0.008 + g * 0.15, 0.004 + g * 0.06, 0.015 + g * 0.2]
    
    return np.clip(img, 0, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  STELLATED 4D POLYTOPES                                   ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    
    shapes = [
        ("Spiky 4D Ball", spiky_ball, 0.82, 0.4, 0.2, 0.15),
        ("Stellated Tesseract", stellated_tesseract, 0.78, 0.5, 0.3, 0.2),
        ("Stellated 16-cell", stellated_16cell, 0.75, 0.45, 0.25, 0.3),
        ("Compound Star", compound_star, 0.85, 0.35, 0.4, 0.1),
    ]
    
    # Render each
    print("  Rendering shapes...\n")
    images = []
    
    for name, sdf_fn, hue, xw, yw, zw in shapes:
        print(f"  {name}...")
        R = rot4d(xw=xw, yw=yw, zw=zw)
        img = render(sdf_fn, res=400, w_slice=0.0, R=R, hue=hue)
        images.append((name, img))
        
        # Save individual
        fname = name.lower().replace(' ', '_').replace('-', '')
        Image.fromarray((img * 255).astype(np.uint8)).save(f'4d_{fname}.png')
        print(f"    Saved: 4d_{fname}.png\n")
    
    # Create 2x2 grid
    print("  Creating grid...")
    cell = 400
    pad = 4
    grid = np.zeros((2 * cell + 3 * pad, 2 * cell + 3 * pad, 3))
    grid[:] = [0.01, 0.005, 0.02]
    
    for idx, (name, img) in enumerate(images):
        row, col = idx // 2, idx % 2
        y = pad + row * (cell + pad)
        x = pad + col * (cell + pad)
        grid[y:y+cell, x:x+cell] = img
    
    Image.fromarray((grid * 255).astype(np.uint8)).save('4d_stellated_grid.png')
    print("  ✓ Saved: 4d_stellated_grid.png\n")
    
    # Quick animation
    print("  Rendering animation...")
    frames = []
    n = 48
    
    for i in range(n):
        t = i / n
        R = rot4d(xw=t * 2 * np.pi, yw=np.sin(t * 4 * np.pi) * 0.3, zw=t * np.pi * 0.5)
        w = np.sin(t * 2 * np.pi) * 0.3
        
        if i % 8 == 0:
            print(f"    Frame {i+1}/{n}")
        
        img = render(spiky_ball, res=350, w_slice=w, R=R, hue=0.82)
        frames.append(Image.fromarray((img * 255).astype(np.uint8)))
    
    frames[0].save('4d_stellated.gif', save_all=True, append_images=frames[1:], duration=50, loop=0)
    print("  ✓ Saved: 4d_stellated.gif\n")


if __name__ == "__main__":
    main()
