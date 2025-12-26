#!/usr/bin/env python3
"""
stellated_clean.py - Clean Stellated 4D Polytopes
Matching 4D Toys aesthetic
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


# ═══════════════════════════════════════════════════════════════════════════════
# STELLATED POLYTOPES
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

def great_stellated_120cell(p, s=0.6):
    """
    Great Stellated 120-cell approximation
    Uses icosahedral symmetry extended to 4D (H4 group)
    """
    # Core sphere
    d = torch.norm(p, dim=-1) - s * 0.25
    
    # Generate spike directions using H4 symmetry
    # 120 vertices of 600-cell (dual) give spike directions
    
    # Icosahedral directions in 4D
    directions = []
    
    # Type 1: (±1, ±1, ±1, ±1) normalized - 16 directions
    for w in [-1, 1]:
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    if (w + x + y + z) % 2 == 0:  # Even parity
                        directions.append([x, y, z, w])
    
    # Type 2: (±φ, ±1, ±1/φ, 0) and permutations - main icosahedral
    perms = [
        [PHI, 1, 1/PHI, 0],
        [1, 1/PHI, 0, PHI],
        [1/PHI, 0, PHI, 1],
        [0, PHI, 1, 1/PHI],
    ]
    
    for perm in perms:
        for s0 in [-1, 1]:
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    d_new = [perm[0]*s0, perm[1]*s1, perm[2]*s2, perm[3]]
                    directions.append(d_new)
                    # Also flip w
                    directions.append([d_new[0], d_new[1], d_new[2], -d_new[3]])
    
    # Normalize and create spikes
    for dir_list in directions[:32]:  # Limit for performance
        dir_vec = torch.tensor(dir_list, device=device, dtype=torch.float32)
        dir_vec = dir_vec / (torch.norm(dir_vec) + 1e-8)
        
        # Spike center (pushed outward)
        center = dir_vec * s * 0.35
        p_local = p - center
        
        # Projection along spike direction
        proj = (p_local * dir_vec).sum(dim=-1)
        
        # Perpendicular distance
        perp_sq = (p_local ** 2).sum(dim=-1) - proj ** 2
        perp = torch.sqrt(torch.clamp(perp_sq, min=1e-8))
        
        # Cone spike - tapers to point
        spike_length = s * 0.55
        taper = 0.18 * (spike_length - proj) / spike_length
        taper = torch.clamp(taper, min=0.01)
        
        cone = perp - taper
        cone = torch.where(proj > 0, cone, torch.full_like(cone, 1e10))
        cone = torch.where(proj < spike_length, cone, torch.full_like(cone, 1e10))
        
        d = torch.minimum(d, cone)
    
    return d


def stellated_icosahedral(p, s=0.5, n_spikes=30):
    """
    Stellated shape with icosahedral symmetry
    Cleaner, faster version
    """
    # Core
    d = torch.norm(p, dim=-1) - s * 0.2
    
    # Golden spiral spike distribution (approximates icosahedral)
    for i in range(n_spikes):
        # Fibonacci sphere point
        y = 1 - (2 * i / (n_spikes - 1))
        r = np.sqrt(1 - y * y)
        theta = np.pi * (1 + np.sqrt(5)) * i
        
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        
        # Extend to 4D with w variation
        w = np.sin(i * 0.5) * 0.5
        
        dir_vec = torch.tensor([x, y, z, w], device=device, dtype=torch.float32)
        dir_vec = dir_vec / torch.norm(dir_vec)
        
        center = dir_vec * s * 0.3
        p_local = p - center
        
        proj = (p_local * dir_vec).sum(dim=-1)
        perp = torch.sqrt(torch.clamp((p_local**2).sum(dim=-1) - proj**2, min=1e-8))
        
        # Sharp cone
        cone = perp - 0.12 * (s * 0.6 - proj)
        cone = torch.where((proj > 0) & (proj < s * 0.6), cone, torch.full_like(cone, 1e10))
        
        d = torch.minimum(d, cone)
    
    return d


def stellated_16cell_clean(p, s=0.55):
    """16-cell with sharp vertex extensions"""
    # 8 vertices at (±1,0,0,0) permutations
    d = torch.full(p.shape[:-1], 1e10, device=device)
    
    for axis in range(4):
        for sign in [-1.0, 1.0]:
            # Spike direction
            dir_vec = torch.zeros(4, device=device)
            dir_vec[axis] = sign
            
            # Spike from origin outward
            proj = (p * dir_vec).sum(dim=-1)
            perp = torch.sqrt(torch.clamp((p**2).sum(dim=-1) - proj**2, min=1e-8))
            
            # Long thin spike
            spike_len = s * 1.0
            taper = 0.08 * (spike_len - proj) / spike_len
            taper = torch.clamp(taper, min=0.01)
            
            cone = perp - taper
            cone = torch.where((proj > 0) & (proj < spike_len), cone, torch.full_like(cone, 1e10))
            
            d = torch.minimum(d, cone)
    
    # Small core sphere
    core = torch.norm(p, dim=-1) - s * 0.15
    d = torch.minimum(d, core)
    
    return d


def compound_5_24cells(p, s=0.5):
    """Compound of 5 24-cells - creates complex star"""
    d = torch.full(p.shape[:-1], 1e10, device=device)
    
    for k in range(5):
        angle = k * 2 * np.pi / 5
        c, ss = np.cos(angle), np.sin(angle)
        
        # Rotate in xw plane
        p_rot = p.clone()
        p_rot[..., 0] = p[..., 0] * c - p[..., 3] * ss
        p_rot[..., 3] = p[..., 0] * ss + p[..., 3] * c
        
        # 24-cell
        q = torch.abs(p_rot)
        d_inf = q.max(dim=-1).values
        sq = torch.sort(q, dim=-1, descending=True).values
        d_12 = (sq[..., 0] + sq[..., 1]) * 0.5
        d_24 = torch.maximum(d_inf, d_12) - s * 0.55
        
        d = torch.minimum(d, d_24)
    
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def render(sdf_fn, res=450, w_slice=0.0, R=None, hue=0.8):
    if R is None:
        R = rot4d(xw=0.4, yw=0.2, zw=0.15)
    R_inv = R.T
    
    fov, cam = 1.1, 2.6
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
    
    for step in range(120):
        if hit.all():
            break
        
        p = origins + t.unsqueeze(-1) * dirs
        p_rot = (p.view(-1, 4) @ R_inv).view(p.shape)
        d = sdf_fn(p_rot)
        
        new_hit = ~hit & (d < 0.0008)
        hit |= new_hit
        hit_pos[new_hit] = p[new_hit]
        
        glow += torch.exp(-torch.clamp(d, min=0) * 6) * 0.008 * (~hit).float()
        t[~hit] += d[~hit].clamp(min=0.0005) * 0.75
        t = torch.clamp(t, max=15.0)
    
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
    
    # Lighting
    L = torch.tensor([0.5, 0.75, -0.35, 0.15], device=device)
    L = L / torch.norm(L)
    diff = torch.clamp((normals * L).sum(dim=-1), 0.08, 1.0)
    
    V = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)
    H = L + V
    H = H / torch.norm(H)
    spec = torch.clamp((normals * H).sum(dim=-1), 0, 1) ** 48
    
    # Fresnel
    NdotV = torch.clamp(torch.abs((normals * V).sum(dim=-1)), 0, 1)
    fresnel = (1 - NdotV) ** 3
    
    w_n = torch.clamp((hit_pos[..., 3] + 0.8) / 1.6, 0, 1)
    fog = torch.exp(-t * 0.08)
    
    # To numpy
    hit_np = hit.cpu().numpy()
    diff_np = diff.cpu().numpy()
    spec_np = spec.cpu().numpy()
    fresnel_np = fresnel.cpu().numpy()
    w_np = w_n.cpu().numpy()
    fog_np = fog.cpu().numpy()
    glow_np = glow.cpu().numpy()
    
    img = np.zeros((res, res, 3))
    
    for y in range(res):
        for x in range(res):
            if hit_np[y, x]:
                # Deep purple stellated material (4D Toys style)
                h = (hue + w_np[y, x] * 0.06) % 1.0
                s = 0.6 + w_np[y, x] * 0.1
                v = 0.12 + 0.5 * diff_np[y, x]
                
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                
                # Specular (pink-white)
                r += spec_np[y, x] * 0.65
                g += spec_np[y, x] * 0.45
                b += spec_np[y, x] * 0.7
                
                # Fresnel rim (magenta)
                rim = fresnel_np[y, x] * 0.35
                r += rim * 0.5
                g += rim * 0.15
                b += rim * 0.55
                
                # Fog to dark background
                bg = [0.008, 0.004, 0.015]
                r = r * fog_np[y, x] + bg[0] * (1 - fog_np[y, x])
                g = g * fog_np[y, x] + bg[1] * (1 - fog_np[y, x])
                b = b * fog_np[y, x] + bg[2] * (1 - fog_np[y, x])
                
                img[y, x] = [r, g, b]
            else:
                # Background with purple glow
                gl = glow_np[y, x]
                gy = (y / res - 0.5) * 0.01
                img[y, x] = [
                    0.006 + gl * 0.12 + abs(gy),
                    0.003 + gl * 0.04,
                    0.012 + gl * 0.18 + abs(gy)
                ]
    
    return np.clip(img, 0, 1)


def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  STELLATED 4D POLYTOPES - 4D TOYS STYLE                   ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    
    # Render single frame of each
    shapes = [
        ("Great Stellated 120-cell", great_stellated_120cell, 0.83),
        ("Stellated Icosahedral", stellated_icosahedral, 0.8),
        ("Stellated 16-cell", stellated_16cell_clean, 0.77),
        ("Compound 5×24-cell", compound_5_24cells, 0.85),
    ]
    
    print("  Static renders:\n")
    for name, sdf_fn, hue in shapes:
        print(f"  {name}...")
        R = rot4d(xw=0.4, yw=0.25, zw=0.2)
        img = render(sdf_fn, res=450, w_slice=0.0, R=R, hue=hue)
        
        fname = name.lower().replace(' ', '_').replace('×', 'x')
        Image.fromarray((img * 255).astype(np.uint8)).save(f'{fname}.png')
        print(f"    ✓ {fname}.png  (hits: {(img.sum(axis=2) > 0.1).sum()})\n")
    
    # Main animation: Great Stellated 120-cell
    print("  ═══════════════════════════════════════════")
    print("  ANIMATING: Great Stellated 120-cell")
    print("  ═══════════════════════════════════════════\n")
    
    frames = []
    n_frames = 72
    
    for i in range(n_frames):
        t = i / n_frames
        
        # Smooth rotation
        xw = t * 2 * np.pi
        yw = np.sin(t * 3 * np.pi) * 0.25
        zw = t * np.pi * 0.6
        xy = np.sin(t * 2 * np.pi) * 0.15
        
        R = rot4d(xw=xw, yw=yw, zw=zw, xy=xy)
        
        # W-slice oscillation
        w = np.sin(t * 2 * np.pi) * 0.35
        
        if i % 12 == 0:
            print(f"    Frame {i+1:02d}/{n_frames} │ w={w:+.2f}")
        
        img = render(great_stellated_120cell, res=400, w_slice=w, R=R, hue=0.83)
        frames.append(Image.fromarray((img * 255).astype(np.uint8)))
    
    frames[0].save('stellated_120cell.gif', save_all=True, 
                   append_images=frames[1:], duration=45, loop=0)
    print(f"\n  ✓ Saved: stellated_120cell.gif")
    
    # Also save still frame
    frames[0].save('stellated_120cell.png')
    print(f"  ✓ Saved: stellated_120cell.png\n")
    
    # Bonus: morphing between shapes
    print("  ═══════════════════════════════════════════")
    print("  BONUS: Shape morphing animation")
    print("  ═══════════════════════════════════════════\n")
    
    def morph_sdf(p, blend):
        d1 = stellated_16cell_clean(p, 0.55)
        d2 = stellated_icosahedral(p, 0.5)
        return d1 * (1 - blend) + d2 * blend
    
    morph_frames = []
    for i in range(48):
        t = i / 48
        blend = (np.sin(t * 2 * np.pi) + 1) / 2
        
        R = rot4d(xw=t * 2 * np.pi, yw=0.2, zw=t * np.pi * 0.5)
        
        img = render(lambda p: morph_sdf(p, blend), res=380, w_slice=0, R=R, hue=0.78)
        morph_frames.append(Image.fromarray((img * 255).astype(np.uint8)))
        
        if i % 12 == 0:
            print(f"    Frame {i+1}/48")
    
    morph_frames[0].save('stellated_morph.gif', save_all=True,
                         append_images=morph_frames[1:], duration=50, loop=0)
    print(f"\n  ✓ Saved: stellated_morph.gif\n")
    
    print("  ═══════════════════════════════════════════")
    print("  COMPLETE!")
    print("  ═══════════════════════════════════════════")
    print("  Files created:")
    print("    • great_stellated_120-cell.png")
    print("    • stellated_icosahedral.png") 
    print("    • stellated_16-cell.png")
    print("    • compound_5x24-cell.png")
    print("    • stellated_120cell.gif (main animation)")
    print("    • stellated_morph.gif (bonus)")


if __name__ == "__main__":
    main()
