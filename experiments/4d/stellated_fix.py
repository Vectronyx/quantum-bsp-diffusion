#!/usr/bin/env python3
"""
stellated_fix.py - Fixed Stellated 4D Polytopes
Continuous SDFs with properly attached spikes
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

def smin(a, b, k=0.1):
    """Smooth minimum - creates smooth blend between shapes"""
    h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1-h) + a * h - k * h * (1-h)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXED STELLATED SDFs - CONTINUOUS WITH SMOOTH CONNECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def sphere(p, r):
    return torch.norm(p, dim=-1) - r

def cone_spike(p, tip, direction, length, base_radius):
    """
    Cone spike SDF that connects smoothly to origin
    tip: where spike tip is
    direction: normalized direction of spike
    length: how long the spike is
    base_radius: radius at base of spike
    """
    # Vector from base to point
    base = tip - direction * length
    p_local = p - base
    
    # Project onto spike axis
    proj = (p_local * direction).sum(dim=-1)
    proj_clamped = torch.clamp(proj, 0, length)
    
    # Perpendicular distance
    closest_on_axis = base + direction * proj_clamped.unsqueeze(-1)
    perp = torch.norm(p - closest_on_axis, dim=-1)
    
    # Radius tapers from base_radius to 0
    taper_ratio = 1 - proj_clamped / length
    current_radius = base_radius * taper_ratio
    
    # Distance to cone surface
    d_cone = perp - current_radius
    
    # Cap the ends
    d_base = -proj  # Negative inside, positive outside base plane
    d_tip = proj - length  # Positive beyond tip
    
    # Combine: inside cone if d_cone < 0 AND between base and tip
    d = torch.maximum(d_cone, torch.maximum(d_base, d_tip))
    
    return d

def stellated_sphere(p, core_r=0.2, spike_len=0.4, spike_r=0.08, n_spikes=20):
    """
    Sphere with smoothly attached conical spikes
    Uses smooth union for seamless connection
    """
    # Core sphere
    d = sphere(p, core_r)
    
    # Add spikes with golden spiral distribution
    phi = (1 + np.sqrt(5)) / 2
    
    for i in range(n_spikes):
        # Fibonacci sphere distribution
        y = 1 - (2.0 * i / (n_spikes - 1))
        radius_at_y = np.sqrt(max(0, 1 - y*y))
        theta = 2 * np.pi * i / phi
        
        x = radius_at_y * np.cos(theta)
        z = radius_at_y * np.sin(theta)
        
        # Add 4D component
        w = np.sin(i * 0.7) * 0.4
        
        # Normalize direction
        dir_4d = torch.tensor([x, y, z, w], device=device, dtype=torch.float32)
        dir_4d = dir_4d / torch.norm(dir_4d)
        
        # Spike tip position
        tip = dir_4d * (core_r + spike_len)
        
        # Create cone spike
        d_spike = cone_spike(p, tip, dir_4d, spike_len, spike_r)
        
        # Smooth union with core - THIS IS THE KEY
        d = smin(d, d_spike, 0.04)
    
    return d

def stellated_16cell(p, s=0.5):
    """
    16-cell (orthoplex) with smoothly attached vertex spikes
    8 spikes along ±x, ±y, ±z, ±w
    """
    # Core: small 16-cell
    core_size = s * 0.25
    d = torch.sum(torch.abs(p), dim=-1) - core_size
    
    # 8 spikes along each axis direction
    spike_len = s * 0.7
    spike_r = s * 0.06
    
    for axis in range(4):
        for sign in [-1.0, 1.0]:
            # Direction along axis
            dir_vec = torch.zeros(4, device=device)
            dir_vec[axis] = sign
            
            # Tip position
            tip = dir_vec * (core_size * 0.5 + spike_len)
            
            # Cone spike
            d_spike = cone_spike(p, tip, dir_vec, spike_len, spike_r)
            
            # Smooth union
            d = smin(d, d_spike, 0.03)
    
    return d

def stellated_tesseract(p, s=0.5):
    """
    Tesseract with pyramidal spikes on each of 8 cubic cells
    Spikes point along ±x, ±y, ±z, ±w (same as 16-cell but with cube core)
    """
    # Core tesseract
    core_size = s * 0.3
    q = torch.abs(p) - core_size
    d = torch.norm(torch.clamp(q, min=0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0)
    
    # Spikes
    spike_len = s * 0.5
    spike_r = s * 0.1
    
    for axis in range(4):
        for sign in [-1.0, 1.0]:
            dir_vec = torch.zeros(4, device=device)
            dir_vec[axis] = sign
            
            tip = dir_vec * (core_size + spike_len)
            d_spike = cone_spike(p, tip, dir_vec, spike_len, spike_r)
            
            d = smin(d, d_spike, 0.04)
    
    return d

def stellated_24cell(p, s=0.5):
    """
    24-cell with spikes at 24 vertices
    Vertices at permutations of (±1, ±1, 0, 0)
    """
    # Core 24-cell
    q = torch.abs(p)
    d_inf = q.max(dim=-1).values
    sq = torch.sort(q, dim=-1, descending=True).values
    d_12 = (sq[..., 0] + sq[..., 1]) * 0.5
    d = torch.maximum(d_inf, d_12) - s * 0.3
    
    # Spikes at 24 vertices
    spike_len = s * 0.35
    spike_r = s * 0.04
    
    for i in range(4):
        for j in range(i + 1, 4):
            for si in [-1.0, 1.0]:
                for sj in [-1.0, 1.0]:
                    # Vertex position
                    v = torch.zeros(4, device=device)
                    v[i] = si
                    v[j] = sj
                    v = v / torch.norm(v)  # Normalize
                    
                    tip = v * (s * 0.35 + spike_len)
                    d_spike = cone_spike(p, tip, v, spike_len, spike_r)
                    
                    d = smin(d, d_spike, 0.025)
    
    return d

def great_stellated_120cell(p, s=0.5):
    """
    Great Stellated 120-cell approximation
    Core sphere with many spikes in H4 symmetry pattern
    """
    PHI = (1 + np.sqrt(5)) / 2
    
    # Core sphere
    d = sphere(p, s * 0.2)
    
    spike_len = s * 0.5
    spike_r = s * 0.035
    
    # Generate directions using icosahedral symmetry extended to 4D
    directions = []
    
    # 8 axis directions
    for axis in range(4):
        for sign in [-1, 1]:
            v = [0, 0, 0, 0]
            v[axis] = sign
            directions.append(v)
    
    # Icosahedral-like directions
    perms = [
        [PHI, 1, 1/PHI, 0],
        [1, PHI, 0, 1/PHI],
        [1/PHI, 0, PHI, 1],
        [0, 1/PHI, 1, PHI],
    ]
    
    for perm in perms:
        for s0 in [-1, 1]:
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    directions.append([perm[0]*s0, perm[1]*s1, perm[2]*s2, perm[3]])
                    directions.append([perm[0]*s0, perm[1]*s1, perm[2]*s2, -perm[3]])
    
    # Use subset for performance
    for dir_list in directions[:40]:
        dir_vec = torch.tensor(dir_list, device=device, dtype=torch.float32)
        dir_vec = dir_vec / (torch.norm(dir_vec) + 1e-8)
        
        tip = dir_vec * (s * 0.2 + spike_len)
        d_spike = cone_spike(p, tip, dir_vec, spike_len, spike_r)
        
        d = smin(d, d_spike, 0.02)
    
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# MORPH FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def morph_stellated(p, t):
    """
    Smooth morph between stellated shapes
    t: 0 = stellated sphere, 0.5 = stellated 16-cell, 1 = back to sphere
    """
    # Use smooth morphing via SDF blending
    cycle = t * 2  # 0 to 2
    
    if cycle < 1:
        # Morph from stellated sphere to stellated 16-cell
        blend = cycle
        d1 = stellated_sphere(p, core_r=0.18, spike_len=0.35, spike_r=0.06, n_spikes=20)
        d2 = stellated_16cell(p, s=0.5)
    else:
        # Morph back
        blend = cycle - 1
        d1 = stellated_16cell(p, s=0.5)
        d2 = stellated_sphere(p, core_r=0.18, spike_len=0.35, spike_r=0.06, n_spikes=20)
    
    # Smooth interpolation
    blend_smooth = blend * blend * (3 - 2 * blend)  # Smoothstep
    return d1 * (1 - blend_smooth) + d2 * blend_smooth


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def render(sdf_fn, res=420, w_slice=0.0, R=None, hue=0.8):
    if R is None:
        R = rot4d(xw=0.4, yw=0.2, zw=0.15)
    R_inv = R.T
    
    fov, cam = 1.05, 2.5
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
        
        new_hit = ~hit & (d < 0.0008)
        hit |= new_hit
        hit_pos[new_hit] = p[new_hit]
        
        glow += torch.exp(-torch.clamp(d, min=0) * 5) * 0.008 * (~hit).float()
        t[~hit] += d[~hit].clamp(min=0.0005) * 0.75
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
                # Purple stellated material
                h = (hue + w_np[y, x] * 0.06) % 1.0
                s = 0.55 + w_np[y, x] * 0.12
                v = 0.12 + 0.52 * diff_np[y, x]
                
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                
                # Specular
                r += spec_np[y, x] * 0.6
                g += spec_np[y, x] * 0.4
                b += spec_np[y, x] * 0.65
                
                # Fresnel rim
                rim = fresnel_np[y, x] * 0.3
                r += rim * 0.45
                g += rim * 0.15
                b += rim * 0.5
                
                # Fog
                bg = [0.006, 0.003, 0.012]
                r = r * fog_np[y, x] + bg[0] * (1 - fog_np[y, x])
                g = g * fog_np[y, x] + bg[1] * (1 - fog_np[y, x])
                b = b * fog_np[y, x] + bg[2] * (1 - fog_np[y, x])
                
                img[y, x] = [r, g, b]
            else:
                gl = glow_np[y, x]
                img[y, x] = [0.005 + gl * 0.1, 0.002 + gl * 0.04, 0.01 + gl * 0.15]
    
    return np.clip(img, 0, 1)


def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  FIXED STELLATED 4D POLYTOPES                             ║")
    print("║  Smooth connections, no gaps                              ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    
    # Test each shape
    shapes = [
        ("Stellated Sphere", lambda p: stellated_sphere(p, 0.18, 0.38, 0.07, 24), 0.82),
        ("Stellated 16-cell", lambda p: stellated_16cell(p, 0.5), 0.77),
        ("Stellated Tesseract", lambda p: stellated_tesseract(p, 0.48), 0.75),
        ("Stellated 24-cell", lambda p: stellated_24cell(p, 0.5), 0.8),
        ("Great Stellated 120-cell", lambda p: great_stellated_120cell(p, 0.5), 0.83),
    ]
    
    print("  Rendering static frames:\n")
    for name, sdf_fn, hue in shapes:
        print(f"  {name}...", end=" ", flush=True)
        R = rot4d(xw=0.4, yw=0.25, zw=0.18)
        img = render(sdf_fn, res=400, w_slice=0.0, R=R, hue=hue)
        
        fname = name.lower().replace(' ', '_').replace('-', '')
        Image.fromarray((img * 255).astype(np.uint8)).save(f'{fname}.png')
        print(f"✓ {fname}.png")
    
    print("\n  ═══════════════════════════════════════════")
    print("  MORPH ANIMATION (fixed - no gaps)")
    print("  ═══════════════════════════════════════════\n")
    
    frames = []
    n_frames = 60
    
    for i in range(n_frames):
        t = i / n_frames
        
        R = rot4d(
            xw=t * 2 * np.pi,
            yw=np.sin(t * 3 * np.pi) * 0.2,
            zw=t * np.pi * 0.5,
            xy=np.sin(t * 2 * np.pi) * 0.12
        )
        
        w = np.sin(t * 2 * np.pi) * 0.25
        
        if i % 10 == 0:
            print(f"    Frame {i+1:02d}/{n_frames} │ morph={t:.2f} │ w={w:+.2f}")
        
        img = render(lambda p: morph_stellated(p, t), res=380, w_slice=w, R=R, hue=0.8)
        frames.append(Image.fromarray((img * 255).astype(np.uint8)))
    
    frames[0].save('stellated_morph_fixed.gif', save_all=True,
                   append_images=frames[1:], duration=50, loop=0)
    print(f"\n  ✓ Saved: stellated_morph_fixed.gif")
    
    # Main 120-cell animation
    print("\n  ═══════════════════════════════════════════")
    print("  120-CELL ANIMATION")
    print("  ═══════════════════════════════════════════\n")
    
    frames2 = []
    for i in range(72):
        t = i / 72
        
        R = rot4d(xw=t * 2 * np.pi, yw=np.sin(t * 4 * np.pi) * 0.22, 
                  zw=t * np.pi * 0.65)
        w = np.sin(t * 2 * np.pi) * 0.3
        
        if i % 12 == 0:
            print(f"    Frame {i+1:02d}/72")
        
        img = render(lambda p: great_stellated_120cell(p, 0.5), res=420, 
                    w_slice=w, R=R, hue=0.83)
        frames2.append(Image.fromarray((img * 255).astype(np.uint8)))
    
    frames2[0].save('great_stellated_120cell.gif', save_all=True,
                    append_images=frames2[1:], duration=45, loop=0)
    print(f"\n  ✓ Saved: great_stellated_120cell.gif\n")
    
    print("  ═══════════════════════════════════════════")
    print("  COMPLETE!")
    print("  ═══════════════════════════════════════════")


if __name__ == "__main__":
    main()
