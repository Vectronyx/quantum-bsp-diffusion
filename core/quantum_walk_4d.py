#!/usr/bin/env python3
"""
4D QUANTUM WALK - Carmack Engineering
Natural walking woman with 4D slice projection
SDF fiber connections, not fake overlays
"""

import torch
import gc
import os
import numpy as np
from PIL import Image
import subprocess

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════════════
# CARMACK FAST MATH
# ═══════════════════════════════════════════════════════════════════════════════

def fast_rsqrt(x):
    """Carmack-style fast inverse sqrt"""
    return 1.0 / np.sqrt(np.maximum(x, 1e-8))

def fast_normalize(v):
    """rsqrt normalization"""
    sq = np.sum(v * v, axis=-1, keepdims=True)
    return v * fast_rsqrt(sq)

def smoothstep(e0, e1, x):
    """C1 Hermite"""
    t = np.clip((x - e0) / (e1 - e0 + 1e-8), 0, 1)
    return t * t * (3 - 2 * t)

def smin(a, b, k=0.1):
    """Smooth minimum for SDF blending"""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)

# ═══════════════════════════════════════════════════════════════════════════════
# 4D SDF PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

def sdf_sphere_4d(p, center, r):
    """4D hypersphere"""
    return np.sqrt(np.sum((p - center)**2, axis=-1)) - r

def sdf_capsule_4d(p, a, b, r):
    """4D capsule for limbs"""
    pa = p - a
    ba = b - a
    h = np.clip(np.sum(pa * ba, axis=-1, keepdims=True) / 
                (np.sum(ba * ba) + 1e-8), 0, 1)
    return np.sqrt(np.sum((pa - ba * h)**2, axis=-1)) - r

def sdf_torus_4d(p, R, r):
    """4D torus slice"""
    dxy = np.sqrt(p[..., 0]**2 + p[..., 1]**2) - R
    return np.sqrt(dxy**2 + p[..., 2]**2 + p[..., 3]**2) - r

# ═══════════════════════════════════════════════════════════════════════════════
# WALKING POSE GENERATOR (procedural, not AI fake)
# ═══════════════════════════════════════════════════════════════════════════════

def walking_pose(t, w_slice=0.0):
    """
    Generate walking skeleton in 4D
    t: animation time [0,1]
    w_slice: 4th dimension slice
    Returns joint positions as 4D vectors
    """
    phase = t * 2 * np.pi
    
    # Base body
    hip_y = 0.0
    hip_bob = 0.02 * np.sin(phase * 2)  # Subtle vertical bob
    
    joints = {}
    
    # Spine
    joints['hip'] = np.array([0, hip_y + hip_bob, 0, w_slice])
    joints['spine'] = np.array([0, hip_y + 0.2 + hip_bob, 0, w_slice])
    joints['chest'] = np.array([0, hip_y + 0.4 + hip_bob * 0.8, 0, w_slice])
    joints['neck'] = np.array([0, hip_y + 0.55 + hip_bob * 0.6, 0, w_slice])
    joints['head'] = np.array([0, hip_y + 0.7 + hip_bob * 0.4, 0, w_slice])
    
    # Arms swing opposite to legs
    arm_swing = 0.15 * np.sin(phase)
    arm_w = 0.05 * np.sin(phase * 0.5)  # 4D arm movement
    
    joints['l_shoulder'] = np.array([-0.15, 0.45 + hip_bob * 0.7, 0, w_slice])
    joints['l_elbow'] = np.array([-0.18, 0.3 + hip_bob * 0.5, arm_swing * 0.5, w_slice + arm_w])
    joints['l_hand'] = np.array([-0.2, 0.15, arm_swing, w_slice + arm_w * 1.5])
    
    joints['r_shoulder'] = np.array([0.15, 0.45 + hip_bob * 0.7, 0, w_slice])
    joints['r_elbow'] = np.array([0.18, 0.3 + hip_bob * 0.5, -arm_swing * 0.5, w_slice - arm_w])
    joints['r_hand'] = np.array([0.2, 0.15, -arm_swing, w_slice - arm_w * 1.5])
    
    # Legs - proper walking cycle
    leg_phase_l = phase
    leg_phase_r = phase + np.pi
    
    # Left leg
    l_hip_angle = 0.3 * np.sin(leg_phase_l)
    l_knee_bend = 0.15 * (1 - np.cos(leg_phase_l)) * (np.sin(leg_phase_l) > 0)
    l_foot_lift = 0.08 * np.maximum(np.sin(leg_phase_l), 0)
    
    joints['l_hip'] = np.array([-0.08, -0.05 + hip_bob, 0, w_slice])
    joints['l_knee'] = np.array([-0.08, -0.3 + hip_bob * 0.3, l_hip_angle * 0.7, w_slice])
    joints['l_foot'] = np.array([-0.08, -0.55 + l_foot_lift, l_hip_angle, w_slice])
    
    # Right leg
    r_hip_angle = 0.3 * np.sin(leg_phase_r)
    r_knee_bend = 0.15 * (1 - np.cos(leg_phase_r)) * (np.sin(leg_phase_r) > 0)
    r_foot_lift = 0.08 * np.maximum(np.sin(leg_phase_r), 0)
    
    joints['r_hip'] = np.array([0.08, -0.05 + hip_bob, 0, w_slice])
    joints['r_knee'] = np.array([0.08, -0.3 + hip_bob * 0.3, r_hip_angle * 0.7, w_slice])
    joints['r_foot'] = np.array([0.08, -0.55 + r_foot_lift, r_hip_angle, w_slice])
    
    return joints

def sdf_walking_figure(p, t, w_slice=0.0):
    """
    Full body SDF with smooth capsule connections
    Carmack-style: smin blending, no hard edges
    """
    joints = walking_pose(t, w_slice)
    
    # Start with large distance
    d = np.full(p.shape[:-1], 1e10)
    
    # Head
    d = smin(d, sdf_sphere_4d(p, joints['head'], 0.08), 0.02)
    
    # Neck
    d = smin(d, sdf_capsule_4d(p, joints['neck'], joints['head'], 0.03), 0.02)
    
    # Torso
    d = smin(d, sdf_capsule_4d(p, joints['hip'], joints['chest'], 0.1), 0.03)
    d = smin(d, sdf_capsule_4d(p, joints['chest'], joints['neck'], 0.07), 0.02)
    
    # Left arm
    d = smin(d, sdf_capsule_4d(p, joints['l_shoulder'], joints['l_elbow'], 0.035), 0.02)
    d = smin(d, sdf_capsule_4d(p, joints['l_elbow'], joints['l_hand'], 0.03), 0.02)
    
    # Right arm
    d = smin(d, sdf_capsule_4d(p, joints['r_shoulder'], joints['r_elbow'], 0.035), 0.02)
    d = smin(d, sdf_capsule_4d(p, joints['r_elbow'], joints['r_hand'], 0.03), 0.02)
    
    # Left leg
    d = smin(d, sdf_capsule_4d(p, joints['l_hip'], joints['l_knee'], 0.05), 0.02)
    d = smin(d, sdf_capsule_4d(p, joints['l_knee'], joints['l_foot'], 0.04), 0.02)
    
    # Right leg
    d = smin(d, sdf_capsule_4d(p, joints['r_hip'], joints['r_knee'], 0.05), 0.02)
    d = smin(d, sdf_capsule_4d(p, joints['r_knee'], joints['r_foot'], 0.04), 0.02)
    
    return d

# ═══════════════════════════════════════════════════════════════════════════════
# 4D SLICE RENDERER (Carmack raymarching)
# ═══════════════════════════════════════════════════════════════════════════════

def render_4d_slice(sdf_func, res=512, w_slice=0.0, t=0.0):
    """
    Raymarch 4D scene with w-slice projection
    Returns depth, normal, and color
    """
    # Camera setup
    fov = 1.2
    cam_dist = 2.0
    
    u = np.linspace(-fov, fov, res)
    v = np.linspace(-fov, fov, res)
    uu, vv = np.meshgrid(u, v)
    
    # 4D ray origins
    origins = np.zeros((res, res, 4))
    origins[..., 2] = -cam_dist
    origins[..., 3] = w_slice
    
    # Ray directions
    dirs = np.zeros((res, res, 4))
    dirs[..., 0] = uu
    dirs[..., 1] = -vv
    dirs[..., 2] = 1.5
    dirs = fast_normalize(dirs)
    
    # Raymarch
    t_dist = np.zeros((res, res))
    hit = np.zeros((res, res), dtype=bool)
    max_steps = 64
    max_dist = 8.0
    epsilon = 0.002
    
    for step in range(max_steps):
        p = origins + t_dist[..., np.newaxis] * dirs
        d = sdf_func(p, t, w_slice)
        
        # Adaptive epsilon (Carmack)
        eps = epsilon * (1 + t_dist * 0.02)
        
        new_hit = (d < eps) & ~hit
        hit |= new_hit
        
        # Miss check
        missed = t_dist > max_dist
        
        # Step with relaxation
        relax = 0.85 + 0.15 * (step / max_steps)
        t_dist = np.where(hit | missed, t_dist, t_dist + d * relax)
    
    # Compute normals via gradient
    hit_pos = origins + t_dist[..., np.newaxis] * dirs
    normals = np.zeros((res, res, 4))
    
    eps_n = 0.005
    for i in range(4):
        p_pos = hit_pos.copy()
        p_neg = hit_pos.copy()
        p_pos[..., i] += eps_n
        p_neg[..., i] -= eps_n
        normals[..., i] = sdf_func(p_pos, t, w_slice) - sdf_func(p_neg, t, w_slice)
    
    normals = fast_normalize(normals)
    
    return hit, t_dist, hit_pos, normals

def shade_figure(hit, t_dist, hit_pos, normals, t=0.0):
    """
    Shade the walking figure with skin tones and cloth
    """
    res = hit.shape[0]
    img = np.zeros((res, res, 3))
    
    # 4D light direction
    light = np.array([0.4, 0.7, -0.5, 0.2])
    light = light / np.linalg.norm(light)
    
    # Diffuse
    diffuse = np.sum(normals * light, axis=-1)
    diffuse = np.clip(diffuse, 0.1, 1.0)
    
    # Specular
    view = np.array([0, 0, 1, 0])
    half_v = light + view
    half_v = half_v / np.linalg.norm(half_v)
    spec = np.sum(normals * half_v, axis=-1)
    spec = np.clip(spec, 0, 1) ** 32
    
    # Fresnel rim
    n_dot_v = np.abs(np.sum(normals * view, axis=-1))
    fresnel = (1 - n_dot_v) ** 3
    
    # W-coordinate for 4D color variation
    w_coord = hit_pos[..., 3]
    w_norm = smoothstep(-0.5, 0.5, w_coord)
    
    # Y-coordinate for body part detection
    y_coord = hit_pos[..., 1]
    
    # Base colors (skin and cloth)
    for y in range(res):
        for x in range(res):
            if hit[y, x]:
                yc = y_coord[y, x]
                
                # Head/neck: skin tone
                if yc > 0.45:
                    base_r, base_g, base_b = 0.92, 0.75, 0.65
                # Arms/hands: skin
                elif abs(hit_pos[y, x, 0]) > 0.12 and yc > 0:
                    base_r, base_g, base_b = 0.90, 0.73, 0.63
                # Torso: dark cloth
                elif yc > -0.1:
                    base_r, base_g, base_b = 0.15, 0.12, 0.18
                # Legs: dark pants
                else:
                    base_r, base_g, base_b = 0.12, 0.10, 0.14
                
                # Apply lighting
                d = diffuse[y, x]
                s = spec[y, x]
                f = fresnel[y, x]
                
                r = base_r * (0.3 + 0.7 * d) + s * 0.3 + f * 0.1
                g = base_g * (0.3 + 0.7 * d) + s * 0.25 + f * 0.05
                b = base_b * (0.3 + 0.7 * d) + s * 0.35 + f * 0.15
                
                # 4D color shift (subtle)
                w_shift = w_norm[y, x] * 0.05
                r += w_shift * 0.5
                b += w_shift
                
                # Depth fog
                fog = np.exp(-t_dist[y, x] * 0.15)
                bg = np.array([0.02, 0.015, 0.03])
                
                r = r * fog + bg[0] * (1 - fog)
                g = g * fog + bg[1] * (1 - fog)
                b = b * fog + bg[2] * (1 - fog)
                
                img[y, x] = [r, g, b]
            else:
                # Background gradient
                grad = y / res
                img[y, x] = [0.02 + grad * 0.01, 0.015 + grad * 0.005, 0.03 + grad * 0.02]
    
    return np.clip(img, 0, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  4D WALKING FIGURE - Carmack SDF Engineering                  ║")
    print("║  Natural motion, 4D slice projection, smooth fibers           ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")
    
    res = 480
    n_frames = 72
    
    os.makedirs('/tmp/walk4d', exist_ok=True)
    
    print(f"  Resolution: {res}x{res}")
    print(f"  Frames: {n_frames}")
    print(f"  SDF: smin-blended capsules (Carmack smooth)\n")
    
    for i in range(n_frames):
        t = i / n_frames
        
        # W-slice oscillates for 4D effect
        w = 0.15 * np.sin(t * 2 * np.pi)
        
        if i % 8 == 0:
            print(f"  Frame {i+1:02d}/{n_frames} │ t={t:.2f} │ w={w:+.3f}")
        
        # Render 4D slice
        hit, t_dist, hit_pos, normals = render_4d_slice(
            sdf_walking_figure, res=res, w_slice=w, t=t
        )
        
        # Shade
        img = shade_figure(hit, t_dist, hit_pos, normals, t)
        
        # Save
        img_u8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_u8).save(f'/tmp/walk4d/frame_{i:04d}.png')
    
    # Encode MP4
    print("\n  Encoding MP4...")
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '24',
        '-i', '/tmp/walk4d/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        'walk_4d.mp4'
    ], check=True, capture_output=True)
    
    # Save single frame
    Image.fromarray((shade_figure(*render_4d_slice(sdf_walking_figure, res, 0, 0.25)) * 255).astype(np.uint8)).save('walk_4d_frame.png')
    
    # Cleanup
    for f in os.listdir('/tmp/walk4d'):
        os.remove(f'/tmp/walk4d/{f}')
    os.rmdir('/tmp/walk4d')
    
    print(f"\n  ✓ walk_4d.mp4")
    print(f"  ✓ walk_4d_frame.png")
    print("\n  CARMACK ENGINEERING:")
    print("    ├─ fast_rsqrt()     │ inverse sqrt")
    print("    ├─ smin()           │ smooth CSG blend")
    print("    ├─ sdf_capsule_4d() │ limb connections")
    print("    ├─ adaptive epsilon │ distance-scaled")
    print("    └─ 4D w-slice       │ hyperplane cross-section")


if __name__ == "__main__":
    main()
