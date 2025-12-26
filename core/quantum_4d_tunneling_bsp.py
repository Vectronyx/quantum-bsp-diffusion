#!/usr/bin/env python3
"""
quantum_4d_polytope.py - Clean 4D Polytope Renderer
Tesseract, 16-cell, 24-cell, 120-cell projections with proper w-slicing
"""

import torch
import numpy as np
from PIL import Image
import colorsys


class Curves:
    @staticmethod
    def smoothstep(e0, e1, x):
        t = torch.clamp((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    
    @staticmethod
    def smin(a, b, k=0.1):
        h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
        return b * (1 - h) + a * h - k * h * (1 - h)


class Rot4D:
    """Clean 4D rotation matrices"""
    
    @staticmethod
    def xy(theta, device):
        c, s = np.cos(theta), np.sin(theta)
        return torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], device=device, dtype=torch.float32)
    
    @staticmethod
    def xz(theta, device):
        c, s = np.cos(theta), np.sin(theta)
        return torch.tensor([
            [c, 0, -s, 0],
            [0, 1, 0, 0],
            [s, 0, c, 0],
            [0, 0, 0, 1]
        ], device=device, dtype=torch.float32)
    
    @staticmethod
    def xw(theta, device):
        c, s = np.cos(theta), np.sin(theta)
        return torch.tensor([
            [c, 0, 0, -s],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [s, 0, 0, c]
        ], device=device, dtype=torch.float32)
    
    @staticmethod
    def yz(theta, device):
        c, s = np.cos(theta), np.sin(theta)
        return torch.tensor([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ], device=device, dtype=torch.float32)
    
    @staticmethod
    def yw(theta, device):
        c, s = np.cos(theta), np.sin(theta)
        return torch.tensor([
            [1, 0, 0, 0],
            [0, c, 0, -s],
            [0, 0, 1, 0],
            [0, s, 0, c]
        ], device=device, dtype=torch.float32)
    
    @staticmethod
    def zw(theta, device):
        c, s = np.cos(theta), np.sin(theta)
        return torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -s],
            [0, 0, s, c]
        ], device=device, dtype=torch.float32)
    
    @staticmethod
    def compose(rotations):
        R = rotations[0]
        for r in rotations[1:]:
            R = R @ r
        return R
    
    @staticmethod
    def apply(R, points):
        return (points.reshape(-1, 4) @ R.T).reshape(points.shape)


class SDF4D:
    """Clean 4D signed distance functions for regular polytopes"""
    
    @staticmethod
    def sphere(p, r=1.0):
        """4D hypersphere / 3-sphere / glome"""
        return p.norm(dim=-1) - r
    
    @staticmethod
    def tesseract(p, s=1.0):
        """8-cell / tesseract / hypercube"""
        q = p.abs() - s
        outside = torch.clamp(q, min=0).norm(dim=-1)
        inside = torch.clamp(q.max(dim=-1).values, max=0)
        return outside + inside
    
    @staticmethod
    def tesseract_edges(p, s=1.0, r=0.03):
        """Tesseract wireframe - edges only"""
        q = p.abs()
        
        # Count how many coordinates are near the edge (close to s)
        near_edge = (q - s).abs() < r * 3
        edge_count = near_edge.float().sum(dim=-1)
        
        # On an edge: 2 coords at ±s, 2 coords free
        # Distance to nearest edge
        d = SDF4D.tesseract(p, s)
        
        # Make it hollow - only show edges
        d_edges = d.abs() - r
        
        # Mask to edges (where at least 2 coords are at boundary)
        edge_mask = edge_count >= 2
        
        return torch.where(edge_mask, d_edges, torch.full_like(d, 1e10))
    
    @staticmethod  
    def cell16(p, s=1.0):
        """16-cell / hexadecachoron / 4D cross-polytope"""
        return p.abs().sum(dim=-1) - s
    
    @staticmethod
    def cell24(p, s=1.0):
        """24-cell / icositetrachoron (self-dual)"""
        # Vertices at permutations of (±1, ±1, 0, 0)
        q = p.abs()
        q_sorted = torch.sort(q, dim=-1, descending=True).values
        
        # The 24-cell is the intersection of a tesseract and 16-cell
        d1 = q.max(dim=-1).values - s  # Tesseract-like
        d2 = (q_sorted[..., 0] + q_sorted[..., 1]) - s * 1.414  # 16-cell-like
        
        return torch.maximum(d1, d2)
    
    @staticmethod
    def duocylinder(p, r1=1.0, r2=1.0):
        """Duocylinder - Cartesian product of two circles"""
        d1 = (p[..., 0]**2 + p[..., 1]**2).sqrt() - r1
        d2 = (p[..., 2]**2 + p[..., 3]**2).sqrt() - r2
        return torch.maximum(d1, d2)
    
    @staticmethod
    def clifford_torus(p, R=1.0, r=0.3):
        """Clifford torus - flat torus in 4D"""
        # Points satisfying x²+y² = R², z²+w² = r²
        d1 = (p[..., 0]**2 + p[..., 1]**2).sqrt() - R
        d2 = (p[..., 2]**2 + p[..., 3]**2).sqrt() - r
        return (d1**2 + d2**2).sqrt() - 0.15
    
    @staticmethod
    def tiger(p, R=0.8, r=0.25):
        """Tiger - a unique 4D torus-like shape"""
        d1 = (p[..., 0]**2 + p[..., 2]**2).sqrt() - R
        d2 = (p[..., 1]**2 + p[..., 3]**2).sqrt() - R
        return (d1**2 + d2**2).sqrt() - r
    
    @staticmethod
    def ditorus(p, R=0.7, r1=0.3, r2=0.1):
        """Double torus - torus of a torus"""
        # First torus in xy-z
        dxy = (p[..., 0]**2 + p[..., 1]**2).sqrt() - R
        d1 = (dxy**2 + p[..., 2]**2).sqrt() - r1
        # Second torus wraps in w
        return (d1**2 + p[..., 3]**2).sqrt() - r2


class Polytope4DRenderer:
    """4D polytope raymarcher with clean w-slice visualization"""
    
    def __init__(self, device='cuda', res=512):
        self.device = device
        self.res = res
        self.max_steps = 120
        self.epsilon = 0.0005
        self.max_dist = 20.0
        
        # Scene parameters
        self.time = 0.0
        self.shape = 'tesseract'  # Current shape
    
    def get_scene_sdf(self, p):
        """Scene with morphing between 4D polytopes"""
        t = self.time
        
        # Morphing weights
        morph = (np.sin(t * 0.5) + 1) * 0.5  # 0 to 1
        
        if self.shape == 'tesseract':
            d1 = SDF4D.tesseract(p, 0.7)
            d2 = SDF4D.cell16(p, 0.9)
            d = d1 * (1 - morph) + d2 * morph
            
        elif self.shape == 'cell24':
            d1 = SDF4D.cell24(p, 0.8)
            d2 = SDF4D.sphere(p, 0.9)
            d = Curves.smin(d1, -d2 + 0.2, 0.1)  # Carved 24-cell
            
        elif self.shape == 'duocylinder':
            d1 = SDF4D.duocylinder(p, 0.7, 0.7)
            d2 = SDF4D.duocylinder(p, 0.5, 0.5)
            d = torch.maximum(d1, -d2)  # Hollow duocylinder
            
        elif self.shape == 'clifford':
            d = SDF4D.clifford_torus(p, 0.8, 0.5)
            
        elif self.shape == 'tiger':
            d = SDF4D.tiger(p, 0.7, 0.2)
            
        elif self.shape == 'ditorus':
            d = SDF4D.ditorus(p, 0.6, 0.25, 0.08)
            
        elif self.shape == 'compound':
            # Compound of tesseract and 16-cell
            d1 = SDF4D.tesseract(p, 0.6)
            d2 = SDF4D.cell16(p, 0.85)
            d3 = SDF4D.sphere(p, 0.5)
            d = Curves.smin(d1, d2, 0.08)
            d = torch.maximum(d, -d3)  # Carve out center
            
        else:  # 'all' - morphing showcase
            phase = (t * 0.3) % (2 * np.pi)
            idx = int((phase / (2 * np.pi)) * 4) % 4
            blend = (phase % (np.pi / 2)) / (np.pi / 2)
            
            shapes = [
                SDF4D.tesseract(p, 0.7),
                SDF4D.cell16(p, 0.9),
                SDF4D.cell24(p, 0.75),
                SDF4D.clifford_torus(p, 0.8, 0.5)
            ]
            
            d = shapes[idx] * (1 - blend) + shapes[(idx + 1) % 4] * blend
        
        return d
    
    def compute_normal(self, p, R_inv, eps=0.001):
        """4D gradient for surface normal"""
        n = torch.zeros_like(p)
        for i in range(4):
            p_pos = p.clone()
            p_neg = p.clone()
            p_pos[..., i] += eps
            p_neg[..., i] -= eps
            d_pos = self.get_scene_sdf(Rot4D.apply(R_inv, p_pos))
            d_neg = self.get_scene_sdf(Rot4D.apply(R_inv, p_neg))
            n[..., i] = d_pos - d_neg
        return n / (n.norm(dim=-1, keepdim=True) + 1e-12)
    
    def march(self, w_slice=0.0, R=None):
        """Raymarch the 4D scene"""
        if R is None:
            R = torch.eye(4, device=self.device)
        R_inv = torch.inverse(R)
        
        H, W = self.res, self.res
        fov = 1.2
        cam_dist = 3.0
        
        # Ray setup
        u = torch.linspace(-fov, fov, W, device=self.device)
        v = torch.linspace(-fov, fov, H, device=self.device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        origins = torch.zeros(H, W, 4, device=self.device)
        origins[..., 2] = -cam_dist
        origins[..., 3] = w_slice
        
        dirs = torch.zeros(H, W, 4, device=self.device)
        dirs[..., 0] = uu
        dirs[..., 1] = -vv
        dirs[..., 2] = 1.8
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        
        # State
        t = torch.zeros(H, W, device=self.device)
        hit = torch.zeros(H, W, dtype=torch.bool, device=self.device)
        active = torch.ones(H, W, dtype=torch.bool, device=self.device)
        hit_pos = torch.zeros(H, W, 4, device=self.device)
        hit_steps = torch.zeros(H, W, device=self.device)
        
        # Volumetric
        glow = torch.zeros(H, W, device=self.device)
        
        for step in range(self.max_steps):
            if not active.any():
                break
            
            p = origins + t.unsqueeze(-1) * dirs
            p_rot = Rot4D.apply(R_inv, p)
            
            d = self.get_scene_sdf(p_rot)
            
            eps = self.epsilon * (1 + t * 0.015)
            
            new_hit = active & (d < eps)
            hit |= new_hit
            hit_pos[new_hit] = p[new_hit]
            hit_steps[new_hit] = step
            
            missed = active & (t > self.max_dist)
            active &= ~new_hit & ~missed
            
            # Glow accumulation
            glow += torch.exp(-d.clamp(min=0) * 8) * 0.008 * active.float()
            
            t[active] += d[active] * 0.85
        
        normals = self.compute_normal(hit_pos, R_inv)
        normals[~hit] = 0
        
        return {
            'hit': hit,
            'pos': hit_pos,
            'dist': t,
            'normal': normals,
            'steps': hit_steps,
            'glow': torch.clamp(glow, 0, 0.6),
            'w': hit_pos[..., 3]
        }
    
    def render(self, result):
        """Render to image with proper 4D lighting"""
        H, W = result['hit'].shape
        
        hit = result['hit'].cpu().numpy()
        norm = result['normal'].cpu().numpy()
        w = result['w'].cpu().numpy()
        dist = result['dist'].cpu().numpy()
        glow = result['glow'].cpu().numpy()
        steps = result['steps'].cpu().numpy()
        
        # Lighting setup
        light1 = np.array([0.6, 0.8, -0.4, 0.1])
        light1 /= np.linalg.norm(light1)
        light2 = np.array([-0.4, 0.3, 0.6, -0.3])
        light2 /= np.linalg.norm(light2)
        
        view = np.array([0, 0, 1, 0])
        
        # Compute lighting
        diff1 = np.clip(np.einsum('ijk,k->ij', norm, light1), 0, 1)
        diff2 = np.clip(np.einsum('ijk,k->ij', norm, light2), 0, 1)
        diff = diff1 * 0.65 + diff2 * 0.35
        
        # Specular
        half_vec = light1 + view
        half_vec /= np.linalg.norm(half_vec)
        spec = np.clip(np.einsum('ijk,k->ij', norm, half_vec), 0, 1) ** 48
        
        # Fresnel rim
        ndotv = np.clip(np.abs(np.einsum('ijk,k->ij', norm, view)), 0, 1)
        fresnel = (1 - ndotv) ** 3
        
        # W-coordinate coloring (4th dimension → hue)
        w_norm = np.clip((w + 1.2) / 2.4, 0, 1)
        
        # Fog
        fog = np.exp(-dist * 0.12)
        
        # AO approximation from step count
        ao = 1.0 - np.clip(steps / 80, 0, 0.4)
        
        img = np.zeros((H, W, 3))
        
        for y in range(H):
            for x in range(W):
                if hit[y, x]:
                    # Hue from w-coordinate
                    hue = (w_norm[y, x] * 0.25 + 0.5) % 1.0  # Cyan-blue-purple range
                    sat = 0.5 + 0.3 * w_norm[y, x]
                    val = (0.15 + 0.55 * diff[y, x]) * ao[y, x]
                    
                    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
                    
                    # Specular highlight (white)
                    r += spec[y, x] * 0.6
                    g += spec[y, x] * 0.6
                    b += spec[y, x] * 0.55
                    
                    # Rim light (cyan)
                    rim = fresnel[y, x] * 0.4
                    r += rim * 0.2
                    g += rim * 0.6
                    b += rim * 0.8
                    
                    # Fog blend
                    fog_col = [0.01, 0.015, 0.03]
                    r = r * fog[y, x] + fog_col[0] * (1 - fog[y, x])
                    g = g * fog[y, x] + fog_col[1] * (1 - fog[y, x])
                    b = b * fog[y, x] + fog_col[2] * (1 - fog[y, x])
                    
                    img[y, x] = [r, g, b]
                else:
                    # Background with glow
                    gl = glow[y, x]
                    
                    # Gradient background
                    gy = y / H
                    
                    img[y, x] = [
                        0.008 + gl * 0.15 + gy * 0.01,
                        0.012 + gl * 0.12,
                        0.025 + gl * 0.25 + gy * 0.015
                    ]
        
        return np.clip(img, 0, 1)


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  4D POLYTOPE RENDERER                                          ║")
    print("║  Regular 4D Polytopes with Clean W-Slice Visualization         ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}\n")
    
    renderer = Polytope4DRenderer(device=device, res=450)
    
    # Available shapes
    shapes = ['compound', 'tesseract', 'cell24', 'clifford', 'tiger', 'ditorus']
    
    print("  4D Polytopes:")
    print("  ├─ Tesseract (8-cell) - 4D hypercube")
    print("  ├─ 16-cell - 4D cross-polytope") 
    print("  ├─ 24-cell - self-dual regular polytope")
    print("  ├─ Duocylinder - product of circles")
    print("  ├─ Clifford torus - flat torus in S³")
    print("  ├─ Tiger - exotic 4D torus")
    print("  └─ Ditorus - double torus\n")
    
    frames = []
    n_frames = 90
    
    # Use 'compound' for a nice complex shape
    renderer.shape = 'compound'
    
    for i in range(n_frames):
        t = i / n_frames
        renderer.time = t * 2 * np.pi
        
        # Smooth double rotation
        # XW rotation - main 4D rotation
        xw = t * 2 * np.pi
        # YW rotation - secondary oscillation
        yw = np.sin(t * 4 * np.pi) * 0.3
        # ZW rotation - tertiary
        zw = t * np.pi * 0.5
        # XY rotation - 3D tumble
        xy = t * np.pi * 0.4
        
        R = Rot4D.compose([
            Rot4D.xw(xw, device),
            Rot4D.yw(yw, device),
            Rot4D.zw(zw, device),
            Rot4D.xy(xy, device)
        ])
        
        # W-slice oscillates through the shape
        w = np.sin(t * 2 * np.pi) * 0.6
        
        if i % 15 == 0:
            print(f"  ├─ Frame {i+1:02d}/{n_frames} │ w={w:+.2f} │ xw={xw:.2f} zw={zw:.2f}")
        
        result = renderer.march(w_slice=w, R=R)
        img = renderer.render(result)
        
        frames.append(Image.fromarray((img * 255).astype(np.uint8)))
    
    # Save animation
    frames[0].save('quantum_tunnel_4d.gif', save_all=True, 
                   append_images=frames[1:], duration=45, loop=0)
    frames[0].save('4d_polytope_test.png')
    
    print(f"  └─ ✓ Saved: quantum_tunnel_4d.gif\n")
    
    # Render individual shapes
    print("  Rendering individual polytopes...")
    for shape in ['tesseract', 'cell24', 'clifford', 'tiger']:
        renderer.shape = shape
        renderer.time = 0
        R = Rot4D.compose([
            Rot4D.xw(0.4, device),
            Rot4D.yw(0.2, device),
            Rot4D.xy(0.3, device)
        ])
        result = renderer.march(w_slice=0.0, R=R)
        img = renderer.render(result)
        Image.fromarray((img * 255).astype(np.uint8)).save(f'4d_{shape}.png')
        print(f"  ├─ 4d_{shape}.png")
    
    print("  └─ Done!\n")
    
    print("  ═══════════════════════════════════════════")
    print("  4D ROTATION PLANES")
    print("  ═══════════════════════════════════════════")
    print("  XY, XZ, YZ - familiar 3D rotations")
    print("  XW, YW, ZW - rotations INTO the 4th dimension")
    print()
    print("  W-SLICING:")
    print("  Like slicing a 3D object to reveal 2D cross-sections,")
    print("  slicing a 4D object reveals 3D cross-sections that")
    print("  morph as we move the slice through w-space.")


if __name__ == "__main__":
    main()
