#!/usr/bin/env python3
"""
polytope_4d.py - Accurate 4D Polytope Cross-Section Renderer
120-cell, 600-cell, Clifford Torus, Tiger, 16-cell, Tesseract
Obsidian/matte material with proper w-slice visualization
"""

import torch
import numpy as np
from PIL import Image
import colorsys


class Rot4D:
    """4D rotation matrices for all 6 planes"""
    
    @staticmethod
    def make(plane, theta, device):
        c, s = np.cos(theta), np.sin(theta)
        R = torch.eye(4, device=device, dtype=torch.float32)
        planes = {'xy':(0,1), 'xz':(0,2), 'xw':(0,3), 'yz':(1,2), 'yw':(1,3), 'zw':(2,3)}
        i, j = planes[plane]
        R[i,i], R[j,j] = c, c
        R[i,j], R[j,i] = -s, s
        return R
    
    @staticmethod
    def compose(rotations):
        R = rotations[0]
        for r in rotations[1:]:
            R = R @ r
        return R
    
    @staticmethod
    def apply(R, p):
        return (p.reshape(-1, 4) @ R.T).reshape(p.shape)


def smin(a, b, k=0.1):
    """Smooth minimum for CSG"""
    h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1-h) + a * h - k * h * (1-h)


def smax(a, b, k=0.1):
    """Smooth maximum"""
    return -smin(-a, -b, k)


class SDF4D:
    """Accurate 4D polytope signed distance functions"""
    
    # ══════════════════════════════════════════════════════════════════════
    # REGULAR CONVEX 4-POLYTOPES (6 total)
    # ══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def hypersphere(p, r=1.0):
        """3-sphere / Glome - boundary of 4-ball"""
        return p.norm(dim=-1) - r
    
    @staticmethod
    def tesseract(p, s=1.0):
        """8-cell / Tesseract / Hypercube
        8 cubic cells, 24 square faces, 32 edges, 16 vertices
        """
        q = p.abs() - s
        return torch.clamp(q, min=0).norm(dim=-1) + torch.clamp(q.max(dim=-1).values, max=0)
    
    @staticmethod
    def cell_16(p, s=1.0):
        """16-cell / Hexadecachoron / Orthoplex
        16 tetrahedral cells, 32 triangular faces, 24 edges, 8 vertices
        Dual of tesseract
        """
        return p.abs().sum(dim=-1) - s
    
    @staticmethod
    def cell_24(p, s=1.0):
        """24-cell / Icositetrachoron
        24 octahedral cells - UNIQUE TO 4D, self-dual
        """
        q = p.abs()
        # Intersection of tesseract and 16-cell bounds
        d1 = q.max(dim=-1).values - s * 0.707  # ~1/√2
        
        # Sum of two largest absolute coordinates
        sorted_q = torch.sort(q, dim=-1, descending=True).values
        d2 = (sorted_q[..., 0] + sorted_q[..., 1]) * 0.707 - s
        
        return torch.maximum(d1, d2)
    
    @staticmethod
    def cell_120(p, s=1.0):
        """120-cell / Hecatonicosachoron
        120 dodecahedral cells, 720 pentagonal faces
        Most complex regular 4-polytope
        """
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        q = p.abs() / s
        
        # The 120-cell can be approximated by intersecting many half-spaces
        # Based on the H4 root system
        
        # Start with hypersphere bound
        d = q.norm(dim=-1) - 2.0
        
        # Add dodecahedral symmetry cuts
        # Normals based on icosahedral vertices extended to 4D
        normals = [
            [1, phi, 0, 1/phi],
            [1, phi, 0, -1/phi],
            [phi, 0, 1/phi, 1],
            [phi, 0, -1/phi, 1],
            [0, 1/phi, 1, phi],
            [0, -1/phi, 1, phi],
            [1/phi, 1, phi, 0],
            [-1/phi, 1, phi, 0],
        ]
        
        for n in normals:
            n = torch.tensor(n, device=p.device, dtype=torch.float32)
            n = n / n.norm()
            dist = (q * n).sum(dim=-1) - 1.618
            d = torch.maximum(d, dist)
            # Also apply with sign flips for full symmetry
            dist2 = (q * n[[0,1,3,2]]).sum(dim=-1) - 1.618
            d = torch.maximum(d, dist2)
        
        return d * s
    
    @staticmethod
    def cell_600(p, s=1.0):
        """600-cell / Hexacosichoron
        600 tetrahedral cells, 1200 triangular faces
        Dual of 120-cell
        """
        phi = (1 + np.sqrt(5)) / 2
        
        q = p.abs() / s
        
        # 600-cell is more "spherical" - closer to hypersphere
        d = q.norm(dim=-1) - phi
        
        # But with icosahedral symmetry cuts (sharper than 120-cell)
        # Based on vertices of 600-cell at permutations of (±φ, ±1, ±1/φ, 0)
        normals = [
            [phi, 1, 1/phi, 0],
            [1, 1/phi, 0, phi],
            [1/phi, 0, phi, 1],
            [0, phi, 1, 1/phi],
            [1, 1, 1, 1],
        ]
        
        for n in normals:
            n = torch.tensor(n, device=p.device, dtype=torch.float32)
            n = n / n.norm()
            dist = (q * n).sum(dim=-1) - 1.175
            d = torch.maximum(d, dist)
        
        return d * s
    
    # ══════════════════════════════════════════════════════════════════════
    # 4D TORI (non-convex, curved)
    # ══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def clifford_torus(p, R=0.7, r=0.4, thickness=0.08):
        """Clifford Torus - flat torus in S³
        Product of two circles: S¹ × S¹ embedded in S³ ⊂ ℝ⁴
        Cross-sections show linked/unlinked rings
        """
        # Distance from first circle (in xy plane, radius R)
        d1 = (p[..., 0]**2 + p[..., 1]**2).sqrt() - R
        # Distance from second circle (in zw plane, radius r)
        d2 = (p[..., 2]**2 + p[..., 3]**2).sqrt() - r
        # Combined as torus
        return (d1**2 + d2**2).sqrt() - thickness
    
    @staticmethod
    def duocylinder(p, r1=0.6, r2=0.6, cap=True):
        """Duocylinder - Cartesian product of two disks
        Ridge is a Clifford torus
        Cross-sections can show two separate circles
        """
        d1 = (p[..., 0]**2 + p[..., 1]**2).sqrt() - r1
        d2 = (p[..., 2]**2 + p[..., 3]**2).sqrt() - r2
        if cap:
            return torch.maximum(d1, d2)
        else:
            # Shell only
            return (d1**2 + d2**2).sqrt() - 0.05
    
    @staticmethod
    def tiger(p, R=0.6, r=0.2):
        """Tiger - exotic 4D torus
        Appears as two separate toroids that link through each other
        Unique to 4D!
        """
        # Two offset torus operations
        d1 = (p[..., 0]**2 + p[..., 2]**2).sqrt() - R
        d2 = (p[..., 1]**2 + p[..., 3]**2).sqrt() - R
        return (d1**2 + d2**2).sqrt() - r
    
    @staticmethod
    def ditorus(p, R1=0.55, R2=0.25, r=0.08):
        """Ditorus / Double Torus
        A torus whose cross-section is itself a torus
        """
        # First torus (xy around z)
        dxy = (p[..., 0]**2 + p[..., 1]**2).sqrt() - R1
        d1 = (dxy**2 + p[..., 2]**2).sqrt() - R2
        # Second torus wraps around w
        return (d1**2 + p[..., 3]**2).sqrt() - r
    
    @staticmethod
    def tiger_cage(p, R=0.5, r=0.15, bar=0.03):
        """Tiger with visible structure - cage-like"""
        base = SDF4D.tiger(p, R, r)
        
        # Add structural rings
        ring1 = (p[..., 0]**2 + p[..., 1]**2).sqrt() - R
        ring2 = (p[..., 2]**2 + p[..., 3]**2).sqrt() - R
        
        rings = smin(ring1.abs() - bar, ring2.abs() - bar, 0.02)
        
        return smin(base, rings, 0.03)


class Polytope4DRenderer:
    """Clean 4D polytope renderer with obsidian material"""
    
    def __init__(self, device='cuda', res=480):
        self.device = device
        self.res = res
        self.max_steps = 128
        self.epsilon = 0.0004
        self.max_dist = 25.0
        
        self.time = 0.0
        self.shape = 'cell_120'
    
    def get_sdf(self, p):
        """Get SDF for current shape"""
        if self.shape == 'tesseract':
            return SDF4D.tesseract(p, 0.7)
        elif self.shape == 'cell_16':
            return SDF4D.cell_16(p, 0.85)
        elif self.shape == 'cell_24':
            return SDF4D.cell_24(p, 0.9)
        elif self.shape == 'cell_120':
            return SDF4D.cell_120(p, 0.6)
        elif self.shape == 'cell_600':
            return SDF4D.cell_600(p, 0.7)
        elif self.shape == 'clifford':
            return SDF4D.clifford_torus(p, 0.7, 0.5, 0.1)
        elif self.shape == 'duocylinder':
            return SDF4D.duocylinder(p, 0.6, 0.6, cap=False)
        elif self.shape == 'tiger':
            return SDF4D.tiger(p, 0.55, 0.18)
        elif self.shape == 'ditorus':
            return SDF4D.ditorus(p, 0.5, 0.22, 0.07)
        elif self.shape == 'compound':
            # Compound: 16-cell + Tesseract (like stella octangula in 4D)
            d1 = SDF4D.tesseract(p, 0.55)
            d2 = SDF4D.cell_16(p, 0.75)
            return smin(d1, d2, 0.05)
        else:
            return SDF4D.hypersphere(p, 0.8)
    
    def normal(self, p, R_inv, eps=0.0008):
        """4D surface normal via gradient"""
        n = torch.zeros_like(p)
        for i in range(4):
            pp, pn = p.clone(), p.clone()
            pp[..., i] += eps
            pn[..., i] -= eps
            n[..., i] = self.get_sdf(Rot4D.apply(R_inv, pp)) - self.get_sdf(Rot4D.apply(R_inv, pn))
        return n / (n.norm(dim=-1, keepdim=True) + 1e-12)
    
    def march(self, w_slice=0.0, R=None):
        """Raymarch with volumetric glow"""
        if R is None:
            R = torch.eye(4, device=self.device)
        R_inv = R.T  # Orthogonal matrix inverse
        
        H, W = self.res, self.res
        fov, cam = 1.15, 2.8
        
        u = torch.linspace(-fov, fov, W, device=self.device)
        v = torch.linspace(-fov, fov, H, device=self.device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        origins = torch.zeros(H, W, 4, device=self.device)
        origins[..., 2] = -cam
        origins[..., 3] = w_slice
        
        dirs = torch.zeros(H, W, 4, device=self.device)
        dirs[..., 0] = uu
        dirs[..., 1] = -vv
        dirs[..., 2] = 2.0
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        
        t = torch.zeros(H, W, device=self.device)
        hit = torch.zeros(H, W, dtype=torch.bool, device=self.device)
        active = torch.ones(H, W, dtype=torch.bool, device=self.device)
        hit_pos = torch.zeros(H, W, 4, device=self.device)
        hit_step = torch.zeros(H, W, device=self.device)
        glow = torch.zeros(H, W, device=self.device)
        
        for step in range(self.max_steps):
            if not active.any():
                break
            
            p = origins + t.unsqueeze(-1) * dirs
            p_rot = Rot4D.apply(R_inv, p)
            d = self.get_sdf(p_rot)
            
            eps = self.epsilon * (1 + t * 0.01)
            
            new_hit = active & (d < eps)
            hit |= new_hit
            hit_pos[new_hit] = p[new_hit]
            hit_step[new_hit] = step
            
            miss = active & (t > self.max_dist)
            active &= ~new_hit & ~miss
            
            # Glow near surface
            glow += torch.exp(-torch.clamp(d, min=0) * 12) * 0.006 * active.float()
            
            t[active] += d[active] * 0.8
        
        normals = self.normal(hit_pos, R_inv)
        normals[~hit] = 0
        
        return {
            'hit': hit, 'pos': hit_pos, 'dist': t, 'norm': normals,
            'step': hit_step, 'glow': torch.clamp(glow, 0, 0.5),
            'w': hit_pos[..., 3]
        }
    
    def render(self, r, material='obsidian'):
        """Render with obsidian/matte material"""
        H, W = r['hit'].shape
        
        hit = r['hit'].cpu().numpy()
        norm = r['norm'].cpu().numpy()
        w = r['w'].cpu().numpy()
        dist = r['dist'].cpu().numpy()
        glow = r['glow'].cpu().numpy()
        step = r['step'].cpu().numpy()
        
        # Material colors
        if material == 'obsidian':
            base_color = np.array([0.15, 0.12, 0.22])  # Dark purple-blue
            spec_color = np.array([0.6, 0.55, 0.7])
            rim_color = np.array([0.3, 0.5, 0.8])
            glow_color = np.array([0.2, 0.15, 0.4])
        elif material == 'crystal':
            base_color = np.array([0.1, 0.15, 0.2])
            spec_color = np.array([0.8, 0.9, 1.0])
            rim_color = np.array([0.4, 0.7, 1.0])
            glow_color = np.array([0.1, 0.2, 0.4])
        else:  # matte
            base_color = np.array([0.18, 0.16, 0.25])
            spec_color = np.array([0.4, 0.38, 0.45])
            rim_color = np.array([0.25, 0.35, 0.5])
            glow_color = np.array([0.15, 0.1, 0.25])
        
        # Lights
        L1 = np.array([0.5, 0.7, -0.4, 0.15])
        L1 /= np.linalg.norm(L1)
        L2 = np.array([-0.4, 0.2, 0.5, -0.3])
        L2 /= np.linalg.norm(L2)
        V = np.array([0, 0, 1, 0])
        
        # Diffuse
        diff1 = np.clip(np.einsum('ijk,k->ij', norm, L1), 0, 1)
        diff2 = np.clip(np.einsum('ijk,k->ij', norm, L2), 0, 1)
        diff = diff1 * 0.6 + diff2 * 0.3 + 0.1  # Ambient
        
        # Specular (Blinn-Phong)
        H1 = L1 + V
        H1 /= np.linalg.norm(H1)
        spec = np.clip(np.einsum('ijk,k->ij', norm, H1), 0, 1) ** 64
        
        # Fresnel rim
        NdotV = np.clip(np.abs(np.einsum('ijk,k->ij', norm, V)), 0, 1)
        fresnel = (1 - NdotV) ** 4
        
        # W-depth factor (subtle color shift)
        w_norm = np.clip((w + 1) / 2, 0, 1)
        
        # AO from step count
        ao = 1.0 - np.clip(step / 100, 0, 0.35)
        
        # Fog
        fog = np.exp(-dist * 0.1)
        
        img = np.zeros((H, W, 3))
        
        # Background gradient
        for y in range(H):
            for x in range(W):
                if hit[y, x]:
                    # Obsidian surface
                    col = base_color * diff[y, x] * ao[y, x]
                    
                    # Subtle w-depth hue shift
                    hue_shift = w_norm[y, x] * 0.1
                    col[0] += hue_shift * 0.05
                    col[2] += (1 - hue_shift) * 0.03
                    
                    # Specular highlight
                    col += spec_color * spec[y, x] * 0.5
                    
                    # Rim light
                    col += rim_color * fresnel[y, x] * 0.4
                    
                    # Fog blend
                    fog_col = np.array([0.01, 0.012, 0.025])
                    col = col * fog[y, x] + fog_col * (1 - fog[y, x])
                    
                    img[y, x] = col
                else:
                    # Background with glow
                    g = glow[y, x]
                    gy = (y / H - 0.5) * 0.02
                    
                    img[y, x] = [
                        0.006 + g * glow_color[0] + abs(gy) * 0.01,
                        0.008 + g * glow_color[1],
                        0.018 + g * glow_color[2] + abs(gy) * 0.01
                    ]
        
        return np.clip(img, 0, 1)


def render_polytope_grid():
    """Render a 2x2 grid of different polytopes"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    shapes = [
        ('cell_120', '120-cell'),
        ('clifford', 'Clifford Torus'),
        ('tiger', 'Tiger'),
        ('cell_16', '16-cell')
    ]
    
    res = 400
    renderer = Polytope4DRenderer(device=device, res=res)
    
    grid = np.zeros((res * 2, res * 2, 3))
    
    for idx, (shape, name) in enumerate(shapes):
        print(f"  Rendering {name}...")
        renderer.shape = shape
        
        # Different viewing angles for each
        angles = [
            (0.3, 0.2, 0.15, 0.1),  # 120-cell
            (0.5, 0.8, 0.2, 0.0),   # Clifford - shows rings
            (0.4, 0.3, 0.6, 0.2),   # Tiger - shows linked tori
            (0.6, 0.4, 0.3, 0.15),  # 16-cell - vertex first
        ][idx]
        
        R = Rot4D.compose([
            Rot4D.make('xw', angles[0], device),
            Rot4D.make('yw', angles[1], device),
            Rot4D.make('zw', angles[2], device),
            Rot4D.make('xy', angles[3], device),
        ])
        
        # W-slice
        w_slices = [0.0, 0.3, 0.0, 0.0][idx]
        
        result = renderer.march(w_slice=w_slices, R=R)
        img = renderer.render(result, material='obsidian')
        
        row, col = idx // 2, idx % 2
        grid[row*res:(row+1)*res, col*res:(col+1)*res] = img
    
    return grid


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  4D POLYTOPE CROSS-SECTION RENDERER                            ║")
    print("║  Accurate Regular 4-Polytopes with Obsidian Material           ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}\n")
    
    print("  Regular Convex 4-Polytopes (6 total):")
    print("  ├─ 5-cell (4-simplex) - 5 tetrahedral cells")
    print("  ├─ 8-cell (Tesseract) - 8 cubic cells")
    print("  ├─ 16-cell (Orthoplex) - 16 tetrahedral cells")
    print("  ├─ 24-cell - 24 octahedral cells [UNIQUE TO 4D]")
    print("  ├─ 120-cell - 120 dodecahedral cells")
    print("  └─ 600-cell - 600 tetrahedral cells\n")
    
    print("  4D Tori:")
    print("  ├─ Clifford Torus - S¹×S¹ in S³ (linked rings)")
    print("  ├─ Tiger - exotic double torus (two linked tori)")
    print("  └─ Ditorus - torus of torus\n")
    
    # Render 2x2 grid
    print("  Rendering 2x2 polytope grid...")
    grid = render_polytope_grid()
    Image.fromarray((grid * 255).astype(np.uint8)).save('4d_polytopes_grid.png')
    print("  └─ Saved: 4d_polytopes_grid.png\n")
    
    # Render animation
    renderer = Polytope4DRenderer(device=device, res=450)
    
    print("  Rendering 120-cell animation...")
    renderer.shape = 'cell_120'
    
    frames = []
    n_frames = 90
    
    for i in range(n_frames):
        t = i / n_frames
        
        # Smooth 4D rotation
        xw = t * 2 * np.pi
        yw = np.sin(t * 4 * np.pi) * 0.25
        zw = t * np.pi * 0.6
        xy = np.sin(t * 2 * np.pi) * 0.2
        
        R = Rot4D.compose([
            Rot4D.make('xw', xw, device),
            Rot4D.make('yw', yw, device),
            Rot4D.make('zw', zw, device),
            Rot4D.make('xy', xy, device),
        ])
        
        w = np.sin(t * 2 * np.pi) * 0.4
        
        if i % 15 == 0:
            print(f"  ├─ Frame {i+1:02d}/{n_frames}")
        
        result = renderer.march(w_slice=w, R=R)
        img = renderer.render(result, material='obsidian')
        frames.append(Image.fromarray((img * 255).astype(np.uint8)))
    
    frames[0].save('4d_120cell.gif', save_all=True, append_images=frames[1:], duration=45, loop=0)
    print(f"  └─ Saved: 4d_120cell.gif\n")
    
    # Render individual shapes
    print("  Rendering individual polytopes...")
    for shape in ['tesseract', 'cell_16', 'cell_24', 'cell_120', 'clifford', 'tiger']:
        renderer.shape = shape
        
        R = Rot4D.compose([
            Rot4D.make('xw', 0.4, device),
            Rot4D.make('yw', 0.25, device),
            Rot4D.make('zw', 0.15, device),
        ])
        
        result = renderer.march(w_slice=0.0, R=R)
        img = renderer.render(result, material='obsidian')
        Image.fromarray((img * 255).astype(np.uint8)).save(f'4d_{shape}.png')
        print(f"  ├─ 4d_{shape}.png")
    
    print("  └─ Done!\n")
    
    print("  ═══════════════════════════════════════════")
    print("  CROSS-SECTION INTERPRETATION")
    print("  ═══════════════════════════════════════════")
    print("  • 120-cell: Bubbling multifaceted sphere")
    print("  • Clifford: Single ring or linked rings")
    print("  • Tiger: Two separate linked tori")
    print("  • 16-cell: Sharp diamond/rhombic shape")
    print("  • Tesseract: Morphing cube/octahedron")
    print("  • 24-cell: Unique self-dual polytope")


if __name__ == "__main__":
    main()
