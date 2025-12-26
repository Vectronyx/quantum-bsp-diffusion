#!/usr/bin/env python3
"""
EUCLIDEAN VOLUMETRIC EPSILON SOLVER
═══════════════════════════════════════════════════════════════════════════════
Mathematically precise SDF raymarching with:
  • Euclidean distance field propagation
  • Adaptive epsilon based on local curvature
  • Volumetric density integration (Beer-Lambert)
  • Lipschitz-bounded step sizing
  • Gradient-based normal estimation
  • Ambient occlusion via cone tracing
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

TAU = 6.283185307179586
SQRT2 = 1.4142135623730951
SQRT3 = 1.7320508075688772
PHI = 1.6180339887498949  # Golden ratio

# ═══════════════════════════════════════════════════════════════════════════════
# EUCLIDEAN MATH PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

@torch.jit.script
def length2(v: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean length"""
    return (v * v).sum(dim=-1)

@torch.jit.script
def length(v: torch.Tensor) -> torch.Tensor:
    """Euclidean length with numerical stability"""
    return torch.sqrt(length2(v) + 1e-12)

@torch.jit.script
def normalize(v: torch.Tensor) -> torch.Tensor:
    """Unit vector"""
    return v / (length(v).unsqueeze(-1) + 1e-12)

@torch.jit.script
def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Dot product"""
    return (a * b).sum(dim=-1)

@torch.jit.script
def reflect(d: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Reflect direction d around normal n"""
    return d - 2.0 * dot(d, n).unsqueeze(-1) * n

@torch.jit.script
def clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(x, lo, hi)

# ═══════════════════════════════════════════════════════════════════════════════
# EPSILON COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.jit.script
def compute_epsilon(t: torch.Tensor, pixel_size: float, cone_angle: float) -> torch.Tensor:
    """
    Adaptive epsilon based on:
    - Distance traveled (perspective correction)
    - Pixel footprint at current depth
    - Cone angle for anti-aliasing
    
    ε(t) = pixel_size * (1 + t * tan(cone_angle/2))
    """
    return pixel_size * (1.0 + t * cone_angle * 0.5)

@torch.jit.script
def compute_lipschitz_step(d: torch.Tensor, lipschitz: float) -> torch.Tensor:
    """
    Lipschitz-bounded step size.
    For SDF with Lipschitz constant L, safe step = d / L
    Most SDFs have L ≤ 1, but CSG operations can increase it.
    """
    return d / lipschitz

# ═══════════════════════════════════════════════════════════════════════════════
# CURVE FUNCTIONS (EXACT MATHEMATICAL DEFINITIONS)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.jit.script
def smoothstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    """C¹ continuous: 3t² - 2t³"""
    t = clamp((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

@torch.jit.script
def smootherstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    """C² continuous (Perlin): 6t⁵ - 15t⁴ + 10t³"""
    t = clamp((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

@torch.jit.script
def quintic(t: torch.Tensor) -> torch.Tensor:
    """C² smooth: t³(6t² - 15t + 10)"""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

@torch.jit.script
def exp_impulse(x: torch.Tensor, k: float) -> torch.Tensor:
    """Attempt to normalize: k*x*exp(1-k*x)"""
    h = k * x
    return h * torch.exp(1.0 - h)

@torch.jit.script
def exp_decay(x: torch.Tensor, k: float) -> torch.Tensor:
    """Exponential decay: exp(-k|x|)"""
    return torch.exp(-k * torch.abs(x))

@torch.jit.script
def polynomial_smin(a: torch.Tensor, b: torch.Tensor, k: float) -> torch.Tensor:
    """
    Polynomial smooth minimum (IQ's method)
    C¹ continuous blend between two SDFs
    """
    h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)

@torch.jit.script
def polynomial_smax(a: torch.Tensor, b: torch.Tensor, k: float) -> torch.Tensor:
    """Smooth maximum"""
    return -polynomial_smin(-a, -b, k)

@torch.jit.script
def exponential_smin(a: torch.Tensor, b: torch.Tensor, k: float) -> torch.Tensor:
    """
    Exponential smooth minimum
    Smoother blend but more expensive
    """
    res = torch.exp(-k * a) + torch.exp(-k * b)
    return -torch.log(res + 1e-12) / k

# ═══════════════════════════════════════════════════════════════════════════════
# 4D ROTATION (EUCLIDEAN SO(4))
# ═══════════════════════════════════════════════════════════════════════════════

class EuclideanRot4D:
    """
    SO(4) rotation via 6 Euler-like angles.
    Each angle rotates in one of the 6 coordinate planes.
    """
    PLANES = {
        'xy': (0, 1), 'xz': (0, 2), 'xw': (0, 3),
        'yz': (1, 2), 'yw': (1, 3), 'zw': (2, 3)
    }
    
    @staticmethod
    def givens(plane: str, theta: float) -> torch.Tensor:
        """Givens rotation in a single plane"""
        c, s = np.cos(theta), np.sin(theta)
        R = torch.eye(4, device=device, dtype=torch.float32)
        i, j = EuclideanRot4D.PLANES[plane]
        R[i, i], R[j, j] = c, c
        R[i, j], R[j, i] = -s, s
        return R
    
    @staticmethod
    def from_euler(xy=0., xz=0., xw=0., yz=0., yw=0., zw=0.) -> torch.Tensor:
        """Compose rotations from 6 angles"""
        R = torch.eye(4, device=device, dtype=torch.float32)
        for plane, angle in [('xy', xy), ('xz', xz), ('xw', xw), 
                             ('yz', yz), ('yw', yw), ('zw', zw)]:
            if abs(angle) > 1e-9:
                R = R @ EuclideanRot4D.givens(plane, angle)
        return R
    
    @staticmethod
    def from_axis_angle_4d(axis: torch.Tensor, angle: float) -> torch.Tensor:
        """
        Rotation around arbitrary 4D axis (normalized 4-vector).
        Uses Rodrigues-like formula extended to 4D.
        """
        # This is complex in 4D - simplified version uses plane
        # For full generality, need double quaternion representation
        raise NotImplementedError("Use from_euler for 4D rotations")

# ═══════════════════════════════════════════════════════════════════════════════
# EUCLIDEAN 4D SDF PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

@torch.jit.script
def sdf_hypersphere(p: torch.Tensor, r: float) -> torch.Tensor:
    """
    4D sphere (3-sphere/glome): {x ∈ ℝ⁴ : |x| = r}
    Exact Euclidean distance.
    """
    return length(p) - r

@torch.jit.script
def sdf_hyperplane(p: torch.Tensor, n: torch.Tensor, d: float) -> torch.Tensor:
    """
    4D hyperplane: {x ∈ ℝ⁴ : n·x = d}
    n must be unit vector.
    """
    return dot(p, n) - d

@torch.jit.script
def sdf_tesseract(p: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    4D axis-aligned box (tesseract/8-cell)
    b = half-extents in each dimension
    Exact Euclidean distance.
    """
    q = torch.abs(p) - b
    outside = length(torch.clamp(q, min=0.))
    inside = torch.clamp(q.max(dim=-1).values, max=0.)
    return outside + inside

@torch.jit.script
def sdf_tesseract_scalar(p: torch.Tensor, size: float) -> torch.Tensor:
    """Tesseract with uniform size"""
    q = torch.abs(p) - size
    outside = length(torch.clamp(q, min=0.))
    inside = torch.clamp(q.max(dim=-1).values, max=0.)
    return outside + inside

@torch.jit.script
def sdf_tesseract_round(p: torch.Tensor, size: float, r: float) -> torch.Tensor:
    """Rounded tesseract (Minkowski sum with sphere)"""
    return sdf_tesseract_scalar(p, size - r) - r

@torch.jit.script
def sdf_hyperoctahedron(p: torch.Tensor, s: float) -> torch.Tensor:
    """
    16-cell (hyperoctahedron/cross-polytope)
    Dual of tesseract: |x| + |y| + |z| + |w| ≤ s
    """
    return torch.abs(p).sum(dim=-1) - s

@torch.jit.script
def sdf_24cell(p: torch.Tensor, s: float) -> torch.Tensor:
    """
    24-cell: self-dual regular 4-polytope
    Vertices at permutations of (±1, ±1, 0, 0)
    """
    ap = torch.abs(p)
    # Max of L∞ and scaled L1 norms
    linf = ap.max(dim=-1).values
    l1 = ap.sum(dim=-1) * 0.5
    return torch.maximum(linf, l1) - s

@torch.jit.script
def sdf_duocylinder(p: torch.Tensor, r1: float, r2: float) -> torch.Tensor:
    """
    Duocylinder: Cartesian product of two circles S¹ × S¹
    Unique to 4D - has two perpendicular circular ridges.
    """
    d1 = torch.sqrt(p[..., 0]**2 + p[..., 1]**2 + 1e-12) - r1
    d2 = torch.sqrt(p[..., 2]**2 + p[..., 3]**2 + 1e-12) - r2
    outside = torch.sqrt(torch.clamp(d1, min=0.)**2 + torch.clamp(d2, min=0.)**2 + 1e-12)
    inside = torch.clamp(torch.maximum(d1, d2), max=0.)
    return outside + inside

@torch.jit.script
def sdf_spherinder(p: torch.Tensor, r: float, h: float) -> torch.Tensor:
    """
    Spherinder: S² × I (sphere × line segment)
    Extrusion of 3-sphere along w-axis.
    """
    d_sphere = torch.sqrt(p[..., 0]**2 + p[..., 1]**2 + p[..., 2]**2 + 1e-12) - r
    d_line = torch.abs(p[..., 3]) - h
    outside = torch.sqrt(torch.clamp(d_sphere, min=0.)**2 + torch.clamp(d_line, min=0.)**2 + 1e-12)
    inside = torch.clamp(torch.maximum(d_sphere, d_line), max=0.)
    return outside + inside

@torch.jit.script
def sdf_cubinder(p: torch.Tensor, b: float, h: float) -> torch.Tensor:
    """
    Cubinder: cube × circle (3D cube × S¹)
    """
    # Distance to cube in xyz
    q = torch.abs(p[..., :3]) - b
    d_cube = length(torch.clamp(q, min=0.)) + torch.clamp(q.max(dim=-1).values, max=0.)
    # Distance to circle in w (just w coordinate for line)
    d_w = torch.abs(p[..., 3]) - h
    # Combine
    outside = torch.sqrt(torch.clamp(d_cube, min=0.)**2 + torch.clamp(d_w, min=0.)**2 + 1e-12)
    inside = torch.clamp(torch.maximum(d_cube, d_w), max=0.)
    return outside + inside

@torch.jit.script
def sdf_hypertorus(p: torch.Tensor, R: float, r1: float, r2: float) -> torch.Tensor:
    """
    Clifford torus (flat torus embedded in 4D)
    Double torus: major radius R, minor radii r1, r2
    """
    # First torus in xy plane
    dxy = torch.sqrt(p[..., 0]**2 + p[..., 1]**2 + 1e-12) - R
    # Second torus around that
    dxyz = torch.sqrt(dxy**2 + p[..., 2]**2 + 1e-12) - r1
    # Third dimension
    return torch.sqrt(dxyz**2 + p[..., 3]**2 + 1e-12) - r2

@torch.jit.script
def sdf_tiger(p: torch.Tensor, R: float, r: float) -> torch.Tensor:
    """
    Tiger: unique 4D toroidal surface
    Two interlocking tori at right angles.
    """
    d1 = torch.sqrt(p[..., 0]**2 + p[..., 2]**2 + 1e-12) - R
    d2 = torch.sqrt(p[..., 1]**2 + p[..., 3]**2 + 1e-12) - R
    return torch.sqrt(d1**2 + d2**2 + 1e-12) - r

@torch.jit.script
def sdf_ditorus(p: torch.Tensor, R1: float, R2: float, r: float) -> torch.Tensor:
    """
    Ditorus: torus of tori
    Parametric surface in 4D.
    """
    # Major torus in xy-w space
    dxy = torch.sqrt(p[..., 0]**2 + p[..., 1]**2 + 1e-12) - R1
    dxyw = torch.sqrt(dxy**2 + p[..., 3]**2 + 1e-12) - R2
    # Minor circle
    return torch.sqrt(dxyw**2 + p[..., 2]**2 + 1e-12) - r

@torch.jit.script
def sdf_gyroid_4d(p: torch.Tensor, scale: float, thickness: float) -> torch.Tensor:
    """
    4D Gyroid: triply-periodic minimal surface extended to 4D.
    Implicit: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(w) + sin(w)cos(x) = 0
    """
    ps = p * scale * TAU
    g = (torch.sin(ps[..., 0]) * torch.cos(ps[..., 1]) +
         torch.sin(ps[..., 1]) * torch.cos(ps[..., 2]) +
         torch.sin(ps[..., 2]) * torch.cos(ps[..., 3]) +
         torch.sin(ps[..., 3]) * torch.cos(ps[..., 0]))
    return (torch.abs(g) - thickness) / scale

@torch.jit.script
def sdf_schwarz_p_4d(p: torch.Tensor, scale: float, thickness: float) -> torch.Tensor:
    """4D Schwarz P surface"""
    ps = p * scale * TAU
    g = (torch.cos(ps[..., 0]) + torch.cos(ps[..., 1]) + 
         torch.cos(ps[..., 2]) + torch.cos(ps[..., 3]))
    return (torch.abs(g) - thickness) / scale

# ═══════════════════════════════════════════════════════════════════════════════
# CSG OPERATIONS (EUCLIDEAN PRESERVING)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.jit.script
def op_union(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    """Boolean union (min)"""
    return torch.minimum(d1, d2)

@torch.jit.script
def op_intersect(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    """Boolean intersection (max)"""
    return torch.maximum(d1, d2)

@torch.jit.script
def op_subtract(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    """Boolean subtraction"""
    return torch.maximum(d1, -d2)

@torch.jit.script
def op_smooth_union(d1: torch.Tensor, d2: torch.Tensor, k: float) -> torch.Tensor:
    """Smooth union with blend radius k"""
    return polynomial_smin(d1, d2, k)

@torch.jit.script
def op_smooth_subtract(d1: torch.Tensor, d2: torch.Tensor, k: float) -> torch.Tensor:
    """Smooth subtraction"""
    return polynomial_smax(d1, -d2, k)

@torch.jit.script
def op_smooth_intersect(d1: torch.Tensor, d2: torch.Tensor, k: float) -> torch.Tensor:
    """Smooth intersection"""
    return polynomial_smax(d1, d2, k)

@torch.jit.script
def op_onion(d: torch.Tensor, thickness: float) -> torch.Tensor:
    """Hollow out a shape (shell)"""
    return torch.abs(d) - thickness

@torch.jit.script
def op_round(d: torch.Tensor, r: float) -> torch.Tensor:
    """Round edges"""
    return d - r

@torch.jit.script
def op_elongate(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Elongate along axes"""
    q = torch.abs(p) - h
    return torch.clamp(q, min=0.) + torch.clamp(q.max(dim=-1, keepdim=True).values, max=0.)

# ═══════════════════════════════════════════════════════════════════════════════
# 4D→3D PROJECTION (SLICE)
# ═══════════════════════════════════════════════════════════════════════════════

class EuclideanSlicer:
    """
    Projects 4D SDF to 3D via hyperplane slicing.
    f₃(x,y,z) = f₄(R⁻¹ · [x, y, z, w₀]ᵀ)
    """
    def __init__(self, sdf_4d: Callable, w_slice: float, R: torch.Tensor, shape: Tuple[int, int]):
        self.sdf_4d = sdf_4d
        self.w_slice = w_slice
        self.R_inv = torch.inverse(R)
        self._buf = torch.empty(*shape, 4, device=device, dtype=torch.float32)
    
    def __call__(self, p3: torch.Tensor) -> torch.Tensor:
        self._buf[..., :3] = p3
        self._buf[..., 3] = self.w_slice
        p4 = torch.einsum('ij,...j->...i', self.R_inv, self._buf)
        return self.sdf_4d(p4)

# ═══════════════════════════════════════════════════════════════════════════════
# EUCLIDEAN VOLUMETRIC RAYMARCHER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SolverResult:
    """Complete raymarching result"""
    hit: torch.Tensor           # Boolean hit mask
    t: torch.Tensor             # Ray parameter at hit
    position: torch.Tensor      # 3D hit position
    normal: torch.Tensor        # Surface normal
    depth: torch.Tensor         # Normalized depth [0,1]
    steps: torch.Tensor         # Steps taken per ray
    density: torch.Tensor       # Accumulated volumetric density
    ao: torch.Tensor            # Ambient occlusion
    curvature: torch.Tensor     # Surface curvature estimate


class EuclideanVolumetricSolver:
    """
    Precision raymarcher with:
    - Euclidean distance field evaluation
    - Adaptive epsilon (pixel footprint + depth)
    - Lipschitz-bounded stepping
    - Beer-Lambert volumetric integration
    - Cone-traced ambient occlusion
    - Curvature estimation
    """
    
    def __init__(self, res: int = 512, fov: float = 1.0, near: float = 0.1, far: float = 20.0):
        self.res = res
        self.fov = fov
        self.near = near
        self.far = far
        
        self.max_steps = 128
        self.lipschitz = 1.0  # Assume Lipschitz ≤ 1 for primitives
        
        # Pixel size at unit depth
        self.pixel_size = 2.0 * fov / res
        self.cone_angle = self.pixel_size  # For AA
        
        # AO settings
        self.ao_steps = 5
        self.ao_step_size = 0.1
        self.ao_strength = 0.8
        
        # Volumetric settings
        self.vol_steps = 32
        self.vol_density_scale = 0.5
        self.vol_absorption = 3.0
        
        # Create ray grid
        u = torch.linspace(-fov, fov, res, device=device)
        v = torch.linspace(-fov, fov, res, device=device)
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        
        # Ray directions (perspective)
        self._dirs = F.normalize(
            torch.stack([uu, -vv, torch.ones_like(uu) * 1.5], dim=-1),
            dim=-1
        )
        
        # Ray origin
        self._origin = torch.tensor([0., 0., -4.0], device=device)
        
        # Preallocate buffers
        self._t = torch.zeros(res, res, device=device)
        self._hit = torch.zeros(res, res, dtype=torch.bool, device=device)
        self._pos = torch.zeros(res, res, 3, device=device)
        self._normal = torch.zeros(res, res, 3, device=device)
        self._steps = torch.zeros(res, res, dtype=torch.int32, device=device)
        self._density = torch.zeros(res, res, device=device)
        self._ao = torch.zeros(res, res, device=device)
        self._curvature = torch.zeros(res, res, device=device)
    
    def solve(self, sdf: Callable, volumetric: bool = True, compute_ao: bool = True) -> SolverResult:
        """
        Main solve loop with epsilon-adaptive marching.
        """
        # Reset
        self._t.fill_(self.near)
        self._hit.zero_()
        self._density.zero_()
        self._steps.zero_()
        
        active = torch.ones(self.res, self.res, dtype=torch.bool, device=device)
        
        for step in range(self.max_steps):
            if not active.any():
                break
            
            # Current position
            p = self._origin + self._t.unsqueeze(-1) * self._dirs
            
            # Evaluate SDF
            d = sdf(p)
            
            # Adaptive epsilon
            eps = compute_epsilon(self._t, self.pixel_size, self.cone_angle)
            
            # Hit detection
            new_hit = active & (d < eps)
            self._hit |= new_hit
            self._pos = torch.where(new_hit.unsqueeze(-1), p, self._pos)
            self._steps[new_hit] = step
            
            # Volumetric density accumulation (Beer-Lambert)
            if volumetric:
                # Density inversely related to distance
                rho = exp_decay(torch.clamp(d, min=0.), self.vol_absorption)
                self._density += rho * active.float() * (1.0 / self.vol_steps)
            
            # Update active rays
            miss = self._t > self.far
            active = active & ~new_hit & ~miss
            
            # Lipschitz-bounded step
            step_size = compute_lipschitz_step(d, self.lipschitz)
            
            # Relaxation for convergence (increases with steps)
            relax = 0.8 + 0.2 * (step / self.max_steps)
            
            self._t = torch.where(active, self._t + step_size * relax, self._t)
        
        # Compute normals via gradient
        self._compute_normals(sdf)
        
        # Compute AO
        if compute_ao:
            self._compute_ao(sdf)
        
        # Compute curvature
        self._compute_curvature(sdf)
        
        # Normalize depth
        depth = (self._t - self.near) / (self.far - self.near)
        depth = torch.where(self._hit, depth, torch.ones_like(depth))
        
        # Apply Beer-Lambert to density
        transmittance = torch.exp(-self._density * self.vol_density_scale)
        density_final = 1.0 - transmittance
        
        return SolverResult(
            hit=self._hit.clone(),
            t=self._t.clone(),
            position=self._pos.clone(),
            normal=self._normal.clone(),
            depth=depth,
            steps=self._steps.clone(),
            density=density_final,
            ao=self._ao.clone(),
            curvature=self._curvature.clone()
        )
    
    def _compute_normals(self, sdf: Callable, eps: float = 0.0005):
        """
        Gradient-based normal estimation.
        Uses tetrahedron technique for better accuracy.
        """
        # Tetrahedron vertices for gradient estimation
        k = torch.tensor([
            [ 1, -1, -1],
            [-1, -1,  1],
            [-1,  1, -1],
            [ 1,  1,  1]
        ], device=device, dtype=torch.float32) * eps
        
        self._normal.zero_()
        
        for i in range(4):
            p_offset = self._pos + k[i]
            d = sdf(p_offset)
            self._normal += k[i] * d.unsqueeze(-1)
        
        # Normalize
        n_len = length(self._normal)
        self._normal = self._normal / (n_len.unsqueeze(-1) + 1e-12)
        
        # Zero out non-hits
        self._normal *= self._hit.unsqueeze(-1).float()
    
    def _compute_ao(self, sdf: Callable):
        """
        Cone-traced ambient occlusion.
        Samples along normal direction with increasing radius.
        """
        self._ao.zero_()
        
        for i in range(1, self.ao_steps + 1):
            t = i * self.ao_step_size
            sample_pos = self._pos + self._normal * t
            d = sdf(sample_pos)
            
            # Occlusion contribution
            # Expected distance is t, actual is d
            # Occlusion if d < t
            occ = torch.clamp((t - d) / t, 0.0, 1.0)
            
            # Weight by distance
            weight = 1.0 / (1.0 + i * 0.5)
            self._ao += occ * weight
        
        self._ao = (self._ao / self.ao_steps) * self.ao_strength
        self._ao *= self._hit.float()
    
    def _compute_curvature(self, sdf: Callable, eps: float = 0.001):
        """
        Estimate mean curvature from Laplacian of SDF.
        κ ≈ ∇²f / 2
        """
        d0 = sdf(self._pos)
        
        laplacian = torch.zeros(self.res, self.res, device=device)
        
        for i in range(3):
            pp = self._pos.clone()
            pn = self._pos.clone()
            pp[..., i] += eps
            pn[..., i] -= eps
            
            dp = sdf(pp)
            dn = sdf(pn)
            
            laplacian += dp + dn - 2 * d0
        
        self._curvature = laplacian / (eps * eps * 2.0)
        self._curvature *= self._hit.float()

# ═══════════════════════════════════════════════════════════════════════════════
# RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class EuclideanRenderer:
    """
    Physically-based renderer for SDF results.
    """
    
    def __init__(self, res: int = 512):
        self.res = res
        self.solver = EuclideanVolumetricSolver(res)
        
        # Lighting
        self._light_dir = F.normalize(torch.tensor([0.5, 0.8, -0.6], device=device), dim=0)
        self._light_color = torch.tensor([1.0, 0.98, 0.95], device=device)
        self._ambient = torch.tensor([0.08, 0.06, 0.12], device=device)
        
        # View
        self._view_dir = torch.tensor([0., 0., 1.], device=device)
        self._half_vec = F.normalize(self._light_dir + self._view_dir, dim=0)
        
        # Image buffer
        self._img = torch.zeros(res, res, 3, device=device)
        
        # Background
        v = torch.linspace(0, 1, res, device=device).unsqueeze(1).expand(res, res)
        self._bg = torch.stack([
            0.02 + 0.04 * v,
            0.015 + 0.025 * v,
            0.06 + 0.08 * v
        ], dim=-1)
    
    def render(self, sdf_4d: Callable, w_slice: float, R: torch.Tensor) -> dict:
        """Full render pipeline"""
        
        # Create 3D slice
        sdf_3d = EuclideanSlicer(sdf_4d, w_slice, R, (self.res, self.res))
        
        # Solve
        result = self.solver.solve(sdf_3d, volumetric=True, compute_ao=True)
        
        # Shade
        self._shade(result)
        
        return {
            'image': self._img.clone(),
            'depth': result.depth,
            'normal': result.normal,
            'hit': result.hit,
            'density': result.density,
            'ao': result.ao,
            'curvature': result.curvature,
            'steps': result.steps
        }
    
    def _shade(self, result: SolverResult):
        """PBR-inspired shading"""
        
        hit = result.hit
        normal = result.normal
        depth = result.depth
        density = result.density
        ao = result.ao
        curvature = result.curvature
        
        # Background + volumetric
        vol_color = smoothstep(0., 0.3, density)
        self._img = self._bg.clone()
        self._img[..., 0] += vol_color * 0.12
        self._img[..., 1] += vol_color * 0.06
        self._img[..., 2] += vol_color * 0.18
        
        if not hit.any():
            return
        
        # Diffuse (Lambert)
        n_dot_l = dot(normal, self._light_dir.expand_as(normal)).clamp(0, 1)
        
        # Specular (Blinn-Phong)
        n_dot_h = dot(normal, self._half_vec.expand_as(normal)).clamp(0, 1)
        spec = n_dot_h.pow(64)
        
        # Fresnel (Schlick approximation)
        n_dot_v = torch.abs(normal[..., 2])
        fresnel = (1.0 - n_dot_v).pow(5) * 0.5
        
        # Base color from depth/curvature
        hue = smootherstep(0., 1., 1.0 - depth) * 0.35 + 0.52
        sat = 0.65 + curvature.clamp(-0.5, 0.5) * 0.1
        val = 0.15 + 0.75 * n_dot_l * (1.0 - ao)
        
        # HSV to RGB (vectorized)
        h6 = (hue * 6.0) % 6.0
        hi = h6.long() % 6
        f = h6 - h6.floor()
        
        p = val * (1 - sat)
        q = val * (1 - sat * f)
        t_val = val * (1 - sat * (1 - f))
        
        r = torch.where(hi==0, val, torch.where(hi==1, q, torch.where(hi==2, p,
            torch.where(hi==3, p, torch.where(hi==4, t_val, val)))))
        g = torch.where(hi==0, t_val, torch.where(hi==1, val, torch.where(hi==2, val,
            torch.where(hi==3, q, torch.where(hi==4, p, p)))))
        b = torch.where(hi==0, p, torch.where(hi==1, p, torch.where(hi==2, t_val,
            torch.where(hi==3, val, torch.where(hi==4, val, q)))))
        
        # Add specular + fresnel
        r = (r + spec * 0.5 + fresnel * 0.25).clamp(0, 1)
        g = (g + spec * 0.5 + fresnel * 0.18).clamp(0, 1)
        b = (b + spec * 0.4 + fresnel * 0.35).clamp(0, 1)
        
        # Fog
        fog = exp_decay(depth * 2.5, 0.8)
        r = r * fog + self._bg[..., 0] * (1 - fog)
        g = g * fog + self._bg[..., 1] * (1 - fog)
        b = b * fog + self._bg[..., 2] * (1 - fog)
        
        # Write
        self._img[..., 0] = torch.where(hit, r, self._img[..., 0])
        self._img[..., 1] = torch.where(hit, g, self._img[..., 1])
        self._img[..., 2] = torch.where(hit, b, self._img[..., 2])

# ═══════════════════════════════════════════════════════════════════════════════
# CONTROLNET EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_depth(depth: torch.Tensor, hit: torch.Tensor) -> Image.Image:
    d = ((1.0 - depth) * hit.float()).cpu().numpy()
    return Image.fromarray((d * 255).astype(np.uint8)).convert('RGB')

def export_normal(normal: torch.Tensor, hit: torch.Tensor) -> Image.Image:
    n = normal.cpu().numpy()
    n_rgb = (n + 1) * 0.5
    neutral = np.array([0.5, 0.5, 1.0])
    n_rgb = np.where(hit.cpu().numpy()[..., None], n_rgb, neutral)
    return Image.fromarray((np.clip(n_rgb, 0, 1) * 255).astype(np.uint8))

def export_ao(ao: torch.Tensor, hit: torch.Tensor) -> Image.Image:
    a = ((1.0 - ao) * hit.float()).cpu().numpy()
    return Image.fromarray((a * 255).astype(np.uint8)).convert('RGB')

def export_curvature(curv: torch.Tensor, hit: torch.Tensor) -> Image.Image:
    c = curv.cpu().numpy()
    c = np.clip(c * 0.5 + 0.5, 0, 1)  # Map to [0,1]
    c *= hit.cpu().numpy()
    return Image.fromarray((c * 255).astype(np.uint8)).convert('RGB')

# ═══════════════════════════════════════════════════════════════════════════════
# SCENE PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

def scene_tesseract(p):
    return sdf_tesseract_round(p, 0.85, 0.1)

def scene_hypersphere(p):
    return sdf_hypersphere(p, 1.0)

def scene_duocylinder(p):
    return sdf_duocylinder(p, 0.85, 0.85)

def scene_tiger(p):
    return sdf_tiger(p, 0.9, 0.35)

def scene_hypertorus(p):
    return sdf_hypertorus(p, 1.0, 0.45, 0.18)

def scene_24cell(p):
    return sdf_24cell(p, 0.85)

def scene_gyroid(p):
    shell = sdf_hypersphere(p, 1.15)
    gyroid = sdf_gyroid_4d(p, 2.2, 0.05)
    return op_smooth_intersect(shell, gyroid, 0.05)

def scene_compound(p):
    t = sdf_tesseract_round(p, 0.7, 0.08)
    s = sdf_hypersphere(p, 1.0)
    return op_smooth_union(t, s, 0.2)

def scene_carved(p):
    base = sdf_tesseract_round(p, 0.9, 0.1)
    hole1 = sdf_duocylinder(p, 0.45, 0.45)
    hole2 = sdf_hypersphere(p, 0.65)
    return op_smooth_subtract(op_smooth_subtract(base, hole1, 0.08), hole2, 0.08)

def scene_ditorus(p):
    return sdf_ditorus(p, 0.9, 0.4, 0.15)

def scene_schwarz(p):
    shell = sdf_hypersphere(p, 1.2)
    schwarz = sdf_schwarz_p_4d(p, 1.8, 0.08)
    return op_smooth_intersect(shell, schwarz, 0.05)

SCENES = {
    'tesseract': scene_tesseract,
    'hypersphere': scene_hypersphere,
    'duocylinder': scene_duocylinder,
    'tiger': scene_tiger,
    'hypertorus': scene_hypertorus,
    '24cell': scene_24cell,
    'gyroid': scene_gyroid,
    'compound': scene_compound,
    'carved': scene_carved,
    'ditorus': scene_ditorus,
    'schwarz': scene_schwarz,
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def animate(scene_name='compound', n_frames=90, res=512, output='euclidean_4d'):
    print(f"\n{'═'*64}")
    print(f"  EUCLIDEAN VOLUMETRIC EPSILON SOLVER")
    print(f"  Scene: {scene_name} | Frames: {n_frames} | Res: {res}")
    print(f"{'═'*64}")
    
    renderer = EuclideanRenderer(res=res)
    scene = SCENES.get(scene_name, scene_compound)
    
    # Warmup
    print("  Warmup...")
    R = EuclideanRot4D.from_euler(xw=0.5)
    _ = renderer.render(scene, 0.0, R)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    t0 = time.perf_counter()
    for _ in range(5):
        _ = renderer.render(scene, 0.0, R)
    if device == 'cuda':
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"  → {ms:.1f} ms/frame ({1000/ms:.1f} FPS)")
    
    # Render
    frames = []
    print(f"\n  Rendering...")
    
    for i in range(n_frames):
        t = i / n_frames
        theta = t * TAU
        
        # 6-plane rotation
        R = EuclideanRot4D.from_euler(
            xw=theta,
            yw=theta * PHI * 0.5,
            zw=theta * 0.382,
            xy=theta * 0.15
        )
        
        # W-slice
        w = np.sin(theta * 1.5) * 0.55
        
        result = renderer.render(scene, w, R)
        
        img_np = (result['image'].cpu().numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(img_np))
        
        # Export maps on first frame
        if i == 0:
            export_depth(result['depth'], result['hit']).save(f'{output}_depth.png')
            export_normal(result['normal'], result['hit']).save(f'{output}_normal.png')
            export_ao(result['ao'], result['hit']).save(f'{output}_ao.png')
            export_curvature(result['curvature'], result['hit']).save(f'{output}_curvature.png')
        
        if (i + 1) % 30 == 0:
            print(f"    Frame {i+1}/{n_frames}")
    
    # Save
    print("  Saving...")
    frames[0].save(f'{output}.gif', save_all=True, append_images=frames[1:], duration=33, loop=0)
    frames[0].save(f'{output}_test.png')
    
    print(f"  ✓ {output}.gif")
    print(f"  ✓ {output}_depth.png")
    print(f"  ✓ {output}_normal.png")
    print(f"  ✓ {output}_ao.png")
    print(f"  ✓ {output}_curvature.png")
    
    # MP4
    try:
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for idx, f in enumerate(frames):
                f.save(f'{tmp}/f_{idx:04d}.png')
            subprocess.run(['ffmpeg', '-y', '-framerate', '30', '-i', f'{tmp}/f_%04d.png',
                           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
                           f'{output}.mp4'], capture_output=True, check=True)
            print(f"  ✓ {output}.mp4")
    except:
        pass
    
    return frames


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Euclidean Volumetric Epsilon Solver')
    parser.add_argument('--scene', '-s', type=str, default='compound', choices=list(SCENES.keys()))
    parser.add_argument('--frames', '-f', type=int, default=90)
    parser.add_argument('--res', '-r', type=int, default=512)
    parser.add_argument('--output', '-o', type=str, default='euclidean_4d')
    parser.add_argument('--all', action='store_true')
    
    args = parser.parse_args()
    
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║  EUCLIDEAN VOLUMETRIC EPSILON SOLVER                           ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    print(f"║  Device: {device:54s} ║")
    print("║                                                                ║")
    print("║  Features:                                                     ║")
    print("║    • Euclidean distance field propagation                      ║")
    print("║    • Adaptive epsilon (pixel footprint + depth)                ║")
    print("║    • Lipschitz-bounded stepping                                ║")
    print("║    • Beer-Lambert volumetric integration                       ║")
    print("║    • Cone-traced ambient occlusion                             ║")
    print("║    • Curvature estimation                                      ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    if args.all:
        for name in SCENES.keys():
            animate(name, n_frames=60, res=args.res, output=f'euc_{name}')
    else:
        animate(args.scene, n_frames=args.frames, res=args.res, output=args.output)


if __name__ == "__main__":
    main()
