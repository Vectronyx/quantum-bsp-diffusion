#!/usr/bin/env python3
"""
QUANTUM SPACE PARTITION - SQRT ASSIMILATION DECREMENT PROTOCOL
Root-refined BSP matrix with quantum state propagation
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable
from enum import Enum
from PIL import Image
import colorsys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ═══════════════════════════════════════════════════════════════════════════════
# SQRT ASSIMILATION PROTOCOLS
# ═══════════════════════════════════════════════════════════════════════════════

class SqrtProtocol(Enum):
    """Square root assimilation decrement protocols"""
    CARMACK = "carmack"           # 0x5f3759df magic
    NEWTON_1 = "newton_1"         # Single Newton iteration
    NEWTON_2 = "newton_2"         # Double Newton iteration
    RSQRT = "rsqrt"               # GPU rsqrt instruction
    QUAKE = "quake"               # Original Q3 fast inverse sqrt
    GOLDSCHMIDT = "goldschmidt"   # Goldschmidt division-free
    BABYLONIAN = "babylonian"     # Babylonian method
    QUANTUM = "quantum"           # Quantum superposition estimate


class SqrtAssimilator:
    """
    Square root assimilation with decrement refinement
    Manages precision vs speed tradeoff per partition depth
    """
    
    def __init__(self, protocol: SqrtProtocol = SqrtProtocol.RSQRT):
        self.protocol = protocol
        self.refinement_depth = 0
        self.error_accumulator = 0.0
        
        # Magic constants
        self.CARMACK_MAGIC = 0x5f3759df
        self.CARMACK_MAGIC_64 = 0x5fe6eb50c7b537a9
    
    def _carmack_scalar(self, x: float) -> float:
        """Original Quake III fast inverse sqrt"""
        import struct
        
        x2 = x * 0.5
        packed = struct.pack('f', x)
        i = struct.unpack('i', packed)[0]
        i = self.CARMACK_MAGIC - (i >> 1)
        packed = struct.pack('i', i)
        y = struct.unpack('f', packed)[0]
        
        # Newton iteration
        y = y * (1.5 - (x2 * y * y))
        return y
    
    def inverse_sqrt(self, x: torch.Tensor, depth: int = 0) -> torch.Tensor:
        """
        Inverse square root with protocol selection
        Depth controls refinement iterations
        """
        x_safe = x.clamp(min=1e-12)
        
        if self.protocol == SqrtProtocol.RSQRT:
            return torch.rsqrt(x_safe)
        
        elif self.protocol == SqrtProtocol.NEWTON_1:
            # Initial estimate via rsqrt, one Newton refinement
            y = torch.rsqrt(x_safe)
            y = y * (1.5 - 0.5 * x_safe * y * y)
            return y
        
        elif self.protocol == SqrtProtocol.NEWTON_2:
            # Two Newton iterations for higher precision
            y = torch.rsqrt(x_safe)
            for _ in range(2):
                y = y * (1.5 - 0.5 * x_safe * y * y)
            return y
        
        elif self.protocol == SqrtProtocol.GOLDSCHMIDT:
            # Division-free Goldschmidt algorithm
            y = torch.rsqrt(x_safe)
            h = x_safe * 0.5
            for _ in range(depth + 1):
                r = 0.5 - h * y * y
                y = y + y * r
            return y
        
        elif self.protocol == SqrtProtocol.BABYLONIAN:
            # Babylonian / Heron's method
            y = torch.ones_like(x_safe)
            for _ in range(depth + 3):
                y = 0.5 * (y + x_safe / y.clamp(min=1e-12))
            return 1.0 / y.clamp(min=1e-12)
        
        elif self.protocol == SqrtProtocol.QUANTUM:
            # Quantum superposition: multiple estimates averaged
            estimates = []
            for i in range(4):
                noise = torch.randn_like(x_safe) * 0.01 * (1.0 / (i + 1))
                y = torch.rsqrt(x_safe + noise)
                estimates.append(y)
            return torch.stack(estimates).mean(dim=0)
        
        else:
            return torch.rsqrt(x_safe)
    
    def sqrt(self, x: torch.Tensor, depth: int = 0) -> torch.Tensor:
        """Square root via inverse sqrt: sqrt(x) = x * rsqrt(x)"""
        x_safe = x.clamp(min=1e-12)
        return x_safe * self.inverse_sqrt(x_safe, depth)
    
    def length(self, v: torch.Tensor, dim: int = -1, depth: int = 0) -> torch.Tensor:
        """Vector length using selected sqrt protocol"""
        sq_sum = (v * v).sum(dim=dim)
        return self.sqrt(sq_sum, depth)
    
    def normalize(self, v: torch.Tensor, dim: int = -1, depth: int = 0) -> torch.Tensor:
        """Normalize using inverse sqrt"""
        sq_sum = (v * v).sum(dim=dim, keepdim=True)
        return v * self.inverse_sqrt(sq_sum, depth)
    
    def decrement_refine(self, x: torch.Tensor, target_error: float = 1e-6) -> torch.Tensor:
        """
        Decrement refinement protocol
        Iteratively refine until error threshold met
        """
        x_safe = x.clamp(min=1e-12)
        y = torch.rsqrt(x_safe)
        
        for iteration in range(8):
            # Compute current error
            err = torch.abs(y * y * x_safe - 1.0).max().item()
            self.error_accumulator = err
            
            if err < target_error:
                self.refinement_depth = iteration
                break
            
            # Newton decrement
            y = y * (1.5 - 0.5 * x_safe * y * y)
        
        return y


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM SPACE PARTITION NODE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantumPartitionState:
    """Quantum state for BSP node"""
    amplitude: complex = 1.0 + 0j
    phase: float = 0.0
    coherence: float = 1.0
    entangled_nodes: List[int] = field(default_factory=list)
    
    @property
    def probability(self) -> float:
        return abs(self.amplitude) ** 2 * self.coherence
    
    def apply_sqrt_gate(self, assimilator: SqrtAssimilator) -> 'QuantumPartitionState':
        """Apply sqrt-based quantum gate"""
        # Amplitude modification via sqrt
        mag = abs(self.amplitude)
        phase = np.angle(self.amplitude)
        
        mag_tensor = torch.tensor([mag], device=device)
        new_mag = float(assimilator.sqrt(mag_tensor, depth=1))
        
        # Phase evolution
        new_phase = self.phase + np.pi * new_mag
        
        return QuantumPartitionState(
            amplitude=complex(new_mag * np.cos(new_phase), 
                            new_mag * np.sin(new_phase)),
            phase=new_phase,
            coherence=self.coherence * 0.99,
            entangled_nodes=self.entangled_nodes.copy()
        )
    
    def collapse(self, rng: np.random.Generator) -> Tuple[bool, 'QuantumPartitionState']:
        """Collapse wavefunction"""
        measured = rng.random() < self.probability
        new_amp = 1.0 + 0j if measured else 0j
        return measured, QuantumPartitionState(new_amp, 0, 1.0, [])


@dataclass 
class Hyperplane4D:
    """4D splitting hyperplane with sqrt-refined distance"""
    normal: torch.Tensor
    distance: float
    sqrt_assimilator: SqrtAssimilator = field(default_factory=SqrtAssimilator)
    
    def __post_init__(self):
        # Normalize using sqrt assimilator
        self.normal = self.sqrt_assimilator.normalize(
            self.normal.unsqueeze(0), dim=-1
        ).squeeze(0)
    
    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        """Signed distance with sqrt-refined normal"""
        return (points * self.normal).sum(dim=-1) - self.distance
    
    def classify(self, points: torch.Tensor, epsilon: float = 0.001) -> torch.Tensor:
        """Classify points: -1 back, 0 on plane, +1 front"""
        d = self.signed_distance(points)
        result = torch.zeros_like(d, dtype=torch.int8)
        result[d > epsilon] = 1
        result[d < -epsilon] = -1
        return result


@dataclass
class QuantumBSPNode:
    """Quantum BSP node with sqrt-refined geometry"""
    node_id: int
    depth: int = 0
    hyperplane: Optional[Hyperplane4D] = None
    front: Optional['QuantumBSPNode'] = None
    back: Optional['QuantumBSPNode'] = None
    quantum_state: QuantumPartitionState = field(default_factory=QuantumPartitionState)
    sqrt_protocol: SqrtProtocol = SqrtProtocol.RSQRT
    sdf_func: Optional[Callable] = None
    
    def is_leaf(self) -> bool:
        return self.front is None and self.back is None


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM SPACE PARTITION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumSpacePartitionMatrix:
    """
    Quantum BSP with sqrt assimilation protocol
    - Partition matrix stores node relationships
    - Sqrt refinement per depth level
    - Quantum state propagation through tree
    """
    
    def __init__(self, 
                 sqrt_protocol: SqrtProtocol = SqrtProtocol.NEWTON_1,
                 max_depth: int = 6,
                 seed: int = 42):
        self.sqrt_assimilator = SqrtAssimilator(sqrt_protocol)
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)
        
        self.root: Optional[QuantumBSPNode] = None
        self.node_count = 0
        self.partition_matrix: torch.Tensor = None
        self.quantum_matrix: torch.Tensor = None
        
        # Sqrt refinement schedule per depth
        self.refinement_schedule = [
            SqrtProtocol.NEWTON_2,  # Depth 0: highest precision
            SqrtProtocol.NEWTON_1,  # Depth 1: high precision
            SqrtProtocol.RSQRT,     # Depth 2+: fast
            SqrtProtocol.RSQRT,
            SqrtProtocol.RSQRT,
            SqrtProtocol.RSQRT,
        ]
    
    def _new_node(self, depth: int = 0) -> QuantumBSPNode:
        node = QuantumBSPNode(
            node_id=self.node_count,
            depth=depth,
            sqrt_protocol=self.refinement_schedule[min(depth, len(self.refinement_schedule)-1)]
        )
        self.node_count += 1
        return node
    
    def build(self, bounds: Tuple[torch.Tensor, torch.Tensor], 
              depth: int = 0,
              parent_quantum: QuantumPartitionState = None) -> QuantumBSPNode:
        """Build quantum BSP tree with sqrt-refined partitions"""
        
        node = self._new_node(depth)
        
        # Propagate quantum state
        if parent_quantum:
            node.quantum_state = parent_quantum.apply_sqrt_gate(self.sqrt_assimilator)
        else:
            node.quantum_state = QuantumPartitionState(
                amplitude=complex(1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                phase=self.rng.random() * 2 * np.pi
            )
        
        # Leaf condition
        if depth >= self.max_depth or self.rng.random() < 0.25:
            # Assign SDF at leaf
            center = (bounds[0] + bounds[1]) / 2
            size = (bounds[1] - bounds[0]).min().item() * 0.35
            
            node.sdf_func = lambda p, c=center, s=size: self._leaf_sdf(p, c, s)
            return node
        
        # Create splitting hyperplane
        min_b, max_b = bounds
        center = (min_b + max_b) / 2
        
        # Random normal with sqrt-normalized components
        normal = torch.randn(4, device=device)
        
        # Use depth-appropriate sqrt protocol for normalization
        assimilator = SqrtAssimilator(node.sqrt_protocol)
        normal = assimilator.normalize(normal.unsqueeze(0)).squeeze(0)
        
        node.hyperplane = Hyperplane4D(
            normal=normal,
            distance=float((normal * center).sum()),
            sqrt_assimilator=assimilator
        )
        
        # Quantum-weighted child construction
        front_quantum = node.quantum_state.apply_sqrt_gate(self.sqrt_assimilator)
        back_quantum = QuantumPartitionState(
            amplitude=node.quantum_state.amplitude * 1j,
            phase=node.quantum_state.phase + np.pi / 2,
            coherence=node.quantum_state.coherence * 0.98
        )
        
        node.front = self.build(bounds, depth + 1, front_quantum)
        node.back = self.build(bounds, depth + 1, back_quantum)
        
        # Entangle siblings
        node.front.quantum_state.entangled_nodes.append(node.back.node_id)
        node.back.quantum_state.entangled_nodes.append(node.front.node_id)
        
        return node
    
    def _leaf_sdf(self, p: torch.Tensor, center: torch.Tensor, size: float) -> torch.Tensor:
        """Leaf SDF with sqrt-refined distance"""
        diff = p - center
        return self.sqrt_assimilator.length(diff) - size
    
    def build_partition_matrix(self):
        """Build adjacency matrix for partition nodes"""
        n = self.node_count
        self.partition_matrix = torch.zeros(n, n, device=device)
        self.quantum_matrix = torch.zeros(n, n, dtype=torch.complex64, device=device)
        
        def fill_matrix(node: QuantumBSPNode):
            if node is None:
                return
            
            nid = node.node_id
            
            if node.front:
                fid = node.front.node_id
                self.partition_matrix[nid, fid] = 1
                self.partition_matrix[fid, nid] = 1
                
                # Quantum coupling
                coupling = node.quantum_state.amplitude * np.conj(node.front.quantum_state.amplitude)
                self.quantum_matrix[nid, fid] = coupling
                self.quantum_matrix[fid, nid] = np.conj(coupling)
                
                fill_matrix(node.front)
            
            if node.back:
                bid = node.back.node_id
                self.partition_matrix[nid, bid] = 1
                self.partition_matrix[bid, nid] = 1
                
                coupling = node.quantum_state.amplitude * np.conj(node.back.quantum_state.amplitude)
                self.quantum_matrix[nid, bid] = coupling
                self.quantum_matrix[bid, nid] = np.conj(coupling)
                
                fill_matrix(node.back)
        
        fill_matrix(self.root)
    
    def traverse_sdf(self, node: QuantumBSPNode, points: torch.Tensor) -> torch.Tensor:
        """Traverse BSP evaluating SDF with sqrt-refined blending"""
        if node is None:
            return torch.full((points.shape[:-1]), 1e10, device=device)
        
        if node.is_leaf():
            if node.sdf_func:
                d = node.sdf_func(points)
                # Weight by quantum probability
                return d * (0.5 + 0.5 * node.quantum_state.probability)
            return torch.full((points.shape[:-1]), 1e10, device=device)
        
        d_front = self.traverse_sdf(node.front, points)
        d_back = self.traverse_sdf(node.back, points)
        
        # Quantum-weighted smooth blend
        prob = node.quantum_state.probability
        k = 0.1 + 0.1 * (1 - prob)  # Blend sharpness from quantum state
        
        # Smooth min with sqrt-refined k
        h = torch.clamp(0.5 + 0.5 * (d_back - d_front) / k, 0, 1)
        return d_back * (1 - h) + d_front * h - k * h * (1 - h)
    
    def evaluate(self, points: torch.Tensor) -> torch.Tensor:
        """Evaluate SDF at points"""
        return self.traverse_sdf(self.root, points)


# ═══════════════════════════════════════════════════════════════════════════════
# 4D ROTATION WITH SQRT REFINEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class SqrtRefinedRot4D:
    """4D rotation with sqrt-refined trigonometry"""
    
    PLANES = {'xy':(0,1), 'xz':(0,2), 'xw':(0,3), 
              'yz':(1,2), 'yw':(1,3), 'zw':(2,3)}
    
    def __init__(self, sqrt_protocol: SqrtProtocol = SqrtProtocol.NEWTON_1):
        self.assimilator = SqrtAssimilator(sqrt_protocol)
    
    def matrix(self, plane: str, theta: float) -> torch.Tensor:
        c, s = np.cos(theta), np.sin(theta)
        R = torch.eye(4, device=device, dtype=torch.float32)
        i, j = self.PLANES[plane]
        R[i,i], R[j,j] = c, c
        R[i,j], R[j,i] = -s, s
        return R
    
    def from_6angles(self, xy=0, xz=0, xw=0, yz=0, yw=0, zw=0) -> torch.Tensor:
        R = torch.eye(4, device=device, dtype=torch.float32)
        for plane, angle in [('xy',xy),('xz',xz),('xw',xw),
                             ('yz',yz),('yw',yw),('zw',zw)]:
            if angle != 0:
                R = R @ self.matrix(plane, angle)
        
        # Orthonormalize with sqrt refinement
        R = self._orthonormalize(R)
        return R
    
    def _orthonormalize(self, R: torch.Tensor) -> torch.Tensor:
        """Gram-Schmidt with sqrt-refined normalization"""
        Q = torch.zeros_like(R)
        
        for i in range(4):
            v = R[:, i].clone()
            for j in range(i):
                proj = (Q[:, j] * v).sum()
                v = v - proj * Q[:, j]
            Q[:, i] = self.assimilator.normalize(v.unsqueeze(0)).squeeze(0)
        
        return Q


# ═══════════════════════════════════════════════════════════════════════════════
# VOLUMETRIC MARCHER WITH SQRT EPSILON
# ═══════════════════════════════════════════════════════════════════════════════

class SqrtVolumetricMarcher:
    """Raymarcher with sqrt-refined epsilon adaptation"""
    
    def __init__(self, res: int = 400, sqrt_protocol: SqrtProtocol = SqrtProtocol.NEWTON_1):
        self.res = res
        self.max_steps = 80
        self.max_dist = 12.0
        self.epsilon_base = 0.0006
        self.assimilator = SqrtAssimilator(sqrt_protocol)
    
    def adaptive_epsilon(self, t: torch.Tensor, step: int) -> torch.Tensor:
        """Sqrt-refined adaptive epsilon"""
        # Sqrt of distance for sub-linear scaling
        t_sqrt = self.assimilator.sqrt(t.clamp(min=0.01))
        dist_scale = 1.0 + t_sqrt * 0.08
        
        step_frac = step / self.max_steps
        step_scale = step_frac * step_frac
        
        return self.epsilon_base * dist_scale * (1.0 + step_scale * 0.4)
    
    def march(self, sdf_func: Callable, w_slice: float = 0.0,
              R: torch.Tensor = None) -> dict:
        
        if R is None:
            R = torch.eye(4, device=device)
        R_inv = R.T
        
        fov, cam = 1.05, 3.5
        u = torch.linspace(-fov, fov, self.res, device=device)
        uu, vv = torch.meshgrid(u, u, indexing='xy')
        
        origins = torch.zeros(self.res, self.res, 4, device=device)
        origins[..., 2] = -cam
        origins[..., 3] = w_slice
        
        dirs = torch.stack([uu, -vv, torch.full_like(uu, 1.6), 
                           torch.zeros_like(uu)], dim=-1)
        dirs = self.assimilator.normalize(dirs)
        
        t = torch.zeros(self.res, self.res, device=device)
        hit = torch.zeros(self.res, self.res, dtype=torch.bool, device=device)
        hit_pos = torch.zeros(self.res, self.res, 4, device=device)
        glow = torch.zeros(self.res, self.res, device=device)
        
        for step in range(self.max_steps):
            if hit.all():
                break
            
            p = origins + t.unsqueeze(-1) * dirs
            p_rot = (p.view(-1, 4) @ R_inv).view(p.shape)
            d = sdf_func(p_rot)
            
            eps = self.adaptive_epsilon(t, step)
            
            new_hit = ~hit & (d < eps)
            hit |= new_hit
            hit_pos[new_hit] = p[new_hit]
            
            # Glow with sqrt falloff
            glow += torch.exp(-self.assimilator.sqrt(d.clamp(min=0)) * 3) * 0.008 * (~hit).float()
            
            # Sqrt-refined step relaxation
            relax = 0.8 + 0.2 * (step / self.max_steps) ** 2
            t[~hit] += d[~hit].clamp(min=eps[~hit]) * relax
            t = t.clamp(max=self.max_dist)
        
        # Normals with sqrt-refined normalization
        normals = torch.zeros_like(hit_pos)
        eps_n = 0.001
        for i in range(4):
            pp, pn = hit_pos.clone(), hit_pos.clone()
            pp[..., i] += eps_n
            pn[..., i] -= eps_n
            normals[..., i] = sdf_func((pp.view(-1,4) @ R_inv).view(pp.shape)) - \
                              sdf_func((pn.view(-1,4) @ R_inv).view(pn.shape))
        normals = self.assimilator.normalize(normals)
        
        return {
            'hit': hit,
            'hit_pos': hit_pos,
            't': t,
            'normals': normals,
            'glow': glow
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def render(result: dict, hue: float = 0.75) -> np.ndarray:
    res = result['hit'].shape[0]
    
    hit = result['hit']
    normals = result['normals']
    hit_pos = result['hit_pos']
    t = result['t']
    glow = result['glow']
    
    L = torch.tensor([0.5, 0.7, -0.4, 0.2], device=device)
    L = L / torch.norm(L)
    diff = torch.clamp((normals * L).sum(dim=-1), 0.1, 1.0)
    
    V = torch.tensor([0, 0, 1, 0], device=device, dtype=torch.float32)
    H = L + V
    H = H / torch.norm(H)
    spec = torch.clamp((normals * H).sum(dim=-1), 0, 1) ** 32
    
    w_n = torch.clamp((hit_pos[..., 3] + 1) / 2, 0, 1)
    fog = torch.exp(-t * 0.1)
    
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
                h = (hue + w_np[y, x] * 0.08) % 1.0
                s = 0.6 + w_np[y, x] * 0.15
                v = 0.15 + 0.6 * diff_np[y, x]
                
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                r += spec_np[y, x] * 0.5
                g += spec_np[y, x] * 0.4
                b += spec_np[y, x] * 0.55
                
                r = r * fog_np[y, x]
                g = g * fog_np[y, x]
                b = b * fog_np[y, x]
                
                img[y, x] = [r, g, b]
            else:
                gl = glow_np[y, x]
                img[y, x] = [0.008 + gl * 0.08, 0.004 + gl * 0.03, 0.015 + gl * 0.12]
    
    return np.clip(img, 0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  QUANTUM SPACE PARTITION - SQRT ASSIMILATION PROTOCOL         ║")
    print("║  Root-refined BSP matrix with quantum state propagation       ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")
    
    print(f"  Device: {device}\n")
    
    # Test sqrt protocols
    print("  ─── SQRT PROTOCOL COMPARISON ───\n")
    
    x = torch.rand(1000, device=device) + 0.1
    
    for protocol in [SqrtProtocol.RSQRT, SqrtProtocol.NEWTON_1, 
                     SqrtProtocol.NEWTON_2, SqrtProtocol.GOLDSCHMIDT]:
        assimilator = SqrtAssimilator(protocol)
        y = assimilator.inverse_sqrt(x)
        
        # Error vs torch.rsqrt
        err = torch.abs(y - torch.rsqrt(x)).max().item()
        print(f"    {protocol.value:12s} │ max error: {err:.2e}")
    
    print()
    
    # Build quantum BSP
    print("  ─── BUILDING QUANTUM BSP ───\n")
    
    qsp = QuantumSpacePartitionMatrix(
        sqrt_protocol=SqrtProtocol.NEWTON_1,
        max_depth=4,
        seed=2024
    )
    
    bounds = (
        torch.tensor([-1.5, -1.5, -1.5, -1.5], device=device),
        torch.tensor([1.5, 1.5, 1.5, 1.5], device=device)
    )
    
    qsp.root = qsp.build(bounds)
    qsp.build_partition_matrix()
    
    print(f"    Nodes: {qsp.node_count}")
    print(f"    Partition matrix: {qsp.partition_matrix.shape}")
    print(f"    Quantum couplings: {(qsp.quantum_matrix.abs() > 0.01).sum().item()}")
    print()
    
    # Render animation
    print("  ─── RENDERING ───\n")
    
    rot = SqrtRefinedRot4D(SqrtProtocol.NEWTON_1)
    marcher = SqrtVolumetricMarcher(res=380, sqrt_protocol=SqrtProtocol.NEWTON_1)
    
    frames = []
    n_frames = 48
    
    for i in range(n_frames):
        t = i / n_frames
        theta = t * 2 * np.pi
        
        R = rot.from_6angles(
            xw=theta,
            yw=np.sin(theta * 2) * 0.3,
            zw=t * np.pi * 0.5,
            xy=np.sin(theta) * 0.15
        )
        
        w = np.sin(theta * 2) * 0.4
        
        if i % 8 == 0:
            print(f"    Frame {i+1:02d}/{n_frames} │ w={w:+.2f}")
        
        result = marcher.march(qsp.evaluate, w_slice=w, R=R)
        img = render(result, hue=0.72)
        
        frames.append(Image.fromarray((img * 255).astype(np.uint8)))
    
    frames[0].save('quantum_sqrt_partition.gif', save_all=True,
                   append_images=frames[1:], duration=50, loop=0)
    frames[0].save('quantum_sqrt_partition_test.png')
    
    print(f"\n  ✓ Saved: quantum_sqrt_partition.gif\n")
    
    print("  SQRT PROTOCOLS:")
    print("    ├─ CARMACK:     0x5f3759df magic constant")
    print("    ├─ NEWTON_1:    rsqrt + 1 Newton iteration")
    print("    ├─ NEWTON_2:    rsqrt + 2 Newton iterations")
    print("    ├─ GOLDSCHMIDT: division-free refinement")
    print("    ├─ BABYLONIAN:  Heron's method")
    print("    └─ QUANTUM:     superposition averaging")
    print()
    print("  REFINEMENT SCHEDULE:")
    print("    ├─ Depth 0: NEWTON_2 (highest precision)")
    print("    ├─ Depth 1: NEWTON_1 (high precision)")
    print("    └─ Depth 2+: RSQRT (fast)")


if __name__ == "__main__":
    main()
