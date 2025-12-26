#!/usr/bin/env python3
"""
NEWTONIAN PHYSICS + SOFT BODY EUCLIDEAN SHAPER
Proper physics: F=ma, buoyancy, drag, no magic repulsion
Soft body: distance constraints, shape preservation
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Tuple

EPSILON = 1e-4
MAX_STEPS = 100
MAX_DIST = 50.0

# ============================================
# EUCLIDEAN DISTANCE UTILITIES
# ============================================

def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Proper Euclidean distance"""
    return torch.sqrt(torch.sum((a - b) ** 2, dim=-1) + 1e-8)

def euclidean_direction(from_pt: torch.Tensor, to_pt: torch.Tensor) -> torch.Tensor:
    """Unit vector from a to b"""
    diff = to_pt - from_pt
    dist = torch.sqrt(torch.sum(diff ** 2) + 1e-8)
    return diff / dist

def point_to_plane_distance(point: torch.Tensor, plane_y: float) -> float:
    """Signed distance from point to horizontal plane"""
    return point[1].item() - plane_y

# ============================================
# SOFT BODY PARTICLE
# ============================================

@dataclass
class Particle:
    pos: torch.Tensor          # Position
    vel: torch.Tensor          # Velocity
    acc: torch.Tensor          # Acceleration (computed each frame)
    mass: float = 1.0
    radius: float = 0.1        # Collision radius
    pinned: bool = False       # If true, doesn't move
    
    def __post_init__(self):
        if self.acc is None:
            self.acc = torch.zeros_like(self.pos)

@dataclass 
class DistanceConstraint:
    """Maintains distance between two particles"""
    p1_idx: int
    p2_idx: int
    rest_length: float
    stiffness: float = 0.5     # 0-1, how rigid

@dataclass
class SoftBody:
    """Collection of particles with constraints"""
    particles: List[Particle]
    constraints: List[DistanceConstraint]
    center_of_mass: torch.Tensor = None
    color: torch.Tensor = None
    
    def compute_center_of_mass(self):
        total_mass = sum(p.mass for p in self.particles)
        com = torch.zeros(3, device=self.particles[0].pos.device)
        for p in self.particles:
            com += p.pos * p.mass
        self.center_of_mass = com / total_mass
        return self.center_of_mass
    
    def get_bounding_radius(self):
        self.compute_center_of_mass()
        max_dist = 0.0
        for p in self.particles:
            d = euclidean_distance(p.pos, self.center_of_mass).item()
            max_dist = max(max_dist, d + p.radius)
        return max_dist

# ============================================
# SOFT BODY FACTORY - EUCLIDEAN SHAPES
# ============================================

class EuclideanShaper:
    """Creates soft bodies from Euclidean primitives"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def create_sphere(self, center: List[float], radius: float, 
                      resolution: int = 8, mass: float = 1.0,
                      stiffness: float = 0.6, color: List[float] = None) -> SoftBody:
        """
        Sphere from particles on surface + center
        Constraints maintain spherical shape
        """
        particles = []
        constraints = []
        
        center_t = torch.tensor(center, device=self.device, dtype=torch.float32)
        
        # Center particle
        particles.append(Particle(
            pos=center_t.clone(),
            vel=torch.zeros(3, device=self.device),
            acc=torch.zeros(3, device=self.device),
            mass=mass * 0.5,
            radius=radius * 0.3
        ))
        center_idx = 0
        
        # Surface particles using fibonacci sphere
        n_surface = resolution * resolution
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(n_surface):
            theta = 2 * math.pi * i / golden_ratio
            phi = math.acos(1 - 2 * (i + 0.5) / n_surface)
            
            x = center[0] + radius * math.sin(phi) * math.cos(theta)
            y = center[1] + radius * math.sin(phi) * math.sin(theta)
            z = center[2] + radius * math.cos(phi)
            
            particles.append(Particle(
                pos=torch.tensor([x, y, z], device=self.device, dtype=torch.float32),
                vel=torch.zeros(3, device=self.device),
                acc=torch.zeros(3, device=self.device),
                mass=mass / n_surface,
                radius=radius * 0.15
            ))
            
            # Constraint to center
            constraints.append(DistanceConstraint(
                p1_idx=center_idx,
                p2_idx=len(particles) - 1,
                rest_length=radius,
                stiffness=stiffness
            ))
        
        # Surface-to-surface constraints (neighbors)
        for i in range(1, len(particles)):
            for j in range(i + 1, len(particles)):
                dist = euclidean_distance(particles[i].pos, particles[j].pos).item()
                # Only connect nearby particles
                if dist < radius * 0.8:
                    constraints.append(DistanceConstraint(
                        p1_idx=i,
                        p2_idx=j,
                        rest_length=dist,
                        stiffness=stiffness * 0.5
                    ))
        
        if color is None:
            color = [0.8, 0.4, 0.4]
        
        return SoftBody(
            particles=particles,
            constraints=constraints,
            color=torch.tensor(color, device=self.device, dtype=torch.float32)
        )
    
    def create_box(self, center: List[float], size: List[float],
                   mass: float = 1.0, stiffness: float = 0.7,
                   color: List[float] = None) -> SoftBody:
        """
        Box from 8 corner particles + center
        """
        particles = []
        constraints = []
        
        cx, cy, cz = center
        sx, sy, sz = [s/2 for s in size]
        
        # Center
        particles.append(Particle(
            pos=torch.tensor(center, device=self.device, dtype=torch.float32),
            vel=torch.zeros(3, device=self.device),
            acc=torch.zeros(3, device=self.device),
            mass=mass * 0.3,
            radius=min(size) * 0.2
        ))
        
        # 8 corners
        corners = [
            [cx-sx, cy-sy, cz-sz],
            [cx+sx, cy-sy, cz-sz],
            [cx-sx, cy+sy, cz-sz],
            [cx+sx, cy+sy, cz-sz],
            [cx-sx, cy-sy, cz+sz],
            [cx+sx, cy-sy, cz+sz],
            [cx-sx, cy+sy, cz+sz],
            [cx+sx, cy+sy, cz+sz],
        ]
        
        for corner in corners:
            particles.append(Particle(
                pos=torch.tensor(corner, device=self.device, dtype=torch.float32),
                vel=torch.zeros(3, device=self.device),
                acc=torch.zeros(3, device=self.device),
                mass=mass / 8,
                radius=min(size) * 0.15
            ))
        
        # Connect all corners to center
        for i in range(1, 9):
            dist = euclidean_distance(particles[0].pos, particles[i].pos).item()
            constraints.append(DistanceConstraint(0, i, dist, stiffness))
        
        # Connect edges (12 edges of cube)
        edges = [
            (1,2), (3,4), (5,6), (7,8),  # x-aligned
            (1,3), (2,4), (5,7), (6,8),  # y-aligned
            (1,5), (2,6), (3,7), (4,8),  # z-aligned
        ]
        for i, j in edges:
            dist = euclidean_distance(particles[i].pos, particles[j].pos).item()
            constraints.append(DistanceConstraint(i, j, dist, stiffness))
        
        # Face diagonals for rigidity
        diags = [(1,4), (2,3), (5,8), (6,7), (1,7), (2,8), (3,5), (4,6)]
        for i, j in diags:
            dist = euclidean_distance(particles[i].pos, particles[j].pos).item()
            constraints.append(DistanceConstraint(i, j, dist, stiffness * 0.8))
        
        if color is None:
            color = [0.6, 0.5, 0.3]
        
        return SoftBody(
            particles=particles,
            constraints=constraints,
            color=torch.tensor(color, device=self.device, dtype=torch.float32)
        )
    
    def create_blob(self, center: List[float], radius: float,
                    n_particles: int = 20, mass: float = 1.0,
                    stiffness: float = 0.3, color: List[float] = None) -> SoftBody:
        """
        Soft blob - loosely connected particles
        More organic, jellyfish-like
        """
        particles = []
        constraints = []
        
        # Random-ish distribution inside sphere
        torch.manual_seed(hash(tuple(center)) % (2**32))
        
        for i in range(n_particles):
            # Random point in sphere
            theta = torch.rand(1).item() * 2 * math.pi
            phi = torch.rand(1).item() * math.pi
            r = radius * (0.3 + 0.7 * torch.rand(1).item())
            
            x = center[0] + r * math.sin(phi) * math.cos(theta)
            y = center[1] + r * math.sin(phi) * math.sin(theta)
            z = center[2] + r * math.cos(phi)
            
            particles.append(Particle(
                pos=torch.tensor([x, y, z], device=self.device, dtype=torch.float32),
                vel=torch.zeros(3, device=self.device),
                acc=torch.zeros(3, device=self.device),
                mass=mass / n_particles,
                radius=radius * 0.1
            ))
        
        # Connect nearby particles
        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                dist = euclidean_distance(particles[i].pos, particles[j].pos).item()
                if dist < radius * 0.7:
                    constraints.append(DistanceConstraint(
                        i, j, dist, stiffness
                    ))
        
        if color is None:
            color = [0.4, 0.7, 0.8]
        
        return SoftBody(
            particles=particles,
            constraints=constraints,
            color=torch.tensor(color, device=self.device, dtype=torch.float32)
        )

# ============================================
# NEWTONIAN PHYSICS ENGINE
# ============================================

class NewtonianPhysics:
    """
    Proper Newtonian mechanics:
    F = ma
    Gravity, buoyancy, drag - no magic forces
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Constants
        self.gravity = torch.tensor([0., -9.81, 0.], device=device)  # m/s²
        self.water_density = 1000.0  # kg/m³
        self.air_density = 1.2       # kg/m³
        self.water_level = 0.0
        
        # Bodies
        self.soft_bodies: List[SoftBody] = []
        self.time = 0.0
    
    def add_soft_body(self, body: SoftBody):
        self.soft_bodies.append(body)
    
    def get_water_height(self, x: float, z: float, t: float) -> float:
        """Smooth wave function"""
        h = 0.0
        # Primary swells
        h += 0.4 * math.sin(x * 0.3 + t * 0.7)
        h += 0.35 * math.sin(z * 0.35 + t * 0.8)
        h += 0.25 * math.sin((x + z) * 0.2 + t * 0.5)
        # Secondary waves
        h += 0.15 * math.sin(x * 0.7 - z * 0.5 + t * 1.2)
        h += 0.1 * math.sin(x * 1.0 + z * 0.8 + t * 1.5)
        # Ripples
        h += 0.04 * math.sin(x * 2.5 + t * 2.5)
        h += 0.03 * math.sin(z * 3.0 + t * 3.0)
        return h
    
    def get_water_velocity(self, x: float, z: float, t: float) -> torch.Tensor:
        """Water current at position (simplified orbital motion)"""
        # Derivative of wave gives horizontal velocity
        vx = 0.3 * 0.3 * math.cos(x * 0.3 + t * 0.7) * 0.7
        vz = 0.35 * 0.35 * math.cos(z * 0.35 + t * 0.8) * 0.8
        vy = 0.1 * math.sin(t * 1.5)  # Small vertical component
        return torch.tensor([vx, vy, vz], device=self.device)
    
    def compute_buoyancy(self, particle: Particle, water_height: float) -> torch.Tensor:
        """
        Archimedes' principle: F_buoy = ρ_water * V_submerged * g
        """
        pos_y = particle.pos[1].item()
        radius = particle.radius
        
        # How much of sphere is submerged
        if pos_y - radius >= water_height:
            # Fully above water
            return torch.zeros(3, device=self.device)
        elif pos_y + radius <= water_height:
            # Fully submerged
            submerged_fraction = 1.0
        else:
            # Partially submerged
            submerged_depth = water_height - (pos_y - radius)
            submerged_fraction = min(1.0, submerged_depth / (2 * radius))
        
        # Volume of sphere
        volume = (4/3) * math.pi * radius**3
        submerged_volume = volume * submerged_fraction
        
        # Buoyancy force (upward)
        buoyancy_magnitude = self.water_density * submerged_volume * 9.81
        
        return torch.tensor([0., buoyancy_magnitude, 0.], device=self.device)
    
    def compute_drag(self, particle: Particle, medium_velocity: torch.Tensor, 
                     is_underwater: bool) -> torch.Tensor:
        """
        Drag force: F_drag = -0.5 * ρ * Cd * A * v²
        """
        relative_vel = particle.vel - medium_velocity
        speed = torch.norm(relative_vel)
        
        if speed < 1e-6:
            return torch.zeros(3, device=self.device)
        
        # Drag coefficient (sphere ≈ 0.47)
        Cd = 0.47
        
        # Cross-sectional area
        A = math.pi * particle.radius**2
        
        # Density of medium
        rho = self.water_density if is_underwater else self.air_density
        
        # Drag magnitude
        drag_mag = 0.5 * rho * Cd * A * speed.item()**2
        
        # Direction opposite to velocity
        drag_dir = -relative_vel / speed
        
        return drag_dir * drag_mag
    
    def apply_constraint(self, body: SoftBody, constraint: DistanceConstraint):
        """
        Position-based constraint solving
        Move particles to satisfy distance constraint
        """
        p1 = body.particles[constraint.p1_idx]
        p2 = body.particles[constraint.p2_idx]
        
        if p1.pinned and p2.pinned:
            return
        
        delta = p2.pos - p1.pos
        dist = torch.norm(delta)
        
        if dist < 1e-6:
            return
        
        # How much to correct
        error = dist - constraint.rest_length
        correction = (delta / dist) * error * constraint.stiffness
        
        # Apply based on mass ratio
        total_mass = p1.mass + p2.mass
        
        if not p1.pinned:
            p1.pos = p1.pos + correction * (p2.mass / total_mass)
        if not p2.pinned:
            p2.pos = p2.pos - correction * (p1.mass / total_mass)
    
    def step(self, dt: float):
        """
        Physics step:
        1. Compute forces (gravity, buoyancy, drag)
        2. Integrate (Verlet or Euler)
        3. Solve constraints
        4. Handle collisions
        """
        self.time += dt
        t = self.time
        
        for body in self.soft_bodies:
            # === FORCE ACCUMULATION ===
            for particle in body.particles:
                if particle.pinned:
                    continue
                
                # Reset acceleration
                particle.acc = torch.zeros(3, device=self.device)
                
                # Gravity: F = mg
                gravity_force = self.gravity * particle.mass
                particle.acc = particle.acc + gravity_force / particle.mass
                
                # Water height at particle position
                water_h = self.get_water_height(
                    particle.pos[0].item(),
                    particle.pos[2].item(),
                    t
                )
                
                # Buoyancy
                buoyancy_force = self.compute_buoyancy(particle, water_h)
                particle.acc = particle.acc + buoyancy_force / particle.mass
                
                # Drag
                is_underwater = particle.pos[1].item() < water_h
                if is_underwater:
                    water_vel = self.get_water_velocity(
                        particle.pos[0].item(),
                        particle.pos[2].item(),
                        t
                    )
                    drag_force = self.compute_drag(particle, water_vel, True)
                else:
                    drag_force = self.compute_drag(particle, torch.zeros(3, device=self.device), False)
                
                particle.acc = particle.acc + drag_force / particle.mass
            
            # === INTEGRATION (Semi-implicit Euler) ===
            for particle in body.particles:
                if particle.pinned:
                    continue
                
                # Update velocity
                particle.vel = particle.vel + particle.acc * dt
                
                # Velocity damping (numerical stability)
                particle.vel = particle.vel * 0.995
                
                # Update position
                particle.pos = particle.pos + particle.vel * dt
            
            # === CONSTRAINT SOLVING (multiple iterations) ===
            for _ in range(4):  # More iterations = more rigid
                for constraint in body.constraints:
                    self.apply_constraint(body, constraint)
            
            # === FLOOR COLLISION ===
            floor_y = -4.0
            for particle in body.particles:
                if particle.pos[1].item() < floor_y + particle.radius:
                    particle.pos[1] = floor_y + particle.radius
                    particle.vel[1] = abs(particle.vel[1].item()) * 0.3  # Bounce
                    particle.vel[0] = particle.vel[0] * 0.8  # Friction
                    particle.vel[2] = particle.vel[2] * 0.8
            
            # === WORLD BOUNDS ===
            for particle in body.particles:
                for dim in [0, 2]:
                    if particle.pos[dim].item() > 15:
                        particle.pos[dim] = 15.0
                        particle.vel[dim] = -particle.vel[dim] * 0.5
                    elif particle.pos[dim].item() < -15:
                        particle.pos[dim] = -15.0
                        particle.vel[dim] = -particle.vel[dim] * 0.5
                
                if particle.pos[1].item() > 10:
                    particle.pos[1] = 10.0
                    particle.vel[1] = -particle.vel[1] * 0.3
            
            # Update center of mass
            body.compute_center_of_mass()

# ============================================
# SDF FROM SOFT BODIES
# ============================================

def sdf_soft_body(p: torch.Tensor, body: SoftBody, smoothness: float = 0.15) -> torch.Tensor:
    """
    Create SDF from soft body particles using smooth union of spheres
    """
    device = p.device
    result = torch.full((p.shape[0],), 999.0, device=device)
    
    for particle in body.particles:
        # Distance to particle sphere
        d = torch.norm(p - particle.pos, dim=-1) - particle.radius
        
        # Smooth union
        k = smoothness
        h = torch.clamp(0.5 + 0.5 * (result - d) / k, 0.0, 1.0)
        result = torch.lerp(result, d, h) - k * h * (1.0 - h)
    
    return result

def sdf_floor(p: torch.Tensor) -> torch.Tensor:
    return p[..., 1] + 4.0

def op_smooth_union(d1, d2, k=0.2):
    h = torch.clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return torch.lerp(d2, d1, h) - k * h * (1.0 - h)

# ============================================
# SCENE
# ============================================

class NewtonianScene:
    def __init__(self, device='cuda'):
        self.device = device
        self.physics = NewtonianPhysics(device)
        self.shaper = EuclideanShaper(device)
        self.time = 0.0
        
        self._spawn_bodies()
    
    def _spawn_bodies(self):
        # Soft sphere - floats
        sphere1 = self.shaper.create_sphere(
            center=[0., 2., 0.],
            radius=0.6,
            resolution=6,
            mass=0.5,  # Light, floats well
            stiffness=0.5,
            color=[1.0, 0.4, 0.4]
        )
        self.physics.add_soft_body(sphere1)
        
        # Another sphere
        sphere2 = self.shaper.create_sphere(
            center=[3., 1., 2.],
            radius=0.5,
            resolution=5,
            mass=0.4,
            stiffness=0.6,
            color=[0.4, 0.8, 1.0]
        )
        self.physics.add_soft_body(sphere2)
        
        # Heavy box - sinks
        box1 = self.shaper.create_box(
            center=[-2., 3., -1.],
            size=[0.8, 0.6, 0.8],
            mass=3.0,  # Heavy, sinks
            stiffness=0.8,
            color=[0.6, 0.5, 0.3]
        )
        self.physics.add_soft_body(box1)
        
        # Light box - floats
        box2 = self.shaper.create_box(
            center=[2., 1., -3.],
            size=[0.5, 0.4, 0.5],
            mass=0.3,
            stiffness=0.7,
            color=[0.8, 0.7, 0.4]
        )
        self.physics.add_soft_body(box2)
        
        # Soft blob - jellyfish-like
        blob1 = self.shaper.create_blob(
            center=[-3., 0.5, 3.],
            radius=0.7,
            n_particles=15,
            mass=0.2,
            stiffness=0.2,  # Very soft
            color=[0.6, 0.9, 0.7]
        )
        self.physics.add_soft_body(blob1)
        
        # Another blob
        blob2 = self.shaper.create_blob(
            center=[4., 2., 0.],
            radius=0.5,
            n_particles=12,
            mass=0.15,
            stiffness=0.25,
            color=[0.9, 0.5, 0.8]
        )
        self.physics.add_soft_body(blob2)
    
    def scene_water(self, p):
        t = self.time
        h = torch.zeros(p.shape[0], device=self.device)
        
        h += 0.4 * torch.sin(p[..., 0] * 0.3 + t * 0.7)
        h += 0.35 * torch.sin(p[..., 2] * 0.35 + t * 0.8)
        h += 0.25 * torch.sin((p[..., 0] + p[..., 2]) * 0.2 + t * 0.5)
        h += 0.15 * torch.sin(p[..., 0] * 0.7 - p[..., 2] * 0.5 + t * 1.2)
        h += 0.1 * torch.sin(p[..., 0] * 1.0 + p[..., 2] * 0.8 + t * 1.5)
        h += 0.04 * torch.sin(p[..., 0] * 2.5 + t * 2.5)
        h += 0.03 * torch.sin(p[..., 2] * 3.0 + t * 3.0)
        
        return p[..., 1] - h
    
    def scene_bodies(self, p):
        result = torch.full((p.shape[0],), 999.0, device=self.device)
        
        for body in self.physics.soft_bodies:
            d = sdf_soft_body(p, body, smoothness=0.12)
            result = op_smooth_union(result, d, 0.1)
        
        return result
    
    def scene_solid(self, p):
        floor = sdf_floor(p)
        bodies = self.scene_bodies(p)
        return op_smooth_union(floor, bodies, 0.15)
    
    def get_body_color(self, p):
        colors = torch.full((p.shape[0], 3), 0.4, device=self.device)
        min_dist = torch.full((p.shape[0],), 999.0, device=self.device)
        
        for body in self.physics.soft_bodies:
            d = sdf_soft_body(p, body)
            mask = d < min_dist
            min_dist = torch.where(mask, d, min_dist)
            colors = torch.where(mask.unsqueeze(-1), body.color.expand(p.shape[0], 3), colors)
        
        return colors

# ============================================
# RENDER
# ============================================

def calc_normal(sdf, p, eps=EPSILON):
    e = torch.tensor([eps, 0., 0.], device=p.device)
    return F.normalize(torch.stack([
        sdf(p + e.roll(0)) - sdf(p - e.roll(0)),
        sdf(p + e.roll(1)) - sdf(p - e.roll(1)),
        sdf(p + e.roll(2)) - sdf(p - e.roll(2))
    ], dim=-1), dim=-1)

def ray_march(ro, rd, sdf, max_steps=MAX_STEPS):
    t = torch.zeros(ro.shape[0], device=ro.device)
    for _ in range(max_steps):
        d = sdf(ro + t.unsqueeze(-1) * rd)
        t = t + d * 0.6
        if (d < EPSILON).all() or (t > MAX_DIST).all():
            break
    return t, d < EPSILON

def fresnel(cos_theta, f0=0.02):
    return f0 + (1.0 - f0) * torch.pow(torch.clamp(1.0 - cos_theta, 0.0, 1.0), 5.0)

def reflect(rd, n):
    return rd - 2.0 * (rd * n).sum(-1, keepdim=True) * n

def render(scene, width=640, height=480):
    device = scene.device
    t = scene.time
    
    # Camera
    cam_angle = t * 0.1
    cam_dist = 18.0
    cam_height = 5.0 + math.sin(t * 0.15) * 1.0
    
    cam_pos = torch.tensor([
        math.sin(cam_angle) * cam_dist,
        cam_height,
        math.cos(cam_angle) * cam_dist
    ], device=device)
    
    cam_target = torch.tensor([0., 0., 0.], device=device)
    cam_fwd = F.normalize(cam_target - cam_pos, dim=0)
    cam_right = F.normalize(torch.linalg.cross(torch.tensor([0., 1., 0.], device=device), cam_fwd), dim=0)
    cam_up = torch.linalg.cross(cam_fwd, cam_right)
    
    aspect = width / height
    y, x = torch.meshgrid(
        torch.linspace(1, -1, height, device=device),
        torch.linspace(-aspect, aspect, width, device=device), indexing='ij')
    
    n_pix = height * width
    ro = cam_pos.unsqueeze(0).expand(n_pix, 3)
    rd = F.normalize(x.flatten().unsqueeze(-1) * cam_right + 
                     y.flatten().unsqueeze(-1) * cam_up + 
                     cam_fwd * 1.8, dim=-1)
    
    # March
    t_water, hit_water = ray_march(ro, rd, scene.scene_water)
    p_water = ro + t_water.unsqueeze(-1) * rd
    n_water = calc_normal(scene.scene_water, p_water)
    
    t_solid, hit_solid = ray_march(ro, rd, scene.scene_solid)
    p_solid = ro + t_solid.unsqueeze(-1) * rd
    n_solid = calc_normal(scene.scene_solid, p_solid)
    
    # Water shading
    view = -rd
    cos_theta = torch.clamp((n_water * view).sum(-1), 0.0, 1.0)
    fres = fresnel(cos_theta)
    
    # Reflection
    refl_dir = reflect(rd, n_water)
    refl_ro = p_water + n_water * 0.02
    t_refl, hit_refl = ray_march(refl_ro, refl_dir, scene.scene_solid, max_steps=60)
    
    # Sky
    sky_blend = (refl_dir[..., 1] + 1.0) * 0.5
    sky_color = torch.stack([
        0.4 + sky_blend * 0.3,
        0.6 + sky_blend * 0.3,
        0.9 + sky_blend * 0.1
    ], dim=-1)
    
    # Sun
    sun_dir = F.normalize(torch.tensor([0.4, 0.8, -0.3], device=device), dim=0)
    sun_spec = torch.pow(torch.clamp((refl_dir * sun_dir).sum(-1), 0.0, 1.0), 96.0)
    sun_color = torch.tensor([1.0, 0.95, 0.85], device=device)
    
    # Reflected solid
    light = F.normalize(torch.tensor([0.4, 0.9, -0.3], device=device), dim=0)
    p_refl = refl_ro + t_refl.unsqueeze(-1) * refl_dir
    n_refl = calc_normal(scene.scene_solid, p_refl)
    refl_diff = torch.clamp((n_refl * light).sum(-1), 0.15, 1.0)
    
    body_colors = scene.get_body_color(p_refl)
    floor_color = torch.tensor([0.35, 0.3, 0.25], device=device)
    
    body_dist = scene.scene_bodies(p_refl)
    floor_dist = sdf_floor(p_refl)
    is_body = body_dist < floor_dist
    
    refl_base = torch.where(is_body.unsqueeze(-1), body_colors, floor_color.expand(n_pix, 3))
    refl_solid = refl_diff.unsqueeze(-1) * refl_base
    
    refl_color = torch.where(hit_refl.unsqueeze(-1), refl_solid, sky_color)
    refl_color = refl_color + sun_spec.unsqueeze(-1) * sun_color * 2.5
    
    # Refraction
    water_tint = torch.tensor([0.1, 0.25, 0.35], device=device)
    underwater_depth = torch.clamp(t_water * 0.2, 0.0, 3.0)
    absorption = torch.exp(-underwater_depth.unsqueeze(-1) * torch.tensor([0.2, 0.1, 0.05], device=device))
    
    refr_dir = F.normalize(rd + n_water * 0.15, dim=-1)
    refr_ro = p_water - n_water * 0.02
    t_refr, hit_refr = ray_march(refr_ro, refr_dir, scene.scene_solid, max_steps=50)
    
    p_under = refr_ro + t_refr.unsqueeze(-1) * refr_dir
    n_under = calc_normal(scene.scene_solid, p_under)
    under_diff = torch.clamp((n_under * light).sum(-1), 0.1, 1.0)
    
    under_colors = scene.get_body_color(p_under)
    body_dist_u = scene.scene_bodies(p_under)
    floor_dist_u = sdf_floor(p_under)
    is_body_u = body_dist_u < floor_dist_u
    
    under_base = torch.where(is_body_u.unsqueeze(-1), under_colors, floor_color.expand(n_pix, 3))
    under_color = under_diff.unsqueeze(-1) * under_base * absorption
    
    deep = water_tint * 0.25
    refr_color = torch.where(hit_refr.unsqueeze(-1), under_color, deep.expand(n_pix, 3))
    
    # Caustics
    c1 = torch.sin(p_water[..., 0] * 2.5 + t * 1.5)
    c2 = torch.sin(p_water[..., 2] * 3.0 + t * 1.3)
    caustics = torch.clamp((c1 * c2) * 0.4 + 0.5, 0.0, 1.0) ** 2 * 0.3 * torch.exp(-t_water * 0.12)
    refr_color = refr_color + caustics.unsqueeze(-1) * torch.tensor([0.15, 0.25, 0.35], device=device)
    
    # Foam
    wave_h = -scene.scene_water(p_water)
    foam = torch.clamp((wave_h - 0.4) * 3.0, 0.0, 1.0)
    foam_color = torch.tensor([0.95, 0.97, 1.0], device=device)
    
    # Combine water
    water_color = fres.unsqueeze(-1) * refl_color + (1.0 - fres.unsqueeze(-1)) * refr_color
    water_color = water_color * (1.0 - foam.unsqueeze(-1)) + foam_color * foam.unsqueeze(-1)
    water_color = water_color * 0.9 + water_tint * 0.1
    
    # Solid above water
    solid_diff = torch.clamp((n_solid * light).sum(-1), 0.15, 1.0)
    solid_colors = scene.get_body_color(p_solid)
    body_dist_s = scene.scene_bodies(p_solid)
    floor_dist_s = sdf_floor(p_solid)
    is_body_s = body_dist_s < floor_dist_s
    
    solid_base = torch.where(is_body_s.unsqueeze(-1), solid_colors, floor_color.expand(n_pix, 3))
    solid_color = solid_diff.unsqueeze(-1) * solid_base
    
    # Specular
    half_vec = F.normalize(light + view, dim=-1)
    solid_spec = torch.pow(torch.clamp((n_solid * half_vec).sum(-1), 0.0, 1.0), 32.0)
    solid_color = solid_color + solid_spec.unsqueeze(-1) * 0.3
    
    # Composite
    water_front = (t_water < t_solid) & hit_water
    solid_vis = hit_solid & ~water_front
    
    # Sky bg
    sky_y = (y.flatten() + 1.0) * 0.5
    sky_bg = torch.stack([
        0.45 + sky_y * 0.2,
        0.6 + sky_y * 0.3,
        0.88 + sky_y * 0.12
    ], dim=-1)
    
    sun_disk = torch.clamp(1.0 - torch.norm(rd - sun_dir, dim=-1) * 6.0, 0.0, 1.0) ** 0.5
    sky_bg = sky_bg + sun_disk.unsqueeze(-1) * sun_color * 2.0
    
    rgb = sky_bg
    rgb = torch.where(solid_vis.unsqueeze(-1), solid_color, rgb)
    rgb = torch.where(water_front.unsqueeze(-1), water_color, rgb)
    
    # Fog
    min_t = torch.where(water_front, t_water, torch.where(hit_solid, t_solid, torch.full_like(t_water, MAX_DIST)))
    fog = torch.exp(-min_t * 0.012)
    fog_color = torch.tensor([0.6, 0.7, 0.85], device=device)
    rgb = rgb * fog.unsqueeze(-1) + fog_color * (1.0 - fog.unsqueeze(-1))
    
    # Tone map
    rgb = rgb / (rgb + 1.0)
    
    return torch.clamp(rgb.reshape(height, width, 3), 0.0, 1.0)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("⚛️ NEWTONIAN SOFT BODY PHYSICS ⚛️")
    
    scene = NewtonianScene(device=device)
    
    # Warm up physics
    print("\nWarm up physics...")
    for i in range(60):
        scene.physics.step(1/30)
    scene.time = scene.physics.time
    
    # Test frame
    print("Test frame...")
    img = render(scene, 800, 600)
    img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np, 'RGB').save('newtonian_test.png')
    print("Saved: newtonian_test.png")
    
    # Reset
    scene = NewtonianScene(device=device)
    
    # Animation
    print("\nRendering 300 frames (10 seconds)...")
    frames = []
    dt = 1.0 / 30.0
    
    for i in range(300):
        # Physics substeps for stability
        for _ in range(2):
            scene.physics.step(dt / 2)
        scene.time = scene.physics.time
        
        img = render(scene, 640, 480)
        img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        frames.append(Image.fromarray(img_np, 'RGB'))
        
        if i % 50 == 0:
            print(f"  Frame {i:3d}/300")
            for j, body in enumerate(scene.physics.soft_bodies[:3]):
                com = body.compute_center_of_mass()
                print(f"    Body {j}: pos=[{com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}]")
    
    print("\nSaving GIF...")
    frames[0].save('newtonian.gif', save_all=True, append_images=frames[1:], duration=33, loop=0)
    print("Saved: newtonian.gif")
    
    try:
        import imageio
        print("Saving MP4...")
        writer = imageio.get_writer('newtonian.mp4', fps=30, codec='libx264', quality=8)
        for f in frames:
            writer.append_data(np.array(f))
        writer.close()
        print("Saved: newtonian.mp4")
    except:
        print("(imageio not available)")
    
    print("\n✅ Done!")
    print("mpv newtonian.mp4")
