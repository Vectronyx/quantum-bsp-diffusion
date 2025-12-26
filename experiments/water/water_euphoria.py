#!/usr/bin/env python3
"""
EUPHORIA WATER - Reactive, floaty, dreamy motion
Audio-reactive style physics without actual audio
Simulated frequency bands drive everything
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List

EPSILON = 1e-4
MAX_STEPS = 120
MAX_DIST = 60.0

# ============================================
# EUPHORIA MOTION SYSTEM
# ============================================

class EuphoriaEngine:
    """
    Simulates audio-reactive motion without audio
    Generates smooth, evolving frequency bands
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.time = 0.0
        
        # Simulated frequency bands (sub, bass, mid, high, presence)
        self.bands = torch.zeros(5, device=device)
        self.band_targets = torch.zeros(5, device=device)
        self.band_velocities = torch.zeros(5, device=device)
        
        # Smoothing factors per band (lower = smoother)
        self.band_smoothing = torch.tensor([0.92, 0.88, 0.75, 0.6, 0.5], device=device)
        
        # Phase accumulators for organic motion
        self.phases = torch.zeros(8, device=device)
        self.phase_speeds = torch.tensor([0.3, 0.5, 0.7, 1.1, 1.7, 2.3, 3.1, 4.3], device=device)
        
        # Global energy (overall intensity)
        self.energy = 0.0
        self.energy_target = 0.0
        
        # Pulse system (beat-like moments)
        self.pulse = 0.0
        self.pulse_decay = 0.92
        self.next_pulse = 0.0
        
        # Sway (slow global movement)
        self.sway = torch.zeros(3, device=device)
        
    def update(self, dt):
        self.time += dt
        t = self.time
        
        # Update phases
        self.phases += self.phase_speeds * dt
        
        # Generate fake "audio" from layered sine waves
        # This creates organic, evolving patterns
        
        # Sub bass - very slow, powerful
        self.band_targets[0] = (
            0.5 + 0.3 * math.sin(t * 0.4) +
            0.2 * math.sin(t * 0.17 + 1.0)
        ) ** 2
        
        # Bass - medium slow
        self.band_targets[1] = (
            0.5 + 0.25 * math.sin(t * 0.8 + 0.5) +
            0.15 * math.sin(t * 0.33) +
            0.1 * math.sin(t * 1.7)
        ) ** 1.5
        
        # Mid - more movement
        self.band_targets[2] = (
            0.4 + 0.2 * math.sin(t * 1.3) +
            0.15 * math.sin(t * 2.1 + 2.0) +
            0.1 * math.sin(t * 0.7) +
            0.05 * math.sin(t * 4.3)
        )
        
        # High - fast, sparkly
        self.band_targets[3] = (
            0.3 + 0.2 * math.sin(t * 2.5 + 1.0) +
            0.15 * math.sin(t * 3.7) +
            0.1 * math.sin(t * 5.3 + 3.0)
        )
        
        # Presence - very fast, shimmery
        self.band_targets[4] = (
            0.2 + 0.15 * math.sin(t * 4.0) +
            0.1 * math.sin(t * 6.1 + 2.0) +
            0.08 * math.sin(t * 8.7)
        )
        
        # Smooth band transitions (spring-damper style)
        for i in range(5):
            diff = self.band_targets[i] - self.bands[i]
            self.band_velocities[i] = self.band_velocities[i] * self.band_smoothing[i] + diff * (1 - self.band_smoothing[i]) * 2
            self.bands[i] = self.bands[i] + self.band_velocities[i] * dt * 10
        
        # Clamp bands
        self.bands = torch.clamp(self.bands, 0.0, 1.0)
        
        # Overall energy
        self.energy_target = (self.bands[0] * 0.4 + self.bands[1] * 0.3 + self.bands[2] * 0.2 + self.bands[3] * 0.1).item()
        self.energy = self.energy * 0.95 + self.energy_target * 0.05
        
        # Pulse system - occasional "beats"
        self.pulse *= self.pulse_decay
        if t >= self.next_pulse:
            # Random-ish pulse timing based on sub bass
            pulse_strength = 0.3 + self.bands[0].item() * 0.7
            self.pulse = pulse_strength
            # Next pulse timing - synced to low frequency
            base_interval = 0.5 + (1.0 - self.bands[0].item()) * 1.0
            self.next_pulse = t + base_interval + math.sin(t * 0.1) * 0.3
        
        # Global sway
        self.sway[0] = math.sin(t * 0.3) * 0.5 + math.sin(t * 0.17) * 0.3
        self.sway[1] = math.sin(t * 0.25 + 1.0) * 0.3 + math.sin(t * 0.13) * 0.2
        self.sway[2] = math.sin(t * 0.35 + 2.0) * 0.4 + math.sin(t * 0.21) * 0.25
    
    def get_sub(self):
        return self.bands[0].item()
    
    def get_bass(self):
        return self.bands[1].item()
    
    def get_mid(self):
        return self.bands[2].item()
    
    def get_high(self):
        return self.bands[3].item()
    
    def get_presence(self):
        return self.bands[4].item()
    
    def get_energy(self):
        return self.energy
    
    def get_pulse(self):
        return self.pulse
    
    def get_sway(self):
        return self.sway

# ============================================
# EUPHORIA PHYSICS BODY
# ============================================

@dataclass
class EuphoriaBody:
    pos: torch.Tensor
    vel: torch.Tensor
    target_pos: torch.Tensor  # Where it wants to float to
    home_pos: torch.Tensor    # Original spawn position
    
    # Motion characteristics
    mass: float = 1.0
    buoyancy: float = 0.7
    drag: float = 0.3
    spring: float = 0.5       # How strongly it returns to target
    damping: float = 0.8      # Velocity damping
    
    # Reactive multipliers
    react_sub: float = 1.0    # How much sub bass affects it
    react_bass: float = 1.0
    react_mid: float = 1.0
    react_high: float = 0.5
    react_pulse: float = 1.0
    
    # Visual
    shape: str = 'sphere'
    size: torch.Tensor = None
    color: torch.Tensor = None
    glow: float = 0.0         # Current glow intensity
    rotation: float = 0.0
    spin: float = 0.0         # Angular velocity
    
    # Phase offsets for unique motion
    phase_offset: float = 0.0

class EuphoriaWorld:
    def __init__(self, device='cuda'):
        self.device = device
        self.euphoria = EuphoriaEngine(device)
        self.bodies: List[EuphoriaBody] = []
        self.time = 0.0
        self.gravity = torch.tensor([0., -3.0, 0.], device=device)  # Lighter gravity
    
    def add_body(self, pos, shape='sphere', size=None, color=None, **kwargs):
        if size is None:
            size = torch.tensor([0.5], device=self.device)
        if color is None:
            color = torch.rand(3, device=self.device) * 0.5 + 0.4
        
        pos_t = torch.tensor(pos, device=self.device, dtype=torch.float32)
        
        body = EuphoriaBody(
            pos=pos_t.clone(),
            vel=torch.zeros(3, device=self.device),
            target_pos=pos_t.clone(),
            home_pos=pos_t.clone(),
            shape=shape,
            size=size if isinstance(size, torch.Tensor) else torch.tensor(size, device=self.device, dtype=torch.float32),
            color=color if isinstance(color, torch.Tensor) else torch.tensor(color, device=self.device, dtype=torch.float32),
            phase_offset=len(self.bodies) * 0.7,  # Unique phase per body
            **kwargs
        )
        self.bodies.append(body)
        return body
    
    def get_water_height(self, x, z, t, euphoria):
        """Water height influenced by euphoria bands"""
        h = 0.0
        
        sub = euphoria.get_sub()
        bass = euphoria.get_bass()
        mid = euphoria.get_mid()
        high = euphoria.get_high()
        pulse = euphoria.get_pulse()
        
        # Sub bass = big slow swells
        h += (0.4 + sub * 0.6) * math.sin(x * 0.2 + t * 0.5)
        h += (0.3 + sub * 0.5) * math.sin(z * 0.25 + t * 0.6)
        
        # Bass = medium waves
        h += (0.2 + bass * 0.3) * math.sin(x * 0.5 + z * 0.3 + t * 1.0)
        h += (0.15 + bass * 0.25) * math.sin(x * 0.7 - z * 0.4 + t * 1.2)
        
        # Mid = choppier waves
        h += (0.1 + mid * 0.15) * math.sin(x * 1.2 + t * 1.8)
        h += (0.08 + mid * 0.12) * math.sin(z * 1.5 + t * 2.0)
        
        # High = ripples
        h += (0.03 + high * 0.05) * math.sin(x * 3.0 + t * 3.5)
        h += (0.02 + high * 0.04) * math.sin(z * 3.5 + t * 4.0)
        
        # Pulse = sudden bump
        h += pulse * 0.3 * math.exp(-((x*x + z*z) * 0.05))
        
        return h
    
    def step(self, dt):
        self.time += dt
        t = self.time
        
        # Update euphoria engine
        self.euphoria.update(dt)
        
        sub = self.euphoria.get_sub()
        bass = self.euphoria.get_bass()
        mid = self.euphoria.get_mid()
        high = self.euphoria.get_high()
        pulse = self.euphoria.get_pulse()
        energy = self.euphoria.get_energy()
        sway = self.euphoria.get_sway()
        
        for body in self.bodies:
            phase = t + body.phase_offset
            
            # Update target position based on euphoria
            # Bodies float around their home position
            target_offset = torch.zeros(3, device=self.device)
            
            # Sub bass = slow vertical bob
            target_offset[1] += math.sin(phase * 0.5) * sub * body.react_sub * 0.8
            
            # Bass = lateral sway
            target_offset[0] += math.sin(phase * 0.7) * bass * body.react_bass * 0.5
            target_offset[2] += math.cos(phase * 0.6) * bass * body.react_bass * 0.4
            
            # Mid = faster movement
            target_offset[0] += math.sin(phase * 1.3) * mid * body.react_mid * 0.3
            target_offset[1] += math.sin(phase * 1.5 + 1.0) * mid * body.react_mid * 0.2
            target_offset[2] += math.sin(phase * 1.1 + 2.0) * mid * body.react_mid * 0.25
            
            # High = jitter
            target_offset[0] += math.sin(phase * 3.0) * high * body.react_high * 0.1
            target_offset[1] += math.sin(phase * 3.5) * high * body.react_high * 0.08
            target_offset[2] += math.sin(phase * 2.7) * high * body.react_high * 0.1
            
            # Global sway
            target_offset[0] += sway[0].item() * 0.3
            target_offset[2] += sway[2].item() * 0.3
            
            body.target_pos = body.home_pos + target_offset
            
            # Water interaction
            water_h = self.get_water_height(body.pos[0].item(), body.pos[2].item(), t, self.euphoria)
            submerged = water_h - body.pos[1].item()
            submerged_ratio = max(0.0, min(1.0, submerged / (body.size[0].item() * 2) + 0.5))
            
            # Forces
            force = self.gravity * body.mass * (1.0 - energy * 0.5)  # Less gravity when energetic
            
            # Buoyancy
            if submerged > -1.0:
                buoy = body.buoyancy * submerged_ratio * 9.8 * body.mass * (1.0 + sub * 0.5)
                force[1] += buoy
            
            # Spring force toward target (dreamy float)
            to_target = body.target_pos - body.pos
            spring_force = to_target * body.spring * (1.0 + energy * 0.5)
            force = force + spring_force
            
            # Pulse kick
            if pulse > 0.1:
                kick_dir = torch.tensor([
                    math.sin(body.phase_offset),
                    0.5 + body.react_pulse * 0.3,
                    math.cos(body.phase_offset)
                ], device=self.device)
                force = force + kick_dir * pulse * body.react_pulse * 15.0
            
            # Water drag
            drag_mult = 1.0 + submerged_ratio * 3.0
            drag_force = -body.vel * body.drag * drag_mult
            force = force + drag_force
            
            # Wave push
            if submerged > -0.5:
                wave_push = torch.tensor([
                    math.sin(body.pos[0].item() * 0.3 + t) * (0.5 + bass * 0.5),
                    0.0,
                    math.cos(body.pos[2].item() * 0.4 + t * 1.1) * (0.4 + bass * 0.4)
                ], device=self.device)
                force = force + wave_push * submerged_ratio
            
            # Integration
            acc = force / body.mass
            body.vel = body.vel * body.damping + acc * dt
            body.pos = body.pos + body.vel * dt
            
            # Spin based on mid frequencies
            body.spin = body.spin * 0.95 + mid * 0.5 * math.sin(phase * 0.8)
            body.rotation += body.spin * dt
            
            # Glow based on high + pulse
            target_glow = high * 0.5 + pulse * 0.8
            body.glow = body.glow * 0.9 + target_glow * 0.1
            
            # Soft world bounds
            for dim in [0, 2]:
                if abs(body.pos[dim].item()) > 12:
                    body.vel[dim] *= 0.8
                    body.pos[dim] = torch.clamp(body.pos[dim], -12., 12.)
            body.pos[1] = torch.clamp(body.pos[1], -4., 6.)

# ============================================
# SDF
# ============================================

def sdf_sphere(p, r=1.0):
    return torch.norm(p, dim=-1) - r

def sdf_box(p, b):
    q = torch.abs(p) - b
    return torch.norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0.0)

def sdf_capsule(p, h=1.0, r=0.3):
    p_y = torch.clamp(p[..., 1], -h, h)
    return torch.norm(p - torch.stack([torch.zeros_like(p_y), p_y, torch.zeros_like(p_y)], dim=-1), dim=-1) - r

def sdf_torus(p, R=1.0, r=0.25):
    q = torch.stack([torch.norm(p[..., [0,2]], dim=-1) - R, p[..., 1]], dim=-1)
    return torch.norm(q, dim=-1) - r

def op_smooth_union(d1, d2, k=0.2):
    h = torch.clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return torch.lerp(d2, d1, h) - k * h * (1.0 - h)

def op_rotate_y(p, angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.stack([c * p[..., 0] + s * p[..., 2], p[..., 1], -s * p[..., 0] + c * p[..., 2]], dim=-1)

# ============================================
# SCENE
# ============================================

class EuphoriaScene:
    def __init__(self, device='cuda'):
        self.device = device
        self.world = EuphoriaWorld(device)
        self.time = 0.0
        self._spawn_bodies()
    
    def _spawn_bodies(self):
        # Orbs - highly reactive to different frequencies
        self.world.add_body([0., 0.5, 0.], shape='sphere', size=[0.6],
            color=[1.0, 0.4, 0.6], react_sub=1.5, react_pulse=1.2, buoyancy=0.85, spring=0.3)
        
        self.world.add_body([3., 1., 2.], shape='sphere', size=[0.45],
            color=[0.4, 0.8, 1.0], react_bass=1.5, react_mid=1.2, buoyancy=0.9, spring=0.4)
        
        self.world.add_body([-2., 0.8, -3.], shape='sphere', size=[0.5],
            color=[0.6, 1.0, 0.5], react_mid=1.5, react_high=1.5, buoyancy=0.88, spring=0.35)
        
        self.world.add_body([4., 0.3, -2.], shape='sphere', size=[0.35],
            color=[1.0, 0.8, 0.3], react_high=2.0, react_pulse=0.8, buoyancy=0.95, spring=0.5)
        
        self.world.add_body([-4., 0.6, 1.], shape='sphere', size=[0.55],
            color=[0.8, 0.4, 1.0], react_sub=1.2, react_bass=1.3, buoyancy=0.82, spring=0.25)
        
        # Cubes - heavier, react to bass
        self.world.add_body([2., 0.4, -4.], shape='box', size=[0.4, 0.3, 0.4],
            color=[0.9, 0.5, 0.3], react_sub=1.8, react_bass=1.5, buoyancy=0.65, spring=0.2, mass=1.5)
        
        self.world.add_body([-3., 0.5, 3.], shape='box', size=[0.35, 0.35, 0.35],
            color=[0.4, 0.6, 0.9], react_bass=1.6, react_mid=1.0, buoyancy=0.7, spring=0.25, mass=1.2)
        
        # Torus - floaty, mid reactive
        self.world.add_body([0., 1.2, 5.], shape='torus', size=[0.6, 0.15],
            color=[1.0, 0.6, 0.8], react_mid=1.8, react_high=1.3, buoyancy=0.92, spring=0.4, mass=0.6)
        
        # Capsules - buoys
        self.world.add_body([-5., 0.3, -1.], shape='capsule', size=[0.5, 0.2],
            color=[1.0, 0.9, 0.4], react_pulse=1.5, react_sub=1.0, buoyancy=0.88, spring=0.3)
        
        self.world.add_body([5., 0.4, 3.], shape='capsule', size=[0.4, 0.15],
            color=[0.5, 1.0, 0.9], react_bass=1.4, react_pulse=1.3, buoyancy=0.9, spring=0.35)
    
    def scene_water(self, p):
        t = self.time
        eu = self.world.euphoria
        
        sub = eu.get_sub()
        bass = eu.get_bass()
        mid = eu.get_mid()
        high = eu.get_high()
        pulse = eu.get_pulse()
        
        h = torch.zeros(p.shape[0], device=self.device)
        
        # Sub swells
        h += (0.5 + sub * 0.5) * torch.sin(p[..., 0] * 0.2 + t * 0.5)
        h += (0.4 + sub * 0.4) * torch.sin(p[..., 2] * 0.25 + t * 0.6)
        h += (0.3 + sub * 0.3) * torch.sin((p[..., 0] + p[..., 2]) * 0.15 + t * 0.4)
        
        # Bass waves
        h += (0.25 + bass * 0.25) * torch.sin(p[..., 0] * 0.5 + p[..., 2] * 0.3 + t * 1.0)
        h += (0.2 + bass * 0.2) * torch.sin(p[..., 0] * 0.7 - p[..., 2] * 0.5 + t * 1.3)
        
        # Mid chop
        h += (0.1 + mid * 0.12) * torch.sin(p[..., 0] * 1.3 + t * 1.8)
        h += (0.08 + mid * 0.1) * torch.sin(p[..., 2] * 1.5 + t * 2.2)
        h += (0.06 + mid * 0.08) * torch.sin((p[..., 0] - p[..., 2]) * 1.1 + t * 1.6)
        
        # High shimmer
        h += (0.03 + high * 0.04) * torch.sin(p[..., 0] * 3.5 + t * 4.0)
        h += (0.02 + high * 0.03) * torch.sin(p[..., 2] * 4.0 + t * 4.5)
        
        # Pulse ripple
        dist_sq = p[..., 0]**2 + p[..., 2]**2
        h += pulse * 0.4 * torch.exp(-dist_sq * 0.03) * torch.sin(torch.sqrt(dist_sq + 0.1) * 2.0 - t * 8.0)
        
        return p[..., 1] - h
    
    def scene_bodies(self, p):
        if not self.world.bodies:
            return torch.full((p.shape[0],), 999.0, device=self.device)
        
        scene = torch.full((p.shape[0],), 999.0, device=self.device)
        
        for body in self.world.bodies:
            p_local = op_rotate_y(p - body.pos, -body.rotation)
            
            if body.shape == 'sphere':
                # Glow expands the sphere slightly
                d = sdf_sphere(p_local, body.size[0].item() * (1.0 + body.glow * 0.2))
            elif body.shape == 'box':
                d = sdf_box(p_local, body.size * (1.0 + body.glow * 0.1))
            elif body.shape == 'capsule':
                d = sdf_capsule(p_local, body.size[0].item(), body.size[1].item() if len(body.size) > 1 else 0.2)
            elif body.shape == 'torus':
                d = sdf_torus(p_local, body.size[0].item(), body.size[1].item() if len(body.size) > 1 else 0.15)
            else:
                d = sdf_sphere(p_local, 0.5)
            
            scene = op_smooth_union(scene, d, 0.08)
        
        return scene
    
    def scene_floor(self, p):
        return p[..., 1] + 5.0 + 0.3 * torch.sin(p[..., 0] * 0.3) * torch.sin(p[..., 2] * 0.35)
    
    def scene_solid(self, p):
        return op_smooth_union(self.scene_floor(p), self.scene_bodies(p), 0.15)
    
    def get_body_color_and_glow(self, p):
        colors = torch.full((p.shape[0], 3), 0.4, device=self.device)
        glows = torch.zeros(p.shape[0], device=self.device)
        min_dist = torch.full((p.shape[0],), 999.0, device=self.device)
        
        for body in self.world.bodies:
            p_local = op_rotate_y(p - body.pos, -body.rotation)
            
            if body.shape == 'sphere':
                d = sdf_sphere(p_local, body.size[0].item())
            elif body.shape == 'box':
                d = sdf_box(p_local, body.size)
            elif body.shape == 'capsule':
                d = sdf_capsule(p_local, body.size[0].item(), body.size[1].item() if len(body.size) > 1 else 0.2)
            elif body.shape == 'torus':
                d = sdf_torus(p_local, body.size[0].item(), body.size[1].item() if len(body.size) > 1 else 0.15)
            else:
                d = sdf_sphere(p_local, 0.5)
            
            mask = d < min_dist
            min_dist = torch.where(mask, d, min_dist)
            colors = torch.where(mask.unsqueeze(-1), body.color.expand(p.shape[0], 3), colors)
            glows = torch.where(mask, torch.full_like(glows, body.glow), glows)
        
        return colors, glows

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
    eu = scene.world.euphoria
    
    energy = eu.get_energy()
    pulse = eu.get_pulse()
    sub = eu.get_sub()
    
    # Camera - dreamy sway
    sway = eu.get_sway()
    cam_angle = t * 0.1 + sway[0].item() * 0.1
    cam_dist = 16.0 + math.sin(t * 0.15) * 2.0 - energy * 2.0  # Zoom in when energetic
    cam_height = 4.5 + math.sin(t * 0.12) * 1.0 + sway[1].item() * 0.3
    
    cam_pos = torch.tensor([
        math.sin(cam_angle) * cam_dist,
        cam_height,
        math.cos(cam_angle) * cam_dist
    ], device=device)
    
    # Slight camera shake on pulse
    cam_pos[0] += pulse * (math.sin(t * 50) * 0.1)
    cam_pos[1] += pulse * (math.cos(t * 47) * 0.08)
    
    cam_target = torch.tensor([sway[0].item() * 0.5, -0.3, sway[2].item() * 0.5], device=device)
    cam_fwd = F.normalize(cam_target - cam_pos, dim=0)
    cam_right = F.normalize(torch.linalg.cross(torch.tensor([0., 1., 0.], device=device), cam_fwd), dim=0)
    cam_up = torch.linalg.cross(cam_fwd, cam_right)
    
    # FOV changes with energy
    fov_mult = 1.8 - energy * 0.3
    
    aspect = width / height
    y, x = torch.meshgrid(
        torch.linspace(1, -1, height, device=device),
        torch.linspace(-aspect, aspect, width, device=device), indexing='ij')
    
    n_pix = height * width
    ro = cam_pos.unsqueeze(0).expand(n_pix, 3)
    rd = F.normalize(x.flatten().unsqueeze(-1) * cam_right + 
                     y.flatten().unsqueeze(-1) * cam_up + 
                     cam_fwd * fov_mult, dim=-1)
    
    # March
    t_water, hit_water = ray_march(ro, rd, scene.scene_water)
    p_water = ro + t_water.unsqueeze(-1) * rd
    n_water = calc_normal(scene.scene_water, p_water)
    
    t_solid, hit_solid = ray_march(ro, rd, scene.scene_solid)
    p_solid = ro + t_solid.unsqueeze(-1) * rd
    n_solid = calc_normal(scene.scene_solid, p_solid)
    
    # Water
    view = -rd
    cos_theta = torch.clamp((n_water * view).sum(-1), 0.0, 1.0)
    fres = fresnel(cos_theta, f0=0.02 + energy * 0.02)
    
    # Reflection
    refl_dir = reflect(rd, n_water)
    refl_ro = p_water + n_water * 0.02
    t_refl, hit_refl = ray_march(refl_ro, refl_dir, scene.scene_solid, max_steps=60)
    
    # Sky - hue shifts with sub bass
    sky_blend = (refl_dir[..., 1] + 1.0) * 0.5
    hue_shift = sub * 0.15
    sky_color = torch.stack([
        0.3 + sky_blend * 0.3 + hue_shift * 0.5,
        0.5 + sky_blend * 0.35 - hue_shift * 0.2,
        0.85 + sky_blend * 0.15 - hue_shift * 0.3
    ], dim=-1)
    
    # Sun
    sun_dir = F.normalize(torch.tensor([0.3, 0.7, -0.4], device=device), dim=0)
    sun_spec = torch.pow(torch.clamp((refl_dir * sun_dir).sum(-1), 0.0, 1.0), 64.0 + energy * 64.0)
    sun_color = torch.tensor([1.0, 0.95 - sub * 0.1, 0.85 - sub * 0.15], device=device)
    
    # Reflected solid with glow
    light = F.normalize(torch.tensor([0.3, 0.9, -0.2], device=device), dim=0)
    p_refl = refl_ro + t_refl.unsqueeze(-1) * refl_dir
    n_refl = calc_normal(scene.scene_solid, p_refl)
    refl_diff = torch.clamp((n_refl * light).sum(-1), 0.15, 1.0)
    
    body_colors, body_glows = scene.get_body_color_and_glow(p_refl)
    floor_color = torch.tensor([0.35, 0.3, 0.4], device=device)
    
    body_dist = scene.scene_bodies(p_refl)
    floor_dist = scene.scene_floor(p_refl)
    is_body = body_dist < floor_dist
    
    refl_base = torch.where(is_body.unsqueeze(-1), body_colors, floor_color.expand(n_pix, 3))
    refl_glow = torch.where(is_body, body_glows, torch.zeros_like(body_glows))
    
    # Add glow emission
    refl_solid = refl_diff.unsqueeze(-1) * refl_base + refl_glow.unsqueeze(-1) * refl_base * 2.0
    
    refl_color = torch.where(hit_refl.unsqueeze(-1), refl_solid, sky_color)
    refl_color = refl_color + sun_spec.unsqueeze(-1) * sun_color * 2.5
    
    # Refraction / underwater
    water_tint = torch.tensor([0.08 + sub * 0.05, 0.22 + energy * 0.05, 0.35], device=device)
    underwater_depth = torch.clamp(t_water * 0.2, 0.0, 3.0)
    absorption = torch.exp(-underwater_depth.unsqueeze(-1) * torch.tensor([0.2, 0.1, 0.05], device=device))
    
    refr_dir = F.normalize(rd + n_water * 0.15, dim=-1)
    refr_ro = p_water - n_water * 0.02
    t_refr, hit_refr = ray_march(refr_ro, refr_dir, scene.scene_solid, max_steps=50)
    
    p_under = refr_ro + t_refr.unsqueeze(-1) * refr_dir
    n_under = calc_normal(scene.scene_solid, p_under)
    under_diff = torch.clamp((n_under * light).sum(-1), 0.1, 1.0)
    
    under_colors, under_glows = scene.get_body_color_and_glow(p_under)
    body_dist_u = scene.scene_bodies(p_under)
    floor_dist_u = scene.scene_floor(p_under)
    is_body_u = body_dist_u < floor_dist_u
    
    under_base = torch.where(is_body_u.unsqueeze(-1), under_colors, floor_color.expand(n_pix, 3))
    under_glow = torch.where(is_body_u, under_glows, torch.zeros_like(under_glows))
    
    under_color = (under_diff.unsqueeze(-1) * under_base + under_glow.unsqueeze(-1) * under_base * 1.5) * absorption
    
    deep = water_tint * 0.25
    refr_color = torch.where(hit_refr.unsqueeze(-1), under_color, deep.expand(n_pix, 3))
    
    # Caustics - reactive
    mid = eu.get_mid()
    high = eu.get_high()
    c1 = torch.sin(p_water[..., 0] * (2.0 + mid) + t * 1.5)
    c2 = torch.sin(p_water[..., 2] * (2.5 + mid) + t * 1.3)
    c3 = torch.sin((p_water[..., 0] + p_water[..., 2]) * (1.8 + high) - t * 2.0)
    caustics = torch.clamp((c1 * c2 + c3 * 0.5) * 0.35 + 0.5, 0.0, 1.0)
    caustics = caustics ** 1.5 * (0.3 + high * 0.3) * torch.exp(-t_water * 0.12)
    refr_color = refr_color + caustics.unsqueeze(-1) * torch.tensor([0.15, 0.3, 0.45], device=device)
    
    # Foam
    wave_h = -scene.scene_water(p_water)
    foam_thresh = 0.35 - energy * 0.1
    foam = torch.clamp((wave_h - foam_thresh) * 4.0, 0.0, 1.0)
    foam_color = torch.tensor([0.92, 0.96, 1.0], device=device)
    
    # Water combine
    water_color = fres.unsqueeze(-1) * refl_color + (1.0 - fres.unsqueeze(-1)) * refr_color
    water_color = water_color * (1.0 - foam.unsqueeze(-1)) + foam_color * foam.unsqueeze(-1)
    water_color = water_color * 0.88 + water_tint * 0.12
    
    # Solid
    solid_diff = torch.clamp((n_solid * light).sum(-1), 0.15, 1.0)
    solid_colors, solid_glows = scene.get_body_color_and_glow(p_solid)
    body_dist_s = scene.scene_bodies(p_solid)
    floor_dist_s = scene.scene_floor(p_solid)
    is_body_s = body_dist_s < floor_dist_s
    
    solid_base = torch.where(is_body_s.unsqueeze(-1), solid_colors, floor_color.expand(n_pix, 3))
    solid_glow = torch.where(is_body_s, solid_glows, torch.zeros_like(solid_glows))
    
    solid_color = solid_diff.unsqueeze(-1) * solid_base + solid_glow.unsqueeze(-1) * solid_base * 2.5
    
    # Specular
    half_vec = F.normalize(light + view, dim=-1)
    solid_spec = torch.pow(torch.clamp((n_solid * half_vec).sum(-1), 0.0, 1.0), 24.0)
    solid_color = solid_color + solid_spec.unsqueeze(-1) * 0.4
    
    # Composite
    water_front = (t_water < t_solid) & hit_water
    solid_vis = hit_solid & ~water_front
    
    # Sky bg
    sky_y = (y.flatten() + 1.0) * 0.5
    sky_bg = torch.stack([
        0.35 + sky_y * 0.2 + hue_shift * 0.3,
        0.5 + sky_y * 0.3 - hue_shift * 0.1,
        0.82 + sky_y * 0.18 - hue_shift * 0.2
    ], dim=-1)
    
    # Sun disk
    sun_disk = torch.clamp(1.0 - torch.norm(rd - sun_dir, dim=-1) * 6.0, 0.0, 1.0)
    sun_disk = sun_disk ** 0.4
    sky_bg = sky_bg + sun_disk.unsqueeze(-1) * sun_color * (2.0 + pulse * 2.0)
    
    rgb = sky_bg
    rgb = torch.where(solid_vis.unsqueeze(-1), solid_color, rgb)
    rgb = torch.where(water_front.unsqueeze(-1), water_color, rgb)
    
    # Fog
    min_t = torch.where(water_front, t_water, torch.where(hit_solid, t_solid, torch.full_like(t_water, MAX_DIST)))
    fog = torch.exp(-min_t * 0.012)
    fog_color = torch.tensor([0.55 + hue_shift * 0.2, 0.65, 0.82 - hue_shift * 0.1], device=device)
    rgb = rgb * fog.unsqueeze(-1) + fog_color * (1.0 - fog.unsqueeze(-1))
    
    # Vignette
    uv_x = x.flatten()
    uv_y = y.flatten()
    vignette = 1.0 - (uv_x**2 + uv_y**2) * 0.3
    vignette = torch.clamp(vignette, 0.5, 1.0)
    rgb = rgb * vignette.unsqueeze(-1)
    
    # Tone map + slight bloom simulation on pulse
    rgb = rgb * (1.0 + pulse * 0.3)
    rgb = rgb / (rgb + 1.0)
    
    # Contrast
    rgb = (rgb - 0.5) * (1.05 + energy * 0.1) + 0.5
    
    return torch.clamp(rgb.reshape(height, width, 3), 0.0, 1.0)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("🎵 EUPHORIA MODE 🎵")
    
    scene = EuphoriaScene(device=device)
    
    # Test frame
    print("\nTest frame...")
    scene.time = 5.0
    for _ in range(150):  # Warm up physics
        scene.world.step(1/30)
    scene.time = scene.world.time
    
    img = render(scene, 800, 600)
    img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np, 'RGB').save('euphoria_test.png')
    print("Saved: euphoria_test.png")
    
    # Reset for animation
    scene = EuphoriaScene(device=device)
    
    # Animation
    print("\nRendering 360 frames (12 seconds)...")
    frames = []
    dt = 1.0 / 30.0
    
    for i in range(360):
        scene.world.step(dt)
        scene.time = scene.world.time
        
        img = render(scene, 640, 480)
        img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        frames.append(Image.fromarray(img_np, 'RGB'))
        
        if i % 60 == 0:
            eu = scene.world.euphoria
            print(f"  Frame {i:3d}/360 | Energy: {eu.get_energy():.2f} | Pulse: {eu.get_pulse():.2f} | Sub: {eu.get_sub():.2f}")
    
    # GIF
    print("\nSaving GIF...")
    frames[0].save('euphoria.gif', save_all=True, append_images=frames[1:], duration=33, loop=0)
    print("Saved: euphoria.gif")
    
    # MP4
    try:
        import imageio
        print("Saving MP4...")
        writer = imageio.get_writer('euphoria.mp4', fps=30, codec='libx264', quality=8)
        for f in frames:
            writer.append_data(np.array(f))
        writer.close()
        print("Saved: euphoria.mp4")
    except:
        print("(imageio not available)")
    
    print("\n✅ Done!")
    print("mpv euphoria.mp4")
