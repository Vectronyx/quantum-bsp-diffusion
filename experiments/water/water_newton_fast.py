#!/usr/bin/env python3
"""
FAST Newtonian Physics - Rigid bodies, proper forces
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
from dataclasses import dataclass

EPSILON = 1e-4
MAX_STEPS = 80
MAX_DIST = 50.0

@dataclass
class RigidBody:
    pos: torch.Tensor
    vel: torch.Tensor
    rotation: float
    angular_vel: float
    mass: float
    volume: float
    drag_coeff: float
    shape: str
    size: torch.Tensor
    color: torch.Tensor

class NewtonianPhysics:
    def __init__(self, device='cuda'):
        self.device = device
        self.g = 9.81
        self.water_density = 1000.0
        self.bodies = []
        self.time = 0.0
    
    def add_sphere(self, pos, radius, mass, color=None):
        volume = (4/3) * math.pi * radius**3
        body = RigidBody(
            pos=torch.tensor(pos, device=self.device, dtype=torch.float32),
            vel=torch.zeros(3, device=self.device),
            rotation=0.0, angular_vel=0.0,
            mass=mass, volume=volume, drag_coeff=0.47,
            shape='sphere',
            size=torch.tensor([radius], device=self.device),
            color=torch.tensor(color or [0.8, 0.4, 0.4], device=self.device)
        )
        self.bodies.append(body)
    
    def add_box(self, pos, size, mass, color=None):
        volume = size[0] * size[1] * size[2]
        body = RigidBody(
            pos=torch.tensor(pos, device=self.device, dtype=torch.float32),
            vel=torch.zeros(3, device=self.device),
            rotation=0.0, angular_vel=0.0,
            mass=mass, volume=volume, drag_coeff=1.05,
            shape='box',
            size=torch.tensor([s/2 for s in size], device=self.device),
            color=torch.tensor(color or [0.6, 0.5, 0.3], device=self.device)
        )
        self.bodies.append(body)
    
    def water_height(self, x, z):
        t = self.time
        h = 0.4 * math.sin(x * 0.3 + t * 0.7)
        h += 0.35 * math.sin(z * 0.35 + t * 0.8)
        h += 0.25 * math.sin((x + z) * 0.2 + t * 0.5)
        h += 0.15 * math.sin(x * 0.7 - z * 0.5 + t * 1.2)
        return h
    
    def water_velocity(self, x, z):
        t = self.time
        vx = 0.3 * math.cos(x * 0.3 + t * 0.7)
        vz = 0.3 * math.cos(z * 0.35 + t * 0.8)
        return torch.tensor([vx, 0., vz], device=self.device)
    
    def submerged_fraction(self, body):
        water_h = self.water_height(body.pos[0].item(), body.pos[2].item())
        if body.shape == 'sphere':
            r = body.size[0].item()
            bottom = body.pos[1].item() - r
            top = body.pos[1].item() + r
            if bottom >= water_h:
                return 0.0
            elif top <= water_h:
                return 1.0
            else:
                return min(1.0, (water_h - bottom) / (2 * r))
        else:
            half_h = body.size[1].item()
            bottom = body.pos[1].item() - half_h
            top = body.pos[1].item() + half_h
            if bottom >= water_h:
                return 0.0
            elif top <= water_h:
                return 1.0
            else:
                return (water_h - bottom) / (2 * half_h)
    
    def step(self, dt):
        self.time += dt
        for body in self.bodies:
            force = torch.zeros(3, device=self.device)
            
            # Gravity
            force[1] -= body.mass * self.g
            
            # Buoyancy
            submerged = self.submerged_fraction(body)
            if submerged > 0:
                buoyancy = self.water_density * body.volume * submerged * self.g
                force[1] += buoyancy
            
            # Drag
            speed = torch.norm(body.vel).item()
            if speed > 0.001:
                if body.shape == 'sphere':
                    area = math.pi * body.size[0].item()**2
                else:
                    area = body.size[0].item() * body.size[1].item() * 4
                
                rho = self.water_density if submerged > 0.5 else 1.2
                
                if submerged > 0:
                    water_vel = self.water_velocity(body.pos[0].item(), body.pos[2].item())
                    relative_vel = body.vel - water_vel * submerged
                else:
                    relative_vel = body.vel
                
                rel_speed = torch.norm(relative_vel).item()
                if rel_speed > 0.001:
                    drag_mag = 0.5 * rho * body.drag_coeff * area * rel_speed**2
                    drag_dir = -relative_vel / rel_speed
                    drag_scale = 0.1 + submerged * 0.9
                    force += drag_dir * drag_mag * drag_scale
            
            # Integration
            acc = force / body.mass
            body.vel += acc * dt
            body.pos += body.vel * dt
            
            # Rotation from waves
            if submerged > 0.1:
                wave_tilt = math.sin(body.pos[0].item() * 0.5 + self.time)
                body.angular_vel = body.angular_vel * 0.98 + wave_tilt * 0.1
                body.rotation += body.angular_vel * dt
            
            # Floor collision
            floor_y = -4.0
            min_y = floor_y + (body.size[0].item() if body.shape == 'sphere' else body.size[1].item())
            if body.pos[1].item() < min_y:
                body.pos[1] = min_y
                body.vel[1] = abs(body.vel[1].item()) * 0.3
                body.vel[0] *= 0.9
                body.vel[2] *= 0.9
            
            # Walls
            for dim in [0, 2]:
                if body.pos[dim].item() > 12:
                    body.pos[dim] = 12.0
                    body.vel[dim] *= -0.5
                elif body.pos[dim].item() < -12:
                    body.pos[dim] = -12.0
                    body.vel[dim] *= -0.5

def sdf_sphere(p, center, radius):
    return torch.norm(p - center, dim=-1) - radius

def sdf_box(p, center, half_size, rotation=0.0):
    c, s = math.cos(-rotation), math.sin(-rotation)
    local = p - center
    rotated = torch.stack([c * local[..., 0] + s * local[..., 2], local[..., 1], -s * local[..., 0] + c * local[..., 2]], dim=-1)
    q = torch.abs(rotated) - half_size
    return torch.norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0.0)

def op_smooth_union(d1, d2, k=0.2):
    h = torch.clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return torch.lerp(d2, d1, h) - k * h * (1.0 - h)

class Scene:
    def __init__(self, device='cuda'):
        self.device = device
        self.physics = NewtonianPhysics(device)
        self.time = 0.0
        self._spawn()
    
    def _spawn(self):
        self.physics.add_sphere([0., 2., 0.], 0.5, 0.3, [1.0, 0.4, 0.4])
        self.physics.add_sphere([3., 1., 2.], 0.4, 0.2, [0.4, 0.8, 1.0])
        self.physics.add_sphere([-2., 1.5, -2.], 0.45, 0.25, [0.5, 1.0, 0.5])
        self.physics.add_sphere([4., 0.5, -3.], 0.35, 0.15, [1.0, 0.8, 0.3])
        self.physics.add_sphere([0., 4., 3.], 0.6, 2.0, [0.3, 0.3, 0.4])  # Heavy
        self.physics.add_box([-3., 2., 1.], [0.6, 0.4, 0.6], 0.4, [0.7, 0.5, 0.3])
        self.physics.add_box([2., 1., -4.], [0.5, 0.5, 0.5], 0.3, [0.8, 0.6, 0.4])
        self.physics.add_box([-4., 3., -2.], [0.7, 0.5, 0.7], 1.5, [0.5, 0.4, 0.35])  # Heavy
    
    def scene_water(self, p):
        t = self.time
        h = torch.zeros(p.shape[0], device=self.device)
        h += 0.4 * torch.sin(p[..., 0] * 0.3 + t * 0.7)
        h += 0.35 * torch.sin(p[..., 2] * 0.35 + t * 0.8)
        h += 0.25 * torch.sin((p[..., 0] + p[..., 2]) * 0.2 + t * 0.5)
        h += 0.15 * torch.sin(p[..., 0] * 0.7 - p[..., 2] * 0.5 + t * 1.2)
        h += 0.1 * torch.sin(p[..., 0] * 1.0 + p[..., 2] * 0.8 + t * 1.5)
        h += 0.04 * torch.sin(p[..., 0] * 2.5 + t * 2.5)
        return p[..., 1] - h
    
    def scene_bodies(self, p):
        result = torch.full((p.shape[0],), 999.0, device=self.device)
        for body in self.physics.bodies:
            if body.shape == 'sphere':
                d = sdf_sphere(p, body.pos, body.size[0].item())
            else:
                d = sdf_box(p, body.pos, body.size, body.rotation)
            result = op_smooth_union(result, d, 0.1)
        return result
    
    def scene_floor(self, p):
        return p[..., 1] + 4.0
    
    def scene_solid(self, p):
        return op_smooth_union(self.scene_floor(p), self.scene_bodies(p), 0.15)
    
    def get_body_color(self, p):
        colors = torch.full((p.shape[0], 3), 0.35, device=self.device)
        min_dist = torch.full((p.shape[0],), 999.0, device=self.device)
        for body in self.physics.bodies:
            if body.shape == 'sphere':
                d = sdf_sphere(p, body.pos, body.size[0].item())
            else:
                d = sdf_box(p, body.pos, body.size, body.rotation)
            mask = d < min_dist
            min_dist = torch.where(mask, d, min_dist)
            colors = torch.where(mask.unsqueeze(-1), body.color.expand(p.shape[0], 3), colors)
        return colors

def calc_normal(sdf, p):
    e = torch.tensor([EPSILON, 0., 0.], device=p.device)
    return F.normalize(torch.stack([
        sdf(p + e.roll(0)) - sdf(p - e.roll(0)),
        sdf(p + e.roll(1)) - sdf(p - e.roll(1)),
        sdf(p + e.roll(2)) - sdf(p - e.roll(2))
    ], dim=-1), dim=-1)

def ray_march(ro, rd, sdf):
    t = torch.zeros(ro.shape[0], device=ro.device)
    for _ in range(MAX_STEPS):
        d = sdf(ro + t.unsqueeze(-1) * rd)
        t = t + d * 0.7
        if (d < EPSILON).all() or (t > MAX_DIST).all():
            break
    return t, d < EPSILON

def render(scene, W=640, H=480):
    dev = scene.device
    t = scene.time
    
    angle = t * 0.12
    dist = 16.0
    cam = torch.tensor([math.sin(angle)*dist, 4.5 + math.sin(t*0.15), math.cos(angle)*dist], device=dev)
    fwd = F.normalize(torch.tensor([0., -0.3, 0.], device=dev) - cam, dim=0)
    right = F.normalize(torch.linalg.cross(torch.tensor([0., 1., 0.], device=dev), fwd), dim=0)
    up = torch.linalg.cross(fwd, right)
    
    asp = W / H
    y, x = torch.meshgrid(torch.linspace(1, -1, H, device=dev), torch.linspace(-asp, asp, W, device=dev), indexing='ij')
    
    N = H * W
    ro = cam.unsqueeze(0).expand(N, 3)
    rd = F.normalize(x.flatten().unsqueeze(-1)*right + y.flatten().unsqueeze(-1)*up + fwd*1.8, dim=-1)
    
    tw, hitw = ray_march(ro, rd, scene.scene_water)
    pw = ro + tw.unsqueeze(-1) * rd
    nw = calc_normal(scene.scene_water, pw)
    
    ts, hits = ray_march(ro, rd, scene.scene_solid)
    ps = ro + ts.unsqueeze(-1) * rd
    ns = calc_normal(scene.scene_solid, ps)
    
    view = -rd
    fres = 0.02 + 0.98 * torch.pow(1.0 - torch.clamp((nw * view).sum(-1), 0, 1), 5)
    
    refl = rd - 2.0 * (rd * nw).sum(-1, keepdim=True) * nw
    
    # FIX: All tensor elements
    sky_r = 0.4 + (refl[..., 1] + 1) * 0.15
    sky_g = 0.6 + (refl[..., 1] + 1) * 0.15
    sky_b = torch.full_like(sky_r, 0.9)
    sky = torch.stack([sky_r, sky_g, sky_b], dim=-1)
    
    sun = F.normalize(torch.tensor([0.4, 0.8, -0.3], device=dev), dim=0)
    spec = torch.pow(torch.clamp((refl * sun).sum(-1), 0, 1), 96)
    
    light = F.normalize(torch.tensor([0.4, 0.9, -0.3], device=dev), dim=0)
    
    refl_ro = pw + nw * 0.02
    tr, hitr = ray_march(refl_ro, refl, scene.scene_solid)
    pr = refl_ro + tr.unsqueeze(-1) * refl
    nr = calc_normal(scene.scene_solid, pr)
    rdiff = torch.clamp((nr * light).sum(-1), 0.15, 1)
    rcol = scene.get_body_color(pr)
    rsolid = rdiff.unsqueeze(-1) * rcol
    rcolor = torch.where(hitr.unsqueeze(-1), rsolid, sky) + spec.unsqueeze(-1) * 2
    
    tint = torch.tensor([0.1, 0.25, 0.35], device=dev)
    absorb = torch.exp(-tw.unsqueeze(-1) * 0.15 * torch.tensor([0.2, 0.1, 0.05], device=dev))
    refr_d = F.normalize(rd + nw * 0.12, dim=-1)
    refr_ro = pw - nw * 0.02
    tref, hitref = ray_march(refr_ro, refr_d, scene.scene_solid)
    pref = refr_ro + tref.unsqueeze(-1) * refr_d
    nref = calc_normal(scene.scene_solid, pref)
    udiff = torch.clamp((nref * light).sum(-1), 0.1, 1)
    ucol = scene.get_body_color(pref)
    ucolor = udiff.unsqueeze(-1) * ucol * absorb
    deep = tint * 0.2
    refr_color = torch.where(hitref.unsqueeze(-1), ucolor, deep.expand(N, 3))
    
    caust = torch.sin(pw[..., 0]*2.5 + t*1.5) * torch.sin(pw[..., 2]*3 + t*1.3)
    caust = torch.clamp(caust * 0.4 + 0.5, 0, 1)**2 * 0.25 * torch.exp(-tw * 0.1)
    refr_color = refr_color + caust.unsqueeze(-1) * torch.tensor([0.1, 0.2, 0.3], device=dev)
    
    wh = -scene.scene_water(pw)
    foam = torch.clamp((wh - 0.35) * 3, 0, 1)
    
    water = fres.unsqueeze(-1) * rcolor + (1 - fres.unsqueeze(-1)) * refr_color
    water = water * (1 - foam.unsqueeze(-1)) + foam.unsqueeze(-1) * 0.95
    water = water * 0.9 + tint * 0.1
    
    sdiff = torch.clamp((ns * light).sum(-1), 0.15, 1)
    scol = scene.get_body_color(ps)
    solid = sdiff.unsqueeze(-1) * scol
    half = F.normalize(light + view, dim=-1)
    sspec = torch.pow(torch.clamp((ns * half).sum(-1), 0, 1), 32)
    solid = solid + sspec.unsqueeze(-1) * 0.3
    
    wfront = (tw < ts) & hitw
    svis = hits & ~wfront
    
    skyy = (y.flatten() + 1) * 0.5
    skybg = torch.stack([0.45 + skyy*0.2, 0.6 + skyy*0.3, 0.88 + skyy*0.12], dim=-1)
    sundisk = torch.clamp(1 - torch.norm(rd - sun, dim=-1) * 6, 0, 1)**0.5
    skybg = skybg + sundisk.unsqueeze(-1) * 2
    
    rgb = skybg
    rgb = torch.where(svis.unsqueeze(-1), solid, rgb)
    rgb = torch.where(wfront.unsqueeze(-1), water, rgb)
    
    mint = torch.where(wfront, tw, torch.where(hits, ts, torch.full_like(tw, MAX_DIST)))
    fog = torch.exp(-mint * 0.01)
    rgb = rgb * fog.unsqueeze(-1) + torch.tensor([0.6, 0.7, 0.85], device=dev) * (1 - fog.unsqueeze(-1))
    
    rgb = rgb / (rgb + 1)
    return torch.clamp(rgb.reshape(H, W, 3), 0, 1)

if __name__ == "__main__":
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {dev}")
    
    scene = Scene(dev)
    
    print("Test frame...")
    for _ in range(60):
        scene.physics.step(1/30)
    scene.time = scene.physics.time
    img = render(scene, 800, 600)
    Image.fromarray((img.cpu().numpy()*255).clip(0,255).astype(np.uint8), 'RGB').save('newton_test.png')
    print("Saved: newton_test.png")
    
    scene = Scene(dev)
    print("\nRendering 240 frames...")
    frames = []
    
    for i in range(240):
        scene.physics.step(1/30)
        scene.time = scene.physics.time
        img = render(scene, 640, 480)
        frames.append(Image.fromarray((img.cpu().numpy()*255).clip(0,255).astype(np.uint8), 'RGB'))
        if i % 40 == 0:
            print(f"  Frame {i}/240")
            for j, b in enumerate(scene.physics.bodies[:3]):
                print(f"    {j}: y={b.pos[1].item():.2f} vy={b.vel[1].item():.2f}")
    
    print("Saving...")
    frames[0].save('newton.gif', save_all=True, append_images=frames[1:], duration=33, loop=0)
    
    try:
        import imageio
        w = imageio.get_writer('newton.mp4', fps=30, codec='libx264', quality=8)
        for f in frames:
            w.append_data(np.array(f))
        w.close()
        print("Saved: newton.mp4")
    except:
        pass
    
    print("Done! mpv newton.mp4")
