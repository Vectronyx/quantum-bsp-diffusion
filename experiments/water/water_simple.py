#!/usr/bin/env python3
"""
Simple water + floating objects
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image

EPSILON = 1e-4
MAX_STEPS = 80
MAX_DIST = 50.0

class Body:
    def __init__(self, pos, size, shape, color, float_height=0.0, bob_speed=1.0, bob_amp=0.3, device='cuda'):
        self.device = device
        self.pos = torch.tensor(pos, device=device, dtype=torch.float32)
        self.size = torch.tensor([size] if isinstance(size, float) else size, device=device, dtype=torch.float32)
        self.shape = shape
        self.color = torch.tensor(color, device=device, dtype=torch.float32)
        self.float_height = float_height
        self.bob_speed = bob_speed
        self.bob_amp = bob_amp
        self.phase = torch.rand(1).item() * 6.28
        self.rotation = 0.0
    
    def update(self, t):
        wave_h = 0.4 * math.sin(self.pos[0].item() * 0.3 + t * 0.7)
        wave_h += 0.35 * math.sin(self.pos[2].item() * 0.35 + t * 0.8)
        target_y = wave_h + self.float_height
        self.pos[1] = self.pos[1] * 0.9 + target_y * 0.1
        self.pos[1] += math.sin(t * self.bob_speed + self.phase) * self.bob_amp * 0.1
        self.rotation += math.sin(t * 0.5 + self.phase) * 0.01

class Scene:
    def __init__(self, device='cuda'):
        self.device = device
        self.time = 0.0
        self.bodies = []
        self._spawn()
    
    def _spawn(self):
        self.bodies.append(Body([0., 0., 0.], 0.5, 'sphere', [1.0, 0.4, 0.4], 0.3, 1.2, device=self.device))
        self.bodies.append(Body([3., 0., 2.], 0.4, 'sphere', [0.4, 0.8, 1.0], 0.25, 1.0, device=self.device))
        self.bodies.append(Body([-2., 0., -2.], 0.45, 'sphere', [0.5, 1.0, 0.5], 0.28, 1.4, device=self.device))
        self.bodies.append(Body([4., 0., -3.], 0.35, 'sphere', [1.0, 0.8, 0.3], 0.2, 0.9, device=self.device))
        self.bodies.append(Body([-3., 0., 3.], 0.55, 'sphere', [0.8, 0.5, 0.9], 0.35, 1.1, device=self.device))
        self.bodies.append(Body([-4., 0., 0.], [0.3, 0.2, 0.3], 'box', [0.7, 0.5, 0.3], 0.15, 0.8, device=self.device))
        self.bodies.append(Body([2., 0., -4.], [0.25, 0.25, 0.25], 'box', [0.8, 0.6, 0.4], 0.18, 1.3, device=self.device))
    
    def update(self, dt):
        self.time += dt
        for body in self.bodies:
            body.update(self.time)
    
    def scene_water(self, p):
        t = self.time
        h = torch.zeros(p.shape[0], device=self.device)
        h += 0.4 * torch.sin(p[..., 0] * 0.3 + t * 0.7)
        h += 0.35 * torch.sin(p[..., 2] * 0.35 + t * 0.8)
        h += 0.25 * torch.sin((p[..., 0] + p[..., 2]) * 0.2 + t * 0.5)
        h += 0.15 * torch.sin(p[..., 0] * 0.7 - p[..., 2] * 0.5 + t * 1.2)
        h += 0.1 * torch.sin(p[..., 0] * 1.0 + p[..., 2] * 0.8 + t * 1.5)
        return p[..., 1] - h
    
    def sdf_sphere(self, p, center, radius):
        return torch.norm(p - center, dim=-1) - radius
    
    def sdf_box(self, p, center, half_size, rotation=0.0):
        c, s = math.cos(-rotation), math.sin(-rotation)
        local = p - center
        rotated = torch.stack([c * local[..., 0] + s * local[..., 2], local[..., 1], -s * local[..., 0] + c * local[..., 2]], dim=-1)
        q = torch.abs(rotated) - half_size
        return torch.norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0.0)
    
    def scene_bodies(self, p):
        result = torch.full((p.shape[0],), 999.0, device=self.device)
        for body in self.bodies:
            if body.shape == 'sphere':
                d = self.sdf_sphere(p, body.pos, body.size[0].item())
            else:
                d = self.sdf_box(p, body.pos, body.size, body.rotation)
            k = 0.1
            h = torch.clamp(0.5 + 0.5 * (result - d) / k, 0.0, 1.0)
            result = torch.lerp(result, d, h) - k * h * (1.0 - h)
        return result
    
    def scene_floor(self, p):
        return p[..., 1] + 4.0
    
    def scene_solid(self, p):
        floor = self.scene_floor(p)
        bodies = self.scene_bodies(p)
        k = 0.15
        h = torch.clamp(0.5 + 0.5 * (floor - bodies) / k, 0.0, 1.0)
        return torch.lerp(floor, bodies, h) - k * h * (1.0 - h)
    
    def get_body_color(self, p):
        colors = torch.full((p.shape[0], 3), 0.35, device=self.device)
        min_dist = torch.full((p.shape[0],), 999.0, device=self.device)
        for body in self.bodies:
            if body.shape == 'sphere':
                d = self.sdf_sphere(p, body.pos, body.size[0].item())
            else:
                d = self.sdf_box(p, body.pos, body.size, body.rotation)
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
    
    angle = t * 0.15
    dist = 14.0
    cam = torch.tensor([math.sin(angle) * dist, 4.0 + math.sin(t * 0.1) * 0.5, math.cos(angle) * dist], device=dev)
    target = torch.tensor([0., 0., 0.], device=dev)
    fwd = F.normalize(target - cam, dim=0)
    right = F.normalize(torch.linalg.cross(torch.tensor([0., 1., 0.], device=dev), fwd), dim=0)
    up = torch.linalg.cross(fwd, right)
    
    asp = W / H
    y, x = torch.meshgrid(torch.linspace(1, -1, H, device=dev), torch.linspace(-asp, asp, W, device=dev), indexing='ij')
    
    N = H * W
    ro = cam.unsqueeze(0).expand(N, 3)
    rd = F.normalize(x.flatten().unsqueeze(-1) * right + y.flatten().unsqueeze(-1) * up + fwd * 1.8, dim=-1)
    
    tw, hitw = ray_march(ro, rd, scene.scene_water)
    pw = ro + tw.unsqueeze(-1) * rd
    nw = calc_normal(scene.scene_water, pw)
    
    ts, hits = ray_march(ro, rd, scene.scene_solid)
    ps = ro + ts.unsqueeze(-1) * rd
    ns = calc_normal(scene.scene_solid, ps)
    
    view = -rd
    fres = 0.02 + 0.98 * torch.pow(1.0 - torch.clamp((nw * view).sum(-1), 0, 1), 5)
    
    refl = rd - 2.0 * (rd * nw).sum(-1, keepdim=True) * nw
    
    # Sky - simple gradient, no giant sun
    sky_t = torch.clamp((refl[..., 1] + 1) * 0.5, 0, 1)
    sky = torch.stack([
        0.4 + sky_t * 0.2,
        0.5 + sky_t * 0.3,
        0.7 + sky_t * 0.2
    ], dim=-1)
    
    # Small sun specular only
    sun = F.normalize(torch.tensor([0.5, 0.6, -0.4], device=dev), dim=0)
    sun_spec = torch.pow(torch.clamp((refl * sun).sum(-1), 0, 1), 256) * 3.0  # Tight, bright
    
    light = F.normalize(torch.tensor([0.4, 0.9, -0.3], device=dev), dim=0)
    
    # Reflected solids
    refl_ro = pw + nw * 0.02
    tr, hitr = ray_march(refl_ro, refl, scene.scene_solid)
    pr = refl_ro + tr.unsqueeze(-1) * refl
    nr = calc_normal(scene.scene_solid, pr)
    rdiff = torch.clamp((nr * light).sum(-1), 0.15, 1)
    rcol = scene.get_body_color(pr)
    rsolid = rdiff.unsqueeze(-1) * rcol
    rcolor = torch.where(hitr.unsqueeze(-1), rsolid, sky) + sun_spec.unsqueeze(-1)
    
    # Refraction
    tint = torch.tensor([0.1, 0.2, 0.3], device=dev)
    absorb = torch.exp(-tw.unsqueeze(-1) * 0.1 * torch.tensor([0.3, 0.15, 0.05], device=dev))
    
    refr_d = F.normalize(rd + nw * 0.08, dim=-1)
    refr_ro = pw - nw * 0.02
    tref, hitref = ray_march(refr_ro, refr_d, scene.scene_solid)
    pref = refr_ro + tref.unsqueeze(-1) * refr_d
    nref = calc_normal(scene.scene_solid, pref)
    udiff = torch.clamp((nref * light).sum(-1), 0.1, 1)
    ucol = scene.get_body_color(pref)
    ucolor = udiff.unsqueeze(-1) * ucol * absorb
    deep = tint * 0.15
    refr_color = torch.where(hitref.unsqueeze(-1), ucolor, deep.expand(N, 3))
    
    # Caustics
    caust = torch.sin(pw[..., 0] * 3 + t * 2) * torch.sin(pw[..., 2] * 3.5 + t * 1.5)
    caust = torch.clamp(caust * 0.3 + 0.5, 0, 1) ** 2 * 0.15 * torch.exp(-tw * 0.08)
    refr_color = refr_color + caust.unsqueeze(-1) * torch.tensor([0.05, 0.1, 0.15], device=dev)
    
    # Foam
    wh = -scene.scene_water(pw)
    foam = torch.clamp((wh - 0.4) * 2.5, 0, 1)
    
    # Combine water
    water = fres.unsqueeze(-1) * rcolor + (1 - fres.unsqueeze(-1)) * refr_color
    water = water * (1 - foam.unsqueeze(-1)) + foam.unsqueeze(-1) * 0.9
    water = water * 0.92 + tint * 0.08
    
    # Solid shading
    sdiff = torch.clamp((ns * light).sum(-1), 0.15, 1)
    scol = scene.get_body_color(ps)
    solid = sdiff.unsqueeze(-1) * scol
    half = F.normalize(light + view, dim=-1)
    sspec = torch.pow(torch.clamp((ns * half).sum(-1), 0, 1), 32)
    solid = solid + sspec.unsqueeze(-1) * 0.25
    
    # Composite
    wfront = (tw < ts) & hitw
    svis = hits & ~wfront
    
    # Sky background - simple gradient
    skyy = (y.flatten() + 1) * 0.5
    skybg = torch.stack([
        0.5 + skyy * 0.15,
        0.6 + skyy * 0.2,
        0.8 + skyy * 0.15
    ], dim=-1)
    
    rgb = skybg
    rgb = torch.where(svis.unsqueeze(-1), solid, rgb)
    rgb = torch.where(wfront.unsqueeze(-1), water, rgb)
    
    # Fog
    mint = torch.where(wfront, tw, torch.where(hits, ts, torch.full_like(tw, MAX_DIST)))
    fog = torch.exp(-mint * 0.008)
    rgb = rgb * fog.unsqueeze(-1) + torch.tensor([0.55, 0.65, 0.8], device=dev) * (1 - fog.unsqueeze(-1))
    
    # Tonemap
    rgb = rgb / (rgb + 1)
    
    return torch.clamp(rgb.reshape(H, W, 3), 0, 1)

if __name__ == "__main__":
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {dev}")
    
    scene = Scene(dev)
    
    print("Test frame...")
    scene.time = 2.0
    for _ in range(30):
        scene.update(1/30)
    img = render(scene, 800, 600)
    Image.fromarray((img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8), 'RGB').save('water_test.png')
    print("Saved: water_test.png")
    
    scene = Scene(dev)
    print("\nRendering 240 frames...")
    frames = []
    
    for i in range(240):
        scene.update(1/30)
        img = render(scene, 640, 480)
        frames.append(Image.fromarray((img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8), 'RGB'))
        if i % 40 == 0:
            print(f"  Frame {i}/240")
    
    print("Saving...")
    frames[0].save('water.gif', save_all=True, append_images=frames[1:], duration=33, loop=0)
    
    try:
        import imageio
        w = imageio.get_writer('water.mp4', fps=30, codec='libx264', quality=8)
        for f in frames:
            w.append_data(np.array(f))
        w.close()
        print("Saved: water.mp4")
    except:
        pass
    
    print("Done! mpv water.mp4")
