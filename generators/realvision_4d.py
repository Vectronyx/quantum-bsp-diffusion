#!/usr/bin/env python3
"""
REALVISION 4D ENGINE - Fixed
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

torch.backends.cuda.matmul.allow_tf32 = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TAU = 6.283185307179586

class Rot4D:
    @staticmethod
    def from_angles(xw=0., yw=0., zw=0., xy=0., xz=0., yz=0.):
        R = torch.eye(4, device=device, dtype=torch.float32)
        planes = ((0,1), (0,2), (0,3), (1,2), (1,3), (2,3))
        for idx, a in enumerate((xy, xz, xw, yz, yw, zw)):
            if abs(a) > 1e-8:
                c, s = np.cos(a), np.sin(a)
                i, j = planes[idx]
                Rp = torch.eye(4, device=device, dtype=torch.float32)
                Rp[i,i], Rp[j,j], Rp[i,j], Rp[j,i] = c, c, -s, s
                R = R @ Rp
        return R

class Field4D:
    @staticmethod
    def breathe(uv, t, strength=0.15):
        phase = t * TAU
        w = np.sin(phase) * 0.5
        r = torch.sqrt(uv[...,0]**2 + uv[...,1]**2 + w**2 + 1e-8)
        sdf = r - 0.8
        warp = torch.exp(-sdf**2 * 4) * strength
        angle = torch.atan2(uv[...,1], uv[...,0])
        dx = torch.cos(angle + phase * 0.5) * warp
        dy = torch.sin(angle + phase * 0.5) * warp
        return torch.stack([uv[...,0] + dx, uv[...,1] + dy], dim=-1)
    
    @staticmethod
    def fold(uv, t, strength=0.12):
        phase = t * TAU
        R = Rot4D.from_angles(xw=phase, yw=phase*0.618, zw=phase*0.382)
        p4 = torch.zeros(*uv.shape[:-1], 4, device=device)
        p4[...,0], p4[...,1] = uv[...,0], uv[...,1]
        p4[...,2] = np.sin(phase) * 0.3
        p4[...,3] = np.cos(phase) * 0.3
        p4_rot = torch.einsum('ij,...j->...i', R, p4)
        q = torch.abs(p4_rot) - 0.7
        sdf = torch.sqrt(torch.clamp(q, min=0.).pow(2).sum(dim=-1) + 1e-8) + \
              torch.clamp(q.max(dim=-1).values, max=0.)
        warp = torch.exp(-torch.abs(sdf) * 5) * strength
        dx = (p4_rot[...,0] - uv[...,0]) * warp + torch.sin(p4_rot[...,3] * 10 + phase) * warp * 0.5
        dy = (p4_rot[...,1] - uv[...,1]) * warp + torch.cos(p4_rot[...,3] * 10 + phase) * warp * 0.5
        return torch.stack([uv[...,0] + dx, uv[...,1] + dy], dim=-1)
    
    @staticmethod
    def swirl(uv, t, strength=0.18):
        phase = t * TAU
        r1 = torch.sqrt(uv[...,0]**2 + (np.sin(phase) * 0.5)**2 + 1e-8)
        r2 = torch.sqrt(uv[...,1]**2 + (np.cos(phase) * 0.5)**2 + 1e-8)
        d1, d2 = r1 - 0.6, r2 - 0.6
        sdf = torch.sqrt(torch.clamp(d1, min=0.)**2 + torch.clamp(d2, min=0.)**2 + 1e-8) + \
              torch.clamp(torch.maximum(d1, d2), max=0.)
        warp = torch.exp(-torch.abs(sdf) * 4) * strength
        angle1 = torch.atan2(uv[...,0], np.sin(phase) * 0.5 + 0.01)
        angle2 = torch.atan2(uv[...,1], np.cos(phase) * 0.5 + 0.01)
        dx = torch.sin(angle1 + phase * 2) * warp
        dy = torch.cos(angle2 - phase * 2) * warp
        return torch.stack([uv[...,0] + dx, uv[...,1] + dy], dim=-1)
    
    @staticmethod
    def torus(uv, t, strength=0.2):
        phase = t * TAU
        R, r1, r2 = 0.6, 0.3, 0.15
        dxy = torch.sqrt(uv[...,0]**2 + uv[...,1]**2 + 1e-8) - R
        z, w = np.sin(phase) * 0.4, np.cos(phase) * 0.4
        dxyz = torch.sqrt(dxy**2 + z**2 + 1e-8) - r1
        sdf = torch.sqrt(dxyz**2 + w**2 + 1e-8) - r2
        warp = torch.exp(-torch.abs(sdf) * 6) * strength
        theta = torch.atan2(uv[...,1], uv[...,0])
        phi = torch.atan2(dxy, z + 0.01)
        dx = torch.sin(theta * 2 + phi + phase * 3) * warp
        dy = torch.cos(theta * 2 - phi + phase * 3) * warp
        return torch.stack([uv[...,0] + dx, uv[...,1] + dy], dim=-1)
    
    @staticmethod
    def fractal(uv, t, strength=0.1):
        phase = t * TAU
        dx = torch.zeros_like(uv[...,0])
        dy = torch.zeros_like(uv[...,1])
        for i in range(5):
            freq = 2.0 ** i
            amp = strength / (i + 1)
            w = np.sin(phase + i * 0.7) * 0.3
            angle = phase * (0.5 + i * 0.2)
            c, s = np.cos(angle), np.sin(angle)
            u_rot = uv[...,0] * c - uv[...,1] * s
            v_rot = uv[...,0] * s + uv[...,1] * c
            r4 = torch.sqrt(u_rot**2 + v_rot**2 + w**2 + 1e-8)
            wave = torch.sin(r4 * freq * TAU + phase * (i + 1))
            dx = dx + torch.cos(u_rot * freq * TAU + phase) * wave * amp
            dy = dy + torch.sin(v_rot * freq * TAU - phase) * wave * amp
        return torch.stack([uv[...,0] + dx, uv[...,1] + dy], dim=-1)
    
    @staticmethod
    def quantum(uv, t, strength=0.15):
        phase = t * TAU
        dx = torch.zeros_like(uv[...,0])
        dy = torch.zeros_like(uv[...,1])
        for i in range(4):
            state_phase = phase + i * TAU / 4
            R = Rot4D.from_angles(xw=state_phase, yw=state_phase*0.7, zw=state_phase*0.3)
            p4 = torch.zeros(*uv.shape[:-1], 4, device=device)
            p4[...,0], p4[...,1] = uv[...,0], uv[...,1]
            p4[...,2] = 0.3 * np.sin(state_phase)
            p4[...,3] = 0.3 * np.cos(state_phase)
            p4_rot = torch.einsum('ij,...j->...i', R, p4)
            psi = torch.exp(-torch.sum(p4_rot**2, dim=-1) * 2)
            interference = torch.cos(p4_rot[...,3] * 15 + state_phase * 2)
            amp = psi * interference * strength / 4
            dx = dx + torch.sin(p4_rot[...,0] * 10 + state_phase) * amp
            dy = dy + torch.cos(p4_rot[...,1] * 10 - state_phase) * amp
        return torch.stack([uv[...,0] + dx, uv[...,1] + dy], dim=-1)

EFFECTS = {'breathe': Field4D.breathe, 'fold': Field4D.fold, 'swirl': Field4D.swirl,
           'torus': Field4D.torus, 'fractal': Field4D.fractal, 'quantum': Field4D.quantum}

def chromatic_4d(img, t, strength=0.012):
    H, W = img.shape[:2]
    phase = t * TAU
    u = torch.linspace(-1, 1, W, device=device)
    v = torch.linspace(-1, 1, H, device=device)
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    result = torch.zeros_like(img)
    for c in range(3):
        c_phase = phase + c * TAU / 3
        dx = np.sin(c_phase) * strength * (uu**2 + vv**2)
        dy = np.cos(c_phase) * strength * (uu**2 + vv**2)
        uv_shifted = torch.stack([uu + dx, vv + dy], dim=-1).unsqueeze(0)
        channel = img[..., c:c+1].permute(2, 0, 1).unsqueeze(0)
        shifted = F.grid_sample(channel, uv_shifted, mode='bilinear', padding_mode='reflection', align_corners=True)
        result[..., c] = shifted.squeeze()
    return result

def pulse_4d(img, t, strength=0.15):
    phase = t * TAU
    pulse = sum(np.cos(phase * (i+1) * 0.7 + np.sin(phase * (i+1) + i * 0.5)) / (i+1) for i in range(3))
    pulse = pulse / 2 * strength + 1.0
    return (img * pulse).clamp(0, 1)

class RealVision4D:
    def __init__(self):
        self.pipe = None
    
    def load(self):
        if self.pipe is not None:
            return
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        model_path = Path.cwd() / "realisticVisionV60B1_v51HyperVAE.safetensors"
        if not model_path.exists():
            model_path = Path.home() / "Desktop" / "realisticVisionV60B1_v51HyperVAE.safetensors"
        print(f"  Loading: {model_path.name}")
        self.pipe = StableDiffusionPipeline.from_single_file(str(model_path), torch_dtype=torch.float16, safety_checker=None).to(device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
        self.pipe.enable_attention_slicing()
        print("  ✓ Loaded")
    
    def generate(self, prompt, seed=-1, width=512, height=768, steps=25, cfg=6.0):
        self.load()
        if seed == -1:
            seed = np.random.randint(0, 2**32 - 1)
        generator = torch.Generator(device).manual_seed(seed)
        print(f"  Generating seed {seed}...")
        image = self.pipe(prompt=prompt, negative_prompt="ugly, deformed, blurry, watermark, text, bad anatomy",
                         width=width, height=height, num_inference_steps=steps, guidance_scale=cfg, generator=generator).images[0]
        return image, seed
    
    def warp_frame(self, img_t, t, effect='breathe', strength=0.15):
        H, W = img_t.shape[:2]
        u = torch.linspace(-1, 1, W, device=device)
        v = torch.linspace(-1, 1, H, device=device)
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        uv = torch.stack([uu, vv], dim=-1)
        uv_warped = EFFECTS.get(effect, Field4D.breathe)(uv, t, strength)
        grid = uv_warped.unsqueeze(0)
        img_in = img_t.permute(2, 0, 1).unsqueeze(0)
        warped = F.grid_sample(img_in, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
        return warped.squeeze(0).permute(1, 2, 0)
    
    def animate(self, image, effect='breathe', n_frames=60, strength=0.15):
        img_np = np.array(image).astype(np.float32) / 255.0
        img_t = torch.tensor(img_np, device=device, dtype=torch.float32)
        frames = []
        for i in range(n_frames):
            t = i / n_frames
            warped = self.warp_frame(img_t, t, effect, strength)
            warped = chromatic_4d(warped, t, 0.01)
            warped = pulse_4d(warped, t, 0.12)
            frames.append(Image.fromarray((warped.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)))
            if (i + 1) % 15 == 0:
                print(f"    Frame {i+1}/{n_frames}")
        return frames
    
    def run(self, prompt, effect='breathe', n_frames=60, strength=0.15, seed=-1, output='realvision_4d'):
        print(f"\n{'═'*64}\n  REALVISION 4D | Effect: {effect} | Frames: {n_frames}\n{'═'*64}")
        image, seed = self.generate(prompt, seed=seed)
        image.save(f'{output}_base.png')
        print(f"  ✓ {output}_base.png")
        print("  Animating...")
        frames = self.animate(image, effect=effect, n_frames=n_frames, strength=strength)
        frames[0].save(f'{output}.gif', save_all=True, append_images=frames[1:], duration=33, loop=0)
        print(f"  ✓ {output}.gif")
        try:
            import subprocess, tempfile
            with tempfile.TemporaryDirectory() as tmp:
                for i, f in enumerate(frames): f.save(f'{tmp}/f_{i:04d}.png')
                subprocess.run(['ffmpeg', '-y', '-framerate', '30', '-i', f'{tmp}/f_%04d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', f'{output}.mp4'], capture_output=True, check=True)
                print(f"  ✓ {output}.mp4")
        except: pass
        return {'seed': seed, 'frames': frames}

def animate_existing(image_path, effect='breathe', n_frames=60, strength=0.15, output=None):
    print(f"\n  Loading: {image_path}")
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_t = torch.tensor(img_np, device=device, dtype=torch.float32)
    H, W = img_t.shape[:2]
    u = torch.linspace(-1, 1, W, device=device)
    v = torch.linspace(-1, 1, H, device=device)
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    uv = torch.stack([uu, vv], dim=-1)
    warp_fn = EFFECTS.get(effect, Field4D.breathe)
    print(f"  Animating with '{effect}'...")
    frames = []
    for i in range(n_frames):
        t = i / n_frames
        uv_warped = warp_fn(uv, t, strength)
        grid = uv_warped.unsqueeze(0)
        img_in = img_t.permute(2, 0, 1).unsqueeze(0)
        warped = F.grid_sample(img_in, grid, mode='bilinear', padding_mode='reflection', align_corners=True).squeeze(0).permute(1, 2, 0)
        warped = chromatic_4d(warped, t, 0.01)
        warped = pulse_4d(warped, t, 0.1)
        frames.append(Image.fromarray((warped.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)))
        if (i + 1) % 20 == 0: print(f"    Frame {i+1}/{n_frames}")
    if output is None: output = Path(image_path).stem + '_4d'
    frames[0].save(f'{output}.gif', save_all=True, append_images=frames[1:], duration=33, loop=0)
    print(f"  ✓ {output}.gif")

def main():
    parser = argparse.ArgumentParser(description='RealVision 4D')
    parser.add_argument('prompt', nargs='?', default=None)
    parser.add_argument('--image', '-i', type=str)
    parser.add_argument('--effect', '-e', type=str, default='breathe', choices=list(EFFECTS.keys()))
    parser.add_argument('--frames', '-f', type=int, default=60)
    parser.add_argument('--strength', '-s', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--output', '-o', type=str, default='realvision_4d')
    args = parser.parse_args()
    
    if args.prompt is None and args.image is None:
        print("\n╔════════════════════════════════════════════════════════════════╗")
        print("║  REALVISION 4D ENGINE                                          ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        print("║  Effects: breathe, fold, swirl, torus, fractal, quantum        ║")
        print("║                                                                ║")
        print("║  python realvision_4d.py \"prompt\" --effect fold               ║")
        print("║  python realvision_4d.py -i image.png --effect swirl           ║")
        print("╚════════════════════════════════════════════════════════════════╝\n")
        return
    
    if args.image:
        animate_existing(args.image, effect=args.effect, n_frames=args.frames, strength=args.strength, output=args.output)
    else:
        RealVision4D().run(args.prompt, effect=args.effect, n_frames=args.frames, strength=args.strength, seed=args.seed, output=args.output)

if __name__ == "__main__":
    main()
