#!/usr/bin/env python3
"""
Quantum SDF + RealVision (LOCAL)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import subprocess
import os
import glob

device = 'cuda'

print("╔═══════════════════════════════════════════════════════════════╗")
print("║  QUANTUM SDF + REALVISION (LOCAL)                             ║")
print("╚═══════════════════════════════════════════════════════════════╝\n")

# Find safetensors file
safetensors_paths = glob.glob("/home/*/realisticVision*.safetensors") + \
                    glob.glob("/home/*/*/realisticVision*.safetensors") + \
                    glob.glob(os.path.expanduser("~/realisticVision*.safetensors")) + \
                    glob.glob(os.path.expanduser("~/Desktop/realisticVision*.safetensors"))

if not safetensors_paths:
    print("  Looking for any .safetensors...")
    safetensors_paths = glob.glob("/home/*/*.safetensors")[:5]

if safetensors_paths:
    model_path = safetensors_paths[0]
    print(f"  Found: {model_path}")
else:
    # Fallback to HuggingFace
    model_path = None
    print("  No local safetensors found, using HuggingFace...")

print("  Loading model...")

if model_path and os.path.exists(model_path):
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=True
    ).to(device)
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
print("  ✓ Model loaded\n")

# Generate
print("  Generating woman...")
prompt = """beautiful woman, elegant, purple ambient lighting,
ethereal glow, soft skin, detailed eyes, flowing hair,
cinematic, photorealistic, 8k, mystical aura"""

negative = "ugly, deformed, blurry, low quality, watermark, bad anatomy"

generator = torch.Generator(device).manual_seed(2024)

image = pipe(
    prompt=prompt,
    negative_prompt=negative,
    width=512,
    height=512,
    num_inference_steps=25,
    guidance_scale=7.0,
    generator=generator
).images[0]

image.save('quantum_woman_base.png')
print("  ✓ Base saved\n")

# Convert to tensor
img = torch.tensor(np.array(image) / 255.0, device=device, dtype=torch.float32)

# Quantum SDF post-processing
print("  Rendering quantum effects...")
os.makedirs('/tmp/qw', exist_ok=True)

n_frames = 120
H, W = 512, 512

u = torch.linspace(-1, 1, W, device=device)
v = torch.linspace(-1, 1, H, device=device)
uu, vv = torch.meshgrid(u, v, indexing='xy')

for i in range(n_frames):
    t = i / n_frames
    phase = t * 2 * np.pi
    
    if i % 20 == 0:
        print(f"    Frame {i+1:03d}/{n_frames}")
    
    result = img.clone()
    
    # Quantum aura
    r = torch.sqrt(uu**2 + vv**2)
    aura = torch.exp(-r * 1.8) * (0.5 + 0.5 * torch.sin(torch.tensor(phase)))
    
    result[..., 0] += aura * 0.08
    result[..., 1] += aura * 0.02
    result[..., 2] += aura * 0.12
    
    # Floating orbs
    for j in range(3):
        cx = 0.35 * np.sin(phase + j * 2.1)
        cy = 0.35 * np.cos(phase * 0.7 + j * 1.7)
        orb_r = 0.07 + 0.02 * np.sin(phase * 3 + j)
        
        d = torch.sqrt((uu - cx)**2 + (vv - cy)**2)
        glow = torch.exp(-((d - orb_r).clamp(min=0) / 0.05)**2)
        core = torch.exp(-(d / orb_r)**2 * 3)
        
        result[..., 0] += (core * 0.3 + glow * 0.15) * (0.6 + j * 0.1)
        result[..., 1] += (core * 0.1 + glow * 0.05)
        result[..., 2] += (core * 0.4 + glow * 0.2) * (0.9 - j * 0.1)
    
    result = torch.clamp(result, 0, 1)
    result_np = (result.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(result_np).save(f'/tmp/qw/frame_{i:04d}.png')

# Encode MP4
print("\n  Encoding MP4...")
subprocess.run([
    'ffmpeg', '-y', '-framerate', '24',
    '-i', '/tmp/qw/frame_%04d.png',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
    'quantum_woman.mp4'
], check=True, capture_output=True)

# Cleanup
for f in os.listdir('/tmp/qw'):
    os.remove(f'/tmp/qw/{f}')
os.rmdir('/tmp/qw')

print(f"\n  ✓ quantum_woman.mp4")
print(f"  ✓ quantum_woman_base.png")
