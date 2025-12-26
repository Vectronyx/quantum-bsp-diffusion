#!/usr/bin/env python3
"""
Quantum SDF + SD - 3070Ti SAFE
Low VRAM, no desktop crash
"""

import torch
import gc
import os

# VRAM safety
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.backends.cudnn.benchmark = False

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

print("╔═══════════════════════════════════════════════════════════════╗")
print("║  QUANTUM SDF + SD - 3070Ti SAFE MODE                          ║")
print("╚═══════════════════════════════════════════════════════════════╝\n")

clear_vram()

# Check VRAM before starting
if torch.cuda.is_available():
    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Free VRAM: {free_mem / 1024**3:.1f} GB\n")

import numpy as np
from PIL import Image
import subprocess

device = 'cuda'

# Step 1: Generate base image with aggressive memory management
print("  Loading model (low VRAM mode)...")

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
    low_cpu_mem_usage=True,
    variant="fp16"
).to(device)

# CRITICAL: Enable all memory optimizations
pipe.enable_attention_slicing(1)  # Minimum slice
pipe.enable_vae_slicing()
pipe.enable_sequential_cpu_offload()  # Offload to CPU when not needed

print("  ✓ Model loaded\n")

# Generate with small batch
print("  Generating (512x512)...")

prompt = "beautiful woman, elegant, purple lighting, soft skin, photorealistic"
negative = "ugly, deformed, blurry, low quality"

generator = torch.Generator(device).manual_seed(2024)

with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=512,
        height=512,
        num_inference_steps=20,  # Reduced
        guidance_scale=7.0,
        generator=generator
    ).images[0]

image.save('quantum_woman_base.png')
print("  ✓ Base saved\n")

# CRITICAL: Unload model before post-processing
del pipe
clear_vram()
print("  ✓ Model unloaded, VRAM freed\n")

# Step 2: Post-process on CPU to avoid VRAM issues
print("  Applying quantum effects (CPU safe)...")

img_np = np.array(image).astype(np.float32) / 255.0
H, W = img_np.shape[:2]

# Create coordinate grid on CPU
u = np.linspace(-1, 1, W)
v = np.linspace(-1, 1, H)
uu, vv = np.meshgrid(u, v)

os.makedirs('/tmp/qw', exist_ok=True)
n_frames = 60  # Reduced frame count

for i in range(n_frames):
    t = i / n_frames
    phase = t * 2 * np.pi
    
    if i % 10 == 0:
        print(f"    Frame {i+1:02d}/{n_frames}")
    
    result = img_np.copy()
    
    # Quantum aura (CPU)
    r = np.sqrt(uu**2 + vv**2)
    aura = np.exp(-r * 1.8) * (0.5 + 0.5 * np.sin(phase))
    
    result[..., 0] += aura * 0.08
    result[..., 1] += aura * 0.02
    result[..., 2] += aura * 0.12
    
    # Floating orbs (CPU)
    for j in range(3):
        cx = 0.35 * np.sin(phase + j * 2.1)
        cy = 0.35 * np.cos(phase * 0.7 + j * 1.7)
        orb_r = 0.07 + 0.02 * np.sin(phase * 3 + j)
        
        d = np.sqrt((uu - cx)**2 + (vv - cy)**2)
        glow = np.exp(-np.maximum(d - orb_r, 0)**2 / 0.0025)
        core = np.exp(-(d / orb_r)**2 * 3)
        
        result[..., 0] += (core * 0.3 + glow * 0.15) * (0.6 + j * 0.1)
        result[..., 1] += (core * 0.1 + glow * 0.05)
        result[..., 2] += (core * 0.4 + glow * 0.2) * (0.9 - j * 0.1)
    
    result = np.clip(result, 0, 1)
    result_u8 = (result * 255).astype(np.uint8)
    Image.fromarray(result_u8).save(f'/tmp/qw/frame_{i:04d}.png')

# Encode MP4
print("\n  Encoding MP4...")
subprocess.run([
    'ffmpeg', '-y', '-framerate', '24',
    '-i', '/tmp/qw/frame_%04d.png',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
    'quantum_woman.mp4'
], check=True, capture_output=True)

# Cleanup
for f in os.listdir('/tmp/qw'):
    os.remove(f'/tmp/qw/{f}')
os.rmdir('/tmp/qw')

clear_vram()

print(f"\n  ✓ quantum_woman.mp4")
print(f"  ✓ quantum_woman_base.png")
print("\n  3070Ti safe - no VRAM overflow")
