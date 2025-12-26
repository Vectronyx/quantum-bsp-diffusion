#!/usr/bin/env python3
"""Quick latent march test"""
import torch
import subprocess
from latent_marcher import ControlNetConditioner

print("[*] Raymarching latent space...")
conditioner = ControlNetConditioner()

depth, normal, latent = conditioner.generate_from_noise(
    seed=42069,
    width=512,
    height=512,
    device='cuda'
)

depth.save('/tmp/depth.png')
normal.save('/tmp/normal.png')

print("[✓] Saved /tmp/depth.png and /tmp/normal.png")
subprocess.Popen(['xdg-open', '/tmp/depth.png'])
subprocess.Popen(['xdg-open', '/tmp/normal.png'])
