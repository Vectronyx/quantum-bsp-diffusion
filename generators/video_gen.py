#!/usr/bin/env python3
"""video_gen.py - fullscreen wave animation"""
import sys, numpy as np, imageio
from PIL import Image
from scipy.ndimage import map_coordinates

img_path = sys.argv[1] if len(sys.argv) > 1 else None
if not img_path: sys.exit("Usage: python video_gen.py <image>")

img = np.array(Image.open(img_path)).astype(np.float32)
h, w = img.shape[:2]
y, x = np.mgrid[0:h, 0:w].astype(np.float32)

frames = []
for t in np.linspace(0, 6*np.pi, 120):
    # big traveling waves across whole screen
    dx = 6 * np.sin(y/15 - t*2) + 4 * np.sin(x/20 + t*1.5)
    dy = 5 * np.sin(x/18 - t*1.8) + 3 * np.cos(y/25 + t)
    
    # ripple from center
    dist = np.sqrt((x - w/2)**2 + (y - h/2)**2)
    ripple = 3 * np.sin(dist/20 - t*3)
    
    nx = x + dx + ripple * (x - w/2) / (dist + 1)
    ny = y + dy + ripple * (y - h/2) / (dist + 1)
    
    frame = np.stack([
        map_coordinates(img[...,c], [ny, nx], order=1, mode='reflect')
        for c in range(3)
    ], axis=-1)
    frames.append(np.clip(frame, 0, 255).astype(np.uint8))

out = img_path.rsplit('.', 1)[0] + '_waves.mp4'
imageio.mimsave(out, frames, fps=30)
print(f"✓ {out}")
