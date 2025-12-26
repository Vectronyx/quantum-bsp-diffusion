#!/usr/bin/env python3
"""Simple clean generation - no quantum nonsense, just good portraits"""

import torch
import random
import os
import subprocess
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def generate(prompt: str = None, seed: int = -1):
    device = 'cuda'
    
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    
    if prompt is None:
        prompt = "beautiful woman, portrait, sharp focus, professional photography, soft lighting, 85mm lens, shallow depth of field"
    
    negative = "blurry, distorted, deformed, ugly, bad anatomy, extra limbs, disfigured, malformed, watermark"
    
    print(f"[*] Loading model...")
    pipe = StableDiffusionPipeline.from_single_file(
        os.path.expanduser("~/Desktop/realisticVisionV60B1_v51HyperVAE.safetensors"),
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++"
    )
    pipe.enable_attention_slicing()
    
    generator = torch.Generator(device).manual_seed(seed)
    
    print(f"[*] Generating (seed: {seed}, cfg: 4.5)...")
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=512,
        height=768,
        num_inference_steps=30,
        guidance_scale=4.5,  # Lower = less "cooked"
        generator=generator,
        clip_skip=2
    ).images[0]
    
    path = os.path.expanduser(f'~/Desktop/clear_{seed}.png')
    image.save(path)
    print(f"[✓] Saved: {path}")
    subprocess.Popen(['xdg-open', path])
    
    return image, seed

if __name__ == "__main__":
    import sys
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    generate(prompt)
