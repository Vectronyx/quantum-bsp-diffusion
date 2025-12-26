#!/usr/bin/env python3
"""
CLEAN SD GENERATION - No gimmicks, just good prompts
"""

import os
import torch
import random
import subprocess
from pathlib import Path

def generate(
    prompt: str,
    negative: str = "",
    seed: int = -1,
    steps: int = 30,
    cfg: float = 5.5,
    width: int = 512,
    height: int = 768,
    clip_skip: int = 2
):
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    
    device = 'cuda'
    model_path = Path.home() / "Desktop" / "realisticVisionV60B1_v51HyperVAE.safetensors"
    
    # Load
    pipe = StableDiffusionPipeline.from_single_file(
        str(model_path),
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++"
    )
    pipe.enable_attention_slicing()
    
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    
    generator = torch.Generator(device).manual_seed(seed)
    
    # Natural skin negative - don't fight the model, guide it
    if not negative:
        negative = (
            "cgi, 3d render, doll, plastic, airbrushed, "
            "blurry, deformed, bad anatomy, bad hands, "
            "watermark, signature, oversaturated"
        )
    
    print(f"\n[*] Generating (seed: {seed}, cfg: {cfg})...")
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
        clip_skip=clip_skip
    ).images[0]
    
    # Save - NO post-processing
    output_path = Path.home() / "Desktop" / f"gen_{seed}.png"
    image.save(output_path)
    
    print(f"[✓] Saved: {output_path}")
    subprocess.Popen(['xdg-open', str(output_path)])
    
    return image, seed


if __name__ == "__main__":
    import sys
    
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "woman portrait, natural skin, photography, 85mm, soft natural light"
    
    generate(prompt, cfg=5.5)
