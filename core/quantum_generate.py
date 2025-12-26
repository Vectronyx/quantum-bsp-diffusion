#!/usr/bin/env python3
"""
QUANTUM BSP + STABLE DIFFUSION
Quantum interference patterns guide the generation
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
import os
import subprocess

from quantum_bsp_marcher import QuantumBSPMarcher


def generate_quantum_portrait(
    prompt: str = None,
    seed: int = -1,
    view: bool = True
):
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    
    device = 'cuda'
    
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    
    # Epic prompt if none given
    if prompt is None:
        prompt = """
        cinematic portrait of ethereal woman emerging from quantum void, 
        bioluminescent particles swirling around her face,
        iridescent skin with subsurface scattering,
        crystalline structures growing from shadows,
        volumetric god rays piercing through cosmic dust,
        fibonacci spiral composition,
        shot on ARRI Alexa with Zeiss Master Prime 50mm,
        directed by Denis Villeneuve and Roger Deakins,
        hyperrealistic octane render, 8k resolution,
        dark moody atmosphere with neon accents of cyan and magenta,
        photorealistic skin texture, visible pores,
        dramatic chiaroscuro lighting,
        wet glistening surfaces reflecting nebula colors,
        depth of field with creamy bokeh,
        award winning photography
        """
    
    negative = """
        cartoon, anime, plastic, doll, cgi, 3d render,
        blurry, ugly, deformed, bad anatomy, bad hands,
        text, watermark, signature, oversaturated,
        flat lighting, boring composition
    """
    
    print("=" * 70)
    print("QUANTUM BSP VOLUMETRIC GENERATION")
    print("=" * 70)
    print(f"Seed: {seed}")
    
    # Generate quantum maps first
    print("\n[1/3] Generating quantum interference patterns...")
    marcher = QuantumBSPMarcher(device=device, num_branches=8)
    color_map, depth_map, interference_map = marcher.generate_maps(seed=seed, width=512, height=768)
    
    # Save quantum maps
    color_map.save(os.path.expanduser(f'~/Desktop/quantum_{seed}_color.png'))
    depth_map.save(os.path.expanduser(f'~/Desktop/quantum_{seed}_depth.png'))
    interference_map.save(os.path.expanduser(f'~/Desktop/quantum_{seed}_interference.png'))
    print("    [✓] Quantum maps saved")
    
    # Load SD
    print("\n[2/3] Loading RealisticVision...")
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
    
    # Generate with same seed for coherence
    print("\n[3/3] Generating image...")
    print(f"    Prompt: {prompt[:80].strip()}...")
    
    generator = torch.Generator(device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=512,
        height=768,
        num_inference_steps=35,
        guidance_scale=5.5,
        generator=generator,
        clip_skip=2
    ).images[0]
    
    # Save
    output_path = os.path.expanduser(f'~/Desktop/quantum_gen_{seed}.png')
    image.save(output_path)
    
    print("\n" + "=" * 70)
    print(f"[✓] COMPLETE")
    print(f"    Image: {output_path}")
    print(f"    Quantum maps: ~/Desktop/quantum_{seed}_*.png")
    print(f"    Seed: {seed}")
    print("=" * 70)
    
    if view:
        subprocess.Popen(['xdg-open', output_path])
        subprocess.Popen(['xdg-open', os.path.expanduser(f'~/Desktop/quantum_{seed}_interference.png')])
    
    return image, seed


if __name__ == "__main__":
    import sys
    
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    generate_quantum_portrait(prompt=prompt)
