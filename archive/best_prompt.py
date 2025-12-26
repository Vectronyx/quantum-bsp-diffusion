#!/usr/bin/env python3
"""
ULTIMATE PROMPT ENGINEERING FOR REALVISION
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

device = 'cuda'

# ═══════════════════════════════════════════════════════════════════════════════
# THE PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

PROMPT = """
photograph of a woman, 25 years old, looking at viewer, slight smile,
natural skin texture with pores and subtle imperfections,
honey brown eyes with visible iris detail and catchlights,
soft volumetric lighting from window, golden hour,
shallow depth of field, 85mm f/1.4 lens, bokeh background,
raw photo, film grain, kodak portra 400,
professional photography, editorial vogue,
hyperrealistic, photorealistic, 8k uhd, dslr
""".strip().replace('\n', ' ')

NEGATIVE = """
cgi, 3d render, cartoon, anime, illustration, painting, drawing,
plastic skin, airbrushed, smooth skin, blurry, soft focus,
overexposed, underexposed, oversaturated, undersaturated,
bad anatomy, deformed, ugly, mutated, disfigured,
extra limbs, missing limbs, floating limbs, disconnected limbs,
malformed hands, extra fingers, missing fingers, fused fingers,
long neck, elongated body,
watermark, signature, text, logo, banner,
cropped, out of frame, worst quality, low quality, jpeg artifacts,
duplicate, morbid, mutilated, poorly drawn face, 
mutation, bad proportions, gross proportions, cloned face,
(naked, nude, nsfw:1.5)
""".strip().replace('\n', ' ')

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

SETTINGS = {
    'steps': 30,
    'cfg': 5.5,          # Lower = more natural, less "AI"
    'width': 512,
    'height': 768,       # Portrait ratio
    'clip_skip': 2,      # Skip last CLIP layer for RealVision
}

def load_pipeline():
    model_path = Path.cwd() / "realisticVisionV60B1_v51HyperVAE.safetensors"
    if not model_path.exists():
        model_path = Path.home() / "Desktop" / "realisticVisionV60B1_v51HyperVAE.safetensors"
    
    print(f"Loading: {model_path.name}")
    
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
    
    return pipe

def generate(pipe, seed=-1):
    if seed == -1:
        seed = np.random.randint(0, 2**32 - 1)
    
    generator = torch.Generator(device).manual_seed(seed)
    
    print(f"\n{'═'*60}")
    print(f"  SEED: {seed}")
    print(f"  CFG: {SETTINGS['cfg']} | Steps: {SETTINGS['steps']}")
    print(f"{'═'*60}")
    print(f"\n  PROMPT:\n  {PROMPT[:80]}...")
    print(f"\n  Generating...")
    
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        width=SETTINGS['width'],
        height=SETTINGS['height'],
        num_inference_steps=SETTINGS['steps'],
        guidance_scale=SETTINGS['cfg'],
        generator=generator,
        clip_skip=SETTINGS['clip_skip']
    ).images[0]
    
    output_path = f'best_{seed}.png'
    image.save(output_path)
    print(f"\n  ✓ Saved: {output_path}")
    
    return image, seed

def main():
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║  ULTIMATE REALVISION PROMPT                                    ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    pipe = load_pipeline()
    
    # Generate 4 variations
    for i in range(4):
        image, seed = generate(pipe)
    
    print("\n  Done! Check best_*.png files")

if __name__ == "__main__":
    main()
