#!/usr/bin/env python3
"""
SD Pipeline - Fixed Local File Loading
"""
import torch
import os
import sys
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def find_model():
    """Find the model file"""
    paths = [
        Path.cwd() / "realisticVisionV60B1_v51HyperVAE.safetensors",
        Path.home() / "Desktop" / "realisticVisionV60B1_v51HyperVAE.safetensors",
        Path.home() / "Desktop" / "SDOFFICIAL" / "realisticVisionV60B1_v51HyperVAE.safetensors",
    ]
    for p in paths:
        if p.exists():
            return str(p.resolve())  # Use absolute resolved path
    return None

def generate(prompt, cfg=6, steps=80, width=512, height=512, seed=-1, clip_skip=2):
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    import numpy as np
    
    model_path = find_model()
    if not model_path:
        print("❌ Model not found! Checking locations...")
        for p in [Path.cwd(), Path.home() / "Desktop"]:
            print(f"  {p}:")
            if p.exists():
                for f in p.iterdir():
                    if f.suffix in ['.safetensors', '.ckpt']:
                        print(f"    → {f.name}")
        return
    
    print(f"✓ Model: {model_path}")
    
    # KEY FIX: Use 'file://' prefix or ensure absolute path
    if not model_path.startswith('/'):
        model_path = os.path.abspath(model_path)
    
    print(f"  Loading pipeline...")
    
    try:
        # Method 1: Direct path (should work)
        pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            local_files_only=True,  # Force local
        )
    except Exception as e1:
        print(f"  Method 1 failed: {e1}")
        try:
            # Method 2: Use load_file directly
            from safetensors.torch import load_file
            from diffusers import StableDiffusionPipeline
            
            # Load from pretrained first, then swap weights
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            state_dict = load_file(model_path)
            # This is more complex... let's try method 3
            raise Exception("Try method 3")
        except:
            # Method 3: Use original_config
            pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
                original_config_file=None,
                load_safety_checker=False,
            )
    
    pipe = pipe.to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++"
    )
    pipe.enable_attention_slicing()
    
    if seed == -1:
        seed = np.random.randint(0, 2**32 - 1)
    
    print(f"  Generating: {prompt[:50]}...")
    print(f"  Seed: {seed}, CFG: {cfg}, Steps: {steps}, {width}x{height}")
    
    generator = torch.Generator(device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt="ugly, blurry, deformed, watermark, text",
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
        clip_skip=clip_skip,
    ).images[0]
    
    out_name = f"out_{seed}.png"
    image.save(out_name)
    print(f"✓ Saved: {out_name}")
    return image

def parse_args():
    """Parse: python p.py "prompt" "C(6)" "S(80)" "H(512)" "W(512)" "seed(123)" """
    import re
    
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    params = {
        'prompt': 'beautiful woman, portrait, soft lighting',
        'cfg': 6.0,
        'steps': 80,
        'height': 512,
        'width': 512,
        'seed': -1,
        'clip_skip': 2,
    }
    
    for arg in args:
        if arg.startswith('C('):
            params['cfg'] = float(re.search(r'C\(([\d.]+)\)', arg).group(1))
        elif arg.startswith('S('):
            params['steps'] = int(re.search(r'S\((\d+)\)', arg).group(1))
        elif arg.startswith('H('):
            params['height'] = int(re.search(r'H\((\d+)\)', arg).group(1))
        elif arg.startswith('W('):
            params['width'] = int(re.search(r'W\((\d+)\)', arg).group(1))
        elif arg.startswith('seed('):
            params['seed'] = int(re.search(r'seed\((\d+)\)', arg).group(1))
        elif not arg.startswith(('C(', 'S(', 'H(', 'W(', 'seed(')):
            params['prompt'] = arg
    
    return params

if __name__ == "__main__":
    p = parse_args()
    generate(p['prompt'], p['cfg'], p['steps'], p['width'], p['height'], p['seed'], p['clip_skip'])
