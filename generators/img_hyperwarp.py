#!/usr/bin/env python3
"""RealisticVision + HyperWarp VAE"""
import os, torch, random, time, subprocess
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from hyperwarp_vae import wrap_diffusers_vae

pipe = StableDiffusionPipeline.from_single_file(
    "/home/cyberpunk/Desktop/realisticVisionV60B1_v51HyperVAE.safetensors",
    torch_dtype=torch.float16,
    safety_checker=None,
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True,
    algorithm_type="sde-dpmsolver++"
)

pipe = pipe.to("cuda")

# === WRAP WITH HYPERWARP ===
print("[*] Wrapping VAE with HyperWarp...")
pipe.vae = wrap_diffusers_vae(pipe)
print("[✓] HyperWarp VAE active!")

pipe.enable_attention_slicing()

def gen(prompt, neg="", steps=40, cfg=6.5, seed=-1):
    if seed == -1: seed = random.randint(0, 2**32-1)
    generator = torch.Generator("cuda").manual_seed(seed)
    
    print(f"\n[HyperWarp] Generating with curved latent manifold...")
    
    with torch.autocast("cuda"):
        image = pipe(
            prompt=prompt,
            negative_prompt=neg,
            width=512, height=768,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            clip_skip=2,
        ).images[0]
    
    filename = f"HYPERWARP_{int(time.time())}_{seed}.png"
    image.save(os.path.expanduser(f"~/Desktop/{filename}"))
    print(f"[✓] Saved: ~/Desktop/{filename}")
    subprocess.Popen(["xdg-open", os.path.expanduser(f"~/Desktop/{filename}")])

gen(
    prompt="hyperrealistic portrait, wet skin, water droplets with neon reflections, 85mm f1.4, insanely detailed",
    neg="blurry, ugly, deformed",
    steps=40, cfg=6.5, seed=42069
)
