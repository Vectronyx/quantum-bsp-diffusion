#!/usr/bin/env python3
import os, torch, time, subprocess
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from hyperwarp_vae import wrap_diffusers_vae

pipe = StableDiffusionPipeline.from_single_file(
    "/home/cyberpunk/Desktop/realisticVisionV60B1_v51HyperVAE.safetensors",
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
)

# Wrap with stable hyperwarp
pipe.vae = wrap_diffusers_vae(pipe)
pipe.enable_attention_slicing()

print(f"[✓] VAE config check: scaling_factor={pipe.vae.config.scaling_factor}")

seed = 12345
image = pipe(
    prompt="beautiful woman portrait, studio lighting, photorealistic, 8k",
    negative_prompt="ugly, blurry, deformed",
    width=512, height=768,
    num_inference_steps=30,
    guidance_scale=6.5,
    generator=torch.Generator("cuda").manual_seed(seed),
).images[0]

path = os.path.expanduser(f"~/Desktop/HW_TEST_{seed}.png")
image.save(path)
print(f"[✓] Saved: {path}")
subprocess.Popen(["xdg-open", path])
