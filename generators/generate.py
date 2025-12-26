#!/usr/bin/env python3
"""SD Generator CLI."""

import os, sys, argparse, time
from pathlib import Path
from datetime import datetime

os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import torch
from PIL import Image
from engine import StableDiffusionEngine, EngineConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--prompt", "-p", default="a beautiful landscape")
    p.add_argument("--negative", "-n", default="")
    p.add_argument("--width", "-W", type=int, default=512)
    p.add_argument("--height", "-H", type=int, default=512)
    p.add_argument("--steps", "-s", type=int, default=20)
    p.add_argument("--cfg", "-c", type=float, default=7.5)
    p.add_argument("--seed", "-S", type=int, default=-1)
    p.add_argument("--lora", "-l")
    p.add_argument("--lora-weight", type=float, default=0.8)
    p.add_argument("--output", "-o", default="/ram/outputs")
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args()
    
    print("\n" + "="*50 + "\n SD BARE METAL\n" + "="*50)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    engine = StableDiffusionEngine(EngineConfig())
    engine.load_model(Path(args.model))
    
    if args.lora:
        engine.load_lora(Path(args.lora), args.lora_weight)
        
    if args.benchmark:
        engine.benchmark()
        return
        
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    def progress(step, total, _):
        pct = step / total
        bar = "█" * int(40*pct) + "░" * int(40*(1-pct))
        print(f"\r[{bar}] {step}/{total}", end="", flush=True)
        
    print(f"\nGenerating: {args.prompt[:40]}...")
    img, meta = engine.generate(
        prompt=args.prompt, negative_prompt=args.negative,
        width=args.width, height=args.height, steps=args.steps,
        cfg_scale=args.cfg, seed=args.seed, callback=progress)
    print()
    
    fn = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{meta['seed']}.png"
    img.save(Path(args.output) / fn)
    print(f"\nDone in {meta['time']:.2f}s | Seed: {meta['seed']} | {fn}")


if __name__ == "__main__":
    main()
