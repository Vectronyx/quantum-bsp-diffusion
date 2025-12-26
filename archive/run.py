#!/usr/bin/env python3
"""
SDOFFICIAL Launcher
"""
import sys, os
sys.path.insert(0, 'scripts/generators')

MODEL = "models/realisticVisionV60B1_v51HyperVAE.safetensors"

if __name__ == "__main__":
    from p import generate, parse_args
    p = parse_args()
    generate(p['prompt'], p['cfg'], p['steps'], p['width'], p['height'], p['seed'])
