# Quantum BSP Diffusion Engine

**Sqrt-optimized 4D raymarching with probabilistic BSP traversal**

**Author:** Darrell Brock (vectronyx)  
**Date:** December 27, 2025  
**License:** MIT

## Key Innovations

1. **Sqrt-Free SDFs** - 4D primitives rewritten to eliminate square roots
2. **Adaptive Epsilon** - precision scales with distance and step count
3. **Quantum BSP** - probabilistic early termination for bounded traversal
4. **Diffusion Bridge** - direct output to Stable Diffusion ControlNet/latents

## Structure

- core/ - Main implementations
- generators/ - SD pipeline integration
- experiments/4d/ - 4D volumetric research
- experiments/water/ - Fluid simulation
- archive/ - Development history

## Citation

@software{quantum_bsp_diffusion,
  author = {Brock, Darrell},
  title = {Quantum BSP Diffusion Engine},
  year = {2025},
  url = {https://github.com/Vectronyx/quantum-bsp-diffusion}
}
