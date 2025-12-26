#!/usr/bin/env python3
"""
QUANTUM BSP VOLUMETRIC MARCHER - FIXED
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Tuple
import random
import os


@dataclass
class QuantumState:
    amplitudes: torch.Tensor
    positions: torch.Tensor
    densities: torch.Tensor
    phase: torch.Tensor


class QuantumBSPMarcher:
    
    def __init__(
        self,
        device: str = 'cuda',
        num_branches: int = 8,
        epsilon_base: float = 0.01
    ):
        self.device = device
        self.num_branches = num_branches
        self.epsilon_base = epsilon_base
    
    def initialize_superposition(
        self,
        ray_origins: torch.Tensor,
        ray_dirs: torch.Tensor
    ) -> QuantumState:
        H, W = ray_origins.shape[:2]
        B = self.num_branches
        
        amplitudes = torch.ones(H, W, B, dtype=torch.complex64, device=self.device) / np.sqrt(B)
        positions = ray_origins.unsqueeze(2).expand(H, W, B, 3).clone()
        phase = torch.rand(H, W, B, device=self.device) * 2 * np.pi
        amplitudes = amplitudes * torch.exp(1j * phase)
        densities = torch.zeros(H, W, B, device=self.device)
        
        return QuantumState(amplitudes, positions, densities, phase)
    
    def branch_sdf(self, position: torch.Tensor, branch_idx: int) -> torch.Tensor:
        center = torch.tensor([0., 0., 0.5], device=self.device)
        offset = 0.2 * torch.tensor([
            np.sin(branch_idx * 0.7),
            np.cos(branch_idx * 1.1),
            np.sin(branch_idx * 0.3)
        ], device=self.device)
        
        sdf = (position - center - offset).norm(dim=-1) - 0.5
        noise = torch.sin(position[..., 0] * 10 + branch_idx) * 0.05
        return sdf + noise
    
    def collapse_wavefunction(self, state: QuantumState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = (state.amplitudes.abs() ** 2).real
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        observation_weight = 1.0 - torch.exp(-state.densities)
        collapse_prob = probs * observation_weight
        collapse_prob = collapse_prob / (collapse_prob.sum(dim=-1, keepdim=True) + 1e-8)
        
        branch_depths = state.positions[..., 2]
        depth = (collapse_prob * branch_depths).sum(dim=-1)
        
        phase_diff = state.phase.unsqueeze(-1) - state.phase.unsqueeze(-2)
        interference = torch.cos(phase_diff).mean(dim=(-1, -2))
        
        color = torch.stack([
            0.5 + 0.5 * interference,
            0.5 + 0.3 * torch.sin(state.phase.mean(dim=-1)),
            0.5 + 0.3 * torch.cos(state.phase.mean(dim=-1) * 2)
        ], dim=-1)
        
        total_prob = collapse_prob.sum(dim=-1)
        color = color * total_prob.unsqueeze(-1)
        
        return color, depth, total_prob
    
    def retain_background(self, state: QuantumState, foreground_mask: torch.Tensor) -> QuantumState:
        # Expand mask to match branch dimension
        fg_expanded = foreground_mask.unsqueeze(-1)  # (H, W, 1)
        
        phase_noise = torch.randn_like(state.phase) * 0.1 * fg_expanded
        new_phase = state.phase + phase_noise
        
        decay = 1.0 - 0.1 * fg_expanded
        new_amplitudes = state.amplitudes * decay
        
        return QuantumState(new_amplitudes, state.positions, state.densities, new_phase)
    
    def march(self, width: int = 512, height: int = 512, num_steps: int = 64, seed: int = -1) -> dict:
        if seed != -1:
            torch.manual_seed(seed)
            random.seed(seed)
        
        aspect = width / height
        u = torch.linspace(-1, 1, width, device=self.device) * aspect
        v = torch.linspace(-1, 1, height, device=self.device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        ray_dirs = F.normalize(torch.stack([uu, -vv, torch.ones_like(uu)], dim=-1), dim=-1)
        ray_origins = torch.zeros(height, width, 3, device=self.device)
        ray_origins[..., 2] = -2.0
        
        state = self.initialize_superposition(ray_origins, ray_dirs)
        
        for step in range(num_steps):
            epsilon = self.epsilon_base * (1.0 + step * 0.01)
            
            for b in range(self.num_branches):
                sdf = self.branch_sdf(state.positions[..., b, :], b)
                density = torch.exp(-sdf.abs() * 5.0)
                state.densities[..., b] += density * 0.1
                state.phase[..., b] += sdf * 0.5
                
                step_size = sdf.clamp(min=epsilon)
                state.positions[..., b, :] += ray_dirs * step_size.unsqueeze(-1)
                
                decay = torch.exp(-step_size * 0.1)
                state.amplitudes[..., b] *= decay
        
        max_density = state.densities.max(dim=-1).values
        foreground_mask = (max_density > 0.3).float()
        
        state = self.retain_background(state, foreground_mask)
        color, depth, probability = self.collapse_wavefunction(state)
        
        background_interference = torch.cos(state.phase).mean(dim=-1)
        background_color = torch.stack([
            0.1 + 0.1 * background_interference,
            0.1 + 0.15 * background_interference,
            0.2 + 0.2 * background_interference
        ], dim=-1)
        
        final_color = color * foreground_mask.unsqueeze(-1) + \
                      background_color * (1.0 - foreground_mask.unsqueeze(-1))
        
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return {
            'color': final_color.clamp(0, 1),
            'depth': depth_norm,
            'foreground': foreground_mask,
            'interference': background_interference
        }
    
    def generate_maps(self, seed: int = -1, width: int = 512, height: int = 512):
        result = self.march(width, height, seed=seed)
        
        color_np = (result['color'].cpu().numpy() * 255).astype(np.uint8)
        color_pil = Image.fromarray(color_np)
        
        depth_np = (result['depth'].cpu().numpy() * 255).astype(np.uint8)
        depth_pil = Image.fromarray(depth_np, mode='L')
        
        interf = result['interference'].cpu().numpy()
        interf_rgb = np.stack([
            (interf * 0.5 + 0.5) * 255,
            (np.sin(interf * np.pi) * 0.5 + 0.5) * 255,
            (np.cos(interf * np.pi) * 0.5 + 0.5) * 255
        ], axis=-1).astype(np.uint8)
        interf_pil = Image.fromarray(interf_rgb)
        
        return color_pil, depth_pil, interf_pil


def demo():
    print("=" * 60)
    print("QUANTUM BSP VOLUMETRIC MARCHER")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    marcher = QuantumBSPMarcher(device=device, num_branches=8)
    
    for seed in [42, 1337, 69420]:
        print(f"\n[*] Quantum march seed {seed}...")
        
        color, depth, interference = marcher.generate_maps(seed=seed)
        
        prefix = os.path.expanduser(f'~/Desktop/qbsp_{seed}')
        color.save(f'{prefix}_color.png')
        depth.save(f'{prefix}_depth.png')
        interference.save(f'{prefix}_interference.png')
        
        print(f"    Saved: {prefix}_*.png")
    
    print("\n[✓] Done!")


if __name__ == "__main__":
    demo()
