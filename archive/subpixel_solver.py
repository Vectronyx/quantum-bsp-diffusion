#!/usr/bin/env python3
"""
SUB-PIXEL COLOR SOLVER + HAIR DETAIL ANALYZER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Sub-pixel color decomposition
- Hair strand frequency analysis
- Alterable hue/depth/saturation per region
- Neural detail enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os
import subprocess
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ColorParams:
    hue_shift: float = 0.0        # -180 to 180
    saturation: float = 1.0       # 0 to 2
    value: float = 1.0            # 0 to 2
    depth: float = 1.0            # Shadow/highlight depth
    temperature: float = 0.0      # -1 (cool) to 1 (warm)
    tint: float = 0.0             # -1 (green) to 1 (magenta)


class SubPixelDecomposer(nn.Module):
    """
    Decomposes image into sub-pixel frequency components.
    Separates: base color, micro detail, hair strands, skin texture
    """
    
    def __init__(self):
        super().__init__()
        
        # Gabor-like filters for hair strand detection (multiple orientations)
        self.hair_filters = nn.ParameterList([
            nn.Parameter(self._create_gabor(theta), requires_grad=False)
            for theta in np.linspace(0, np.pi, 8, endpoint=False)
        ])
        
        # Laplacian pyramid kernels
        self.blur_kernels = nn.ParameterList([
            nn.Parameter(self._gaussian_kernel(s), requires_grad=False)
            for s in [1, 2, 4, 8, 16]
        ])
        
    def _create_gabor(self, theta: float, sigma: float = 2.0, lambd: float = 4.0, 
                       gamma: float = 0.5, size: int = 11) -> torch.Tensor:
        """Create oriented Gabor filter for hair detection"""
        half = size // 2
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32),
            torch.arange(-half, half + 1, dtype=torch.float32),
            indexing='ij'
        )
        
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        
        gb = torch.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2)
        gb = gb * torch.cos(2 * np.pi * x_theta / lambd)
        
        return gb.view(1, 1, size, size)
    
    def _gaussian_kernel(self, sigma: float, size: int = None) -> torch.Tensor:
        if size is None:
            size = int(6 * sigma + 1) | 1  # Ensure odd
        
        half = size // 2
        x = torch.arange(-half, half + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_2d = kernel_1d.outer(kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        return kernel_2d.view(1, 1, size, size)
    
    def detect_hair(self, gray: torch.Tensor) -> torch.Tensor:
        """Detect hair strands using oriented Gabor filters"""
        responses = []
        
        for gabor in self.hair_filters:
            gabor_dev = gabor.to(gray.device)
            pad = gabor_dev.shape[-1] // 2
            response = F.conv2d(gray, gabor_dev, padding=pad)
            responses.append(response.abs())
        
        # Max response across orientations = hair likelihood
        hair_map = torch.stack(responses, dim=0).max(dim=0).values
        
        # Normalize
        hair_map = (hair_map - hair_map.min()) / (hair_map.max() - hair_map.min() + 1e-8)
        
        return hair_map
    
    def laplacian_pyramid(self, img: torch.Tensor, levels: int = 5) -> list:
        """Decompose into frequency bands"""
        pyramid = []
        current = img
        
        for i in range(min(levels, len(self.blur_kernels))):
            kernel = self.blur_kernels[i].to(img.device)
            pad = kernel.shape[-1] // 2
            
            # Blur for this level
            blurred = F.conv2d(
                current.view(-1, 1, current.shape[-2], current.shape[-1]),
                kernel, padding=pad
            ).view(current.shape)
            
            # Detail = current - blurred
            detail = current - blurred
            pyramid.append(detail)
            
            current = blurred
        
        # Residual (lowest frequency)
        pyramid.append(current)
        
        return pyramid
    
    def forward(self, image: torch.Tensor) -> dict:
        """
        Decompose image into components.
        Input: (B, 3, H, W) RGB image [0, 1]
        """
        B, C, H, W = image.shape
        device = image.device
        
        # Convert to different color spaces
        rgb = image
        
        # To grayscale for structure analysis
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        
        # Hair detection
        hair_mask = self.detect_hair(gray)
        
        # Laplacian pyramid per channel
        pyramids = {
            'R': self.laplacian_pyramid(rgb[:, 0:1]),
            'G': self.laplacian_pyramid(rgb[:, 1:2]),
            'B': self.laplacian_pyramid(rgb[:, 2:3]),
            'L': self.laplacian_pyramid(gray)
        }
        
        # High frequency = detail (sum of first 2 levels)
        detail = sum(pyramids['L'][:2])
        
        # Low frequency = base color (last level)
        base = pyramids['L'][-1]
        
        # Skin detection (simplified - warm hues, medium saturation)
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        skin_mask = ((r > g) & (g > b) & (r > 0.2) & (r < 0.85)).float()
        skin_mask = F.avg_pool2d(skin_mask, 5, stride=1, padding=2)  # Smooth
        
        return {
            'rgb': rgb,
            'gray': gray,
            'hair_mask': hair_mask,
            'skin_mask': skin_mask,
            'detail': detail,
            'base': base,
            'pyramids': pyramids
        }


class ColorSolver(nn.Module):
    """
    Adjusts colors with sub-pixel precision.
    Different parameters for hair, skin, background.
    """
    
    def __init__(self):
        super().__init__()
        self.decomposer = SubPixelDecomposer()
    
    def rgb_to_hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to HSV"""
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        
        max_c, _ = rgb.max(dim=1)
        min_c, _ = rgb.min(dim=1)
        diff = max_c - min_c
        
        # Value
        v = max_c
        
        # Saturation
        s = torch.where(max_c > 0, diff / (max_c + 1e-8), torch.zeros_like(max_c))
        
        # Hue
        h = torch.zeros_like(max_c)
        
        mask_r = (max_c == r) & (diff > 0)
        mask_g = (max_c == g) & (diff > 0)
        mask_b = (max_c == b) & (diff > 0)
        
        h = torch.where(mask_r, 60 * ((g - b) / (diff + 1e-8) % 6), h)
        h = torch.where(mask_g, 60 * ((b - r) / (diff + 1e-8) + 2), h)
        h = torch.where(mask_b, 60 * ((r - g) / (diff + 1e-8) + 4), h)
        
        return torch.stack([h, s, v], dim=1)
    
    def hsv_to_rgb(self, hsv: torch.Tensor) -> torch.Tensor:
        """Convert HSV to RGB"""
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        
        h = h / 60.0
        i = h.floor()
        f = h - i
        
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        i = i.long() % 6
        
        r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p, 
             torch.where(i == 3, p, torch.where(i == 4, t, v)))))
        g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v,
             torch.where(i == 3, q, torch.where(i == 4, p, p)))))
        b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t,
             torch.where(i == 3, v, torch.where(i == 4, v, q)))))
        
        return torch.stack([r, g, b], dim=1)
    
    def apply_color_params(
        self,
        rgb: torch.Tensor,
        params: ColorParams,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply color adjustments with optional mask"""
        
        hsv = self.rgb_to_hsv(rgb)
        
        # Hue shift
        hsv[:, 0] = (hsv[:, 0] + params.hue_shift) % 360
        
        # Saturation
        hsv[:, 1] = (hsv[:, 1] * params.saturation).clamp(0, 1)
        
        # Value (brightness)
        hsv[:, 2] = (hsv[:, 2] * params.value).clamp(0, 1)
        
        # Depth (shadows/highlights)
        if params.depth != 1.0:
            mid = 0.5
            hsv[:, 2] = mid + (hsv[:, 2] - mid) * params.depth
            hsv[:, 2] = hsv[:, 2].clamp(0, 1)
        
        result = self.hsv_to_rgb(hsv)
        
        # Temperature adjustment (blue-orange axis)
        if params.temperature != 0.0:
            if params.temperature > 0:  # Warm
                result[:, 0] = (result[:, 0] + params.temperature * 0.1).clamp(0, 1)
                result[:, 2] = (result[:, 2] - params.temperature * 0.05).clamp(0, 1)
            else:  # Cool
                result[:, 0] = (result[:, 0] + params.temperature * 0.05).clamp(0, 1)
                result[:, 2] = (result[:, 2] - params.temperature * 0.1).clamp(0, 1)
        
        # Tint adjustment (green-magenta axis)
        if params.tint != 0.0:
            result[:, 1] = (result[:, 1] - params.tint * 0.1).clamp(0, 1)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1) if mask.dim() == 3 else mask
            result = rgb * (1 - mask) + result * mask
        
        return result.clamp(0, 1)
    
    def enhance_hair_detail(
        self,
        image: torch.Tensor,
        hair_mask: torch.Tensor,
        detail_boost: float = 1.5,
        strand_sharpness: float = 1.2
    ) -> torch.Tensor:
        """Enhance hair strand detail"""
        
        # Extract high-frequency detail
        blur = F.avg_pool2d(F.pad(image, (2,2,2,2), mode='reflect'), 5, stride=1)
        detail = image - blur
        
        # Boost detail in hair regions
        hair_mask_3ch = hair_mask.expand(-1, 3, -1, -1)
        boosted_detail = detail * (1 + (detail_boost - 1) * hair_mask_3ch)
        
        # Unsharp mask for strand sharpness
        sharp = image + boosted_detail * strand_sharpness
        
        # Blend based on hair mask
        result = image * (1 - hair_mask_3ch * 0.5) + sharp * (hair_mask_3ch * 0.5)
        
        return result.clamp(0, 1)
    
    def solve(
        self,
        image: torch.Tensor,
        hair_params: ColorParams = None,
        skin_params: ColorParams = None,
        background_params: ColorParams = None,
        hair_detail_boost: float = 1.3
    ) -> dict:
        """
        Full sub-pixel color solve.
        
        Returns dict with processed image and all masks/components.
        """
        # Decompose
        decomp = self.decomposer(image)
        
        result = image.clone()
        
        # Apply hair adjustments
        if hair_params is not None:
            result = self.apply_color_params(result, hair_params, decomp['hair_mask'])
        
        # Apply skin adjustments
        if skin_params is not None:
            result = self.apply_color_params(result, skin_params, decomp['skin_mask'])
        
        # Apply background adjustments (inverse of hair + skin)
        if background_params is not None:
            bg_mask = 1.0 - (decomp['hair_mask'] + decomp['skin_mask']).clamp(0, 1)
            result = self.apply_color_params(result, background_params, bg_mask)
        
        # Enhance hair detail
        if hair_detail_boost > 1.0:
            result = self.enhance_hair_detail(
                result, decomp['hair_mask'], 
                detail_boost=hair_detail_boost
            )
        
        return {
            'result': result,
            'hair_mask': decomp['hair_mask'],
            'skin_mask': decomp['skin_mask'],
            'detail': decomp['detail'],
            'original': image
        }


def process_image(
    image_path: str,
    output_path: str = None,
    # Hair
    hair_hue: float = 0,
    hair_sat: float = 1.0,
    hair_val: float = 1.0,
    hair_detail: float = 1.3,
    # Skin
    skin_hue: float = 0,
    skin_sat: float = 1.0,
    skin_val: float = 1.0,
    skin_temp: float = 0.0,
    # Background
    bg_hue: float = 0,
    bg_sat: float = 1.0,
    bg_val: float = 1.0,
    bg_depth: float = 1.0,
    # View
    view: bool = True,
    save_masks: bool = False
):
    """Process an image with sub-pixel color solving"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load image
    img = Image.open(os.path.expanduser(image_path)).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    
    print(f"[*] Processing: {image_path}")
    print(f"    Size: {img.size}")
    
    # Create solver
    solver = ColorSolver().to(device)
    
    # Set up parameters
    hair_params = ColorParams(
        hue_shift=hair_hue,
        saturation=hair_sat,
        value=hair_val
    )
    
    skin_params = ColorParams(
        hue_shift=skin_hue,
        saturation=skin_sat,
        value=skin_val,
        temperature=skin_temp
    )
    
    bg_params = ColorParams(
        hue_shift=bg_hue,
        saturation=bg_sat,
        value=bg_val,
        depth=bg_depth
    )
    
    # Solve
    with torch.no_grad():
        result = solver.solve(
            img_tensor,
            hair_params=hair_params,
            skin_params=skin_params,
            background_params=bg_params,
            hair_detail_boost=hair_detail
        )
    
    # Convert back to PIL
    result_np = result['result'][0].permute(1, 2, 0).cpu().numpy()
    result_np = (result_np * 255).clip(0, 255).astype(np.uint8)
    result_img = Image.fromarray(result_np)
    
    # Output path
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_solved{ext}"
    
    output_path = os.path.expanduser(output_path)
    result_img.save(output_path)
    print(f"[✓] Saved: {output_path}")
    
    # Save masks if requested
    if save_masks:
        hair_np = (result['hair_mask'][0, 0].cpu().numpy() * 255).astype(np.uint8)
        skin_np = (result['skin_mask'][0, 0].cpu().numpy() * 255).astype(np.uint8)
        
        base, ext = os.path.splitext(output_path)
        Image.fromarray(hair_np).save(f"{base}_hair_mask.png")
        Image.fromarray(skin_np).save(f"{base}_skin_mask.png")
        print(f"    Masks saved")
    
    if view:
        subprocess.Popen(['xdg-open', output_path])
    
    return result_img


def demo():
    """Demo with a generated image"""
    print("=" * 60)
    print("SUB-PIXEL COLOR SOLVER")
    print("=" * 60)
    
    # Find a recent gen image
    desktop = os.path.expanduser("~/Desktop")
    images = [f for f in os.listdir(desktop) if f.startswith(('gen_', 'clear_')) and f.endswith('.png')]
    
    if not images:
        print("No generated images found. Run clear_gen.py first.")
        return
    
    # Use most recent
    images.sort(key=lambda x: os.path.getmtime(os.path.join(desktop, x)), reverse=True)
    image_path = os.path.join(desktop, images[0])
    
    print(f"\nProcessing: {images[0]}")
    
    # Process with various settings
    
    # 1. Original colors, boosted hair detail
    process_image(
        image_path,
        output_path=os.path.join(desktop, "solved_detail.png"),
        hair_detail=1.5,
        save_masks=True,
        view=True
    )
    
    # 2. Warm skin, cooler background
    process_image(
        image_path,
        output_path=os.path.join(desktop, "solved_warm.png"),
        skin_temp=0.3,
        skin_sat=1.1,
        bg_sat=0.8,
        bg_val=0.9,
        view=False
    )
    
    # 3. Dramatic (deep shadows, saturated)
    process_image(
        image_path,
        output_path=os.path.join(desktop, "solved_dramatic.png"),
        hair_sat=1.2,
        hair_val=0.9,
        skin_sat=1.05,
        bg_depth=1.4,
        bg_val=0.7,
        view=False
    )
    
    print("\n" + "=" * 60)
    print("Done! Check ~/Desktop/solved_*.png")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process specific image
        process_image(sys.argv[1], save_masks=True)
    else:
        demo()
