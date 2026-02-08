import torch
import kornia
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import ColorizationDataset, MainConfig


def safe_lab_to_rgb(L, ab):
    """
    Robust converter that handles standardizing ranges
    to prevent 'cyan noise' artifacts.
    """
    # 1. Reverse Normalization (Your math)
    # L: [-1, 1] -> [0, 100]
    L = (L + 1.0) * 50.0
    # ab: [-1, 1] -> [-128, 127]
    ab = ab * 128.0

    # 2. Combine into Lab tensor
    lab_tensor = torch.cat([L, ab], dim=1)

    # 3. Convert to RGB using Kornia
    rgb_tensor = kornia.color.lab_to_rgb(lab_tensor)

    # 4. CRITICAL FIX: Clamp to [0, 1]
    # Matplotlib creates noise if values are -0.001 or 1.001
    return torch.clamp(rgb_tensor, 0.0, 1.0)


def verify_visuals():
    print("🎨 Generating verification image...")
    ds = ColorizationDataset(f"{MainConfig.DATA_PATH}/landscape_images", split="train")
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    # Get a batch
    batch = next(iter(dl))  # [4, 3, 256, 256] RGB Input

    # Simulate the pipeline
    lab = kornia.color.rgb_to_lab(batch)
    L_in = (lab[:, [0], :, :] / 50.0) - 1.0
    ab_in = lab[:, 1:, :, :] / 128.0

    # Reconstruct
    rgb_recon = safe_lab_to_rgb(L_in, ab_in)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Data Pipeline Integrity Check\n(Left: Original | Right: Reconstructed from Lab)", fontsize=16)

    for i in range(2):
        # Original
        axes[i, 0].imshow(batch[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Original RGB")
        axes[i, 0].axis('off')

        # Reconstructed
        axes[i, 1].imshow(rgb_recon[i].permute(1, 2, 0).numpy())
        axes[i, 1].set_title("Pipeline Output (Must match Original)")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig("final_pipeline_check.png")
    print("✅ Saved 'final_pipeline_check.png'. Open this file.")
    print("   If the images match, Phase 1 is 100% COMPLETE.")


if __name__ == "__main__":
    verify_visuals()