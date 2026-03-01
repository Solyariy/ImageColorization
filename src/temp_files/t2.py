import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import kornia

# Import your code
from dataset import ColorizationDataset, DataProcessor, MainConfig
from model import ColorizationModel


def run_untrained_inference():
    print("🚀 Loading 'Untrained' Model...")

    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your custom U-Net (Phase 1)
    model = ColorizationModel().to(device)

    # Load your GPU Data Processor
    processor = DataProcessor().to(device)

    # Load Data
    ds = ColorizationDataset(f"{MainConfig.DATA_PATH}/landscape_images", split="train")
    if len(ds) == 0:
        print("❌ No images found! Check path in dataset.py")
        return
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    # 2. Get a real image
    rgb_input = next(iter(dl)).to(device)  # [1, 3, 256, 256]

    # 3. Prepare the Input (L Channel)
    # We use your processor to get the 'L' channel cleanly
    data = processor(rgb_input)
    L_input = data['L']  # Range [-1, 1]

    # 4. The "Forward Pass"
    # This is the moment of truth. The image goes into the U-Net.
    print("🎨 Running image through the U-Net...")
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        ab_predicted = model(L_input)  # Model tries to guess colors

    # 5. Reconstruct the Image
    # We combine the Real L (Detail) with the Predicted ab (Color)
    lab_predicted = torch.cat([L_input, ab_predicted], dim=1)

    # Reverse the math (Normalize -> 0..100 -> RGB)
    # (Manual un-normalization to be safe)
    L_unscaled = (lab_predicted[:, [0], :, :] + 1.0) * 50.0
    ab_unscaled = lab_predicted[:, 1:, :, :] * 128.0
    lab_unscaled = torch.cat([L_unscaled, ab_unscaled], dim=1)

    # Convert to RGB
    rgb_output = kornia.color.lab_to_rgb(lab_unscaled)
    rgb_output = torch.clamp(rgb_output, 0.0, 1.0)  # Clip weird values

    # 6. Visualize
    print("💾 Saving result to 'untrained_result.png'...")

    rgb_input_np = rgb_input[0].permute(1, 2, 0).cpu().numpy()
    rgb_output_np = rgb_output[0].permute(1, 2, 0).cpu().numpy()
    L_input_np = L_input[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # A. The Input (What the model sees)
    axes[0].imshow(L_input_np, cmap='gray')
    axes[0].set_title("Input (Grayscale)")
    axes[0].axis('off')

    # B. The Output (What the untrained model guesses)
    axes[1].imshow(rgb_output_np)
    axes[1].set_title("Untrained Model Output\n(Should be shaped correctly but gray/brown)")
    axes[1].axis('off')

    # C. The Target (What it SHOULD look like)
    axes[2].imshow(rgb_input_np)
    axes[2].set_title("Target (Ground Truth)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("untrained_result.png")
    plt.show()
    print("✅ Done! Check 'untrained_result.png'.")


if __name__ == "__main__":
    run_untrained_inference()