import torch
from torch.utils.data import DataLoader
from dataset import ColorizationDataset, MainConfig
import kornia


def debug_values():
    print("🔍 Inspecting Data Pipeline...")

    # 1. Load Data
    ds = ColorizationDataset(f"{MainConfig.DATA_PATH}/landscape_images", split="train")
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    # Get one image
    try:
        img = next(iter(dl))
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Check Input Range (Crucial!)
    print(f"\n[Step 1] Raw from DataLoader:")
    print(f"   Shape: {img.shape}")
    print(f"   Min: {img.min():.4f}")
    print(f"   Max: {img.max():.4f}")

    if img.max() > 1.5:
        print("   ❌ CRITICAL ERROR: Your images are 0-255. They MUST be 0-1.")
        print("   Fix: Ensure transforms.ToTensor() is in your dataset.py")
        return
    else:
        print("   ✅ Input range looks correct (0-1).")

    # 3. Check Kornia Conversion
    lab = kornia.color.rgb_to_lab(img)
    print(f"\n[Step 2] After RGB -> Lab Conversion:")
    print(f"   L Channel (Brightness) Min/Max: {lab[:, 0].min():.2f} / {lab[:, 0].max():.2f} (Should be 0 to 100)")
    print(f"   a Channel (Green-Red)  Min/Max: {lab[:, 1].min():.2f} / {lab[:, 1].max():.2f} (Should be ~ -128 to 128)")

    # 4. Check Your Normalization
    L = (lab[:, [0], :, :] / 50.0) - 1.0
    ab = lab[:, 1:, :, :] / 128.0

    print(f"\n[Step 3] After Normalization (Input to Model):")
    print(f"   L Min/Max: {L.min():.4f} / {L.max():.4f} (Should be -1 to 1)")
    print(f"   ab Min/Max: {ab.min():.4f} / {ab.max():.4f} (Should be -1 to 1)")


if __name__ == "__main__":
    debug_values()