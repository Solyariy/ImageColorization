import torch
import matplotlib.pyplot as plt

from src.image_processing.dataset import make_dataloaders
from src.image_processing.lab_convertor import LabConvertor
from src.model import ColorizationModel
from src.setup.config import main_config


def run_untrained_inference():
    device = main_config.DEVICE

    MODEL_ID = "baseline_unet_epoch_10"

    model = ColorizationModel().to(device)
    checkpoint = torch.load(main_config.TRAINED_MODELS / (MODEL_ID + ".pth"), map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    processor = LabConvertor().to(device)

    dl = make_dataloaders(root_dir=main_config.LANDSCAPE_IMAGES, split="test", batch_size=1)

    rgb_input = next(iter(dl)).to(device)

    data = processor(rgb_input)
    L_input = data["L"]

    model.eval()
    with torch.no_grad():
        ab_predicted = model(L_input)

    rgb_output = processor.unscale_to_rgb(L_input, ab_predicted)

    rgb_input_np = rgb_input[0].permute(1, 2, 0).cpu().numpy()
    rgb_output_np = rgb_output[0].permute(1, 2, 0).cpu().numpy()
    L_input_np = L_input[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(L_input_np, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(rgb_output_np)
    axes[1].set_title("Trained Model Output")
    axes[1].axis("off")

    axes[2].imshow(rgb_input_np)
    axes[2].set_title("Target")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(main_config.PREDICTED_IMAGES / f"{MODEL_ID}.jpeg")


if __name__ == "__main__":
    run_untrained_inference()