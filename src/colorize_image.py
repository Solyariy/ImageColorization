import os.path

import torch
import matplotlib.pyplot as plt

from src.image_processing.dataset import make_dataloaders
from src.image_processing.lab_convertor import LabConvertor
from src.model import ColorizationModel
from src.setup.config import main_config
from src.setup.enums import RunTypeEnum
from src.setup.utils import get_uuid_str


def colorize_image(model_id: str, second_model_id, image_path: str = None):
    device = main_config.DEVICE

    model = ColorizationModel().to(device)
    checkpoint = torch.load(main_config.TRAINED_MODELS / (model_id + ".pth"), map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    model_gan = ColorizationModel().to(device)
    checkpoint_gan = torch.load(main_config.TRAINED_MODELS / (second_model_id + ".pth"), map_location=device)
    model_gan.load_state_dict(checkpoint_gan)
    model_gan.eval()

    processor = LabConvertor().to(device)

    dl = make_dataloaders(
        root_dir=main_config.LANDSCAPE_IMAGES,
        split=RunTypeEnum.TEST,
        batch_size=1,
        image_path=image_path
    )

    rgb_input = next(iter(dl)).to(device)

    data = processor(rgb_input)
    L_input = data["L"]

    with torch.no_grad():
        ab_predicted = model(L_input)
        ab_predicted_gan = model_gan(L_input)

    rgb_output = processor.unscale_to_rgb(L_input, ab_predicted)
    rgb_output_gan = processor.unscale_to_rgb(L_input, ab_predicted_gan)

    rgb_input_np = rgb_input[0].permute(1, 2, 0).cpu().numpy()
    rgb_output_np = rgb_output[0].permute(1, 2, 0).cpu().numpy()
    rgb_output_np_gan = rgb_output_gan[0].permute(1, 2, 0).cpu().numpy()
    L_input_np = L_input[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    axes[0].imshow(L_input_np, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(rgb_output_np)
    axes[1].set_title("Phase 1 Output")
    axes[1].axis("off")

    axes[2].imshow(rgb_output_np_gan)
    axes[2].set_title("Phase 2 GAN Output")
    axes[2].axis("off")

    axes[3].imshow(rgb_input_np)
    axes[3].set_title("Target")
    axes[3].axis("off")

    plt.tight_layout()
    if not os.path.exists(main_config.PREDICTED_IMAGES):
        os.mkdir(main_config.PREDICTED_IMAGES)
    plt.savefig(main_config.PREDICTED_IMAGES / f"{model_id}_{get_uuid_str()[:8]}.jpeg")


if __name__ == "__main__":
    colorize_image("2026-03-15_14-03-28_baseline_epoch_30", "2026-03-15_16-40-28_gan_gen_epoch_100", "2992.jpg")