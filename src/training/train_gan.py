import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.image_processing.lab_convertor import LabConvertor
from src.model import ColorizationModel
from src.models.discriminator import Discriminator

from src.setup.config import main_config, TrainingConfig
from src.setup.utils import get_datetime
from src.training.training_pipeline import setup_data


def train_step_discriminator(gen, disc, L, ab_real, opt_d, criterion):
    opt_d.zero_grad(set_to_none=True)

    pred_real = disc(L, ab_real)
    loss_d_real = criterion(pred_real, torch.ones_like(pred_real))

    with torch.no_grad():
        ab_fake = gen(L)

    pred_fake = disc(L, ab_fake)
    loss_d_fake = criterion(pred_fake, torch.zeros_like(pred_fake))

    loss_d = (loss_d_real + loss_d_fake) * 0.5
    loss_d.backward()
    opt_d.step()

    return loss_d.item()


def train_step_generator(gen, disc, L, ab_real, opt_g, crit_gan, crit_l1, lambda_l1):
    opt_g.zero_grad(set_to_none=True)

    ab_fake = gen(L)
    pred_fake = disc(L, ab_fake)

    loss_gan = crit_gan(pred_fake, torch.ones_like(pred_fake))
    loss_l1 = crit_l1(ab_fake, ab_real)

    loss_g = loss_gan + (lambda_l1 * loss_l1)
    loss_g.backward()
    opt_g.step()

    return loss_g.item()


def train_one_gan_epoch(
    gen, disc, loader, processor, opt_g, opt_d, crit_gan, crit_l1, lambda_l1, epoch
):
    gen.train()
    disc.train()

    running_d_loss = 0
    running_g_loss = 0

    loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{TrainingConfig.GAN_EPOCHS}")

    for rgb_batch in loop:
        rgb_batch = rgb_batch.to(main_config.DEVICE, non_blocking=True)
        processed_data = processor(rgb_batch)
        L, real_ab = processed_data["L"], processed_data["ab"]

        d_loss = train_step_discriminator(gen, disc, L, real_ab, opt_d, crit_gan)
        g_loss = train_step_generator(
            gen, disc, L, real_ab, opt_g, crit_gan, crit_l1, lambda_l1
        )

        running_d_loss += d_loss
        running_g_loss += g_loss
        loop.set_postfix(D_loss=f"{d_loss:.4f}", G_loss=f"{g_loss:.4f}")

    return (running_d_loss / len(loader)), (running_g_loss / len(loader))


def save_gan_checkpoints(gen, disc, epoch):
    timestamp = get_datetime()

    gen_path = main_config.TRAINED_MODELS / f"{timestamp}_gan_gen_epoch_{epoch + 1}.pth"
    disc_path = (
        main_config.TRAINED_MODELS / f"{timestamp}_gan_disc_epoch_{epoch + 1}.pth"
    )

    torch.save(gen.state_dict(), gen_path)
    torch.save(disc.state_dict(), disc_path)
    print(f"Saved GAN checkpoints to {main_config.TRAINED_MODELS}\n")


def run_gan_session(baseline_name: str):
    device = main_config.DEVICE
    print(f"GAN Training on device: {device}")

    train_loader = setup_data()
    processor = LabConvertor().to(device)

    generator = ColorizationModel().to(device)
    discriminator = Discriminator().to(device)

    baseline_path = main_config.TRAINED_MODELS / baseline_name

    try:
        generator.load_state_dict(torch.load(baseline_path, map_location=device))
        print(f"[*] Baseline loaded: {baseline_name}")
    except FileNotFoundError:
        print("[!] No baseline found. Training GAN from scratch (Not recommended).")

    opt_g = optim.Adam(
        generator.parameters(),
        lr=TrainingConfig.LEARNING_RATE,
        betas=TrainingConfig.BETAS,
    )
    opt_d = optim.Adam(
        discriminator.parameters(),
        lr=TrainingConfig.LEARNING_RATE,
        betas=TrainingConfig.BETAS,
    )

    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    for epoch in range(TrainingConfig.GAN_EPOCHS):
        avg_d_loss, avg_g_loss = train_one_gan_epoch(
            generator,
            discriminator,
            train_loader,
            processor,
            opt_g,
            opt_d,
            criterion_gan,
            criterion_l1,
            TrainingConfig.LAMBDA_L1,
            epoch,
        )

        print(
            f"\nEpoch [{epoch + 1}/{TrainingConfig.GAN_EPOCHS}] Completed. "
            f"Avg D Loss: {avg_d_loss:.4f} | Avg G Loss: {avg_g_loss:.4f}"
        )

        if (epoch + 1) % 5 == 0 or (epoch + 1) == TrainingConfig.GAN_EPOCHS:
            save_gan_checkpoints(generator, discriminator, epoch)


if __name__ == "__main__":
    run_gan_session("2026-03-15_14-03-28_baseline_epoch_30.pth")
