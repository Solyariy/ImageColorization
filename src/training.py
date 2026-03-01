import torch
from tqdm import tqdm
from torch import nn, optim

from src.image_processing.dataset import make_dataloaders
from src.image_processing.lab_convertor import LabConvertor
from src.model import ColorizationModel
from src.setup.config import main_config
from src.setup.enums import RunTypeEnum
from src.setup.utils import get_datetime

device = torch.device("mps")
model = ColorizationModel().to(device)

# L1 Loss зазвичай дає яскравіші кольори, ніж MSE
criterion = nn.L1Loss()

# Оптимізатор Adam. Передаючи model.parameters(), ми кажемо йому
# тренувати і енкодер, і декодер одночасно.
optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))

epochs = 20  # Кількість разів, які модель побачить весь твій набір даних

if __name__ == "__main__":
    for epoch in range(epochs):
        model.train()  # Переводимо модель у режим тренування
        running_loss = 0.0
        dataloader = make_dataloaders(
            root_dir=main_config.LANDSCAPE_IMAGES,
            split=RunTypeEnum.TRAIN,
            n_workers=2
        )
        processor = LabConvertor().to(device)

        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        # dataloader - це об'єкт, який видає тобі батчі (порції) зображень
        for rgb_batch in loop:
            rgb_batch = rgb_batch.to(device)

            # Process into Normalized L and ab channels
            processed_data = processor(rgb_batch)
            L = processed_data["L"]
            ab_real = processed_data["ab"]
            # Переносимо дані на відеокарту

            # 1. Очищаємо старі градієнти
            optimizer.zero_grad()

            # 2. Прямий прохід: модель намагається вгадати кольори
            ab_pred = model(L)

            # 3. Рахуємо помилку
            loss = criterion(ab_pred, ab_real)

            # 4. Зворотне поширення (магія навчання)
            loss.backward()

            # 5. Оновлюємо ваги (числа у фільтрах)
            optimizer.step()

            running_loss += loss.item()

            if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
                checkpoint_path = f"{main_config.TRAINED_MODELS}/{get_datetime()}_epoch_{epoch + 1}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}\n")

        print(f"Епоха {epoch + 1}/{epochs}, Середня помилка: {running_loss / len(dataloader):.4f}")
