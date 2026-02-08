from pathlib import Path


class MainConfig:
    DATA_PATH: Path = Path(__file__).parents[1] / "data"
    IMG_SIZE = 256
