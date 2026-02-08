from pathlib import Path


class MainConfig:
    TEMP_PATH: Path = Path(__file__).parents[1] / "temp"
