import tomllib
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "config.toml"


def load(path=DEFAULT):
    with open(path, "rb") as f:
        return tomllib.load(f)
