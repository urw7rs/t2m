from os import PathLike
from pathlib import Path


def search(path: PathLike, file: str, all: bool = False, strict: bool = True):
    matches = list(Path(path).glob(f"**/{file}"))

    if len(matches) == 0:
        if strict:
            raise FileNotFoundError(f"{file} doesn't exist in {path}")
        else:
            return None

    if all:
        return matches
    else:
        return matches[0]
