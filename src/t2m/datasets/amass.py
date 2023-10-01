import pprint
import shutil
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from t2m import io


def search(path: PathLike, file: str, all: bool = False, strict: bool = False):
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


class AMASS(Dataset):
    def __init__(self, root):
        super().__init__()

        self.root = root

        self.valid_paths = []
        for path in Path(root).glob("**/*.npz"):
            assert path.exists()

            bdata = np.load(path, allow_pickle=True)

            fps = int(bdata.get("mocap_framerate", 0))
            frame_number = bdata.get("trans", None)
            if fps == 0 or frame_number is None:
                continue

            self.valid_paths.append(path)

    def __getitem__(self, index):
        path = self.valid_paths[index]
        bdata = np.load(path, allow_pickle=True)

        fps = int(bdata.get("mocap_framerate", 0))
        frame_number = bdata.get("trans", None)
        frame_number = frame_number.shape[0]

        def to_tensor(array):
            return torch.as_tensor(array, dtype=torch.float32)

        poses, betas, trans = [
            to_tensor(bdata[key]) for key in ["poses", "betas", "trans"]
        ]

        # poses: num_frames x 156
        # trans: num_frames x 3
        # betas: 16

        gender = str(bdata["gender"])

        metadata = {
            "fps": fps,
            "gender": gender,
            "path": str(path.relative_to(self.root)),
        }

        return {"trans": trans, "poses": poses, "betas": betas, "metadata": metadata}

    def __len__(self):
        return len(self.valid_paths)

    @staticmethod
    def build(src: PathLike, dst: PathLike):
        dst = Path(dst)

        amass_filenames = [
            "ACCAD.tar.bz2",
            "BMLhandball.tar.bz2",
            "BMLmovi.tar.bz2",
            "BMLrub.tar.bz2",
            "CMU.tar.bz2",
            "DFaust.tar.bz2",
            "EKUT.tar.bz2",
            "EyesJapanDataset.tar.bz2",
            "HDM05.tar.bz2",
            "HumanEva.tar.bz2",
            "KIT.tar.bz2",
            "MoSh.tar.bz2",
            "PosePrior.tar.bz2",
            "SFU.tar.bz2",
            "SSM.tar.bz2",
            "TCDHands.tar.bz2",
            "TotalCapture.tar.bz2",
            "Transitions.tar.bz2",
        ]

        matches = [(file, search(src, file)) for file in amass_filenames]
        missing = [file for file, path in matches if path is None]

        if len(missing) > 0:
            raise FileNotFoundError(
                "Download the SMPL+H G files from https://amass.is.tue.mpg.de/download.php.\n"
                + f"{pprint.pformat(missing)} is missing. "
            )

        tmp_path = dst / "extracted"
        tmp_path.mkdir()

        for file in amass_filenames:
            path = search(src, file)

            io.extract.bz2(path, tmp_path)

            for path in tmp_path.glob("*"):
                shutil.move(path, Path(dst) / path.relative_to(tmp_path))
