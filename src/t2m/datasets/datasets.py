import math
from os import PathLike
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..models.word2vec import Word2Vec


class AMASS(Dataset):
    def __init__(self, amass_paths: Iterable[PathLike]):
        super().__init__()

        self.valid_paths = []
        for path in amass_paths:
            assert Path(path).exists()

            bdata = np.load(path, allow_pickle=True)

            fps = int(bdata.get("mocap_framerate", 0))
            frame_number = bdata.get("trans", None)
            if fps == 0 or frame_number is None:
                continue

            self.valid_paths.append(path)

    def path_at(self, index):
        return self.valid_paths[index]

    def gender_at(self, index):
        bdata = np.load(self.valid_paths[index], allow_pickle=True)
        return str(bdata["gender"])

    def __getitem__(self, index):
        bdata = np.load(self.valid_paths[index], allow_pickle=True)

        fps = int(bdata.get("mocap_framerate", 0))
        frame_number = bdata.get("trans", None)
        frame_number = frame_number.shape[0]

        def to_tensor(array):
            return torch.as_tensor(array, dtype=torch.float32)

        poses, betas, trans = [
            to_tensor(bdata[key]) for key in ["poses", "betas", "trans"]
        ]

        return trans, fps, poses, betas, index

    def __len__(self):
        return len(self.valid_paths)


def load_txt(path: PathLike):
    fps = 20

    metadata = []
    for line in Path(path).read_text().splitlines():
        caption, pos_tags, start, end = line.split("#")
        pos_tags = pos_tags.split(" ")

        start = float(start)
        end = float(end)

        if np.isnan(start):
            start = 0.0

        if np.isnan(end):
            end = 0.0

        annotations = {
            "caption": caption,
            "tokens": pos_tags,
            "start": int(start * fps),
            "end": int(end * fps),
        }

        metadata.append(annotations)

    return metadata


class MotionDataset(Dataset):
    def __init__(
        self,
        root: PathLike,
        *,
        split: str = "train",
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        transforms: Callable = None,
    ):
        abs_root = Path(root).expanduser()

        self.mean_path = abs_root / "Mean.npy"
        self.std_path = abs_root / "Std.npy"

        names = (abs_root / f"{split}.txt").read_text().splitlines()

        if min_length is None:
            min_length = -math.inf

        if max_length is None:
            max_length = math.inf

        self.paths = []
        for name in tqdm(sorted(names), dynamic_ncols=True):
            path = abs_root / "new_joint_vecs" / f"{name}.npy"

            # Some motion may not exist in KIT dataset
            if not path.exists():
                continue

            motion = np.load(path)
            motion_length = len(motion)

            if motion_length < min_length:
                continue

            if motion_length > max_length:
                continue

            self.paths.append(path)

        self.transforms = transforms

    def mean(self):
        mean = np.load(self.mean_path)
        mean = torch.from_numpy(mean).float()
        return mean

    def std(self):
        std = np.load(self.std_path)
        std = torch.from_numpy(std).float()
        return std

    def __getitem__(self, idx):
        path = self.paths[idx]

        motion = np.load(path)
        motion = torch.from_numpy(motion).float()

        if self.transforms is not None:
            motion = self.transforms(motion)

        return motion

    def __len__(self):
        return len(self.paths)


class MotionTextDataset(Dataset):
    max_text_length: int = 20

    def __init__(
        self,
        root: PathLike,
        *,
        split: str = "train",
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        transforms: Optional[Callable] = None,
    ):
        abs_root = Path(root).expanduser()

        self.mean_path = abs_root / "Mean.npy"
        self.std_path = abs_root / "Std.npy"

        names = (abs_root / f"{split}.txt").read_text().splitlines()

        path_pairs = []
        for name in sorted(names):
            path = abs_root / "new_joint_vecs" / f"{name}.npy"

            # Some motion may not exist in KIT dataset
            if path.exists():
                path_pairs.append((path, abs_root / "texts" / f"{name}.txt"))

        word2vec = Word2Vec.download(root)

        if min_length is None:
            min_length = -math.inf

        if max_length is None:
            max_length = math.inf

        self.data = []
        for i, (npy_path, text_path) in enumerate(tqdm(path_pairs, dynamic_ncols=True)):
            all_annotations = load_txt(text_path)

            motion = np.load(npy_path)

            motion_length = len(motion)

            if motion_length < min_length:
                continue

            pool = []
            for annotations in all_annotations:
                caption = annotations["caption"]
                tokens = annotations["tokens"]

                start = annotations["start"]
                end = annotations["end"]

                if start != 0 and end != 0:
                    motion = motion[start:end]

                motion_length = len(motion)

                if motion_length < min_length:
                    continue

                if motion_length > max_length:
                    continue

                if len(tokens) < self.max_text_length:
                    # add sos and eos tags
                    new_tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
                    num_tokens = len(new_tokens)
                    # pad with unk
                    new_tokens += ["unk/OTHER"] * (self.max_text_length - len(tokens))
                else:
                    # crop to match length
                    new_tokens = tokens[: self.max_text_length]
                    # add sos and eos tags
                    new_tokens = ["sos/OTHER"] + new_tokens + ["eos/OTHER"]
                    num_tokens = len(new_tokens)

                emb, one_hot = zip(*[word2vec[token] for token in new_tokens])

                emb = np.stack(emb)
                one_hot = np.stack(one_hot)

                labels = {
                    "caption": caption,
                    "word_embeddings": emb,
                    "pos_one_hots": one_hot,
                    "cap_lens": num_tokens,
                }

                if start == 0 and end == 0:
                    pool.append(labels)
                else:
                    self.data.append((npy_path, start, end, [labels]))

            if len(pool) > 0:
                self.data.append((npy_path, 0, 0, pool))

        self.transforms = transforms

    def mean(self):
        mean = np.load(self.mean_path)
        mean = torch.from_numpy(mean).float()
        return mean

    def std(self):
        std = np.load(self.std_path)
        std = torch.from_numpy(std).float()
        return std

    def __getitem__(self, idx):
        path, start, end, labels = self.data[idx]

        motion = np.load(path)
        if start != 0 and end != 0:
            motion = motion[start:end]
        motion = torch.from_numpy(motion).float()

        data = {"motion": motion, "labels": labels}

        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.data)
