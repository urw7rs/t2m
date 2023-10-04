import os
from os import PathLike
from pathlib import Path
from typing import List

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from t2m import humanml3d_utils, io, skel_utils
from t2m.common import search
from t2m.smpl2joints import SMPL2Joints

from .amass import AMASS


class HumanML3D(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root

    def __getitem__(self, index):
        ...

    def __len__(self):
        ...

    @staticmethod
    def build(src: PathLike, dst: PathLike):
        # setup dir
        src = Path(src)
        pose_path = Path(dst) / "pose_data"
        joints_path = Path(dst) / "joints"
        new_joints_path = Path(dst) / "new_joints"
        new_joint_vecs_path = Path(dst) / "new_joint_vecs"

        # precision needs to be set to highest for accuracy
        torch.set_float32_matmul_precision("highest")

        fabric = L.Fabric(devices=1, precision=32)

        amass_path = Path(dst) / "AMASS"
        if not amass_path.exists():
            amass_path.mkdir()
            AMASS.build(src, amass_path)

        dataset = AMASS(amass_path)

        workers = os.cpu_count()
        if workers is None:
            workers = 1

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, pin_memory=True, num_workers=workers
        )
        dataloader = fabric.setup_dataloaders(dataloader)

        smpl2joints_path = Path(dst) / "smpl2joints"
        if not smpl2joints_path.exists():
            smpl2joints_path.mkdir()
            SMPL2Joints.extract(src, smpl2joints_path)

        smpl2joints = SMPL2Joints.from_dir(smpl2joints_path)
        smpl2joints = fabric.setup_module(smpl2joints)

        for batch in tqdm(dataloader, dynamic_ncols=True):
            metadata = batch["metadata"]

            pose_seq = smpl2joints(
                batch["trans"][0],
                batch["poses"][0],
                batch["betas"][0],
                fps=int(metadata["fps"][0]),
                gender=metadata["gender"][0],
            )

            path = Path(metadata["path"][0]).with_suffix(".npy")
            new_path = pose_path / path

            new_path.parent.mkdir(exist_ok=True, parents=True)
            np.save(new_path, pose_seq.cpu().numpy())

        url = "https://github.com/EricGuo5513/HumanML3D/raw/99b33e1cc7826ae96b0ee11a734453e250e5e75f/pose_data/humanact12.zip"
        zip_path = io.download.url(url, pose_path)
        io.extract.zip(zip_path, pose_path)

        url = "https://github.com/EricGuo5513/HumanML3D/raw/99b33e1cc7826ae96b0ee11a734453e250e5e75f/index.csv"
        index_path = io.download.url(url, dst)

        df = pd.read_csv(index_path)

        for index, row in tqdm(list(df.iterrows()), dynamic_ncols=True):
            path = Path(row["source_path"]).relative_to("pose_data")
            np_path = pose_path / path

            data = np.load(np_path)

            fps = 20
            if "humanact12" not in str(path):
                if "Eyes_Japan_Dataset" in str(path):
                    data = data[3 * fps :]
                if "MPI_HDM05" in str(path):
                    data = data[3 * fps :]
                if "TotalCapture" in str(path):
                    data = data[1 * fps :]
                if "MPI_Limits" in str(path):
                    data = data[1 * fps :]
                if "Transitions_mocap" in str(path):
                    data = data[int(0.5 * fps) :]

                data = data[row["start_frame"] : row["end_frame"]]
                data[..., 0] *= -1

            data_m = humanml3d_utils.swap_left_right(data)

            new_path = joints_path / row["new_name"]
            new_path.parent.mkdir(exist_ok=True, parents=True)

            np.save(new_path, data)

            new_path = new_path.with_name("M" + new_path.name)
            np.save(new_path, data_m)

        raw_offsets = skel_utils.t2m_raw_offsets
        kinematic_chain = skel_utils.t2m_kinematic_chain

        skeleton = skel_utils.Skeleton(
            torch.from_numpy(raw_offsets), kinematic_chain, device="cpu"
        )

        target_skeleton = np.load(search(joints_path, "000021.npy"))
        target_skeleton = target_skeleton.reshape(len(target_skeleton), -1, 3)
        target_skeleton = torch.from_numpy(target_skeleton)

        target_offsets = skeleton.get_offsets_joints(target_skeleton[0])

        new_joints_path.mkdir(exist_ok=True, parents=True)
        new_joint_vecs_path.mkdir(exist_ok=True, parents=True)

        for path in tqdm(search(joints_path, "*.npy", all=True), dynamic_ncols=True):
            array = np.load(path)
            if array.shape[0] == 1:
                print(f"skipping {path}")
                continue

            array = array[:, : skeleton.njoints()]

            (
                data,
                ground_positions,
                positions,
                l_velocity,
            ) = humanml3d_utils.process_file(
                torch.from_numpy(raw_offsets),
                kinematic_chain,
                array,
                0.002,
                target_offsets,
                device="cpu",
            )

            rec_ric_data = humanml3d_utils.recover_from_ric(
                torch.from_numpy(data).unsqueeze(0).float(),
                joints_num=skeleton.njoints(),
            )

            new_path = Path(new_joints_path) / path.name
            np.save(new_path, rec_ric_data.squeeze().numpy())

            new_path = Path(new_joint_vecs_path) / path.name
            np.save(new_path, data)

        arrays: List[np.ndarray] = []
        for path in tqdm(
            search(new_joint_vecs_path, "*.npy", all=True), dynamic_ncols=True
        ):
            data = np.load(path)

            if np.isnan(data).any():
                print(f"skipping {path.name}")
                continue

            arrays.append(data)

        x = np.concatenate(arrays, axis=0)

        mean = x.mean(axis=0)
        std = x.std(axis=0)

        std = humanml3d_utils.compute_std(std)

        np.save(Path(dst) / "Mean.npy", mean)
        np.save(Path(dst) / "Std.npy", std)

        path = io.download.url(
            url="https://raw.githubusercontent.com/EricGuo5513/HumanML3D/ab5b332c3148ec669da4c55ad119e0d73861b867/HumanML3D/texts.zip",
            path=dst,
        )
        io.extract.zip(path, dst)

        io.download.url(
            "https://raw.githubusercontent.com/EricGuo5513/HumanML3D/ab5b332c3148ec669da4c55ad119e0d73861b867/HumanML3D/train.txt",
            dst,
        )
        io.download.url(
            "https://raw.githubusercontent.com/EricGuo5513/HumanML3D/ab5b332c3148ec669da4c55ad119e0d73861b867/HumanML3D/val.txt",
            dst,
        )
        io.download.url(
            "https://raw.githubusercontent.com/EricGuo5513/HumanML3D/ab5b332c3148ec669da4c55ad119e0d73861b867/HumanML3D/train_val.txt",
            dst,
        )
        io.download.url(
            "https://raw.githubusercontent.com/EricGuo5513/HumanML3D/ab5b332c3148ec669da4c55ad119e0d73861b867/HumanML3D/test.txt",
            dst,
        )
        io.download.url(
            "https://raw.githubusercontent.com/EricGuo5513/HumanML3D/ab5b332c3148ec669da4c55ad119e0d73861b867/HumanML3D/all.txt",
            dst,
        )
