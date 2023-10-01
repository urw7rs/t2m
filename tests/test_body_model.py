import shutil

import pytest
import torch

from t2m.smpl2joints import SMPL2Joints


@pytest.fixture()
def smpl2joints_path(data_root, tmp_path):
    path = tmp_path / "joints2smpl"
    path.mkdir()

    SMPL2Joints.extract(data_root, path)

    yield path

    shutil.rmtree(path)


@pytest.fixture()
def smpl2joints(smpl2joints_path):
    return SMPL2Joints.from_dir(smpl2joints_path)


def test_smpl2joints(smpl2joints):
    num_frames = 8

    trans = torch.randn(num_frames, 3)
    poses = torch.randn(num_frames, 156)
    betas = torch.randn(16)

    smpl2joints(trans, poses, betas, fps=20, gender="male")
