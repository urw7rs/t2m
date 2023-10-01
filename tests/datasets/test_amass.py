import shutil
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from t2m.datasets import AMASS


@pytest.fixture()
def amass_path(data_root, tmp_path):
    # HumanML3D requires the SMPL datasets from AMASS and humanact12

    ## Raw SMPL datasets:
    # ACCAD.tar.bz2
    # BMLhandball.tar.bz2
    # BMLmovi.tar.bz2
    # BMLrub.tar.bz2
    # CMU.tar.bz2
    # DFaust.tar.bz2
    # EKUT.tar.bz2
    # EyesJapanDataset.tar.bz2
    # "HDM05.tar.bz2
    # HumanEva.tar.bz2
    # KIT.tar.bz2
    # MoSh.tar.bz2
    # PosePrior.tar.bz2
    # SFU.tar.bz2
    # SSM.tar.bz2
    # TCDHands.tar.bz2
    # TotalCapture.tar.bz2
    # Transitions.tar.bz2

    ## SMPL-H and DMPL models
    # smplh.tar.xz
    # dmpls.tar.xz
    path = Path(tmp_path) / "AMASS"
    path.mkdir()
    yield path
    shutil.rmtree(path)


def test_build(data_root, amass_path):
    AMASS.build(data_root, amass_path)


@pytest.fixture()
def amass(amass_path):
    dataset = AMASS(amass_path)
    return dataset


def test_amass_dataloader(amass):
    dataloader = DataLoader(amass, batch_size=1, pin_memory=True)

    for batch in dataloader:
        assert batch["trans"].shape[-1] == 3
        assert batch["poses"].shape[-1] == 156
        assert batch["betas"].shape[-1] == 16

        metadata = batch["metadata"]
        assert "fps" in metadata.keys()
        assert "gender" in metadata.keys()
        assert "path" in metadata.keys()

        assert metadata["fps"].shape == torch.Size([1])
        assert len(metadata["gender"]) == 1

        break
