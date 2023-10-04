import shutil
from pathlib import Path

import pytest

from t2m.datasets import HumanML3D
from t2m.humanml3d_utils import validate


@pytest.fixture()
def humanml3d_path(tmp_path):
    path = Path(tmp_path) / "HumanML3D"
    path.mkdir()
    yield path
    shutil.rmtree(tmp_path)


def test_humanml3d(data_root, humanml3d_path, tmp_path):
    HumanML3D.build(data_root, humanml3d_path)

    validate(humanml3d_path)

    def check(file):
        assert (humanml3d_path / file).exists()

    check("Mean.npy")
    check("Std.npy")

    check("train.txt")
    check("val.txt")
    check("train_val.txt")
    check("test.txt")
    check("all.txt")

    check("new_joints")
    check("new_joint_vecs")

    check("texts")
