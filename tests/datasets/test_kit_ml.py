import shutil

import pytest

from t2m.datasets import KITML


@pytest.fixture()
def kit_ml_path(tmp_path):
    KITML.download(tmp_path)

    def exists(file):
        return (tmp_path / "train.txt").exists()

    for file in [
        "train.txt",
        "val.txt",
        "train_val.txt",
        "test.txt",
        "all.txt",
        "Mean.npy",
        "Std.npy",
        "new_joints",
        "new_joint_vecs",
        "texts",
    ]:
        assert exists(file)

    yield tmp_path

    shutil.rmtree(tmp_path)


def test_kit_ml(kit_ml_path):
    ...
