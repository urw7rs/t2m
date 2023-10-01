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
