import shutil
from pathlib import Path

import pytest

from t2m.common import search


@pytest.fixture()
def dir(tmp_path):
    path = Path(tmp_path)
    subdir = path / "subdir"
    subdir.mkdir()
    (subdir / "test.txt").write_text("alsdkfjalsdkfjasdl")
    yield path
    shutil.rmtree(tmp_path)


def test_search(dir):
    search(dir, "test.txt")
