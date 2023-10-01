from pathlib import Path

from t2m.humanml3d_utils import validate


def test_validate(data_root):
    path = Path(data_root) / "HumanML3D"
    if path.exists():
        validate(path)
