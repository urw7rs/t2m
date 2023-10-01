from os import PathLike

from jsonargparse import CLI

from t2m.datasets import KITML, HumanML3D


def humanml3d(src: PathLike, root: PathLike = "HumanML3D", workers: int = 0):
    """Build HumanML3D Dataset

    Parameters:
        src (PathLike): path to search for amass files and smpl models
        root (PathLike): path to output dataset to
        workers (int): number of workers to use for parallel processing.
            0 sets workers to the number of cpus

    """

    HumanML3D.build(src, root)


def kit_ml(root: PathLike = "KIT-ML"):
    KITML.download(root, verbose=True)


if __name__ == "__main__":
    CLI([humanml3d, kit_ml], as_positional=False)
