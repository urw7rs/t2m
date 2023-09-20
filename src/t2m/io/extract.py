import os
import subprocess
import tarfile
import zipfile
from os import PathLike
from pathlib import Path

import patoolib


def get_workers(workers: int):
    assert workers >= 0

    if workers == 0:
        num_cpus = os.cpu_count()
        workers = 1 if num_cpus is None else num_cpus

    return workers


def mkdirs(path: PathLike):
    Path(path).mkdir(exist_ok=True, parents=True)


def bz2(src: PathLike, dst: PathLike, workers: int = 0):
    workers = get_workers(workers)

    mkdirs(dst)

    cmd = f"lbzip2 -dc -n {workers} {src} | tar -xf - -C {dst}"
    subprocess.run(cmd, shell=True, check=True)

    return list(Path(dst).glob("**/*"))


def xz(src: PathLike, dst: PathLike, workers: int = 0):
    workers = get_workers(workers)

    mkdirs(dst)

    cmd = f"xz -T{workers} -dc {src} | tar -xf - -C {dst}"
    subprocess.run(cmd, shell=True, check=True)

    return list(Path(dst).glob("**/*"))


def zip(src: PathLike, dst: PathLike):
    mkdirs(dst)

    with zipfile.ZipFile(src) as zip_file:
        zip_file.extractall(dst)

        files = []
        for name in zip_file.namelist():
            files.append(Path(dst) / name)

    return files


def rar(src: PathLike, dst: PathLike):
    mkdirs(dst)

    patoolib.extract_archive(src, outdir=dst)

    return list(Path(dst).glob("**/*"))


def tar(tar_path: PathLike, path: PathLike):
    with tarfile.open(tar_path) as tar_file:
        tar_file.extractall(path)

        paths = []
        for name in tar_file.getnames():
            paths.append(Path(path) / name)

        return paths
