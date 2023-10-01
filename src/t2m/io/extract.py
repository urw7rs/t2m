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


def run_cmd(cmd):
    try:
        subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        output = exc.output.decode()
        raise RuntimeError(output)


def bz2(src: PathLike, dst: PathLike, workers: int = 0):
    workers = get_workers(workers)

    run_cmd(f"lbzip2 -dc -n {workers} {src} | tar -xf - -C {dst}")


def xz(src: PathLike, dst: PathLike, workers: int = 0):
    workers = get_workers(workers)

    run_cmd(f"xz -T{workers} -dc {src} | tar -xf - -C {dst}")


def zip(src: PathLike, dst: PathLike):
    with zipfile.ZipFile(src) as zip_file:
        zip_file.extractall(dst)


def rar(src: PathLike, dst: PathLike):
    patoolib.extract_archive(src, outdir=dst)


def tar(src: PathLike, dst: PathLike):
    with tarfile.open(src) as tar_file:
        tar_file.extractall(dst)
