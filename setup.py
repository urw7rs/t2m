from pathlib import Path

from setuptools import find_packages, setup


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="t2m",
    version="0.0.0",
    description="Text To Motion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chanhyuk Jung",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "numpy==1.23.1",
        "patool",
        "torch",
        "einops",
        "scipy",
        "click",
        "pandas",
        "jsonargparse",
        "tqdm",
        "requests",
        "lightning>=2.0",
        "gdown",
        "requests",
        "matplotlib>=3.6.0",
        "imageio",
        "smplx",
        "chumpy",
        "pyrender",
        "shapely",
        "h5py",
        "mapbox_earcut",
        "pygifsicle",
        "dask",
    ],
    extras_require={
        "test": ["pytest", "pytest-xdist"],
        "dev": ["black", "ruff", "bumpver"],
    },
)
