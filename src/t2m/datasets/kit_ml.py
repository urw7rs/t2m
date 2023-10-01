from pathlib import Path

from torch.utils.data import Dataset

from t2m import io


class KITML(Dataset):
    def __init__(self):
        ...

    def __getitem__(self, index):
        ...

    def __len__(self):
        ...

    @staticmethod
    def download(path, verbose: bool = True):
        print("Downloading and Extracting KIT-ML dataset...")

        # download and extract KIT-ML
        path = Path(path)

        path.mkdir(exist_ok=True, parents=True)

        download_path = Path(path) / "download"
        root = io.download.gdrive_folder(
            id="1MnixfyGfujSP-4t8w_2QvjtTVpEKr97t", path=download_path, verbose=verbose
        )

        # copy and extract files to KIT-ML
        path.mkdir(exist_ok=True, parents=True)

        def cp(src, dst):
            data = src.read_bytes()
            dst.write_bytes(data)

        for file in [
            "train.txt",
            "val.txt",
            "train_val.txt",
            "test.txt",
            "all.txt",
            "Mean.npy",
            "Std.npy",
        ]:
            src = download_path / file
            dst = path / file

            print(f"copying {src} to {dst}...")
            cp(src, dst)

        root = Path(root)
        for file in ["new_joints.rar", "new_joint_vecs.rar", "texts.rar"]:
            rar_path = root / file

            print(f"extracting {rar_path} to {path}...")
            io.extract.rar(rar_path, path)
