import shutil
from os import PathLike
from pathlib import Path

import gdown
import requests
from tqdm import tqdm


def url(
    url: str,
    path: PathLike,
    chunk_size: int = 8192,
    verbose: bool = True,
) -> str:
    """
    This function downloads a file from a given URL and saves it to a specified path.

    Args:
        url (str): The URL of the file to be downloaded.
        path (PathLike, optional): The local path where the downloaded file should be
            saved. Defaults to the current directory.
        chunk_size (int, optional): The size of the chunks in which the file is
            downloaded. Defaults to 8192 bytes.

    Returns:
        Path: The local path where the downloaded file has been saved.

    Raises:
        Exception: If the downloaded file size does not match the expected file size.
    """

    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get the filename from the headers if it exists, else use the url
    if "content-disposition" in response.headers:
        filename = response.headers.get("content-disposition").split("filename=")[1]
    else:
        filename = url.split("/")[-1]

    full_path = Path(path) / filename

    absolute_path = full_path.expanduser().absolute()
    absolute_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = Path(f"{absolute_path}.download")

    file_size = int(response.headers.get("Content-Length", 0))
    if verbose:
        progress = tqdm(total=file_size, unit="iB", unit_scale=True)

    with tmp_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if verbose:
                progress.update(len(chunk))

            f.write(chunk)

    progress.close()

    shutil.move(tmp_path, absolute_path)
    return full_path


def gdrive_folder(url=None, id=None, path=None, verbose: bool = False):
    gdown.download_folder(url=url, id=id, output=str(path), quiet=not verbose)
    return path
