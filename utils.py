import platform
import torch
import uuid
import os


def save_files_to_disk(files) -> None:
    folder_uuid = str(uuid.uuid4())
    folder_path = os.path.join('/tmp', folder_uuid)
    # Create the folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for file in files:
        with open(os.path.join(folder_path, file.name), "wb") as f:
            f.write(file.getbuffer())

    return folder_path


def get_device() -> str:
    """
    Returns the device to use for the language model.

    Returns:
        str: The device to use for the language model.
    """
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"

    return "cuda" if torch.cuda.is_available() else "cpu"
