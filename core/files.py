import os
import uuid


def save_files_to_disk(files) -> str:
    """
    Saves a list of files to disk and returns the path to the folder where they were saved.

    Args:
        files: A list of files to be saved to disk.

    Returns:
        The path to the folder where the files were saved.
    """
    folder_uuid = str(uuid.uuid4())
    folder_path = os.path.join('/tmp', folder_uuid)
    # Create the folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for file in files:
        with open(os.path.join(folder_path, file.name), "wb") as f:
            f.write(file.getbuffer())

    return folder_path
