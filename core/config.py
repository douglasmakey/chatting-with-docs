import platform
import torch
import yaml

CONFIG_FILE_PATH = "config.yaml"


def get_device() -> str:
    """
    Returns the device to use for the language model.

    Returns:
        str: The device to use for the language model.
    """

    print("Getting device")

    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"

    return "cuda" if torch.cuda.is_available() else "cpu"


def read_config() -> object:
    """
    Reads the configuration file and loads it into the `config` attribute of the object.

    :return: None
    """

    print("Loading configuration file...")

    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


CONFIG = read_config()
DEVICE = get_device()
