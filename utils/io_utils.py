import json
import yaml
import os


def txt_loader(path, skip_lines=0):
    with open(path, "r") as f:
        content = f.read().splitlines()
    return content[skip_lines:]


def json_loader(json_path):
    """Loads json from a path

    Args:
        json_path (str): Path to json file

    Returns:
        dict: Dict with json content
    """
    with open(json_path, "r") as f:
        loaded_json = json.load(f)
    return loaded_json


def yaml_loader(yaml_path):
    """Loads yaml from a path

    Args:
        yaml_path (str): Path to yaml file

    Returns:
        dict: Dict with yaml content
    """
    with open(yaml_path, "r") as f:
        loaded_yaml = yaml.safe_load(f)
    return loaded_yaml


def check_if_exists(path, create=True):
    """Checks if a path exists and, if wanted, creates it

    Args:
        path (str): Path to be checked
        create (bool, optional): If path doesn't exists, creates it or not. Defaults to True.

    Returns:
        Bool: Exists or not
    """
    if os.path.exists(path):
        return True
    elif create:
        os.mkdir(path)
        return True
    else:
        return False
