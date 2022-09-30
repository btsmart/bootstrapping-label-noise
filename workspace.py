import copy
import json
import logging
import os
import pprint
import time

import easydict
import git
import yaml

# This file handles work relating to loading the config file and creating the results
# directory. None of this directly relates to the functionality of the code and can be
# omitted.

logger = logging.getLogger(__name__)

# -------------------
# --- JSON / YAML ---
# -------------------


def load_yaml(file_path):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_to_json(dict, save_path):
    with open(save_path, "w") as f:
        json.dump(dict, f, indent=4)


# ---------------------------
# --- Config and Run Info ---
# ---------------------------


def dict_merge(a, b):
    """Merge two dictionaries together recursively, giving precendence to the `b`"""
    if isinstance(a, dict) == False or isinstance(b, dict) == False:
        return b
    result = copy.deepcopy(a)
    for key, value in b.items():
        if key in result:
            result[key] = dict_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_config(config_folder, config_name):
    config_file_path = os.path.join(config_folder, config_name)
    config = load_yaml(config_file_path)
    if "include" in config:
        for include in config["include"]:
            sub_config = get_config(config_folder, include)
            config = dict_merge(sub_config, config)
    return config


# ----------------
# --- Git Info ---
# ----------------


def structtime_to_dict(t):
    return {
        "year": t.tm_year,
        "month": t.tm_mon,
        "day": t.tm_mday,
        "hour": t.tm_hour,
        "minute": t.tm_min,
        "second": t.tm_sec,
    }


def timestamp_info(t):
    return {
        "time": t,
        "date_struct": structtime_to_dict(time.localtime(t)),
        "date_string": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)),
    }


def save_git_commit_info(save_path):
    """Use gitpython to save info about the current git commit to a file"""
    repo = git.Repo(search_parent_directories=True)
    head_commit = repo.head.commit
    git_commit_info = {
        "hexsha": head_commit.hexsha,
        "authored": {
            "author": head_commit.author.name,
            "authored_time": timestamp_info(head_commit.authored_date),
        },
        "committed": {
            "commit": head_commit.committer.name,
            "committed_time": timestamp_info(head_commit.committed_date),
        },
        "message": head_commit.message.strip(),
    }

    with open(save_path, "w") as f:
        yaml.dump(git_commit_info, f)

    return git_commit_info


# -----------------------------
# --- Create Results Folder ---
# -----------------------------


def create_results_folder(config):
    """Create the folder that results will be stored in"""

    # Create results directory
    result_path = time.strftime(config.result_dir, time.localtime())
    config.save_dir = os.path.join(config.current_dir, result_path)
    os.makedirs(config.save_dir)


def create_workspace_from_config(config, print_info=True):
    """Create a workspace from a config"""

    create_results_folder(config)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set up the logger to print messages *and* save them to a file
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(config.save_dir, "output.log")),
            logging.StreamHandler(),
        ],
    )

    # Save config to results folder
    if config.should_save.config:
        save_to_json(dict(config), os.path.join(config.save_dir, "config.json"))

    # Save information about the current git commit to results folder
    if config.should_save.git_info:
        save_git_commit_info(os.path.join(config.save_dir, "git.yaml"))

    # Print the info and save dir
    if print_info:
        logger.info(pprint.pformat(config))
        logger.info(f"Saving results to {config.save_dir}")

    return config


def create_workspace(current_dir, config_file_name, print_info=True):
    """Create a workspace from a config file name"""

    config = get_config(current_dir, config_file_name)
    config = easydict.EasyDict(config)
    config.current_dir = current_dir
    if print_info:
        logger.info(f"Using config from {config_file_name}")

    return create_workspace_from_config(config, print_info)

