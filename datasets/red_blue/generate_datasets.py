import os

import numpy as np

from datasets.red_blue.red_blue import (
    MiniImagenet,
    red_blue_data_dir,
    red_blue_dataset_dir,
    red_blue_label_dir,
)
from datasets.dataset import DS
import utils.utils as utils


def save_array(array, save_file):
    pardir = os.path.abspath(os.path.join(save_file, os.pardir))
    if not os.path.exists(pardir):
        os.makedirs(pardir)
    np.save(save_file, array)


def dataset_from_noise_types(noise_types, targets):

    noisy_label_sets = [
        utils.load_labels(os.path.join(red_blue_label_dir, f"{noise_type}.npy"))
        for noise_type in noise_types
    ]

    num_elements = min([noisy_set.shape[0] for noisy_set in noisy_label_sets])

    ds = DS()
    ds.indices = np.arange(num_elements, dtype=np.int64)
    ds.true_labels = np.full_like(targets, 0, dtype=np.int64)
    ds.learned_labels = np.array(targets, dtype=np.int64)
    ds.noisy_label_sets = noisy_label_sets
    return ds


def generate_datasets():

    noise_rates = ["0.2", "0.4", "0.6", "0.8"]

    for noise_rate in noise_rates:

        targets = MiniImagenet(
            red_blue_data_dir, mode="train", color="red", noise_rate=noise_rate,
        ).targets

        red_blue_labels = utils.one_hot(np.array(targets), 100)

        label_save_dir = os.path.join(red_blue_label_dir, f"labels_{noise_rate}.npy")
        save_array(red_blue_labels, label_save_dir)

        # Create the datasets
        ds = dataset_from_noise_types([f"labels_{noise_rate}"], targets)

        # Save the datasets
        ds_save_dir = os.path.join(red_blue_dataset_dir, f"labels_{noise_rate}")
        if not os.path.exists(ds_save_dir):
            os.makedirs(ds_save_dir)
        ds.save_to_file(os.path.join(ds_save_dir, "ds.json"))


def main():

    generate_datasets()
