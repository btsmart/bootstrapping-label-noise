import os

import numpy as np

from datasets.webvision.webvision import (
    Webvision,
    webvision_data_dir,
    webvision_dataset_dir,
    webvision_label_dir,
)
from datasets.dataset import DS
import utils.utils as utils


webvision_targets = Webvision(webvision_data_dir, mode="train").targets


def save_array(array, save_file):
    pardir = os.path.abspath(os.path.join(save_file, os.pardir))
    if not os.path.exists(pardir):
        os.makedirs(pardir)
    np.save(save_file, array)


def dataset_from_noise_types(noise_types):

    noisy_label_sets = [
        utils.load_labels(os.path.join(webvision_label_dir, f"{noise_type}.npy"))
        for noise_type in noise_types
    ]

    num_elements = min([noisy_set.shape[0] for noisy_set in noisy_label_sets])

    ds = DS()
    ds.indices = np.arange(num_elements, dtype=np.int64)
    ds.true_labels = np.full_like(webvision_targets, 0, dtype=np.int64)
    ds.learned_labels = np.array(webvision_targets, dtype=np.int64)
    ds.noisy_label_sets = noisy_label_sets
    return ds


def generate_dataset():

    webvision_labels = utils.one_hot(np.array(webvision_targets), 50)

    label_save_dir = os.path.join(webvision_label_dir, "labels.npy")
    save_array(webvision_labels, label_save_dir)

    # Create the datasets
    ds = dataset_from_noise_types(["labels"])

    # Save the datasets
    ds_save_dir = os.path.join(webvision_dataset_dir, "labels")
    if not os.path.exists(ds_save_dir):
        os.makedirs(ds_save_dir)
    ds.save_to_file(os.path.join(ds_save_dir, "ds.json"))


def main():

    generate_dataset()
