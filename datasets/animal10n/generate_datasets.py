import os

import numpy as np

from datasets.animal10n.animal10n import (
    Animal10N,
    animal10n_data_dir,
    animal10n_dataset_dir,
    animal10n_label_dir,
)
from datasets.dataset import DS
import utils.utils as utils


animal10n_targets = Animal10N(animal10n_data_dir).targets


def save_array(array, save_file):
    pardir = os.path.abspath(os.path.join(save_file, os.pardir))
    if not os.path.exists(pardir):
        os.makedirs(pardir)
    np.save(save_file, array)


def dataset_from_noise_types(noise_types):

    noisy_label_sets = [
        utils.load_labels(os.path.join(animal10n_label_dir, f"{noise_type}.npy"))
        for noise_type in noise_types
    ]

    num_elements = min([noisy_set.shape[0] for noisy_set in noisy_label_sets])

    ds = DS()
    ds.indices = np.arange(num_elements, dtype=np.int64)
    ds.true_labels = np.full_like(animal10n_targets, 0, dtype=np.int64)
    ds.learned_labels = np.array(animal10n_targets, dtype=np.int64)
    ds.noisy_label_sets = noisy_label_sets
    return ds

def generate_dataset():

    animal10n_labels = utils.one_hot(np.array(animal10n_targets), 10)

    label_save_dir = os.path.join(animal10n_label_dir, "labels.npy")
    save_array(animal10n_labels, label_save_dir)

    # Create the datasets
    ds = dataset_from_noise_types(["labels"])

    # Save the datasets
    ds_save_dir = os.path.join(animal10n_dataset_dir, "labels")
    if not os.path.exists(ds_save_dir):
        os.makedirs(ds_save_dir)
    ds.save_to_file(os.path.join(ds_save_dir, "ds.json"))


def main():

    generate_dataset()
