import json
import math
import os
import random

import numpy as np

from datasets.cifar100.cifar100 import (
    CIFAR100,
    cifar100_data_dir,
    cifar100_dataset_dir,
    cifar100_label_dir,
)
from datasets.dataset import DS
import utils.utils as utils


cifar100_targets = CIFAR100(cifar100_data_dir, download=True).targets


def save_array(array, save_file):
    pardir = os.path.abspath(os.path.join(save_file, os.pardir))
    if not os.path.exists(pardir):
        os.makedirs(pardir)
    np.save(save_file, array)


def dataset_from_noise_types(noise_types):

    noisy_label_sets = [
        utils.load_labels(os.path.join(cifar100_label_dir, f"{noise_type}.npy"))
        for noise_type in noise_types
    ]

    num_elements = min([noisy_set.shape[0] for noisy_set in noisy_label_sets])

    ds = DS()
    ds.indices = np.arange(num_elements, dtype=np.int64)
    ds.true_labels = np.array(cifar100_targets, dtype=np.int64)
    ds.learned_labels = np.array(cifar100_targets, dtype=np.int64)
    ds.noisy_label_sets = noisy_label_sets
    return ds


def create_and_save_dataset(noise_type):

    ds = dataset_from_noise_types([noise_type])

    ds_save_dir = os.path.join(cifar100_dataset_dir, noise_type)
    if not os.path.exists(ds_save_dir):
        os.makedirs(ds_save_dir)
    ds.save_to_file(os.path.join(ds_save_dir, "ds.json"))


def generate_sym_datasets():

    noise_rates = [0.0, 0.2, 0.5, 0.8, 0.9]

    for noise_rate in noise_rates:

        # Generate the noisy labels
        sym_labels = np.copy(cifar100_targets)
        num_noisy_samples = math.floor(sym_labels.shape[0] * noise_rate)
        noisy_samples = utils.get_random_splits(
            sym_labels.shape[0], [num_noisy_samples]
        )[0]
        for i in noisy_samples:
            sym_labels[i] = random.randint(0, 99)
        sym_labels = utils.one_hot(sym_labels, 100)

        # Save the labels
        label_save_dir = os.path.join(cifar100_label_dir, f"sym-{noise_rate}.npy")
        save_array(sym_labels, label_save_dir)

        create_and_save_dataset(f"sym-{noise_rate}")


def generate_pmd_datasets():

    pmd_noises = [
        "pmd-1-0.35",
        "pmd-1-0.70",
        "pmd-2-0.35",
        "pmd-2-0.70",
        "pmd-3-0.35",
        "pmd-3-0.70",
    ]

    for pmd_noise in pmd_noises:

        pmd_labels = utils.load_labels(
            os.path.join(cifar100_label_dir, "pmd_single_label", f"{pmd_noise}.npy")
        ).astype(np.int32)
        pmd_labels = utils.one_hot(pmd_labels, 100)
        label_save_dir = os.path.join(cifar100_label_dir, f"{pmd_noise}.npy")
        save_array(pmd_labels, label_save_dir)

        create_and_save_dataset(pmd_noise)


def generate_rog_datasets():

    rog_noises = [
        "semantic_densenet",
        "semantic_resnet",
        "semantic_vgg",
    ]

    for rog_noise in rog_noises:

        with open(
            os.path.join(cifar100_label_dir, "rog", rog_noise + ".json"), "r"
        ) as f:
            rog_labels = json.load(f)
        rog_labels = utils.one_hot(np.array(rog_labels), 100)
        label_save_dir = os.path.join(cifar100_label_dir, f"{rog_noise}.npy")
        save_array(rog_labels, label_save_dir)

        create_and_save_dataset(rog_noise)


def main():

    utils.set_seed(42)

    generate_sym_datasets()
    generate_pmd_datasets()
    generate_rog_datasets()
