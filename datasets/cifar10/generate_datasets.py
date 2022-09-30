import json
import math
import os
import random

import numpy as np

from datasets.cifar10.cifar10 import (
    CIFAR10,
    cifar10_data_dir,
    cifar10_dataset_dir,
    cifar10_label_dir,
)
from datasets.dataset import DS
import utils.utils as utils


cifar10_targets = CIFAR10(cifar10_data_dir, download=True).targets


def save_array(array, save_file):
    pardir = os.path.abspath(os.path.join(save_file, os.pardir))
    if not os.path.exists(pardir):
        os.makedirs(pardir)
    np.save(save_file, array)


def dataset_from_noise_types(noise_types):

    noisy_label_sets = [
        utils.load_labels(os.path.join(cifar10_label_dir, f"{noise_type}.npy"))
        for noise_type in noise_types
    ]

    num_elements = min([noisy_set.shape[0] for noisy_set in noisy_label_sets])

    ds = DS()
    ds.indices = np.arange(num_elements, dtype=np.int64)
    ds.true_labels = np.array(cifar10_targets, dtype=np.int64)
    ds.learned_labels = np.array(cifar10_targets, dtype=np.int64)
    ds.noisy_label_sets = noisy_label_sets
    return ds


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

        # Create the datasets
        ds = dataset_from_noise_types([pmd_noise])

        # Save the datasets
        ds_save_dir = os.path.join(cifar10_dataset_dir, pmd_noise)
        if not os.path.exists(ds_save_dir):
            os.makedirs(ds_save_dir)
        ds.save_to_file(os.path.join(ds_save_dir, "ds.json"))


def create_and_save_dataset(noise_type):

    ds = dataset_from_noise_types([noise_type])

    ds_save_dir = os.path.join(cifar10_dataset_dir, noise_type)
    if not os.path.exists(ds_save_dir):
        os.makedirs(ds_save_dir)
    ds.save_to_file(os.path.join(ds_save_dir, "ds.json"))


def generate_scan_dataset():

    create_and_save_dataset("scan")


def generate_asym_datasets():

    noise_rates = [0.2, 0.4, 0.49, 0.6]
    transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}

    for noise_rate in noise_rates:

        # Generate the noisy labels
        asym_labels = np.copy(cifar10_targets)
        num_noisy_samples = math.floor(asym_labels.shape[0] * noise_rate)
        noisy_samples = utils.get_random_splits(
            asym_labels.shape[0], [num_noisy_samples]
        )[0]
        for i in noisy_samples:
            asym_labels[i] = transition[asym_labels[i]]
        asym_labels = utils.one_hot(asym_labels, 10)

        # Save the labels
        label_save_dir = os.path.join(cifar10_label_dir, f"asym-{noise_rate}.npy")
        save_array(asym_labels, label_save_dir)

        create_and_save_dataset(f"asym-{noise_rate}")


def generate_sym_datasets():

    noise_rates = [0.0, 0.2, 0.5, 0.8, 0.9]

    for noise_rate in noise_rates:

        # Generate the noisy labels
        sym_labels = np.copy(cifar10_targets)
        num_noisy_samples = math.floor(sym_labels.shape[0] * noise_rate)
        noisy_samples = utils.get_random_splits(
            sym_labels.shape[0], [num_noisy_samples]
        )[0]
        for i in noisy_samples:
            sym_labels[i] = random.randint(0, 9)
        sym_labels = utils.one_hot(sym_labels, 10)

        # Save the labels
        label_save_dir = os.path.join(cifar10_label_dir, f"sym-{noise_rate}.npy")
        save_array(sym_labels, label_save_dir)

        create_and_save_dataset(f"sym-{noise_rate}")


def generate_true_dataset():

    true_labels = utils.one_hot(np.array(cifar10_targets), 10)

    # Save the labels
    label_save_dir = os.path.join(cifar10_label_dir, f"true.npy")
    save_array(true_labels, label_save_dir)

    create_and_save_dataset("true")


def generate_random_dataset():

    labels = np.zeros(len(cifar10_targets), dtype=np.int32)
    for i in range(len(labels)):
        labels[i] = random.randint(0, 9)
    labels = utils.one_hot(labels, 10)

    # Save the labels
    label_save_dir = os.path.join(cifar10_label_dir, f"random.npy")
    save_array(labels, label_save_dir)

    create_and_save_dataset("random")


def generate_rog_datasets():

    rog_noises = [
        "semantic_densenet",
        "semantic_resnet",
        "semantic_vgg",
    ]

    for rog_noise in rog_noises:

        with open(
            os.path.join(cifar10_label_dir, "rog", rog_noise + ".json"), "r"
        ) as f:
            rog_labels = json.load(f)
        rog_labels = utils.one_hot(np.array(rog_labels), 10)
        label_save_dir = os.path.join(cifar10_label_dir, f"{rog_noise}.npy")
        save_array(rog_labels, label_save_dir)

        create_and_save_dataset(rog_noise)


def main():

    utils.set_seed(42)

    generate_asym_datasets()
    generate_sym_datasets()
    generate_pmd_datasets()
    generate_scan_dataset()
    generate_rog_datasets()
    generate_true_dataset()
    generate_random_dataset()
