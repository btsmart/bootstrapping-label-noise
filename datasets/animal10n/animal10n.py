import logging
import os

import torch
from torchvision import transforms
from PIL import Image

from datasets.autoaugment import CIFAR10Policy
from datasets.dataset import DS, BasicDataset, EvalDataset, TransformFixMatch
from datasets.randaugment import RandAugment
from utils.simclr.util import TwoCropTransform

logger = logging.getLogger(__name__)

animal10n_mean = (0.4914, 0.4822, 0.4465)
animal10n_std = (0.2470, 0.2435, 0.2616)

animal10n_class_names = [
    "Cat",
    "Lynx",
    "Wolf",
    "Coyote",
    "Cheetah",
    "jaguar",
    "Chimpanzee",
    "Orangutan",
    "Hamster",
    "Guinea pig",
]

current_dir = os.getcwd()
animal10n_data_dir = os.path.join(current_dir, "data", "animal10n")
animal10n_label_dir = os.path.join(current_dir, "datasets", "animal10n", "noisy_labels")
animal10n_dataset_dir = os.path.join(
    current_dir, "datasets", "animal10n", "saved_datasets"
)

# ---------------------
# --- Augmentations ---
# ---------------------

animal10n_weak_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=8, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(animal10n_mean, animal10n_std),
    ]
)

animal10n_strong_transform = transforms.Compose(
    [
        RandAugment(3, 5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=8, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(animal10n_mean, animal10n_std),
    ]
)

animal10n_fixmatch_transform = TransformFixMatch(
    animal10n_weak_transform, animal10n_strong_transform
)

animal10n_test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(animal10n_mean, animal10n_std)]
)

animal10n_simclr_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=64, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(animal10n_mean, animal10n_std),
    ]
)

animal10n_splitnet_strong_transform = transforms.Compose(
    [
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(animal10n_mean, animal10n_std),
        transforms.RandomErasing(
            p=0.5, scale=(0.125, 0.2), ratio=(0.99, 1.0), value=0, inplace=False,
        ),
    ]
)

animal10n_splitnet_weak_transform = transforms.Compose(
    [
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(animal10n_mean, animal10n_std),
    ]
)

animal10n_splitnet_fixmatch_transform = TransformFixMatch(
    animal10n_splitnet_weak_transform, animal10n_splitnet_strong_transform
)

animal10n_splitnet_test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(animal10n_mean, animal10n_std),]
)

# ----------------
# --- Datasets ---
# ----------------

# Modified from PLC Paper
# https://github.com/pxiangwu/PLC/blob/master/animal10/dataset.py
class Animal10N(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):

        if train:
            split = "train"
        else:
            split = "test"

        self.image_dir = os.path.join(root, split + "ing")
        self.image_files = [
            f
            for f in os.listdir(self.image_dir)
            if os.path.isfile(os.path.join(self.image_dir, f))
        ]
        self.image_files = sorted(self.image_files)

        self.targets = []

        for path in self.image_files:
            label = path.split("_")[0]
            self.targets.append(int(label))

        self.transform = transform

    def get_data(self, index):

        image_path = os.path.join(self.image_dir, self.image_files[index])
        return Image.open(image_path)

    def __getitem__(self, index):

        image = self.get_data(index)
        if self.transform is not None:
            image = self.transform(image)

        label = self.targets[index]
        return image, label

    def __len__(self):

        return len(self.targets)


def animal10n_class_name(index):
    """Return the name of the class corresponding to 'index'"""
    return animal10n_class_names[index]


def animal10n_load_dataset(dataset_type):
    """Load a dataset from the Animal10N saved datasets directory"""
    ds_dir = os.path.join(animal10n_dataset_dir, dataset_type, "ds.json")
    ds = DS.load_from_file(ds_dir)
    return ds


# ---------------
# --- Loaders ---
# ---------------


def dataset_from_ds(ds, transform):
    return BasicDataset(Animal10N(animal10n_data_dir, train=True), ds, transform)


def eval_dataset(transform, train=False):
    return EvalDataset(Animal10N(animal10n_data_dir, train=train), transform,)


def regular_dataset(transform, train=False):
    return Animal10N(animal10n_data_dir, train=train, transform=transform)


def animal10n_ssl_datasets(clean_ds, noisy_ds):
    logger.info("Getting Animal10N SSL Datasets")
    train_clean_dataset = dataset_from_ds(clean_ds, animal10n_weak_transform)
    train_noisy_dataset = dataset_from_ds(noisy_ds, animal10n_fixmatch_transform)
    test_noisy_dataset = dataset_from_ds(noisy_ds, animal10n_weak_transform)
    test_dataset = regular_dataset(train=False, transform=animal10n_test_transform)
    return train_clean_dataset, train_noisy_dataset, test_noisy_dataset, test_dataset


def animal10n_bootstrapping_datasets(ds, train_aug):

    augs = {
        "strong": animal10n_splitnet_strong_transform,
        "weak": animal10n_splitnet_weak_transform,
        "eval": animal10n_splitnet_test_transform,
    }

    logger.info("Getting Animal10N Clean Set Datasets")
    train_dataset = dataset_from_ds(ds, augs[train_aug])
    test_dataset = regular_dataset(animal10n_splitnet_test_transform, train=False)
    return train_dataset, test_dataset


def animal10n_split_dataset_datasets(ds, eval_aug):

    augs = {
        "strong": animal10n_splitnet_strong_transform,
        "weak": animal10n_splitnet_weak_transform,
        "eval": animal10n_splitnet_test_transform,
    }

    logger.info("Getting Animal10N Clean Set Datasets")
    eval_dataset = dataset_from_ds(ds, augs[eval_aug])
    return eval_dataset


def animal10n_test_aug_datasets():
    logger.info("Getting Animal10n Test Aug Dataset")
    test_dataset = eval_dataset(animal10n_splitnet_weak_transform, train=False)
    return test_dataset


def animal10n_final_datasets(ds):
    logger.info("Getting Animal10N Final Datasets")
    train_dataset = dataset_from_ds(ds, animal10n_splitnet_strong_transform)
    test_dataset = regular_dataset(animal10n_splitnet_test_transform, train=False)
    return train_dataset, test_dataset


def animal10n_pretraining_dataset(ds):
    logger.info("Getting animal10n Pretraining Datasets")
    train_dataset = dataset_from_ds(ds, TwoCropTransform(animal10n_simclr_transform))
    return train_dataset
