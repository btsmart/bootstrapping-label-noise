# Adapted from https://github.com/filipe-research/PropMix/blob/main/data/webvision.py

import logging
import os

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from datasets.autoaugment import CIFAR10Policy
from datasets.dataset import DS, BasicDataset, EvalDataset, TransformFixMatch
from datasets.randaugment import RandAugment
from utils.simclr.util import TwoCropTransform


logger = logging.getLogger(__name__)

red_blue_mean = (0.485, 0.456, 0.406)
red_blue_std = (0.229, 0.224, 0.225)

current_dir = os.getcwd()
red_blue_data_dir = os.path.join(current_dir, "data", "red_blue")
red_blue_label_dir = os.path.join(current_dir, "datasets", "red_blue", "noisy_labels")
red_blue_dataset_dir = os.path.join(
    current_dir, "datasets", "red_blue", "saved_datasets"
)

# ---------------------
# --- Augmentations ---
# ---------------------

red_blue_weak_transform = transforms.Compose(
    [
        transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(red_blue_mean, red_blue_std),
    ]
)

red_blue_strong_transform = transforms.Compose(
    [
        transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
        RandAugment(3, 5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(red_blue_mean, red_blue_std),
    ]
)

red_blue_fixmatch_transform = TransformFixMatch(
    red_blue_weak_transform, red_blue_strong_transform
)

red_blue_mixmatch_transform = TransformFixMatch(
    red_blue_weak_transform, red_blue_weak_transform
)

red_blue_test_transform = transforms.Compose(
    [
        transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(red_blue_mean, red_blue_std),
    ]
)

red_blue_simclr_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=64, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(red_blue_mean, red_blue_std),
    ]
)

red_blue_splitnet_strong_transform = transforms.Compose(
    [
        transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(red_blue_mean, red_blue_std),
        transforms.RandomErasing(
            p=0.5, scale=(0.125, 0.2), ratio=(0.99, 1.0), value=0, inplace=False,
        ),
    ]
)

red_blue_splitnet_weak_transform = transforms.Compose(
    [
        transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(red_blue_mean, red_blue_std),
    ]
)

red_blue_splitnet_fixmatch_transform = TransformFixMatch(
    red_blue_splitnet_weak_transform, red_blue_splitnet_strong_transform
)

red_blue_splitnet_test_transform = transforms.Compose(
    [
        transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(red_blue_mean, red_blue_std),
    ]
)

# ----------------
# --- Datasets ---
# ----------------


class MiniImagenet(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode, color, noise_rate, transform=None):

        self.root = root_dir
        self.transform = transform
        self.mode = mode

        if self.mode == "train":
            noise_file = "{}_noise_nl_{}".format(color, noise_rate)
            with open(os.path.join(self.root, "split", noise_file)) as f:
                lines = f.readlines()
        elif self.mode == "test":
            with open(os.path.join(self.root, "split", "clean_validation")) as f:
                lines = f.readlines()

        self.imgs = []
        self.targets = []
        for line in lines:
            img, target = line.split()
            target = int(target)
            if self.mode == "train":
                img_path = os.path.join("all_images", img)
            elif self.mode == "test":
                img_path = os.path.join("validation", str(target), img)
            self.imgs.append(img_path)
            self.targets.append(target)

    def get_data(self, index):
        img_path = os.path.join(self.root, self.imgs[index])
        image = Image.open(img_path).convert("RGB")
        return image

    def __getitem__(self, index):
        image = self.get_data(index)
        if self.transform is not None:
            image = self.transform(image)
        target = self.targets[index]
        return image, target

    def __len__(self):
        return len(self.imgs)


def red_blue_class_name(index, noise):
    """Return the name of the class corresponding to 'index'"""
    return str(index)


def red_blue_load_dataset(dataset_type, noise):
    """Load a dataset from the Red Mini Imagenet saved datasets directory"""
    ds_dir = os.path.join(red_blue_dataset_dir, dataset_type, "ds.json")
    ds = DS.load_from_file(ds_dir)
    return ds


# ---------------
# --- Loaders ---
# ---------------


def dataset_from_ds(ds, noise_rate, transform):
    return BasicDataset(
        MiniImagenet(
            red_blue_data_dir, mode="train", color="red", noise_rate=noise_rate
        ),
        ds,
        transform,
    )


def eval_dataset(transform, noise_rate, train=False):
    mode = "train" if train else "test"
    return EvalDataset(
        MiniImagenet(red_blue_data_dir, mode=mode, color="red", noise_rate=noise_rate),
        transform,
    )


def regular_dataset(transform, noise_rate, train=False):
    mode = "train" if train else "test"
    return MiniImagenet(
        red_blue_data_dir,
        mode=mode,
        transform=transform,
        color="red",
        noise_rate=noise_rate,
    )


def red_blue_ssl_datasets(clean_ds, noisy_ds, noise):
    logger.info("Getting Red Mini Imagenet SSL Datasets")
    train_clean_dataset = dataset_from_ds(clean_ds, noise, red_blue_weak_transform)
    train_noisy_dataset = dataset_from_ds(noisy_ds, noise, red_blue_fixmatch_transform)
    test_noisy_dataset = dataset_from_ds(noisy_ds, noise, red_blue_weak_transform)
    test_dataset = regular_dataset(red_blue_test_transform, noise, train=False)
    return train_clean_dataset, train_noisy_dataset, test_noisy_dataset, test_dataset


def red_blue_bootstrapping_datasets(ds, train_aug, noise):

    augs = {
        "strong": red_blue_splitnet_strong_transform,
        "weak": red_blue_splitnet_weak_transform,
        "eval": red_blue_splitnet_test_transform,
    }

    logger.info("Getting Red Mini-Imagenet Bootstrapping Datasets")
    train_dataset = dataset_from_ds(ds, noise, augs[train_aug])
    test_dataset = regular_dataset(red_blue_test_transform, noise, train=False)
    return train_dataset, test_dataset


def red_blue_split_dataset_datasets(ds, eval_aug, noise):

    augs = {
        "strong": red_blue_splitnet_strong_transform,
        "weak": red_blue_splitnet_weak_transform,
        "eval": red_blue_splitnet_test_transform,
    }

    logger.info("Getting Red Mini-Imagenet Clean Set Datasets")
    eval_dataset = dataset_from_ds(ds, noise, augs[eval_aug])
    return eval_dataset


def red_blue_test_aug_datasets(noise):

    logger.info("Getting Red Mini-Imagenet Test Aug Dataset")
    test_dataset = eval_dataset(red_blue_weak_transform, noise, train=False)
    return test_dataset


def red_blue_final_datasets(train_ds, noise):

    logger.info("Getting Red Mini-Imagenet Final Datasets")
    train_dataset = dataset_from_ds(train_ds, noise, red_blue_splitnet_strong_transform)
    test_dataset = regular_dataset(red_blue_splitnet_test_transform, noise, train=False)
    return train_dataset, test_dataset


def red_blue_pretraining_dataset(ds, noise):

    logger.info("Getting Red Mini Imagenet Pretraining Datasets")
    train_dataset = dataset_from_ds(
        ds, noise, TwoCropTransform(red_blue_simclr_transform)
    )
    return train_dataset
