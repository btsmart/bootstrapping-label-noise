# Adapted from https://github.com/filipe-research/PropMix/blob/main/data/webvision.py

import logging
import os

import torch
from torchvision import transforms
from PIL import Image

from datasets.autoaugment import ImageNetPolicy
from datasets.dataset import DS, BasicDataset, EvalDataset, TransformFixMatch
from datasets.randaugment import RandAugment
from utils.simclr.util import TwoCropTransform

logger = logging.getLogger(__name__)

webvision_mean = (0.485, 0.456, 0.406)
webvision_std = (0.229, 0.224, 0.225)

current_dir = os.getcwd()
webvision_data_dir = os.path.join(current_dir, "data", "webvision")
webvision_label_dir = os.path.join(current_dir, "datasets", "webvision", "noisy_labels")
webvision_dataset_dir = os.path.join(
    current_dir, "datasets", "webvision", "saved_datasets"
)

# ---------------------
# --- Augmentations ---
# ---------------------

crop_size = 256
erase_p = 0.5

webvision_splitnet_strong_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            crop_size,
            scale=(0.1, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize(webvision_mean, std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(
            p=erase_p, scale=(0.05, 0.12), ratio=(0.5, 1.5), value=0
        ),
    ]
)

webvision_splitnet_weak_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            crop_size,
            scale=(0.1, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(webvision_mean, std=[0.229, 0.224, 0.225]),
    ]
)

webvision_splitnet_test_transform = transforms.Compose(
    [
        transforms.Resize(
            int(crop_size * 1.15), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(webvision_mean, std=[0.229, 0.224, 0.225]),
    ]
)

webvision_weak_transform = transforms.Compose(
    [
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(227, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(webvision_mean, webvision_std),
    ]
)

webvision_test_transform = transforms.Compose(
    [
        transforms.Resize([227, 227]),
        transforms.ToTensor(),
        transforms.Normalize(webvision_mean, webvision_std),
    ]
)

webvision_strong_transform = transforms.Compose(
    [
        transforms.Resize([256, 256]),
        RandAugment(3, 5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(227, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(webvision_mean, webvision_std),
    ]
)

webvision_fixmatch_transform = TransformFixMatch(
    webvision_weak_transform, webvision_strong_transform
)

webvision_splitnet_fixmatch_transform = TransformFixMatch(
    webvision_splitnet_weak_transform, webvision_splitnet_strong_transform
)

webvision_simclr_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=64, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(webvision_mean, webvision_std),
    ]
)

# ----------------
# --- Datasets ---
# ----------------


class Webvision(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode, transform=None, num_classes=50):

        self.root = root_dir
        self.transform = transform
        self.mode = mode
        self.num_classes = num_classes

        if self.mode == "train":
            with open(
                os.path.join(self.root, "info", "train_filelist_google.txt")
            ) as f:
                lines = f.readlines()
        elif self.mode == "test":
            with open(os.path.join(self.root, "info", "val_filelist.txt")) as f:
                lines = f.readlines()
            lines = ["val_images_256/" + line for line in lines]

        self.imgs = []
        self.targets = []
        for line in lines:
            img, target = line.split()
            target = int(target)
            if target < self.num_classes:
                self.imgs.append(img)
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


def webvision_class_name(index):
    """Return the name of the class corresponding to 'index'"""
    return str(index)


def webvision_load_dataset(dataset_type):
    """Load a dataset from the Webvision saved datasets directory"""
    ds_dir = os.path.join(webvision_dataset_dir, dataset_type, "ds.json")
    ds = DS.load_from_file(ds_dir)
    return ds


# ---------------
# --- Loaders ---
# ---------------


def dataset_from_ds(ds, transform):
    return BasicDataset(Webvision(webvision_data_dir, mode="train"), ds, transform)


def eval_dataset(transform, train=False):
    mode = "train" if train else "test"
    return EvalDataset(Webvision(webvision_data_dir, mode=mode), transform,)


def regular_dataset(transform, train=False):
    mode = "train" if train else "test"
    return Webvision(webvision_data_dir, mode=mode, transform=transform)


def webvision_ssl_datasets(clean_ds, noisy_ds):
    logger.info("Getting Webvision SSL Datasets")
    train_clean_dataset = dataset_from_ds(clean_ds, webvision_weak_transform)
    train_noisy_dataset = dataset_from_ds(noisy_ds, webvision_fixmatch_transform)
    test_noisy_dataset = dataset_from_ds(noisy_ds, webvision_weak_transform)
    test_dataset = regular_dataset(train=False, transform=webvision_test_transform)
    return train_clean_dataset, train_noisy_dataset, test_noisy_dataset, test_dataset


def webvision_bootstrapping_datasets(ds, train_aug):

    augs = {
        "strong": webvision_splitnet_strong_transform,
        "weak": webvision_splitnet_weak_transform,
        "eval": webvision_splitnet_test_transform,
    }

    logger.info("Getting Webvision Bootstrapping Datasets")
    train_dataset = dataset_from_ds(ds, augs[train_aug])
    test_dataset = regular_dataset(webvision_splitnet_test_transform, train=False)
    return train_dataset, test_dataset


def webvision_split_dataset_datasets(ds, eval_aug):

    augs = {
        "strong": webvision_splitnet_strong_transform,
        "weak": webvision_splitnet_weak_transform,
        "eval": webvision_splitnet_test_transform,
    }

    logger.info("Getting webvision Clean Set Datasets")
    eval_dataset = dataset_from_ds(ds, augs[eval_aug])
    return eval_dataset


def webvision_test_aug_datasets():
    logger.info("Getting webvision Test Aug Dataset")
    test_dataset = eval_dataset(webvision_splitnet_weak_transform, train=False)
    return test_dataset


def webvision_final_datasets(train_ds):
    logger.info("Getting webvision Final Datasets")
    train_dataset = dataset_from_ds(train_ds, webvision_splitnet_strong_transform)
    test_dataset = regular_dataset(webvision_splitnet_test_transform, train=False)
    return train_dataset, test_dataset


def webvision_pretraining_dataset(ds):
    logger.info("Getting Webvision Pretraining Datasets")
    train_dataset = dataset_from_ds(ds, TwoCropTransform(webvision_simclr_transform))
    return train_dataset
