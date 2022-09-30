import logging
import os

from PIL import Image
from torchvision import datasets, transforms

from datasets.autoaugment import CIFAR10Policy
from datasets.dataset import DS, BasicDataset, EvalDataset, TransformFixMatch
from datasets.randaugment import RandAugment
from utils.simclr.util import TwoCropTransform

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)

cifar10_class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

current_dir = os.getcwd()
cifar10_data_dir = os.path.join(current_dir, "data", "cifar10")
cifar10_label_dir = os.path.join(current_dir, "datasets", "cifar10", "noisy_labels")
cifar10_dataset_dir = os.path.join(current_dir, "datasets", "cifar10", "saved_datasets")

# --------------------
# --- Augmentation ---
# --------------------

cifar10_splitnet_strong_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(
            p=0.5, scale=(0.125, 0.2), ratio=(0.99, 1.0), value=0, inplace=False,
        ),
    ]
)

cifar10_splitnet_test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

cifar10_splitnet_weak_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

cifar10_splitnet_fixmatch_transform = TransformFixMatch(
    cifar10_splitnet_weak_transform, cifar10_splitnet_strong_transform
)

cifar10_weak_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)

cifar10_strong_transform = transforms.Compose(
    [
        RandAugment(3, 5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)

cifar10_fixmatch_transform = TransformFixMatch(
    cifar10_weak_transform, cifar10_strong_transform
)

cifar10_test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
)

cifar10_simclr_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)

# ----------------
# --- Datasets ---
# ----------------


class CIFAR10(datasets.CIFAR10):
    """Simply exposing the standard CIFAR10 dataset from torchvision in the same
    namespace as the new implementation"""

    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)

    def get_data(self, index):
        return Image.fromarray(self.data[index])


def cifar10_class_name(index):
    """Return the name of the class corresponding to 'index'"""
    return cifar10_class_names[index]


def cifar10_load_dataset(dataset_type):
    """Load a dataset from the CIFAR10 saved datasets directory"""
    ds_dir = os.path.join(cifar10_dataset_dir, dataset_type, "ds.json")
    ds = DS.load_from_file(ds_dir)
    return ds


# ---------------
# --- Loaders ---
# ---------------


def dataset_from_ds(ds, transform):
    return BasicDataset(
        CIFAR10(cifar10_data_dir, train=True, download=True), ds, transform
    )


def eval_dataset(transform, train=False):
    return EvalDataset(
        CIFAR10(cifar10_data_dir, train=train, download=True), transform,
    )


def regular_dataset(transform, train=False):
    return CIFAR10(cifar10_data_dir, train=train, transform=transform, download=True)


def cifar10_ssl_datasets(clean_ds, noisy_ds):
    logger.info("Getting CIFAR10 SSL Datasets")
    train_clean_dataset = dataset_from_ds(clean_ds, cifar10_weak_transform)
    train_noisy_dataset = dataset_from_ds(noisy_ds, cifar10_fixmatch_transform)
    test_noisy_dataset = dataset_from_ds(noisy_ds, cifar10_weak_transform)
    test_dataset = regular_dataset(train=False, transform=cifar10_test_transform)
    return train_clean_dataset, train_noisy_dataset, test_noisy_dataset, test_dataset


def cifar10_bootstrapping_datasets(ds, train_aug):

    augs = {
        "strong": cifar10_strong_transform,
        "weak": cifar10_weak_transform,
        "eval": cifar10_test_transform,
    }

    logger.info("Getting CIFAR10 Bootstrapping Datasets")
    train_dataset = dataset_from_ds(ds, augs[train_aug])
    test_dataset = regular_dataset(cifar10_test_transform, train=False)
    return train_dataset, test_dataset


def cifar10_split_dataset_datasets(ds, eval_aug):

    augs = {
        "strong": cifar10_strong_transform,
        "weak": cifar10_weak_transform,
        "eval": cifar10_test_transform,
    }

    logger.info("Getting CIFAR10 Split Dataset Datasets")
    eval_dataset = dataset_from_ds(ds, augs[eval_aug])
    return eval_dataset


def cifar10_test_aug_datasets():
    logger.info("Getting CIFAR10 Test Aug Dataset")
    test_dataset = eval_dataset(cifar10_splitnet_weak_transform, train=False)
    return test_dataset


def cifar10_final_datasets(train_ds):
    logger.info("Getting CIFAR10 Final Datasets")
    train_dataset = dataset_from_ds(train_ds, cifar10_splitnet_strong_transform)
    test_dataset = regular_dataset(cifar10_splitnet_test_transform, train=False)
    return train_dataset, test_dataset


def cifar10_pretraining_dataset(ds):
    logger.info("Getting CIFAR10 Pretraining Datasets")
    train_dataset = dataset_from_ds(ds, TwoCropTransform(cifar10_simclr_transform))
    return train_dataset


def cifar10_test_dataset():
    logger.info("Getting CIFAR10 Test Dataset")
    test_dataset = eval_dataset(cifar10_splitnet_test_transform, train=False)
    return test_dataset
