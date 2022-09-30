import logging
import os

from PIL import Image
from torchvision import datasets, transforms

from datasets.autoaugment import CIFAR10Policy
from datasets.dataset import DS, BasicDataset, EvalDataset, TransformFixMatch
from datasets.randaugment import RandAugment
from utils.simclr.util import TwoCropTransform


logger = logging.getLogger(__name__)

cifar100_mean = (0.4914, 0.4822, 0.4465)
cifar100_std = (0.2470, 0.2435, 0.2616)

cifar100_class_names = [
    "beaver",
    "dolphin",
    "otter",
    "seal",
    "whale",
    "aquarium fish",
    "flatfish",
    "ray",
    "shark",
    "trout",
    "orchids",
    "poppies",
    "roses",
    "sunflowers",
    "tulips",
    "bottles",
    "bowls",
    "cans",
    "cups",
    "plates",
    "apples",
    "mushrooms",
    "oranges",
    "pears",
    "sweet peppers",
    "clock",
    "computer keyboard",
    "lamp",
    "telephone",
    "television",
    "bed",
    "chair",
    "couch",
    "table",
    "wardrobe",
    "bee",
    "beetle",
    "butterfly",
    "caterpillar",
    "cockroach",
    "bear",
    "leopard",
    "lion",
    "tiger",
    "wolf",
    "bridge",
    "castle",
    "house",
    "road",
    "skyscraper",
    "cloud",
    "forest",
    "mountain",
    "plain",
    "sea",
    "camel",
    "cattle",
    "chimpanzee",
    "elephant",
    "kangaroo",
    "fox",
    "porcupine",
    "possum",
    "raccoon",
    "skunk",
    "crab",
    "lobster",
    "snail",
    "spider",
    "worm",
    "baby",
    "boy",
    "girl",
    "man",
    "woman",
    "crocodile",
    "dinosaur",
    "lizard",
    "snake",
    "turtle",
    "hamster",
    "mouse",
    "rabbit",
    "shrew",
    "squirrel",
    "maple",
    "oak",
    "palm",
    "pine",
    "willow",
    "bicycle",
    "bus",
    "motorcycle",
    "pickup truck",
    "train",
    "lawn-mower",
    "rocket",
    "streetcar",
    "tank",
    "tractor",
]


current_dir = os.getcwd()
cifar100_data_dir = os.path.join(current_dir, "data", "cifar100")
cifar100_label_dir = os.path.join(current_dir, "datasets", "cifar100", "noisy_labels")
cifar100_dataset_dir = os.path.join(
    current_dir, "datasets", "cifar100", "saved_datasets"
)

# --------------------
# --- Augmentation ---
# --------------------

cifar100_weak_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ]
)

cifar100_strong_transform = transforms.Compose(
    [
        RandAugment(3, 5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ]
)

cifar100_fixmatch_transform = TransformFixMatch(
    cifar100_weak_transform, cifar100_strong_transform
)


cifar100_test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(cifar100_mean, cifar100_std)]
)

cifar100_simclr_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ]
)


cifar100_splitnet_weak_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
        transforms.RandomErasing(
            p=0.5, scale=(0.0625, 0.1), ratio=(0.99, 1.0), value=0, inplace=False,
        ),
    ]
)

cifar100_splitnet_strong_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
        transforms.RandomErasing(
            p=0.5, scale=(0.0625, 0.1), ratio=(0.99, 1.0), value=0, inplace=False,
        ),
    ]
)

cifar100_splitnet_test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
    ]
)

cifar100_splitnet_fixmatch_transform = TransformFixMatch(
    cifar100_splitnet_weak_transform, cifar100_splitnet_strong_transform
)

# ----------------
# --- Datasets ---
# ----------------


class CIFAR100(datasets.CIFAR100):
    """Simply exposing the standard CIFAR100 dataset from torchvision in the same
    namespace as the new implementation"""

    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)

    def get_data(self, index):
        return Image.fromarray(self.data[index])


def cifar100_class_name(index):
    """Return the name of the class corresponding to 'index'"""
    return cifar100_class_names[index]


def cifar100_load_dataset(dataset_type):
    """Load a dataset from the CIFAR100 saved datasets directory"""
    ds_dir = os.path.join(cifar100_dataset_dir, dataset_type, "ds.json")
    ds = DS.load_from_file(ds_dir)
    return ds


# ---------------
# --- Loaders ---
# ---------------


def dataset_from_ds(ds, transform):
    return BasicDataset(
        CIFAR100(cifar100_data_dir, train=True, download=True), ds, transform
    )


def eval_dataset(transform, train=False):
    return EvalDataset(
        CIFAR100(cifar100_data_dir, train=train, download=True), transform,
    )


def regular_dataset(transform, train=False):
    return CIFAR100(cifar100_data_dir, train=train, transform=transform, download=True)


def cifar100_ssl_datasets(clean_ds, noisy_ds):
    logger.info("Getting CIFAR100 SSL Datasets")
    train_clean_dataset = dataset_from_ds(clean_ds, cifar100_weak_transform)
    train_noisy_dataset = dataset_from_ds(noisy_ds, cifar100_fixmatch_transform)
    test_noisy_dataset = dataset_from_ds(noisy_ds, cifar100_weak_transform)
    test_dataset = regular_dataset(train=False, transform=cifar100_test_transform)
    return train_clean_dataset, train_noisy_dataset, test_noisy_dataset, test_dataset


def cifar100_bootstrapping_datasets(ds, train_aug):

    augs = {
        "strong": cifar100_strong_transform,
        "weak": cifar100_weak_transform,
        "eval": cifar100_test_transform,
    }

    logger.info("Getting CIFAR100 Clean Set Datasets")
    train_dataset = dataset_from_ds(ds, augs[train_aug])
    test_dataset = regular_dataset(cifar100_test_transform, train=False)
    return train_dataset, test_dataset


def cifar100_split_dataset_datasets(ds, eval_aug):

    augs = {
        "strong": cifar100_strong_transform,
        "weak": cifar100_weak_transform,
        "eval": cifar100_test_transform,
    }

    logger.info("Getting CIFAR100 Clean Set Datasets")
    eval_dataset = dataset_from_ds(ds, augs[eval_aug])
    return eval_dataset


def cifar100_test_aug_datasets():
    logger.info("Getting CIFAR100 Test Aug Dataset")
    test_dataset = eval_dataset(cifar100_weak_transform, train=False)
    return test_dataset


def cifar100_final_datasets(train_ds):
    logger.info("Getting CIFAR100 Final Datasets")
    train_dataset = dataset_from_ds(train_ds, cifar100_splitnet_strong_transform)
    test_dataset = regular_dataset(cifar100_splitnet_test_transform, train=False)
    return train_dataset, test_dataset


def cifar100_pretraining_dataset(ds):
    logger.info("Getting CIFAR100 Pretraining Datasets")
    train_dataset = dataset_from_ds(ds, TwoCropTransform(cifar100_simclr_transform))
    return train_dataset
