import logging
import os

import torch
from torchvision import transforms
from PIL import Image

from datasets.dataset import EvalDataset

logger = logging.getLogger(__name__)

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

current_dir = os.getcwd()
imagenet_data_dir = os.path.join(current_dir, "data", "ILSVRC2012")

# ---------------------
# --- Augmentations ---
# ---------------------

crop_size = 256
erase_p = 0.5

imagenet_splitnet_weak_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            crop_size,
            scale=(0.1, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, std=[0.229, 0.224, 0.225]),
    ]
)

imagenet_splitnet_test_transform = transforms.Compose(
    [
        transforms.Resize(
            int(crop_size * 1.15), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, std=[0.229, 0.224, 0.225]),
    ]
)

# ----------------
# --- Datasets ---
# ----------------


class ILSVRC2012(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):

        self.root = root_dir
        self.transform = transform

        # Code from Filipe
        self.val_imgs = []  # Filtered images containing the first 50 classes
        self.val_labels = {}
        classes1000 = []
        all_50k_images = []
        for x in os.listdir(self.root):
            if os.path.isfile(os.path.join(self.root, x)):
                all_50k_images.append(x)
        all_50k_images.sort()

        # Because the indices of classes in ILSVRC2012 don't align with the inidces of
        # classes in Webvision, we load a mapping between the two classes
        class_ground_truth_path = os.path.join(
            current_dir,
            "datasets",
            "ilsvrc2012",
            "ILSVRC2012_validation_to_webvision_labels_ground_truth.txt",
        )
        with open(class_ground_truth_path) as f:
            lines = f.readlines()
            for li in lines:
                classes1000.append(int(li))

        print(f"Classes1000: {len(classes1000)}")
        print(f"All 50K Images: {len(all_50k_images)}")
        assert len(classes1000) == len(all_50k_images)

        for id, img_name in enumerate(all_50k_images):
            if int(classes1000[id]) < 50:
                self.val_imgs.append(img_name)
                self.val_labels[img_name] = int(classes1000[id])

    def get_data(self, index):
        img_path = os.path.join(self.root, self.val_imgs[index])
        image = Image.open(img_path).convert("RGB")
        return image

    def __getitem__(self, index):
        image = self.get_data(index)
        if self.transform is not None:
            image = self.transform(image)
        target = self.val_labels[self.val_imgs[index]]
        return image, target

    def __len__(self):
        return len(self.val_imgs)


def imagenet_class_name(index):
    """Return the name of the class corresponding to 'index'"""
    return str(index)


# ---------------
# --- Loaders ---
# ---------------


def eval_dataset(transform, train=False):
    mode = "train" if train else "test"
    return EvalDataset(ILSVRC2012(imagenet_data_dir), transform)


def imagenet_test_aug_datasets():
    logger.info("Getting imagenet Test Aug Dataset")
    test_dataset = eval_dataset(imagenet_splitnet_weak_transform)
    return test_dataset


def imagenet_test_dataset():
    logger.info("Getting imagenet Test Aug Dataset")
    test_dataset = ILSVRC2012(imagenet_data_dir, imagenet_splitnet_test_transform)
    return test_dataset
