import json

import numpy as np
import torch

class DS:
    """
    This is a class used to denote a subset of the entire training dataset, and which
    labels we currently believe belong to each sample. For example, this class is used
    to maintain the set of samples which we believe are clean, and what we believe each
    of their labels are.

    Properties:
        - indices: An array of indices referring to which elements of the *entire*
            training set are in this dataset
        - true_labels: The ground truth labels for the samples in the *entire* training
            set (for inspection and testing)
        - learned_labels: What labels we believe every sample has (-1 represents that
            we aren't sure)
        - noisy_labels: An array containing each set of noisy labels (we may have
            multiple sets of noisy labels)

    The true_labels are only used to inspect and test our methods, and don't effect the
    final trained models
    """

    def __init__(self):
        self.indices = np.array([], dtype=np.int64)
        self.true_labels = np.array([], dtype=np.int64)
        self.learned_labels = np.array([], dtype=np.int64)
        self.noisy_label_sets = []
        self.weights = None
        self.noise_distribution = None

    def limit_to_indices(self, indices):
        """Limit the indices, and reset the learned labels"""
        self.indices = indices
        self.learned_labels.fill(-1)

    def save_to_file(self, save_path):
        """Convert each numpy array into a Python list and save to a JSON file"""
        data = {
            "indices": self.indices.tolist(),
            "true_labels": self.true_labels.tolist(),
            "learned_labels": self.learned_labels.tolist(),
            "noisy_label_sets": [labels.tolist() for labels in self.noisy_label_sets],
            "weights": self.weights.tolist() if self.weights is not None else None,
            "noise_distribution": self.noise_distribution.tolist() if self.noise_distribution is not None else None,
        }
        with open(save_path, "w") as f:
            json.dump(data, f, ensure_ascii=True, indent=4)

    @classmethod
    def load_from_file(cls, load_path):
        """Load a previously saved dataset from the saved JSON file"""
        ds = cls()
        with open(load_path, "r") as f:
            data = json.load(f)
        ds.indices = np.array(data["indices"], dtype=np.int64)
        ds.true_labels = np.array(data["true_labels"], dtype=np.int64)
        ds.learned_labels = np.array(data["learned_labels"], dtype=np.int64)
        ds.noisy_label_sets = [
            np.array(labels, dtype=np.float32) for labels in data["noisy_label_sets"]
        ]
        if "weights" in data:
            ds.weights = np.array(data["weights"], dtype=np.float32)
        else:
            ds.weights = None

        if "noise_distribution" in data:
            ds.noise_distribution = np.array(data["noise_distribution"], dtype=np.float32)
        else:
            ds.noise_distribution = None
        return ds


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, ds, transform):
        super(BasicDataset, self).__init__()
        self.dataset = dataset
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds.indices)

    def __getitem__(self, index):
        ds_index = self.ds.indices[index]
        data = self.dataset.get_data(ds_index)
        true_label = self.ds.true_labels[ds_index]
        learned_label = self.ds.learned_labels[ds_index]

        noisy_label = self.ds.noisy_label_sets[0][ds_index]
        if self.ds.noise_distribution is not None:
            noisy_label = np.zeros_like(noisy_label)
            num_classes = noisy_label.shape[0]
            new_label = np.random.choice(num_classes, 1, p=self.ds.noise_distribution[ds_index])
            noisy_label[new_label] = 1

        img = data
        if self.transform is not None:
            img = self.transform(img)

        return index, img, true_label, learned_label, noisy_label


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        super(EvalDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset.__getitem__(index)
        if self.transform is not None:
            img = self.transform(img)
        return index, img, label


class TransformFixMatch(object):
    def __init__(self, weak_transform, strong_transform):
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __call__(self, x):
        return self.weak_transform(x), self.strong_transform(x)
