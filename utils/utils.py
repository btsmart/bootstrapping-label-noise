import copy
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from utils.log_dataset_info import log_dataset

logger = logging.getLogger(__name__)


def set_seed(seed):
    """Seed Python's, Numpy's and PyTorch's random number generation"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(0)


def set_device(config):
    """Set the visible CUDA devices and create the device to be used for training"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpus)
    logger.info(f"Using CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def get_random_splits(samples, section_sizes):
    """Randomly split the elements in 'samples' in multiple lists"""
    idxs = np.random.permutation(samples)
    out = []
    pos = 0
    for section_size in section_sizes:
        out.append(idxs[pos : pos + section_size])
        pos += section_size
    return out


def count_parameters(model):
    """Return the total number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def class_distribution(labels, num_classes):
    """Calculate the distribution of samples across classes given the sample labels"""
    class_dist = np.zeros(num_classes, dtype=np.float32)
    for i in range(labels.shape[0]):
        class_dist[labels[i]] += 1
    class_dist /= np.sum(class_dist)
    return class_dist


def noise_matrix(labels_a, labels_b, num_classes):
    """Calculate the distribution of samples across noise transitions"""
    label_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    for i in range(labels_a.shape[0]):
        label_matrix[labels_a[i], labels_b[i]] += 1
    print(f"Label Matrix before normalization:\n{label_matrix.astype(np.int32)}")
    label_matrix /= np.sum(label_matrix)
    return label_matrix


def one_hot(values, classes):
    """Convert values into one-hot numpy vectors"""
    out = np.zeros((values.size, classes))
    out[np.arange(values.size), values] = 1
    return out


def load_labels(file_path):
    """Load a series of labels from a file"""
    return np.load(file_path).astype(np.float32)


# -----------------------
# --- Data / Datasets ---
# -----------------------


def save_datasets(clean_ds, noisy_ds, save_dir):
    """Save and log the clean and noisy datasets"""

    logger.info(f"Logging and saving the current clean/noisy datasets to {save_dir}")
    log_dataset(clean_ds, os.path.join(save_dir, "clean_ds"))
    log_dataset(noisy_ds, os.path.join(save_dir, "noisy_ds"))


def split_dataset(ds, clean_size):
    """Randomly the dataset into a clean set and a remaining noisy dataset"""

    noisy_size = ds.indices.shape[0] - clean_size
    clean_indices, noisy_indices = get_random_splits(
        ds.indices, [clean_size, noisy_size]
    )

    clean_ds = copy.deepcopy(ds)
    clean_ds.limit_to_indices(clean_indices)
    clean_ds.learned_labels[clean_ds.indices] = clean_ds.true_labels[clean_ds.indices]

    noisy_ds = copy.deepcopy(ds)
    noisy_ds.limit_to_indices(noisy_indices)

    return clean_ds, noisy_ds


# -------------------
# --- Checkpoints ---
# -------------------


def save_checkpoint(state, save_file):
    """Save the model, creating the appropriate directories if they don't exist"""

    pardir = os.path.abspath(os.path.join(save_file, os.pardir))
    if not os.path.exists(pardir):
        os.makedirs(pardir)
    torch.save(state, save_file)


# -------------------
# --- Null Labels ---
# -------------------


def null_labels_like(n_labels, type):
    """Convert a set of noisy labels into a set of null labels"""
    if type == "zeros":
        return torch.zeros_like(n_labels)
    elif type == "ones":
        return torch.ones_like(n_labels)
    elif type == "average":
        return torch.full_like(n_labels, 1.0 / n_labels.size(-1))


def null_labels(shape, type):
    """Create a set of null labels with the given shape"""
    if type == "zeros":
        return torch.zeros(shape)
    elif type == "ones":
        return torch.ones(shape)
    elif type == "average":
        return torch.full(shape, 1.0 / shape[1])


def set_rows_to_null(noisy_labels, null_type, nl_change_frac=0.5):
    """Convert `nl_change_frac` of the noisy labels to null labels"""
    new_labels = noisy_labels.clone().detach()
    batch_size = new_labels.size(0)
    random_vals = torch.rand(batch_size)
    for i, val in enumerate(random_vals):
        if val < nl_change_frac:
            new_labels[i] = null_labels_like(new_labels[i], null_type)
    return new_labels


# ---------------
# --- FixMatch ---
# ----------------

# Adapted from https://github.com/LeeDoYup/FixMatch-pytorch/blob/main/models/fixmatch/fixmatch_utils.py
def consistency_loss(logits_s, logits_w, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        select = max_probs.ge(p_cutoff).long()
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), select, max_idx.long()

    else:
        assert Exception('Not Implemented consistency_loss')
            
# Adapted from https://github.com/LeeDoYup/FixMatch-pytorch/blob/main/train_utils.py
def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss

# Adapted from https://fyubang.com/2019/06/01/ema/
class EMA:

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
