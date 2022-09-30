from scipy.special import softmax
from tqdm import tqdm
import numpy as np
import torch

import utils.utils as utils

def enable_dropout(model):
    """Enable the dropout layers even when the model is in eval mode"""
    
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def get_logits(model, data_loader, num_samples, output_size, device, label_iterations=12, encodings=False, use_noisy_labels=True, null_label_type="zeros"):
    """Get the logit predictions from multiple augmentations of each image. This one
    loads data in the form (idx, image, _, _, noisy_labels)."""

    logits = np.full((num_samples, label_iterations, output_size), -1, dtype=np.float32)

    model.eval()
    enable_dropout(model)

    with torch.no_grad():

        for label_itr in range(label_iterations):

            data_iterator = tqdm(data_loader, ascii=True, ncols=88)
            for idxs, images, _, _, noisy_labels in data_iterator:

                # Cast to device
                images = images.to(device)
                noisy_labels = noisy_labels.to(device)

                if not use_noisy_labels:
                    noisy_labels = utils.null_labels_like(noisy_labels, null_label_type)

                # Get predictions from model
                if encodings:
                    outputs = model.encoder(images).detach().cpu().numpy()
                else:
                    outputs = model(images, noisy_labels).detach().cpu().numpy()
                logits[idxs, label_itr] = outputs

    return logits



def get_logits_labels_eval(model, data_loader, num_samples, output_size, device, label_iterations=12, null_label_type="zeros", generate_noisy_labels=False):
    """Get the logit predictions from multiple augmentations of each image. This one
    loads data in the form (idx, image, label)."""

    logits = np.full((num_samples, label_iterations, output_size), -1, dtype=np.float32)
    all_labels = np.full(num_samples, -1, dtype=np.int32)

    model.eval()

    if label_iterations != 1:
        enable_dropout(model)

    with torch.no_grad():

        for label_itr in range(label_iterations):

            data_iterator = tqdm(data_loader, ascii=True, ncols=88)
            for idxs, images, labels in data_iterator:

                # Cast to device
                images = images.to(device)
                labels = labels.detach().cpu().numpy()
                noisy_labels = utils.null_labels((images.size(0), output_size), null_label_type).to(device)

                if generate_noisy_labels:
                    import random
                    transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}
                    for i, x in enumerate(labels):
                        if random.random() <= 0.4:
                            noisy_labels[i][transition[x]] = 1
                        else:
                            noisy_labels[i][x] = 1

                # Get predictions from model
                outputs = model(images, noisy_labels).detach().cpu().numpy()
                logits[idxs, label_itr] = outputs
                all_labels[idxs] = labels

    return logits, all_labels


def get_logits_labels_combined(model_a, model_b, data_loader, num_samples, output_size, device, label_iterations=12, null_label_type="zeros"):
    """Get the logit predictions from multiple augmentations of each image. This one
    loads data in the form (idx, image, label). It also uses a second model to generate
    plausible noisy labels to feed into the first model"""

    logits = np.full((num_samples, label_iterations, output_size), -1, dtype=np.float32)
    all_labels = np.full(num_samples, -1, dtype=np.int32)

    model_a.eval()
    model_b.eval()
    enable_dropout(model_a)
    enable_dropout(model_b)

    with torch.no_grad():

        for label_itr in range(label_iterations):

            data_iterator = tqdm(data_loader, ascii=True, ncols=88)
            for idxs, images, labels in data_iterator:

                # Cast to device
                images = images.to(device)
                labels = labels.detach().cpu().numpy()
                noisy_labels = utils.null_labels((images.size(0), output_size), null_label_type).to(device)
                
                # Get predictions from model
                outputs = model_a(images, noisy_labels)
                predictions = torch.argmax(outputs, dim=1)
                noisy_labels = torch.nn.functional.one_hot(predictions, output_size).float()
                outputs = model_b(images, noisy_labels).detach().cpu().numpy()
                logits[idxs, label_itr] = outputs
                all_labels[idxs] = labels

    return logits, all_labels


def get_predictions(logits, use_softmax=True):
    """Get statistical information and predictions from the lists of logits"""

    if use_softmax:
        logits = softmax(logits, axis=2)

    means = np.mean(logits, axis=1)
    stds = np.std(logits, axis=1)
    predictions = np.argmax(means, axis=1)
    prediction_means = means[range(predictions.shape[0]), predictions]
    prediction_stds = stds[range(predictions.shape[0]), predictions]

    return predictions, prediction_means, prediction_stds
