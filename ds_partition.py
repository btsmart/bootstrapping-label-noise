import copy
import logging
import os

import math
import numpy as np

import main
from utils.evaluate_selection import evaluate_selection
from utils.pseudo_labelling import get_logits, get_predictions
import utils.plotter as plotter
import utils.utils as utils
from datasets.data import get_split_dataset_datasets

logger = logging.getLogger(__name__)


# ------------------
# --- Old Method ---
# ------------------

def get_new_datasets(ds, selections, predictions, weights=None, noise_distribution=None):
    """Split the dataset object into clean/noisy datasets based on the selection"""

    # Create the new clean dataset
    clean_indices = selections
    clean_ds = copy.deepcopy(ds)
    clean_ds.limit_to_indices(clean_indices)
    clean_ds.learned_labels[clean_ds.indices] = predictions[clean_ds.indices]
    if weights is not None:
        clean_ds.weights = weights
    if noise_distribution is not None:
        clean_ds.noise_distribution = noise_distribution

    # Create the new noisy dataset
    noisy_indices = np.setdiff1d(ds.indices, selections)
    noisy_ds = copy.deepcopy(ds)
    noisy_ds.limit_to_indices(noisy_indices)

    return clean_ds, noisy_ds


def get_unbalanced_selection(config, predictions, scores, num_to_select, log=True):
    """Select whichever samples have the lowest scores"""
    selection = np.argsort(scores)[:num_to_select]
    if log:
        threshold = np.min(scores[selection]) if len(selection) != 0 else 0.0
        logger.info(f"Threshold used for selection: {threshold}")
    return selection


def get_class_balanced_selection(config, predictions, scores, num_to_select, log=True):
    """Select the same number of samples from each class"""

    adjusted_scores = config.data.total_samples + np.argsort(scores).astype(
        dtype=np.float32
    )

    for class_a in range(config.data.num_classes):
        selection = np.argwhere(predictions == class_a).ravel()
        selection = selection[np.argsort(scores[selection])]

        adjusted_scores[selection] = np.minimum(
            adjusted_scores[selection],
            config.data.num_classes * np.arange(1, len(selection) + 1),
        )

    selection = np.argsort(adjusted_scores)[:num_to_select]

    if log:
        thresholds = np.ones(config.data.num_classes)
        for class_a in range(config.data.num_classes):
            class_selection = np.intersect1d(
                np.argwhere(predictions == class_a).ravel(), selection
            )
            thresholds[class_a] = np.min(class_selection, initial=1.0)
        logger.info(f"Threshold used for classes: {thresholds}")

    return selection


def get_noise_matrix_balanced_selection(
    config, ds, predictions, scores, noise_matrix, num_to_select, log=True
):
    """Select a proportional number of samples from each noise transition, using the
    provided noise matrix to decide proportionality"""

    noisy_labels = np.argmax(ds.noisy_label_sets[0], axis=1)[ds.indices]

    adjusted_scores = config.data.total_samples + np.argsort(scores)

    for class_a in range(config.data.num_classes):
        for class_b in range(config.data.num_classes):

            selection = np.intersect1d(
                np.argwhere(predictions == class_a),
                np.argwhere(noisy_labels == class_b),
            )
            selection = selection[np.argsort(scores[selection])]
            adjusted_scores[selection] = np.minimum(
                adjusted_scores[selection],
                (1.0 / max(1e-7, noise_matrix[class_a, class_b]))
                * np.arange(1, len(selection) + 1),
            )

    selection = np.argsort(adjusted_scores)[:num_to_select]

    if log:
        thresholds = np.ones((config.data.num_classes, config.data.num_classes))
        for class_a in range(config.data.num_classes):
            for class_b in range(config.data.num_classes):
                noise_selection = np.intersect1d(
                    np.intersect1d(
                        np.argwhere(predictions == class_a),
                        np.argwhere(noisy_labels == class_b),
                    ),
                    selection,
                )
                if len(noise_selection) > 0:
                    thresholds[class_a][class_b] = np.min(
                        scores[noise_selection], initial=1.0
                    )
        logger.info(
            f"Threshold used for noise transitions:\n"
            f"{np.array_str(thresholds, precision=4, suppress_small=True)}"
        )

    return selection




def select_samples(config, model, ds, weak_loader, device, use_noisy_labels=True, null_label_type="zeros"):

    logits = get_logits(
        model,
        weak_loader,
        ds.indices.shape[0],
        config.data.num_classes,
        device,
        config.ds_partition.guessing_label_iterations,
        use_noisy_labels=use_noisy_labels,
        null_label_type=null_label_type
    )
    predictions, confidences, uncertainties = get_predictions(logits)

    print(config.ds_partition.balancing_mode)

    if config.ds_partition.balancing_mode == "unbalanced":
        selected_indices = get_unbalanced_selection(
            config, predictions, -1.0 * confidences, config.ds_partition.samples_to_select
        )
        weights = None
        noise_distribution = None

    elif config.ds_partition.balancing_mode == "class_balanced":
        selected_indices = get_class_balanced_selection(
            config, predictions, -1.0 * confidences, config.ds_partition.samples_to_select
        )
        weights = None
        noise_distribution = None

    elif config.ds_partition.balancing_mode == "noise_balanced":

        # NOTE: The commented code here uses the true labels of the samples for testing
        true_noise_matrix = utils.noise_matrix(
            ds.true_labels[ds.indices],
            np.argmax(ds.noisy_label_sets[0], axis=1)[ds.indices],
            config.data.num_classes,
        )
        logger.info(f"True Noise Matrix:\n{true_noise_matrix}")
        for sample_frac in [0.1, 0.8, 0.9, 1.0]:
            # Estimate the noise matrix
            cb_selected_indices = get_class_balanced_selection(
                config,
                predictions,
                -1.0 * confidences,
                int(sample_frac * config.data.total_samples),
                log=False,
            )
            est_noise_matrix = utils.noise_matrix(
                predictions[cb_selected_indices],
                np.argmax(ds.noisy_label_sets[0], axis=1)[cb_selected_indices],
                config.data.num_classes,
            )
            logger.info(
                "\n" + np.array_str(est_noise_matrix, precision=4, suppress_small=True)
            )
            logger.info(
                "\n" + np.array_str(est_noise_matrix, precision=2, suppress_small=True)
            )
            logger.info(
                f"Est. Noise: {np.sum(est_noise_matrix) - np.trace(est_noise_matrix)}"
            )
            logger.info(
                f"L1 Loss: {np.sum(np.abs(true_noise_matrix - est_noise_matrix))}"
            )
            logger.info(f"")

        # Estimate the noise matrix
        cb_selected_indices = get_class_balanced_selection(
            config,
            predictions,
            -1.0 * confidences,
            int(config.ds_partition.samples_for_noise_estimation * config.data.total_samples),
            log=False,
        )
        est_noise_matrix = utils.noise_matrix(
            predictions[cb_selected_indices],
            np.argmax(ds.noisy_label_sets[0], axis=1)[cb_selected_indices],
            config.data.num_classes,
        )
        logger.info(
            "\n" + np.array_str(est_noise_matrix, precision=4, suppress_small=True)
        )
        logger.info(
            "\n" + np.array_str(est_noise_matrix, precision=2, suppress_small=True)
        )
        logger.info(
            f"Est. Noise: {np.sum(est_noise_matrix) - np.trace(est_noise_matrix)}"
        )
        selected_indices = get_noise_matrix_balanced_selection(
            config,
            ds,
            predictions,
            -1.0 * confidences,
            est_noise_matrix,
            config.ds_partition.samples_to_select,
        )
        weights = None
        noise_distribution = None

    elif config.ds_partition.balancing_mode == "noise_generation":

        # Estimate the noise matrix
        cb_selected_indices = get_class_balanced_selection(
            config,
            predictions,
            -1.0 * confidences,
            int(config.ds_partition.samples_for_noise_estimation * config.data.total_samples),
            log=False,
        )
        est_noise_matrix = utils.noise_matrix(
            predictions[cb_selected_indices],
            np.argmax(ds.noisy_label_sets[0], axis=1)[cb_selected_indices],
            config.data.num_classes,
        )
        logger.info(
            "\n" + np.array_str(est_noise_matrix, precision=4, suppress_small=True)
        )
        logger.info(
            "\n" + np.array_str(est_noise_matrix, precision=2, suppress_small=True)
        )
        logger.info(
            f"Est. Noise: {np.sum(est_noise_matrix) - np.trace(est_noise_matrix)}"
        )
        selected_indices = get_class_balanced_selection(
            config,
            predictions,
            -1.0 * confidences,
            config.ds_partition.samples_to_select,
            log=False,
        )
        weights = None
        noise_distribution = np.zeros_like(ds.noisy_label_sets[0])
        for i in selected_indices:
            noise_dist = est_noise_matrix[predictions[i]]
            noise_distribution[i] = noise_dist / np.sum(noise_dist)
            print(f"Prediction: {predictions[i]}")
            print(noise_distribution[i])


    plotter.prediction_plots(
        ds,
        predictions,
        confidences,
        uncertainties,
        logits,
        os.path.join(config.save_dir, "plots"),
    )

    evaluate_selection(
        config,
        ds,
        selected_indices,
        predictions,
        confidences,
        uncertainties,
        os.path.join(config.save_dir, "saved_results.md"),
    )

    clean_ds, noisy_ds = get_new_datasets(ds, selected_indices, predictions, weights, noise_distribution)
    return clean_ds, noisy_ds




def estimate_noise_matrix(config, ds, predictions, confidences, log=True):
    """Estimate the noise matrix for the dataset using the most confident predictions"""

    # Get the most confident samples from each class
    cb_selected_indices = get_class_balanced_selection(
        config,
        predictions,
        -1.0 * confidences,
        int(
            config.ds_partition.samples_for_noise_estimation
            * config.data.total_samples
        ),
        log=False,
    )

    # Produce the noise matrix from these samples noisy labels and predictions
    est_noise_matrix = utils.noise_matrix(
        predictions[cb_selected_indices],
        np.argmax(ds.noisy_label_sets[0], axis=1)[cb_selected_indices],
        config.data.num_classes,
    )

    if log:
        logger.info(
            "\n" + np.array_str(est_noise_matrix, precision=4, suppress_small=True)
        )
        logger.info(
            "\n" + np.array_str(est_noise_matrix, precision=2, suppress_small=True)
        )
        logger.info(
            f"Est. Noise: {np.sum(est_noise_matrix) - np.trace(est_noise_matrix)}"
        )

    return est_noise_matrix


def split_dataset_2(config, ds, model, forced_frac, confidence_threshold, device, tb_log, save_dir, use_noisy_labels=True, should_plot=True):

    from utils.pseudo_labelling import get_logits, get_predictions

    eval_dataset = get_split_dataset_datasets(
        config.data.dataset, ds, config.bootstrapping.eval_aug
    )
    eval_loader = main.get_split_dataset_loader(config, eval_dataset)

    logits = get_logits(
        model,
        eval_loader,
        ds.indices.shape[0],
        config.data.num_classes,
        device,
        config.ds_partition.guessing_label_iterations,
        use_noisy_labels=use_noisy_labels,
        null_label_type=config.null_label_type,
    )
    predictions, confidences, uncertainties = get_predictions(logits)

    selection = np.argwhere(
        confidences > confidence_threshold
    ).flatten()

    if config.ds_partition.balancing_mode == "unweighted":
        logger.warning("Not using weights")
        weights = None
        noise_distribution = None

    elif config.ds_partition.balancing_mode == "class_weighted" or config.ds_partition.balancing_mode == "noise_generation":

        logger.warning("Using class weighting")
        class_weights = np.full(config.data.num_classes, -1.0, dtype=np.float32)
        samples_per_class = np.zeros(config.data.num_classes, dtype=np.int32)
        weights = np.zeros(ds.noisy_label_sets[0].shape[0], dtype=np.float32)

        target_rate = 1.0 / config.data.num_classes
        num_to_select = math.ceil(
            forced_frac * target_rate * config.data.total_samples
        )

        # Add in at least x% of the samples from each class
        for class_a in range(config.data.num_classes):
            c_samples = np.argwhere(predictions == class_a).flatten()
            ordered_c_samples = c_samples[np.argsort(-1.0 * confidences[c_samples])]
            class_selection = ordered_c_samples[:num_to_select]
            selection = np.union1d(selection, class_selection)

        for class_a in range(config.data.num_classes):

            c_samples = selection[np.argwhere(predictions[selection] == class_a)]
            samples_per_class[class_a] = len(c_samples)

            # Calculate the weight of each sample with this transition
            if len(c_samples) != 0:
                weight = len(selection) * target_rate / len(c_samples)
                logger.warning(
                    f"Prediction is: {class_a}, Selected Samples: {len(c_samples)}, Target Rate: {target_rate}, Weight: {weight}"
                )
                class_weights[class_a] = weight
                for i in c_samples:
                    weights[i] = weight
            else:
                logger.warning(
                    f"Prediction is: {class_a}, No Selected Samples, Target Rate: {target_rate}"
                )
        
        noise_distribution = None
        if config.ds_partition.balancing_mode == "noise_generation":
            est_noise_matrix = estimate_noise_matrix(config, ds, predictions, confidences)
            noise_distribution = np.zeros_like(ds.noisy_label_sets[0])
            for i in selection:
                noise_dist = est_noise_matrix[predictions[i]]
                noise_distribution[i] = noise_dist / np.sum(noise_dist)
                print(f"Prediction: {predictions[i]}")
                print(noise_distribution[i])


    elif config.ds_partition.balancing_mode == "noise_weighted":

        logger.warning("Using noise weighting")
        est_noise_matrix = estimate_noise_matrix(config, ds, predictions, confidences)

        weight_matrix = np.full_like(est_noise_matrix, -1.0, dtype=np.float32)
        weights = np.zeros(ds.noisy_label_sets[0].shape[0], dtype=np.float32)
        noisy_labels = np.argmax(ds.noisy_label_sets[0], axis=1)[ds.indices]
        samples_per_noise = np.zeros_like(est_noise_matrix, dtype=np.int32)


        # Add in at least 5% of the samples from each noise transition
        for class_a in range(config.data.num_classes):
            for class_b in range(config.data.num_classes):
                # print(f"Class A: {class_a}, Class B: {class_b}")
                noise_samples = np.intersect1d(
                    np.argwhere(predictions == class_a),
                    np.argwhere(noisy_labels == class_b),
                ).flatten()
                # print(noise_samples)
                # print(noise_samples.shape)
                target_rate = est_noise_matrix[class_a, class_b]
                ordered_noise_selection = noise_samples[np.argsort(-1.0 * confidences[noise_samples])]
                num_to_select = math.floor(forced_frac * target_rate * config.data.total_samples)
                # print(f"{forced_frac} * {target_rate} * {config.data.total_samples} = {num_to_select}")
                # print(len(noise_samples))
                noise_selection = ordered_noise_selection[:num_to_select]
                # print(len(noise_selection))
                selection = np.union1d(selection, noise_selection)
                # print(len(selection))

        for class_a in range(config.data.num_classes):
            for class_b in range(config.data.num_classes):

                # Calculate the number of samples we have selected with this transition
                noise_selection = selection[
                    np.intersect1d(
                        np.argwhere(predictions[selection] == class_a),
                        np.argwhere(noisy_labels[selection] == class_b),
                    )
                ]
                samples_per_noise[class_a, class_b] = len(noise_selection)

                # Fetch our target transition rate
                target_rate = est_noise_matrix[class_a, class_b]

                # Calculate the weight of each sample with this transition
                if len(noise_selection) != 0:
                    weight = len(selection) * target_rate / len(noise_selection)
                    logger.warning(
                        f"Prediction is: {class_a}, Noisy Label is: {class_b}"
                    )
                    logger.warning(
                        f"Selected Samples: {len(noise_selection)}, Target Rate: {target_rate}, Weight: {weight}"
                    )
                    # weight = min(weight, config.ds_partition.weight_cap)
                    weight_matrix[class_a, class_b] = weight
                    for i in noise_selection:
                        weights[i] = weight

                else:
                    logger.warning(
                        f"Prediction is: {class_a}, Noisy Label is: {class_b}"
                    )
                    logger.warning(
                        f"No Selected Samples where Target Rate is: {target_rate}"
                    )

        logger.warning(f"Weight Matrix:\n{np.array_str(weight_matrix, precision=3, suppress_small=True)}")
        logger.warning(f"Weights: {weights}")
        logger.warning(f"Weights Sum: {np.sum(weights)}")
        logger.warning(f"Samples Per Noise:\n{samples_per_noise}")

        noise_distribution = None


    # logger.warning(selection)
    logger.warning(f"Selection Shape: {selection.shape}")

    if should_plot:

        plotter.prediction_plots(
            ds,
            predictions,
            confidences,
            uncertainties,
            logits,
            os.path.join(save_dir, "plots"),
        )

        evaluate_selection(
            config,
            ds,
            selection,
            predictions,
            confidences,
            uncertainties,
            os.path.join(save_dir, "saved_results.md"),
        )

    if False:
        plotter.label_variation_test(
            config, model, eval_loader, device, save_dir
        )

    clean_ds, noisy_ds = get_new_datasets(ds, selection, predictions, weights, noise_distribution)
    return clean_ds, noisy_ds