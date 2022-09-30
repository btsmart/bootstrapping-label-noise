import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from scipy.special import softmax

from datasets.data import get_class_name

# ---------------------------
# --- Train Logging Plots ---
# ---------------------------


def train_log_plots(train_logs, test_logs, save_dir):
    """Create plots to show the training/testing loss/accuracy"""

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    loss_plot(train_logs, test_logs, os.path.join(save_dir, "loss.png"))
    accuracy_plot(train_logs, test_logs, os.path.join(save_dir, "accuracy.png"))


def loss_plot(train_logs, test_logs, save_path):
    """Create a plot from the training/testing loss, highlighting the best test loss"""

    clean_epochs = [log["epoch"] for log in train_logs]
    clean_losses = [log["loss"] for log in train_logs]
    test_epochs = [log["epoch"] for log in test_logs]
    test_losses = [log["loss"] for log in test_logs]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(clean_epochs, clean_losses, label="Clean Loss")
    ax.plot(test_epochs, test_losses, label="Test Loss")
    ax.grid()

    # Set the axis limits and labels
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel("Epoch")
    plt.ylabel("Average Batch Loss")

    # Create vertical bar for the lowest noisy loss
    if len(test_losses) > 0:
        bestIndex = np.argmin(test_losses)
        label = (
            f"Minimum Noisy Loss is {round(test_losses[bestIndex], 3)} "
            f"at Epoch={round(test_epochs[bestIndex], 3)}"
        )
        plt.axvline(
            test_epochs[bestIndex], 0, 1, color="r", linestyle="--", label=label
        )

    # Add the legend, save, and close
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=400, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def accuracy_plot(train_logs, test_logs, save_path):
    """Create a plot from the training/testing accuracy,
    highlighting the best test accuracy"""

    clean_epochs = [log["epoch"] for log in train_logs]
    clean_accs = [log["top1"] for log in train_logs]
    test_epochs = [log["epoch"] for log in test_logs]
    test_accs = [log["top1"] for log in test_logs]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(clean_epochs, clean_accs, label="Clean Accuracy")
    ax.plot(test_epochs, test_accs, label="Test Accuracy")
    ax.grid()

    # Set the axis limits and labels
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    # Create vertical bar for the highest noisy accuracy
    if len(test_accs) > 0:
        bestIndex = np.argmax(test_accs)
        label = (
            f"Maximum Noisy Accuracy is {round(test_accs[bestIndex], 3)} "
            f"at Epoch={round(test_epochs[bestIndex], 3)}"
        )
        plt.axvline(
            test_epochs[bestIndex], 0, 1, color="r", linestyle="--", label=label
        )

    # Add the legend, save, and close
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=400, bbox_inches="tight", pad_inches=0.2)
    plt.close()


# ------------------------
# --- Prediction Plots ---
# ------------------------


def confidence_histogram(outputs, labels, save_path):
    """A histogram of the confidences of the predictions"""

    predicted_labels = np.argmax(outputs, axis=1)
    confidences = np.max(outputs, axis=1)

    correct_idxs = np.argwhere(predicted_labels == labels).ravel()
    incorrect_idxs = np.argwhere(predicted_labels != labels).ravel()

    fig = plt.figure(figsize=(3.2, 2.4))
    ax = fig.add_subplot(111)
    ax.hist(
        confidences[correct_idxs],
        label="Correct Classifications",
        bins=100,
        range=(0, 1),
        alpha=0.8,
    )
    ax.hist(
        confidences[incorrect_idxs],
        label="Incorrect Classifications",
        bins=100,
        range=(0, 1),
        alpha=0.8,
    )
    # ax.hist(
    #     [confidences[correct_idxs], confidences[incorrect_idxs]],
    #     label=["Correct Classifications", "Incorrect Classifications"],
    #     bins=100,
    #     range=(0, 1),
    #     alpha=0.8,
    #     # stacked=True,
    # )

    plt.xlabel("Confidence of Prediction")
    plt.ylabel("Frequency")

    plt.legend(loc="upper left")
    plt.savefig(save_path, dpi=1200, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def confidence_histogram_clean_noisy(outputs, labels, noisy_labels, save_path):
    """A histogram of the confidences of the predictions"""

    predicted_labels = np.argmax(outputs, axis=1)
    confidences = np.max(outputs, axis=1)

    clean_idxs = np.argwhere(noisy_labels == labels).ravel()
    noisy_idxs = np.argwhere(noisy_labels != labels).ravel()

    fig = plt.figure(figsize=(3.2, 2.4))
    ax = fig.add_subplot(111)
    ax.hist(
        confidences[clean_idxs],
        label="Clean Samples",
        bins=100,
        range=(0, 1),
        alpha=0.8,
        # stacked=True,
    )
    ax.hist(
        confidences[noisy_idxs],
        label="Noisy Samples",
        bins=100,
        range=(0, 1),
        alpha=0.8,
        # stacked=True,
    )

    plt.xlabel("Confidence of Prediction")
    plt.ylabel("Frequency")

    plt.legend(loc="upper left")
    plt.savefig(save_path, dpi=2400, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def uncertainty_histogram(outputs, uncertainties, labels, save_path):
    """A histogram of the standard deviations of the predictions"""

    predicted_labels = np.argmax(outputs, axis=1)

    correct_idxs = np.argwhere(predicted_labels == labels).ravel()
    incorrect_idxs = np.argwhere(predicted_labels != labels).ravel()

    fig, axs = plt.subplots()
    axs.hist(
        [uncertainties[correct_idxs], uncertainties[incorrect_idxs]],
        label=["Correct Classifications", "Incorrect Classifications"],
        bins=100,
        stacked=True,
    )

    plt.xlabel("Uncertainty of Prediction")
    plt.ylabel("Frequency")

    plt.legend(loc="upper left")
    plt.savefig(save_path, dpi=400, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def calibration_plot(outputs, labels, save_path):
    """Create a calibration plot of the predictions"""

    predicted_labels = np.argmax(outputs, axis=1)
    confidences = np.max(outputs, axis=1)
    correct_mask = predicted_labels == labels

    def evaluate_interval(min_conf, max_conf):
        confidence_mask = np.logical_and(
            confidences > min_conf, confidences <= max_conf
        )
        num_conf = np.sum(confidence_mask)
        if num_conf == 0:
            return (0, 0, 0)
        else:
            num_correct = np.sum(np.logical_and(confidence_mask, correct_mask))
            correct_frac = num_correct / num_conf
            return (num_correct, num_conf, correct_frac)

    bins = 20
    mids = [(i + 1 / 2) / bins for i in range(bins)]
    interval_vals = [evaluate_interval(i / bins, (i + 1) / bins) for i in range(bins)]

    accuracies = [val[2] for val in interval_vals]
    bar_labels = [f"{val[2]*100:.2f}%\n{val[0]}/{val[1]}" for val in interval_vals]

    fig, ax = plt.subplots()
    p1 = ax.bar(mids, accuracies, 0.9 / bins, label="Accuracy Given Network Confidence")

    ax.set_xlim(xmin=0, xmax=1)
    ax.set_ylim(ymin=0, ymax=1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")

    for i in range(len(accuracies)):
        plt.text(mids[i], accuracies[i] + 0.01, bar_labels[i], ha="center", size=3.5)

    plt.plot([0, 1], [0, 1], "r--", linewidth=2)

    plt.savefig(save_path, dpi=400, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def num_selected_against_threshold(outputs, save_path):
    """How many samples we will select with different thresholds"""
    confidences = np.max(outputs, axis=1)
    confidence_order = np.flip(np.argsort(confidences))
    points = []
    for i, idx in enumerate(confidence_order):
        points.append((confidences[idx], i))
        points.append((confidences[idx], i + 1))
    points.append((0, points[-1][1]))

    xs, ys = list(zip(*points))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, ys)
    ax.grid()

    # Set the axis limits and labels
    plt.xlim(xmin=1, xmax=0)
    plt.ylim(ymin=0)
    plt.xlabel("Threshold")
    plt.ylabel("Number of Selected Items")

    # Add the legend, save, and close
    plt.savefig(save_path, dpi=400, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def accuracy_against_threshold(outputs, labels, save_path):
    """How accurate we will be with different thresholds"""

    confidences = np.max(outputs, axis=1)
    confidence_order = np.flip(np.argsort(confidences))
    predicted_labels = np.argmax(outputs, axis=1)
    correct_mask = predicted_labels == labels
    correct = np.cumsum(correct_mask[confidence_order])

    points = []
    points.append((confidences[confidence_order[0]], correct[0]))
    for i, idx in enumerate(confidence_order[1:]):
        points.append((confidences[idx], points[-1][1]))
        points.append((confidences[idx], correct[i] / (i + 1)))
    points.append((0, points[-1][1]))

    xs, ys = list(zip(*points))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, ys)
    ax.grid()

    # Set the axis limits and labels
    plt.xlim(xmin=1, xmax=0)
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy of Labels Across Selected Dataset")

    # Add the legend, save, and close
    plt.savefig(save_path, dpi=400, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def accuracy_against_num_selected(outputs, labels, save_path):
    """How accurate we will be with different numbers of selected samples"""

    def get_threshold_point(threshold):
        selected_samples = np.argwhere(confidences > threshold)
        num_correct = np.sum(correct_mask[selected_samples])
        return (num_correct, selected_samples.shape[0])

    confidences = np.max(outputs, axis=1)
    confidence_order = np.flip(np.argsort(confidences))
    predicted_labels = np.argmax(outputs, axis=1)
    correct_mask = predicted_labels == labels
    correct = np.cumsum(correct_mask[confidence_order])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()

    # Plot line graph
    points = []
    points.append((1, correct[0]))
    for i, idx in enumerate(confidence_order[1:]):
        points.append((i, points[-1][1]))
        points.append((i, correct[i] / (i + 1)))
    xs, ys = list(zip(*points))
    ax.plot(xs, ys, label="Cumulative Accuracy")

    # Plot histogram
    hist_bins = 20
    bin_size = correct_mask.shape[0] // hist_bins
    hist_values = [
        np.sum(correct_mask[confidence_order][i * bin_size : (i + 1) * bin_size])
        / bin_size
        for i in range(hist_bins)
    ]
    mids = [(i + 0.5) * bin_size for i in range(hist_bins)]
    p1 = ax.bar(
        mids,
        hist_values,
        correct_mask.shape[0] * 0.9 / hist_bins,
        label="Average Accuracy for Bin",
    )

    # Plot threshold labels
    thresholds = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.0]
    for threshold in thresholds:
        num_correct, num_selected = get_threshold_point(threshold)
        if num_selected == 0:
            continue
        acc = num_correct / num_selected
        ax.plot(num_selected, acc, "ro")
        plt.text(
            num_selected + 0.03,
            acc + 0.03,
            f"$\lambda = {threshold}$\n$Acc. = {acc:.2f}$",
            ha="left",
            multialignment="center",
            size=5,
        )

    # Set the axis limits and labels
    plt.xlabel("Number of Selected Samples")
    plt.ylabel("Accuracy of Labels Across Selected Dataset")

    # Add the legend, save, and close
    plt.legend(loc="lower left")
    plt.savefig(save_path, dpi=400, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def accuracy_against_num_selected_score(
    outputs, scores, labels, save_path, scores_to_highlight=None
):
    """How accurate we will be with different numbers of selected samples (using scores)"""

    def get_threshold_point(threshold):
        selected_samples = np.argwhere(scores < threshold)
        num_correct = np.sum(correct_mask[selected_samples])
        return (num_correct, selected_samples.shape[0])

    score_order = np.argsort(scores)
    predicted_labels = np.argmax(outputs, axis=1)
    correct_mask = predicted_labels == labels
    correct = np.cumsum(correct_mask[score_order])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()

    # Plot line graph
    points = []
    points.append((1, correct[0]))
    for i, idx in enumerate(score_order[1:]):
        points.append((i, points[-1][1]))
        points.append((i, correct[i] / (i + 1)))
    xs, ys = list(zip(*points))
    ax.plot(xs, ys, label="Cumulative Accuracy")

    # Plot histogram
    hist_bins = 20
    bin_size = correct_mask.shape[0] // hist_bins
    hist_values = [
        np.sum(correct_mask[score_order][i * bin_size : (i + 1) * bin_size]) / bin_size
        for i in range(hist_bins)
    ]
    mids = [(i + 0.5) * bin_size for i in range(hist_bins)]
    p1 = ax.bar(
        mids,
        hist_values,
        correct_mask.shape[0] * 0.9 / hist_bins,
        label="Average Accuracy for Bin",
    )

    # Plot threshold labels
    if scores_to_highlight is not None:
        for score in scores_to_highlight:
            num_correct, num_selected = get_threshold_point(score)
            if num_selected == 0:
                continue
            acc = num_correct / num_selected
            ax.plot(num_selected, acc, "ro")
            plt.text(
                num_selected + 0.03,
                acc + 0.03,
                f"$\lambda = {score}$\n$Acc. = {acc:.2f}$",
                ha="left",
                multialignment="center",
                size=5,
            )

    # Set the axis limits and labels
    plt.xlabel("Number of Selected Samples")
    plt.ylabel("Accuracy of Labels Across Selected Dataset")

    # Add the legend, save, and close
    plt.legend(loc="lower left")
    plt.savefig(save_path, dpi=400, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def clean_frac_against_num_selected_score(
    outputs, scores, labels, clean_mask, save_path, scores_to_highlight=None
):
    """What fraction of the selected samples will be clean (using scores)"""

    def get_threshold_point(threshold):
        selected_samples = np.argwhere(scores < threshold)
        num_clean = np.sum(clean_mask[selected_samples])
        return (num_clean, selected_samples.shape[0])

    score_order = np.argsort(scores)
    clean_sum = np.cumsum(clean_mask[score_order])

    fig = plt.figure(figsize=(3.2, 2.4))
    ax = fig.add_subplot(111)
    ax.grid()

    # Plot line graph
    points = []
    points.append((1, clean_sum[0]))
    for i, idx in enumerate(score_order[1:]):
        points.append((i, points[-1][1]))
        points.append((i, clean_sum[i] / (i + 1)))
    xs, ys = list(zip(*points))
    ax.plot(xs, ys, label="Cumulative Clean Fraction")

    # Plot threshold labels
    if scores_to_highlight is not None:
        for score in scores_to_highlight:
            num_clean, num_selected = get_threshold_point(score)
            if num_selected == 0:
                continue
            f = num_clean / num_selected
            ax.plot(num_selected, f, "ro")
            plt.text(
                num_selected + 0.03,
                f + 0.03,
                f"$\lambda = {score}$\n$Clean Frac. = {f:.2f}$",
                ha="left",
                multialignment="center",
                size=5,
            )

    plt.axhline(
        points[-1][1], 0, 1, color="r", linestyle="--", label="Dataset noise rate"
    )

    # Set the axis limits and labels
    plt.xlabel("Number of Selected Samples")
    plt.ylabel("Clean Fraction (%)")
    plt.ylim(bottom=points[-1][1] - 0.1)

    # Add the legend, save, and close
    # plt.legend(loc="lower left", prop={"size": 6})
    plt.legend(loc="lower left")
    plt.savefig(save_path, dpi=1200, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def interpolation_plot(test_ds, predictions, save_path):
    """Test whether interpolating between the predictions and noisy labels helps"""

    def accuracy(outputs, noisy_labels, true_labels, t):
        preds = outputs * (1 - t) + noisy_labels * (t)
        correct_mask = np.argmax(preds, axis=1) == true_labels
        acc = np.sum(correct_mask) / correct_mask.shape[0]
        return acc

    outputs = predictions[test_ds.indices]
    noisy_labels = test_ds.noisy_label_sets[0][test_ds.indices]
    true_labels = test_ds.true_labels[test_ds.indices]

    xs = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    ys = np.zeros_like(xs)
    for i, x in enumerate(xs):
        ys[i] = accuracy(outputs, noisy_labels, true_labels, x)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, ys)
    ax.grid()

    # Set the axis limits and labels
    ax.set_xlim(xmin=0, xmax=1)
    ax.set_ylim(ymin=0, ymax=1)
    plt.xlabel("Interpolation Factor (0 is model output, 1 is noisy label)")
    plt.ylabel("Accuracy")

    # Create vertical bar for the lowest noisy loss
    bestIndex = np.argmax(ys)
    label = (
        f"Maximum Accuracy is {round(ys[bestIndex], 3)} "
        f"at factor t={round(xs[bestIndex], 3)}"
    )
    plt.axvline(xs[bestIndex], 0, 1, color="r", linestyle="--", label=label)

    for i in [0, xs.shape[0] - 1, bestIndex]:
        ax.plot(xs[i], ys[i], "ro")
        plt.text(
            xs[i] + 0.03,
            ys[i] + 0.03,
            f"$\t = {xs[i]}$\n$Clean Frac. = {ys[i]:.2f}$",
            ha="left",
            multialignment="center",
            size=5,
        )

    # Add the legend, save, and close
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=400, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def loss_histogram(losses, outputs, labels, save_path):
    """A histogram of the cross entropy loss of the predictions"""

    # print("START")
    predicted_labels = np.argmax(outputs, axis=1)

    correct_idxs = np.argwhere(predicted_labels == labels).ravel()
    incorrect_idxs = np.argwhere(predicted_labels != labels).ravel()

    # print("CORRECT and INCORRECT")

    fig = plt.figure(figsize=(3.2, 2.4))
    ax = fig.add_subplot(111)
    ax.hist(
        losses[correct_idxs],
        label="Correct Classifications",
        bins=100,
        range=(0, 1),
        alpha=0.8,
    )
    ax.hist(
        losses[incorrect_idxs],
        label="Incorrect Classifications",
        bins=100,
        range=(0, 1),
        alpha=0.8,
    )
    # print("HISTOGRAMS")
    # ax.hist(
    #     [confidences[correct_idxs], confidences[incorrect_idxs]],
    #     label=["Correct Classifications", "Incorrect Classifications"],
    #     bins=100,
    #     range=(0, 1),
    #     alpha=0.8,
    #     # stacked=True,
    # )

    plt.xlabel("Loss of Prediction")
    plt.ylabel("Frequency")

    plt.legend(loc="upper left")
    plt.savefig(save_path, dpi=2400, bbox_inches="tight", pad_inches=0.01)
    plt.close()

    # print("END")

def accuracy_text_file(correct_mask, save_dir):
    """Save some basic information about how accurate our predictions are"""

    with open(save_dir, "w") as f:
        total = correct_mask.shape[0]
        correct = np.sum(correct_mask)
        incorrect = total - correct
        f.write(f"Correct: {correct}\n")
        f.write(f"Incorrect: {incorrect}\n")
        f.write(f"Accuracy: ({correct}/{total}) = {100 * (correct / total):.2f}%\n")


def prediction_plots(ds, predictions, confidences, uncertainties, logits, save_dir):
    """Call all of the above selection plots given a set of predictions"""

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Calculate some statistics about the data that will be used by different plots
    hard_noisy_labels = np.argmax(ds.noisy_label_sets[0], axis=1)[ds.indices]
    true_labels = ds.true_labels[ds.indices]
    clean_mask = hard_noisy_labels == true_labels
    correct_mask = predictions == true_labels
    avg_softmax_outputs = np.mean(softmax(logits, axis=2), axis=1)

    # Note: These indices are with respect to the subset of data, not with respect to
    # the whole dataset
    clean_indices = np.argwhere(clean_mask).ravel()
    noisy_indices = np.argwhere(np.logical_not(clean_mask)).ravel()

    # Calculate statistics about logits
    avg_logits = np.mean(logits, axis=1)
    sorted_logits = np.sort(avg_logits, axis=1)
    logit_scores = sorted_logits[:, 0]
    logit_diff_scores = sorted_logits[:, 0] - sorted_logits[:, 1]

    # Calculate statistics about confidence/uncertainty orderings
    confidence_order = np.argsort(np.argsort(1.0 - confidences))
    uncertainty_order = np.argsort(np.argsort(uncertainties))
    min_order = np.minimum(confidence_order, uncertainty_order)
    max_order = np.maximum(confidence_order, uncertainty_order)
    mean_order = (confidence_order + uncertainty_order) / 2

    # Save some basic results about the accuracy
    accuracy_text_file(correct_mask, os.path.join(save_dir, "total_acc.txt"))

    # Show calibration plots for different subsets of the data
    calibration_plot(
        avg_softmax_outputs, true_labels, os.path.join(save_dir, "calibration.png")
    )
    calibration_plot(
        avg_softmax_outputs[clean_indices],
        true_labels[clean_indices],
        os.path.join(save_dir, "calibration_clean.png"),
    )
    calibration_plot(
        avg_softmax_outputs[noisy_indices],
        true_labels[noisy_indices],
        os.path.join(save_dir, "calibration_noisy.png"),
    )

    # Show confidence plots for different subsets of the data
    confidence_histogram(
        avg_softmax_outputs, true_labels, os.path.join(save_dir, "confidences.png")
    )
    confidence_histogram(
        avg_softmax_outputs[clean_indices],
        true_labels[clean_indices],
        os.path.join(save_dir, "confidences_clean.png"),
    )
    confidence_histogram(
        avg_softmax_outputs[noisy_indices],
        true_labels[noisy_indices],
        os.path.join(save_dir, "confidences_noisy.png"),
    )
    confidence_histogram_clean_noisy(
        avg_softmax_outputs, true_labels, hard_noisy_labels, os.path.join(save_dir, "confidences_clean_noisy.png")
    )


    # Show uncertainty plots for different subsets of the data
    uncertainty_histogram(
        avg_softmax_outputs,
        uncertainties,
        true_labels,
        os.path.join(save_dir, "uncertainties.png"),
    )
    uncertainty_histogram(
        avg_softmax_outputs[clean_indices],
        uncertainties[clean_indices],
        true_labels[clean_indices],
        os.path.join(save_dir, "uncertainties_clean.png"),
    )
    uncertainty_histogram(
        avg_softmax_outputs[noisy_indices],
        uncertainties[noisy_indices],
        true_labels[noisy_indices],
        os.path.join(save_dir, "uncertainties_noisy.png"),
    )

    # Use confidences to perform selection (as the main output)
    num_selected_against_threshold(
        avg_softmax_outputs,
        os.path.join(save_dir, "num_selected_against_threshold.png"),
    )
    accuracy_against_threshold(
        avg_softmax_outputs,
        true_labels,
        os.path.join(save_dir, "accuracy_against_threshold.png"),
    )
    accuracy_against_num_selected(
        avg_softmax_outputs,
        true_labels,
        os.path.join(save_dir, "accuracy_against_num_selected.png"),
    )

    # Use confidence scores to perform selection
    accuracy_against_num_selected_score(
        avg_softmax_outputs,
        1.0 - confidences,
        true_labels,
        os.path.join(save_dir, "accuracy_against_num_selected_confidence.png"),
        [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    )
    # clean_frac_against_num_selected_score(
    #     avg_softmax_outputs,
    #     1.0 - confidences,
    #     true_labels,
    #     clean_mask,
    #     os.path.join(save_dir, "clean_frac_against_num_selected_confidence.png"),
    #     [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    # )
    clean_frac_against_num_selected_score(
        avg_softmax_outputs,
        1.0 - confidences,
        true_labels,
        clean_mask,
        os.path.join(save_dir, "clean_frac_against_num_selected_confidence.png"),
        [],
    )

    # Use uncertainty scores to perform selection
    accuracy_against_num_selected_score(
        avg_softmax_outputs,
        uncertainties,
        true_labels,
        os.path.join(save_dir, "accuracy_against_num_selected_uncertainty.png"),
        [0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0],
    )
    clean_frac_against_num_selected_score(
        avg_softmax_outputs,
        uncertainties,
        true_labels,
        clean_mask,
        os.path.join(save_dir, "clean_frac_against_num_selected_uncertainty.png"),
        [0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0],
    )

    # Use highest logit value scores to perform selection
    accuracy_against_num_selected_score(
        avg_softmax_outputs,
        logit_scores,
        true_labels,
        os.path.join(save_dir, "accuracy_against_num_selected_logit.png"),
        [0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0],
    )
    clean_frac_against_num_selected_score(
        avg_softmax_outputs,
        logit_scores,
        true_labels,
        clean_mask,
        os.path.join(save_dir, "clean_frac_against_num_selected_logit.png"),
        [0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0],
    )

    # Use difference between highest and second highest logit value to perform selection
    accuracy_against_num_selected_score(
        avg_softmax_outputs,
        logit_diff_scores,
        true_labels,
        os.path.join(save_dir, "accuracy_against_num_selected_logit_diff.png"),
        [0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0],
    )
    clean_frac_against_num_selected_score(
        avg_softmax_outputs,
        logit_diff_scores,
        true_labels,
        clean_mask,
        os.path.join(save_dir, "clean_frac_against_num_selected_logit_diff.png"),
        [0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0],
    )

    # Use random scores to perform selection
    random_scores = np.random.rand(confidences.shape[0])
    accuracy_against_num_selected_score(
        avg_softmax_outputs,
        random_scores,
        true_labels,
        os.path.join(save_dir, "accuracy_against_num_selected_random.png"),
    )
    clean_frac_against_num_selected_score(
        avg_softmax_outputs,
        random_scores,
        true_labels,
        clean_mask,
        os.path.join(save_dir, "clean_frac_against_num_selected_random.png"),
    )

    # Use max value from confidence/uncertainty ordering
    accuracy_against_num_selected_score(
        avg_softmax_outputs,
        min_order,
        true_labels,
        os.path.join(save_dir, "accuracy_against_num_selected_min.png"),
        [1000, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000],
    )
    clean_frac_against_num_selected_score(
        avg_softmax_outputs,
        min_order,
        true_labels,
        clean_mask,
        os.path.join(save_dir, "clean_frac_against_num_selected_min.png"),
    )

    # Use min value from confidence/uncertainty ordering
    accuracy_against_num_selected_score(
        avg_softmax_outputs,
        max_order,
        true_labels,
        os.path.join(save_dir, "accuracy_against_num_selected_max.png"),
        [1000, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000],
    )
    clean_frac_against_num_selected_score(
        avg_softmax_outputs,
        max_order,
        true_labels,
        clean_mask,
        os.path.join(save_dir, "clean_frac_against_num_selected_max.png"),
    )

    # Use mean value frmo confidence/uncertainty ordering
    accuracy_against_num_selected_score(
        avg_softmax_outputs,
        mean_order,
        true_labels,
        os.path.join(save_dir, "accuracy_against_num_selected_mean.png"),
        [1000, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000],
    )
    clean_frac_against_num_selected_score(
        avg_softmax_outputs,
        mean_order,
        true_labels,
        clean_mask,
        os.path.join(save_dir, "clean_frac_against_num_selected_mean.png"),
    )

    interpolation_plot(
        ds, avg_softmax_outputs, os.path.join(save_dir, "interpolation.png"),
    )

    cross_entropy_per_sample = torch.nn.functional.cross_entropy(torch.tensor(avg_logits), torch.tensor(hard_noisy_labels), reduction='none').numpy()
    loss_histogram(cross_entropy_per_sample, avg_softmax_outputs, true_labels, os.path.join(save_dir, "cross_entropy.png"))
    # print(cross_entropy_per_sample)
    # print(cross_entropy_per_sample.size())


# ---------------
# --- Archive ---
# ---------------


@torch.no_grad()
def label_variation_test(config, model, data_loader, device, save_dir):
    def name(class_idx):
        return get_class_name(config.data.dataset, class_idx).capitalize()

    os.makedirs(os.path.join(save_dir, "clean"))
    os.makedirs(os.path.join(save_dir, "noisy"))

    model.eval()

    labels_to_test = []
    labels_to_test.append(torch.zeros(config.data.num_classes))  # Add the zero label
    for k in range(0, config.data.num_classes):  # Add the one hot labels
        label = torch.zeros(config.data.num_classes)
        label[k] = 1
        labels_to_test.append(label)

    data_iterator = enumerate(tqdm(data_loader, ascii=True, ncols=80))
    for batch_idx, (idxs, images, t_labels, _, *n_labels) in data_iterator:

        images = images.to(device)
        n_labels = [n_label.to(device) for n_label in n_labels]

        outputs = model(images, *n_labels)
        outputs = torch.softmax(outputs, dim=1)

        if batch_idx < 6:

            all_outputs = np.empty(
                (idxs.shape[0], len(labels_to_test), config.data.num_classes),
                dtype=np.float32,
            )
            for j, label in enumerate(labels_to_test):
                labels = torch.zeros(
                    [idxs.shape[0], config.data.num_classes], device=device
                )
                for i in range(0, idxs.shape[0]):
                    labels[i] = label
                outputs = model(images, labels)
                outputs = torch.softmax(outputs, dim=1)
                outputs = outputs.cpu().detach().numpy()
                all_outputs[:, j, :] = outputs

            idxs = idxs.detach().numpy()
            images = images.cpu()
            t_labels = t_labels.detach().numpy()
            n_labels = [n_label.cpu().detach().numpy() for n_label in n_labels]

            for i, idx in enumerate(idxs):

                idx = idx.item()
                noisy_label = np.argmax(n_labels[0][i])
                true_label = t_labels[i]

                subfolder = "clean"
                if noisy_label != true_label:
                    subfolder = "noisy"

                torchvision.utils.save_image(
                    images[i, :, :, :],
                    os.path.join(save_dir, subfolder, f"{idx}.png"),
                    normalize=True,
                )

                with open(os.path.join(save_dir, subfolder, f"{idx}.txt"), "w+") as f:

                    f.write(f"# Labels\n")
                    f.write(f"True label: {t_labels[i]} ({name(t_labels[i])})\n")
                    f.write(f"Noisy label: {noisy_label} ({name(noisy_label)})\n\n")

                    f.write(f"# Prediction Table\n")
                    predicted_classes = np.flip(np.argsort(all_outputs[i][0]))
                    f.write(f"Zero Label |")
                    for k in range(3):
                        f.write(
                            f"{name(predicted_classes[k]):10s} ({(100 * all_outputs[i][0][predicted_classes[k]]):.2f}%) |"
                        )
                    f.write("\n")
                    for j in range(config.data.num_classes):
                        predicted_classes = np.flip(np.argsort(all_outputs[i][j + 1]))
                        f.write(f"{name(j):10s} |")
                        for k in range(3):
                            f.write(
                                f"{name(predicted_classes[k]):10s} ({(100 * all_outputs[i][j+1][predicted_classes[k]]):.2f}%) |"
                            )
                        f.write("\n")
                    f.write("\n")


@torch.no_grad()
def label_variation_test(config, model, data_loader, device, save_dir):
    def name(class_idx):
        return get_class_name(config.data.dataset, class_idx).capitalize()

    os.makedirs(os.path.join(save_dir, "clean"))
    os.makedirs(os.path.join(save_dir, "noisy"))

    model.eval()

    labels_to_test = []
    labels_to_test.append(torch.zeros(config.data.num_classes))  # Add the zero label
    for k in range(0, config.data.num_classes):  # Add the one hot labels
        label = torch.zeros(config.data.num_classes)
        label[k] = 1
        labels_to_test.append(label)

    data_iterator = enumerate(tqdm(data_loader, ascii=True, ncols=80))
    for batch_idx, (idxs, images, t_labels) in data_iterator:

        images = images.to(device)
        # n_labels = [n_label.to(device) for n_label in n_labels]

        # outputs = model(images, *n_labels)
        # outputs = torch.softmax(outputs, dim=1)

        if batch_idx < 20:

            all_outputs = np.empty(
                (idxs.shape[0], len(labels_to_test), config.data.num_classes),
                dtype=np.float32,
            )
            for j, label in enumerate(labels_to_test):
                labels = torch.zeros(
                    [idxs.shape[0], config.data.num_classes], device=device
                )
                for i in range(0, idxs.shape[0]):
                    labels[i] = label
                outputs = model(images, labels)
                outputs = torch.softmax(outputs, dim=1)
                outputs = outputs.cpu().detach().numpy()
                all_outputs[:, j, :] = outputs

            idxs = idxs.detach().numpy()
            images = images.cpu()
            t_labels = t_labels.detach().numpy()
            # n_labels = [n_label.cpu().detach().numpy() for n_label in n_labels]

            for i, idx in enumerate(idxs):

                idx = idx.item()
                # noisy_label = np.argmax(n_labels[0][i])
                true_label = t_labels[i]

                subfolder = "clean"
                # if noisy_label != true_label:
                #     subfolder = "noisy"

                torchvision.utils.save_image(
                    images[i, :, :, :],
                    os.path.join(save_dir, subfolder, f"{idx}.png"),
                    normalize=True,
                )

                with open(os.path.join(save_dir, subfolder, f"{idx}.txt"), "w+") as f:

                    f.write(f"# Labels\n")
                    f.write(f"True label: {t_labels[i]} ({name(t_labels[i])})\n")
                    # f.write(f"Noisy label: {noisy_label} ({name(noisy_label)})\n\n")

                    f.write(f"# Prediction Table\n")
                    predicted_classes = np.flip(np.argsort(all_outputs[i][0]))
                    f.write(f"Zero Label |")
                    for k in range(3):
                        f.write(
                            f"{name(predicted_classes[k]):10s} ({(100 * all_outputs[i][0][predicted_classes[k]]):.2f}%) |"
                        )
                    f.write("\n")
                    for j in range(config.data.num_classes):
                        predicted_classes = np.flip(np.argsort(all_outputs[i][j + 1]))
                        f.write(f"{name(j):10s} |")
                        for k in range(3):
                            f.write(
                                f"{name(predicted_classes[k]):10s} ({(100 * all_outputs[i][j+1][predicted_classes[k]]):.2f}%) |"
                            )
                        f.write("\n")
                    f.write("\n")

                    # f.write(f"# Raw Predictions\n")
                    # for j in range(config.data.num_classes):
                    #     pseudo_label = np.zeros(config.data.num_classes)
                    #     pseudo_label[j] = 1
                    #     print(f"\tWith pseudo-label: {pseudo_label}", file=f)
                    #     pred = list(all_outputs[i][j])
                    #     pred = [f"{p:.3f}" for p in pred]
                    #     print(f"\t\tPrediction is: {pred}", file=f)

    # with open(os.path.join(save_dir, "accuracy.txt"), "w") as f:
    #     f.write(f"Correct: {correct}\n")
    #     f.write(f"Incorrect: {total - correct}\n")
    #     f.write(f"Accuracy: ({correct}/{total}) = {100 * (correct / total):.2f}%\n")


# def label_variant_predictions(config, run_info, model, data_loader, device, save_dir):
#     def name(class_idx):
#         return get_class_name(config.dataset, class_idx).capitalize()

#     if not os.path.exists(save_dir):
#         os.makedirs(os.path.join(save_dir, "clean"))
#         os.makedirs(os.path.join(save_dir, "noisy"))

#     model.eval()

#     correct = 0
#     total = 0

#     with torch.no_grad():

#         data_iterator = enumerate(tqdm(data_loader, ascii=True, ncols=80))
#         for batch_idx, (idxs, images, t_labels, _, *n_labels) in data_iterator:

#             images = images.to(device)
#             n_labels = [n_label.to(device) for n_label in n_labels]

#             outputs = model(images, *n_labels)
#             outputs = torch.softmax(outputs, dim=1)
#             for i, _ in enumerate(idxs):
#                 total += 1
#                 if torch.argmax(outputs[i]) == t_labels[i]:
#                     correct += 1

#             if batch_idx < 6:

#                 all_outputs = np.empty((idxs.shape[0], run_info.num_classes, run_info.num_classes), dtype=np.float32)
#                 for j in range(0, run_info.num_classes):
#                     fake_labels = torch.zeros([idxs.shape[0], run_info.num_classes])
#                     for k in range(0, fake_labels.shape[0]):
#                         fake_labels[k, j] = 1
#                     outputs = model(images, fake_labels)
#                     outputs = torch.softmax(outputs, dim=1)
#                     outputs = outputs.cpu().detach().numpy()
#                     all_outputs[:, j] = outputs

#                 idxs = idxs.detach().numpy()
#                 images = images.cpu()
#                 t_labels = t_labels.detach().numpy()
#                 n_labels = [n_label.cpu().detach().numpy() for n_label in n_labels]

#                 for i, idx in enumerate(idxs):

#                     idx = idx.item()
#                     noisy_label = np.argmax(n_labels[0][i])
#                     true_label = t_labels[i]

#                     subfolder = "clean"
#                     if noisy_label != true_label:
#                         subfolder = "noisy"

#                     torchvision.utils.save_image(
#                         images[i, :, :, :],
#                         os.path.join(save_dir, subfolder, f"{idx}.png"),
#                         normalize=True,
#                     )

#                     with open(
#                         os.path.join(save_dir, subfolder, f"{idx}.txt"), "w+"
#                     ) as f:

#                         f.write(f"# Labels\n")
#                         f.write(f"True label: {t_labels[i]} ({name(t_labels[i])})\n")
#                         f.write(f"Noisy label: {noisy_label} ({name(noisy_label)})\n\n")

#                         f.write(f"# Prediction Table\n")
#                         for j in range(run_info.num_classes):
#                             predicted_classes = np.flip(np.argsort(all_outputs[i][j]))
#                             f.write(f"{name(j):10s} |")
#                             for k in range(3):
#                                 f.write(
#                                     f"{name(predicted_classes[k]):10s} ({(100 * all_outputs[i][j][predicted_classes[k]]):.2f}%) |"
#                                 )
#                             f.write("\n")
#                         f.write("\n")

#                         f.write(f"# Raw Predictions\n")
#                         for j in range(run_info.num_classes):
#                             pseudo_label = np.zeros(run_info.num_classes)
#                             pseudo_label[j] = 1
#                             print(f"\tWith pseudo-label: {pseudo_label}", file=f)
#                             pred = list(all_outputs[i][j])
#                             pred = [f"{p:.3f}" for p in pred]
#                             print(f"\t\tPrediction is: {pred}", file=f)

#     with open(os.path.join(save_dir, "accuracy.txt"), "w") as f:
#         f.write(f"Correct: {correct}\n")
#         f.write(f"Incorrect: {total - correct}\n")
#         f.write(f"Accuracy: ({correct}/{total}) = {100 * (correct / total):.2f}%\n")


def visualise_ssl_batch(config, x, x_ulb_w, x_ulb_s):
    import torchvision
    x_grid = torchvision.utils.make_grid(x, nrow=8, padding=2, normalize=True)
    torchvision.utils.save_image(x_grid, os.path.join(config.save_dir, f"x.png"))
    weak_grid = torchvision.utils.make_grid(x_ulb_w, nrow=8, padding=2, normalize=True)
    torchvision.utils.save_image(weak_grid, os.path.join(config.save_dir, f"weak.png"))
    strong_grid = torchvision.utils.make_grid(x_ulb_s, nrow=8, padding=2, normalize=True)
    torchvision.utils.save_image(strong_grid, os.path.join(config.save_dir, f"strong.png"))
    return