import numpy as np


def percentage_noise_matrix(noise_matrix):
    """Convert a noise matrix into human readable format"""

    if np.sum(noise_matrix) == 0:
        return 0.0
    return 100.0 * noise_matrix / np.sum(noise_matrix)


def evaluate_selection(
    config, ds, selected_indices, predictions, confidences, uncertainties, save_file
):
    """Save a Markdown file with statistics about the selection"""

    # Create empty arrays
    num_per_guessed_class = np.zeros(config.data.num_classes, dtype=np.int32)
    num_per_true_class = np.zeros(config.data.num_classes, dtype=np.int32)
    num_per_noisy_class = np.zeros(config.data.num_classes, dtype=np.int32)
    full_noise_matrix = np.zeros((2, 2, 2, config.data.num_classes), dtype=np.int32)

    # Update the statistics using information from each sample
    for idx in selected_indices:

        guessed_label = predictions[idx]
        noisy_label = np.argmax(ds.noisy_label_sets[0][idx])
        true_label = ds.true_labels[idx]

        num_per_guessed_class[guessed_label] += 1
        num_per_true_class[true_label] += 1
        num_per_noisy_class[noisy_label] += 1

        g_eq_t = int(guessed_label == true_label)
        n_eq_t = int(noisy_label == true_label)
        g_eq_n = int(guessed_label == noisy_label)

        full_noise_matrix[g_eq_t][n_eq_t][g_eq_n][guessed_label] += 1

    # Calculate statistics about the dataset as a whole
    num_selected = len(selected_indices)
    correct_samples = np.sum(full_noise_matrix[1, :, :, :])
    incorrect_samples = np.sum(full_noise_matrix[0, :, :, :])
    clean_samples = np.sum(full_noise_matrix[:, 1, :, :])
    noisy_samples = np.sum(full_noise_matrix[:, 0, :, :])

    with open(save_file, "w") as f:

        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=5)

        f.write(
            f"# Details\n"
            f"Correct Samples: {correct_samples} ({(100.0 * correct_samples/num_selected):.5f}%)\n"
            f"Incorrect Samples: {incorrect_samples} ({(100.0 * incorrect_samples/num_selected):.5f}%)\n"
            f"\n"
            f"Samples where the noisy label matches the true label: {clean_samples} ({(100.0 * clean_samples/num_selected):.5f}%)\n"
            f"\tWe labelled correctly: {np.sum(full_noise_matrix[1, 1, :, :])}: ({(100.0 * np.sum(full_noise_matrix[1, 1, :, :])/num_selected):.5f}%)\n"
            f"\tWe labelled incorrectly: {np.sum(full_noise_matrix[0, 1, :, :])} ({(100.0 * np.sum(full_noise_matrix[0, 1, :, :])/num_selected):.5f}%)\n"
            f"Samples where the noisy label is different from the true label: {noisy_samples} ({(100.0 * noisy_samples/num_selected):.5f}%)\n"
            f"\tWe labelled as the true label: {np.sum(full_noise_matrix[1, 0, :, :])} ({(100.0 * np.sum(full_noise_matrix[1, 0, :, :])/num_selected):.5f}%)\n"
            f"\tWe labelled as the noisy label: {np.sum(full_noise_matrix[:, 0, 1, :])} ({(100.0 * np.sum(full_noise_matrix[:, 0, 1, :])/num_selected):.5f}%)\n"
            f"\tWe labelled as another label: {np.sum(full_noise_matrix[0, 0, 0, :])} ({(100.0 * np.sum(full_noise_matrix[0, 0, 0, :])/num_selected):.5f}%)\n"
            f"\n"
            # f"Thresholds:\n{threshold}\n"
            f"\n"
            f"# Samples Per Class\n"
            f"Number of Samples per Guessed Class: {num_per_guessed_class.tolist()}\n"
            f"Number of Samples per True Class: {num_per_true_class.tolist()}\n"
            f"Number of Samples per Noisy Class: {num_per_noisy_class.tolist()}\n"
            f"\n"
        )

        f.write(f"# Noise Matrix Per Class\n")
        for i in range(0, config.data.num_classes):
            f.write(
                f"\n"
                f"Class {i}:\n"
                f"\tSamples where the noisy label matches the true label: {np.sum(full_noise_matrix[:, 1, :, i])} ({(100.0 * np.sum(full_noise_matrix[:, 1, :, i])/num_selected):.5f}%)\n"
                f"\t\tWe labelled correctly: {np.sum(full_noise_matrix[1, 1, :, i])}: ({(100.0 * np.sum(full_noise_matrix[1, 1, :, i])/num_selected):.5f}%)\n"
                f"\t\tWe labelled incorrectly: {np.sum(full_noise_matrix[0, 1, :, i])} ({(100.0 * np.sum(full_noise_matrix[0, 1, :, i])/num_selected):.5f}%)\n"
                f"\tSamples where the noisy label is different from the true label: {np.sum(full_noise_matrix[:, 0, : ,i])} ({(100.0 * np.sum(full_noise_matrix[:, 0, :, i])/num_selected):.5f}%)\n"
                f"\t\tWe labelled as the true label: {np.sum(full_noise_matrix[1, 0, :, i])} ({(100.0 * np.sum(full_noise_matrix[1, 0, :, i])/num_selected):.5f}%)\n"
                f"\t\tWe labelled as the noisy label: {np.sum(full_noise_matrix[:, 0, 1, i])} ({(100.0 * np.sum(full_noise_matrix[:, 0, 1, i])/num_selected):.5f}%)\n"
                f"\t\tWe labelled as another label: {np.sum(full_noise_matrix[0, 0, 0, i])} ({(100.0 * np.sum(full_noise_matrix[0, 0, 0, i])/num_selected):.5f}%)\n"
            )

        if True:
            f.write(f"\n# Samples\n\n")
            for idx in selected_indices:
                f.write(
                    f"{idx}"
                    f", Confidence: {confidences[idx]:5f}"
                    f", Uncertainty: {uncertainties[idx]:5f}\n"
                    f"\t Noisy Label: {np.argmax(ds.noisy_label_sets[0][idx])}"
                    f", Guessed Label: {predictions[idx]}"
                    f", True Label: {ds.true_labels[idx]}\n"
                )