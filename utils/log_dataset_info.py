import os
import numpy as np


def log_dataset(ds, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    ds.save_to_file(os.path.join(save_dir, "saved_dataset.json"))
    log_dataset_info(ds, os.path.join(save_dir, "logged_info.md"))

# --------------------------------
# --- Logging Helper Functions ---
# --------------------------------

def assess_labels(ds, indices, labels):
    """Given a dataset (which contains the true labels) and a set of provided labels,
    calculate some statistics on how good the labelling is"""
    correct_indices = []
    incorrect_indices = []
    for i in indices:
        if labels[i] == ds.true_labels[i]:
            correct_indices.append(i)
        else:
            incorrect_indices.append(i)
    correct_indices = np.array(correct_indices)
    incorrect_indices = np.array(incorrect_indices)
    noise_ratio = 0
    if len(indices != 0):
        noise_ratio = len(incorrect_indices) / len(indices)
    return correct_indices, incorrect_indices, noise_ratio


def compare_labels(ds, indices, labels_a, labels_b):
    num_classes = ds.noisy_label_sets[0].shape[1]
    label_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    correct_matrix = np.zeros((2, 2), dtype=np.int32)
    compared_to_true_matrix = np.zeros((num_classes, num_classes, num_classes), dtype=np.int32)
    for i in indices:
        label_matrix[labels_a[i], labels_b[i]] += 1
        compared_to_true_matrix[ds.true_labels[i], labels_a[i], labels_b[i]] += 1
        a_correct = 1 if labels_a[i] == ds.true_labels[i] else 0
        b_correct = 1 if labels_b[i] == ds.true_labels[i] else 0
        correct_matrix[a_correct][b_correct] += 1
    return label_matrix, correct_matrix, compared_to_true_matrix


def create_2d_matrix(m):
    output = ""
    output += "    |" + "|".join(f" {i:5d} " for i in range(0, m.shape[1])) + "\n"
    output += "*" * (4 + (5 + 3) * m.shape[1]) + "\n"
    for i in range(m.shape[0]):
        output += f"{i:3d} |" + "|".join(f" {j:5d} " for j in m[i]) + "\n"
    return output


def split_based_on_clean_labels(ds, noisy_predictions):
    num_clean_labels_sets = [[] for i in range(0, len(noisy_predictions) + 1)]
    for i in ds.indices:
        clean_labels = 0
        for noisy_labels in noisy_predictions:
            if ds.true_labels[i] == noisy_labels[i]:
                clean_labels += 1
        num_clean_labels_sets[clean_labels].append(i)
    num_clean_labels_sets = [np.array(s) for s in num_clean_labels_sets]
    return num_clean_labels_sets


def accuracy_of_labels(ds, indices, noisy_predictions):
    l_labels = ds.learned_labels
    if len(l_labels.shape) == 2:
        l_labels = np.argmax(l_labels, axis=1)
    res = [[0, 0], [0, 0]]
    for i in indices:
        is_renamed_as_noisy_label = False
        for j in range(len(noisy_predictions)):
            if noisy_predictions[j][i] == l_labels[i]:
                is_renamed_as_noisy_label = True
        if ds.true_labels[i] == l_labels[i]:
            if is_renamed_as_noisy_label:
                res[0][0] += 1
            else:
                res[0][1] += 1
        else:
            if is_renamed_as_noisy_label:
                res[1][0] += 1
            else:
                res[1][1] += 1
    return res


def log_dataset_info(ds, save_file):
    """Report a number of statistics about the provided dataset, and save them in
    markdown format to the provided file"""

    num_classes = ds.noisy_label_sets[0].shape[1]

    with open(save_file, "w") as f:

        noisy_predictions = [
            np.argmax(label_set, axis=1) for label_set in ds.noisy_label_sets
        ]
        l_labels = ds.learned_labels
        if len(l_labels.shape) == 2:
            l_labels = np.argmax(l_labels, axis=1)

        num_elements = len(ds.indices)
        f.write(("# Basic Info\n" f"Number of elements: {num_elements}\n"))

        # With learned labels
        correct_indices, incorrect_indices, noise_ratio = assess_labels(
            ds, ds.indices, l_labels
        )
        f.write(
            (
                "\n"
                "# Accuracy of Learned Labels\n"
                f"Number of correct elements: {len(correct_indices)}\n"
                f"Number of incorrect elements: {len(incorrect_indices)}\n"
                f"Noise Ratio: {noise_ratio}\n"
                f"Incorrect elements: {incorrect_indices}\n"
            )
        )

        samples_by_true_label = [
            np.sum(ds.true_labels[ds.indices] == i) for i in range(0, num_classes)
        ]
        f.write(f"\nSamples in each true class: {samples_by_true_label}\n")
        samples_by_learned_label = [
            np.sum(l_labels[ds.indices] == i) for i in range(0, num_classes)
        ]
        f.write(f"\nSamples in each true class: {samples_by_learned_label}\n")

        label_matrix, correct_matrix, _ = compare_labels(
            ds, ds.indices, ds.true_labels, l_labels
        )
        f.write("\nTrue Labels (Columns) vs Learned Labels (Rows)\n")
        f.write(create_2d_matrix(label_matrix))
        f.write("\nCorrectness of True Labels (Columns) vs Learned Labels (Rows)\n")
        f.write(create_2d_matrix(correct_matrix))

        # Get the indices of the elements in this dataset with 0, 1, 2, etc. correct labels
        num_clean_labels_sets = split_based_on_clean_labels(ds, noisy_predictions)
        f.write("\n# Results for Different Numbers of Clean Labels\n")
        for i in range(len(num_clean_labels_sets)):
            res = accuracy_of_labels(ds, num_clean_labels_sets[i], noisy_predictions)
            f.write(
                (
                    f"\nNumber of elements with {i} clean labels: {num_clean_labels_sets[i].shape[0]}\n"
                    f"\t- Number labelled with their true label: {res[0][0] + res[0][1]}\n"
                    f"\t- Number labelled with their noisy label: {res[1][0]}\n"
                    f"\t- Number labelled with a different label: {res[1][1]}\n"
                )
            )

        # With each type of noisy label
        for noise_type_idx in range(len(ds.noisy_label_sets)):
            labels = noisy_predictions[noise_type_idx]
            correct_indices, incorrect_indices, noise_ratio = assess_labels(
                ds, ds.indices, labels
            )
            f.write(
                (
                    "\n"
                    f"# Accuracy of Noisy Label Set {noise_type_idx}\n"
                    f"Number of correct elements: {len(correct_indices)}\n"
                    f"Number of incorrect elements: {len(incorrect_indices)}\n"
                    f"Noise Ratio: {noise_ratio}\n"
                    f"Incorrect elements: {incorrect_indices}\n"
                )
            )
            label_matrix, correct_matrix, _ = compare_labels(
                ds, ds.indices, ds.true_labels, labels
            )
            f.write(f"\nTrue Labels (Columns) vs Noisy Label {noise_type_idx} (Rows)\n")
            f.write(create_2d_matrix(label_matrix))
            f.write(
                f"\nCorrectness of True Labels (Columns) vs Noisy Label {noise_type_idx} (Rows)\n"
            )
            f.write(create_2d_matrix(correct_matrix))

        # Compare the learned labels to each type of noise (and the true labels)
        for noise_type_idx in range(len(ds.noisy_label_sets)):
            labels = noisy_predictions[noise_type_idx]
            label_matrix, correct_matrix, compared_to_true_matrix = compare_labels(
                ds, ds.indices, l_labels, labels
            )
            f.write(f"\n# Comparing Learned Labels to Noisy Label {noise_type_idx}")
            f.write(
                f"\nLearned Labels (Columns) vs Noisy Label {noise_type_idx} (Rows)\n"
            )
            f.write(create_2d_matrix(label_matrix))
            f.write(
                f"\nCorrectness of Learned Labels (Columns) vs Noisy Label {noise_type_idx} (Rows)\n"
            )
            f.write(create_2d_matrix(correct_matrix))

            for i in range(0, num_classes):
                f.write(
                    f"\n## For True Label {i} - Learned Labels (Columns) vs Noisy Label {noise_type_idx} (Rows)\n"
                )
                f.write(create_2d_matrix(compared_to_true_matrix[i]))

        # Compare each type of noise to every other type of noise
        for noise_type_a in range(len(ds.noisy_label_sets)):
            for noise_type_b in range(noise_type_a + 1, len(ds.noisy_label_sets)):
                labels_a = noisy_predictions[noise_type_a]
                labels_b = noisy_predictions[noise_type_b]
                label_matrix, correct_matrix, compared_to_true_matrix = compare_labels(
                    ds, ds.indices, labels_a, labels_b
                )
                f.write(
                    f"\n# Comparing Noisy Label {noise_type_a} to Noisy Label {noise_type_b}"
                )
                f.write(
                    f"\nNoisy Label {noise_type_a} (Columns) vs Noisy Label {noise_type_b} (Rows)\n"
                )
                f.write(create_2d_matrix(label_matrix))
                f.write(
                    f"\nCorrectness of  Noisy Label {noise_type_a} (Columns) vs Noisy Label {noise_type_b} (Rows)\n"
                )
                f.write(create_2d_matrix(correct_matrix))

                for i in range(0, num_classes):
                    f.write(
                        f"\n## For True Label {i} -  Noisy Label {noise_type_a} (Columns) vs Noisy Label {noise_type_b} (Rows)\n"
                    )
                    f.write(create_2d_matrix(compared_to_true_matrix[i]))
