import os
import time
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

DEBUG_FILE = "debug_info.npz"
OUTPUT_DIR = "assignment3_outputs"
CACHE_DIR = "assignment3_cache"
REPORT_ASSET_DIR = "assignment3_report_assets"


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(REPORT_ASSET_DIR, exist_ok=True)


def load_debug_info(debug_file=DEBUG_FILE, verbose=False):
    if not os.path.exists(debug_file):
        raise FileNotFoundError(
            f"Cannot find {debug_file}. Please place debug_info.npz in the project root."
        )

    data = np.load(debug_file)

    if verbose:
        print("Keys in debug file:")
        for key in data.files:
            print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")

    return data

def conv_slow(X_ims, Fs): 
    # Slow but clear implementation of the patchify convolution layer
    img_size = X_ims.shape[0]
    n = X_ims.shape[3]

    f = Fs.shape[0]
    nf = Fs.shape[3]

    patches_per_side = img_size // f
    conv_outputs = np.zeros((patches_per_side, patches_per_side, nf, n))

    for i in range(n):
        for filter_idx in range(nf):
            for row in range(patches_per_side):
                for col in range(patches_per_side):
                    row_start = row * f
                    row_end = row_start + f

                    col_start = col * f
                    col_end = col_start + f

                    X_patch = X_ims[row_start:row_end, col_start:col_end, :, i]
                    F = Fs[:, :, :, filter_idx]

                    conv_outputs[row, col, filter_idx, i] = np.sum(np.multiply(X_patch, F))

    return conv_outputs

def build_MX(X_ims, f):
    # Construct the patch matrix representation MX for a batch of images.
    img_size = X_ims.shape[0]
    channels = X_ims.shape[2]
    n = X_ims.shape[3]

    patches_per_side = img_size // f
    n_p = patches_per_side * patches_per_side
    patch_dim = f * f * channels

    MX = np.zeros((n_p, patch_dim, n), dtype=X_ims.dtype)

    for i in range(n):
        patch_idx = 0

        for row in range(patches_per_side):
            for col in range(patches_per_side):
                row_start = row * f
                row_end = row_start + f

                col_start = col * f
                col_end = col_start + f

                X_patch = X_ims[row_start:row_end, col_start:col_end, :, i]

                MX[patch_idx, :, i] = X_patch.reshape((patch_dim,), order="C")

                patch_idx += 1

    return MX

def flatten_filters(Fs):
    # Flatten convolution filters into a matrix.
    f = Fs.shape[0]
    nf = Fs.shape[3]

    return Fs.reshape((f * f * 3, nf), order="C")


def conv_matrix_multiplication(MX, Fs_flat):
    # Compute convolution outputs using matrix multiplication.
    n_p = MX.shape[0]
    n = MX.shape[2]
    nf = Fs_flat.shape[1]

    conv_outputs_mat = np.zeros((n_p, nf, n), dtype=MX.dtype)

    for i in range(n):
        conv_outputs_mat[:, :, i] = np.matmul(MX[:, :, i], Fs_flat)

    return conv_outputs_mat

def conv_einsum(MX, Fs_flat):
    # Compute convolution outputs using Einstein summation.
    return np.einsum("ijn,jl->iln", MX, Fs_flat, optimize=True)

def relative_error(a, b, eps=1e-12): 
    # Compute relative error between two arrays.
    numerator = np.max(np.abs(a - b))
    denominator = max(eps, np.max(np.abs(a)) + np.max(np.abs(b)))
    return numerator / denominator

def softmax(S):
    # Compute softmax column-wise.

    S_shifted = S - np.max(S, axis=0, keepdims=True)
    exp_S = np.exp(S_shifted)
    return exp_S / np.sum(exp_S, axis=0, keepdims=True)

def evaluate_network(MX, params):
    # Forward pass for the three-layer network with an initial patchify layer.
    Fs_flat = params["Fs_flat"]
    W1 = params["W1"]
    W2 = params["W2"]
    b1 = params["b1"]
    b2 = params["b2"]

    n_p = MX.shape[0]
    nf = Fs_flat.shape[1]
    n = MX.shape[2]

    conv_outputs_mat = conv_einsum(MX, Fs_flat)

    if "b0" in params and params["b0"] is not None:
        conv_outputs_mat = conv_outputs_mat + params["b0"].reshape((1, nf, 1))

    conv_flat = np.fmax(conv_outputs_mat.reshape((n_p * nf, n), order="C"), 0)

    S1 = W1 @ conv_flat + b1
    X1 = np.fmax(S1, 0)

    S = W2 @ X1 + b2
    P = softmax(S)

    cache = {
        "conv_outputs_mat": conv_outputs_mat,
        "conv_flat": conv_flat,
        "S1": S1,
        "X1": X1,
        "S": S,
        "P": P,
    }

    return P, cache

def smooth_labels(Y, epsilon=0.1):
    # Apply label smoothing to one-hot labels.
    if epsilon <= 0.0:
        return Y

    K = Y.shape[0]
    Y_smooth = (1.0 - epsilon) * Y + (epsilon / (K - 1)) * (1.0 - Y)

    return Y_smooth

def compute_gradients(MX, Y, params, cache, lam=0.0, label_smoothing=0.0):
    """
    Compute analytical gradients for the three-layer network.

    L2 regularization is applied to Fs_flat, W1, and W2.
    Bias terms are not regularized.

    If label_smoothing > 0, the one-hot labels Y are replaced by smoothed
    labels in the softmax + cross-entropy gradient.
    """
    W1 = params["W1"]
    W2 = params["W2"]
    Fs_flat = params["Fs_flat"]

    P = cache["P"]
    X1 = cache["X1"]
    conv_flat = cache["conv_flat"]

    n = Y.shape[1]
    n_p = MX.shape[0]
    nf = Fs_flat.shape[1]

    Y_target = smooth_labels(Y, epsilon=label_smoothing)

    # Gradient through softmax + cross-entropy
    G = -(Y_target - P)

    # Gradients for W2 and b2
    grad_W2 = (G @ X1.T) / n
    grad_b2 = np.sum(G, axis=1, keepdims=True) / n

    # Backpropagate to hidden layer X1
    G = W2.T @ G

    # Backpropagate through ReLU after W1
    G = G * (X1 > 0)

    # Gradients for W1 and b1
    grad_W1 = (G @ conv_flat.T) / n
    grad_b1 = np.sum(G, axis=1, keepdims=True) / n

    # Backpropagate to conv_flat
    G_batch = W1.T @ G

    # Backpropagate through ReLU after convolution
    G_batch = G_batch * (conv_flat > 0)

    # Undo flattening: (n_p*nf, n) -> (n_p, nf, n)
    GG = G_batch.reshape((n_p, nf, n), order="C")

    # Gradient for convolution bias b0
    grad_b0 = np.sum(GG, axis=(0, 2), keepdims=True) / n
    grad_b0 = grad_b0.reshape((nf, 1))

    # Gradient for flattened convolution filters
    MXt = np.transpose(MX, (1, 0, 2))
    grad_Fs_flat = np.einsum("ijn,jln->il", MXt, GG, optimize=True) / n

    # L2 regularization gradients
    grad_Fs_flat += 2 * lam * Fs_flat
    grad_W1 += 2 * lam * W1
    grad_W2 += 2 * lam * W2

    grads = {
        "Fs_flat": grad_Fs_flat,
        "b0": grad_b0,
        "W1": grad_W1,
        "W2": grad_W2,
        "b1": grad_b1,
        "b2": grad_b2,
    }

    return grads

def compute_loss(P, Y):
    # Compute the average cross-entropy loss.
    n = Y.shape[1]
    eps = 1e-15

    log_likelihood = -np.log(np.sum(Y * P, axis=0) + eps)
    loss = np.sum(log_likelihood) / n

    return loss

def compute_cost(P, Y, params, lam=0.0):
    # Compute the average cross-entropy loss plus L2 regularization.
    loss = compute_loss(P, Y)

    reg = lam * (
        np.sum(params["Fs_flat"] ** 2)
        + np.sum(params["W1"] ** 2)
        + np.sum(params["W2"] ** 2)
    )

    return loss + reg


def compute_accuracy(P, y):
    # Compute classification accuracy.
    predictions = np.argmax(P, axis=0)
    accuracy = np.mean(predictions == y)

    return accuracy

def initialize_parameters(f, nf, nh, K=10, img_size=32, channels=3, seed=None, dtype=np.float32):
    # Initialize parameters for the three-layer network using He initialization.
    rng = np.random.default_rng(seed)

    patch_dim = f * f * channels
    patches_per_side = img_size // f
    n_p = patches_per_side * patches_per_side
    d0 = n_p * nf

    params = {}

    # Convolution filters: each column is one flattened filter.
    # Fan-in for each filter is patch_dim.
    params["Fs_flat"] = (
        rng.normal(0.0, np.sqrt(2.0 / patch_dim), size=(patch_dim, nf))
        .astype(dtype)
    )

    params["b0"] = np.zeros((nf, 1), dtype=dtype)

    # Fully connected hidden layer.
    # Fan-in is d0.
    params["W1"] = (
        rng.normal(0.0, np.sqrt(2.0 / d0), size=(nh, d0))
        .astype(dtype)
    )

    params["b1"] = np.zeros((nh, 1), dtype=dtype)

    # Output layer.
    # Fan-in is nh.
    params["W2"] = (
        rng.normal(0.0, np.sqrt(2.0 / nh), size=(K, nh))
        .astype(dtype)
    )

    params["b2"] = np.zeros((K, 1), dtype=dtype)

    return params

def load_experiment_summary(result_name):
    # Load one saved experiment summary JSON file.
    summary_path = os.path.join(OUTPUT_DIR, f"{result_name}_summary.json")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Cannot find summary file: {summary_path}")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    return summary

def plot_short_run_barcharts():
    # Plot bar charts for Exercise 3 short training runs.
    # The plots are saved to assignment3_report_assets/.
    os.makedirs(REPORT_ASSET_DIR, exist_ok=True)

    experiments = [
        {
            "label": "Arch 1\nf=2, nf=3",
            "result_name": "ex3_arch1_f2_nf3_nh50",
        },
        {
            "label": "Arch 2\nf=4, nf=10",
            "result_name": "ex3_initial_f4_nf10_nh50",
        },
        {
            "label": "Arch 3\nf=8, nf=40",
            "result_name": "ex3_arch3_f8_nf40_nh50",
        },
        {
            "label": "Arch 4\nf=16, nf=160",
            "result_name": "ex3_arch4_f16_nf160_nh50",
        },
    ]

    labels = []
    test_accuracies = []
    training_times = []

    for exp in experiments:
        summary = load_experiment_summary(exp["result_name"])

        labels.append(exp["label"])
        test_accuracies.append(summary["final_metrics"]["test_acc"])
        training_times.append(summary["training_time"])

    # Bar chart 1: final test accuracy
    plt.figure(figsize=(8, 5))
    plt.bar(labels, test_accuracies)
    plt.ylabel("Final test accuracy")
    plt.xlabel("Architecture")
    plt.title("Final Test Accuracy for Short Training Runs")
    plt.ylim(0.0, max(test_accuracies) + 0.08)

    for idx, value in enumerate(test_accuracies):
        plt.text(idx, value + 0.01, f"{value:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    acc_path = os.path.join(REPORT_ASSET_DIR, "ex3_short_runs_test_accuracy.png")
    plt.savefig(acc_path, dpi=200)
    plt.close()

    # Bar chart 2: training time
    plt.figure(figsize=(8, 5))
    plt.bar(labels, training_times)
    plt.ylabel("Training time (seconds)")
    plt.xlabel("Architecture")
    plt.title("Training Time for Short Training Runs")
    plt.ylim(0.0, max(training_times) + 5.0)

    for idx, value in enumerate(training_times):
        plt.text(idx, value + 0.5, f"{value:.2f}s", ha="center", va="bottom")

    plt.tight_layout()
    time_path = os.path.join(REPORT_ASSET_DIR, "ex3_short_runs_training_time.png")
    plt.savefig(time_path, dpi=200)
    plt.close()

    print("\n[Exercise 3] Saved short-run bar charts")
    print(f"  Test accuracy chart: {acc_path}")
    print(f"  Training time chart: {time_path}")

    return acc_path, time_path

def plot_long_training_loss_curves():
    """
    Plot training and validation loss curves for Exercise 3 longer training runs.

    The saved curves use validation loss as the held-out loss during training.
    Final test metrics are still computed after training.
    """
    os.makedirs(REPORT_ASSET_DIR, exist_ok=True)

    experiments = [
        {
            "label": "f=4, nf=10",
            "result_name": "ex3_long_f4_nf10_nh50",
        },
        {
            "label": "f=8, nf=40",
            "result_name": "ex3_long_f8_nf40_nh50",
        },
        {
            "label": "f=4, nf=40",
            "result_name": "ex3_long_f4_nf40_nh50",
        },
    ]

    for exp in experiments:
        summary = load_experiment_summary(exp["result_name"])
        history = summary["history"]

        update_steps = history["update_steps"]
        train_loss = history["train_loss"]
        val_loss = history["val_loss"]

        plt.figure(figsize=(8, 5))
        plt.plot(update_steps, train_loss, marker="o", markersize=3, label="Training loss")
        plt.plot(update_steps, val_loss, marker="o", markersize=3, label="Validation loss")

        plt.xlabel("Update step")
        plt.ylabel("Loss")
        plt.title(f"Long Training Loss Curves ({exp['label']})")
        plt.legend()
        plt.tight_layout()

        safe_label = exp["label"].replace(", ", "_").replace("=", "").replace(" ", "")
        fig_path = os.path.join(
            REPORT_ASSET_DIR,
            f"ex3_long_loss_{safe_label}.png",
        )

        plt.savefig(fig_path, dpi=200)
        plt.close()

        print(f"  Saved loss curve: {fig_path}")

    print("\n[Exercise 3] Saved long training loss curves")

def plot_exercise4_loss_curves():
    """
    Plot training and test loss curves for Exercise 4.

    Compares the large network trained without label smoothing and with label smoothing.
    """
    os.makedirs(REPORT_ASSET_DIR, exist_ok=True)

    experiments = [
        {
            "label": "No label smoothing",
            "result_name": "ex4_large_no_label_smoothing",
        },
        {
            "label": "Label smoothing",
            "result_name": "ex4_large_label_smoothing",
        },
    ]

    for exp in experiments:
        summary = load_experiment_summary(exp["result_name"])
        history = summary["history"]

        update_steps = history["update_steps"]
        train_loss = history["train_loss"]
        test_loss = history["test_loss"]

        plt.figure(figsize=(8, 5))
        plt.plot(update_steps, train_loss, marker="o", markersize=3, label="Training loss")
        plt.plot(update_steps, test_loss, marker="o", markersize=3, label="Test loss")

        plt.xlabel("Update step")
        plt.ylabel("Loss")
        plt.title(f"Exercise 4 Loss Curves ({exp['label']})")
        plt.legend()
        plt.tight_layout()

        if exp["result_name"] == "ex4_large_no_label_smoothing":
            fig_name = "ex4_loss_no_label_smoothing.png"
        else:
            fig_name = "ex4_loss_label_smoothing.png"

        fig_path = os.path.join(REPORT_ASSET_DIR, fig_name)
        plt.savefig(fig_path, dpi=200)
        plt.close()

        print(f"  Saved loss curve: {fig_path}")

    print("\n[Exercise 4] Saved label smoothing loss curves")

def plot_bonus_results():
    """
    Plot final bonus experiment results.

    Generates:
    1. Training/test loss curves for the best random horizontal flip experiment.
    2. Validation/test accuracy comparison for selected baseline bonus models.
    3. Final test accuracy comparison for all bonus variants.
    4. Final test loss comparison for all bonus variants.
    """
    os.makedirs(REPORT_ASSET_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot 1: best bonus flip training/test loss curves
    # ------------------------------------------------------------------
    best_summary = load_experiment_summary("bonus_flip_f4_nf40_nh300_5cycles")
    best_history = best_summary["history"]

    update_steps = best_history["update_steps"]
    train_loss = best_history["train_loss"]
    test_loss = best_history["test_loss"]

    plt.figure(figsize=(8, 5))
    plt.plot(update_steps, train_loss, marker="o", markersize=3, label="Training loss")
    plt.plot(update_steps, test_loss, marker="o", markersize=3, label="Test loss")
    plt.xlabel("Update step")
    plt.ylabel("Loss")
    plt.title("Bonus: Best Flip Augmentation Loss Curves")
    plt.legend()
    plt.tight_layout()

    loss_path = os.path.join(REPORT_ASSET_DIR, "bonus_flip_loss_curves.png")
    plt.savefig(loss_path, dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: compact comparison among Exercise 4 and main flip models
    # ------------------------------------------------------------------
    compact_experiments = [
        {
            "label": "No LS",
            "result_name": "ex4_large_no_label_smoothing",
        },
        {
            "label": "LS ε=0.1",
            "result_name": "ex4_large_label_smoothing",
        },
        {
            "label": "Flip 4 cycles",
            "result_name": "bonus_flip_f4_nf40_nh300",
        },
        {
            "label": "Flip 5 cycles",
            "result_name": "bonus_flip_f4_nf40_nh300_5cycles",
        },
    ]

    labels = []
    test_accuracies = []
    val_accuracies = []

    for exp in compact_experiments:
        summary = load_experiment_summary(exp["result_name"])
        labels.append(exp["label"])
        test_accuracies.append(summary["final_metrics"]["test_acc"])
        val_accuracies.append(summary["final_metrics"]["val_acc"])

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, val_accuracies, width, label="Validation accuracy")
    plt.bar(x + width / 2, test_accuracies, width, label="Test accuracy")

    plt.xticks(x, labels)
    plt.ylabel("Accuracy")
    plt.xlabel("Experiment")
    plt.title("Bonus Comparison: Validation and Test Accuracy")
    plt.ylim(0.0, max(max(val_accuracies), max(test_accuracies)) + 0.08)
    plt.legend()

    for idx, value in enumerate(val_accuracies):
        plt.text(idx - width / 2, value + 0.01, f"{value:.4f}", ha="center", va="bottom")

    for idx, value in enumerate(test_accuracies):
        plt.text(idx + width / 2, value + 0.01, f"{value:.4f}", ha="center", va="bottom")

    plt.tight_layout()

    acc_path = os.path.join(REPORT_ASSET_DIR, "bonus_comparison_accuracy.png")
    plt.savefig(acc_path, dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 3: final test accuracy comparison for all bonus variants
    # ------------------------------------------------------------------
    all_experiments = [
        {
            "label": "No aug",
            "result_name": "ex4_large_no_label_smoothing",
        },
        {
            "label": "LS",
            "result_name": "ex4_large_label_smoothing",
        },
        {
            "label": "Flip\n4 cyc",
            "result_name": "bonus_flip_f4_nf40_nh300",
        },
        {
            "label": "Flip+LS",
            "result_name": "bonus_flip_label_smoothing_f4_nf40_nh300",
        },
        {
            "label": "Flip\nλ=0.0015",
            "result_name": "bonus_flip_f4_nf40_nh300_lam0015",
        },
        {
            "label": "Flip\nλ=0.0010",
            "result_name": "bonus_flip_f4_nf40_nh300_lam0010",
        },
        {
            "label": "Flip\nλ=0.0030",
            "result_name": "bonus_flip_f4_nf40_nh300_lam0030",
        },
        {
            "label": "Flip\n5 cyc",
            "result_name": "bonus_flip_f4_nf40_nh300_5cycles",
        },
        {
            "label": "5 cyc\nηmax=0.075",
            "result_name": "bonus_flip_f4_nf40_nh300_5cycles_etamax0075",
        },
        {
            "label": "5 cyc\nηmax=0.05",
            "result_name": "bonus_flip_f4_nf40_nh300_5cycles_etamax005",
        },
    ]

    labels = []
    test_accuracies = []
    test_losses = []

    for exp in all_experiments:
        summary = load_experiment_summary(exp["result_name"])
        labels.append(exp["label"])
        test_accuracies.append(summary["final_metrics"]["test_acc"])
        test_losses.append(summary["final_metrics"]["test_loss"])

    plt.figure(figsize=(12, 5))
    plt.bar(labels, test_accuracies)
    plt.ylabel("Test accuracy")
    plt.xlabel("Experiment")
    plt.title("Final Bonus Test Accuracy Comparison")
    plt.ylim(0.62, max(test_accuracies) + 0.03)
    plt.xticks(rotation=25, ha="right")

    for idx, value in enumerate(test_accuracies):
        plt.text(idx, value + 0.005, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    final_acc_path = os.path.join(REPORT_ASSET_DIR, "bonus_final_accuracy_comparison.png")
    plt.savefig(final_acc_path, dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 4: final test loss comparison for all bonus variants
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.bar(labels, test_losses)
    plt.ylabel("Test loss")
    plt.xlabel("Experiment")
    plt.title("Final Bonus Test Loss Comparison")
    plt.ylim(0.0, max(test_losses) + 0.15)
    plt.xticks(rotation=25, ha="right")

    for idx, value in enumerate(test_losses):
        plt.text(idx, value + 0.02, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    final_loss_path = os.path.join(REPORT_ASSET_DIR, "bonus_final_loss_comparison.png")
    plt.savefig(final_loss_path, dpi=200)
    plt.close()

    print("\n[Bonus] Saved bonus plots")
    print(f"  Best flip loss curves: {loss_path}")
    print(f"  Compact accuracy comparison: {acc_path}")
    print(f"  Final accuracy comparison: {final_acc_path}")
    print(f"  Final loss comparison: {final_loss_path}")

    return loss_path, acc_path, final_acc_path, final_loss_path
      
def compute_cyclic_learning_rate(t, step_size, eta_min, eta_max):
    # Compute cyclical learning rate for update step t.
    cycle_position = t % (2 * step_size)

    if cycle_position <= step_size:
        eta = eta_min + (cycle_position / step_size) * (eta_max - eta_min)
    else:
        eta = eta_max - ((cycle_position - step_size) / step_size) * (eta_max - eta_min)

    return eta
    
def compute_cyclic_learning_rate_increasing(t, step_size_1, eta_min, eta_max, n_cycles):
    """
    Compute cyclical learning rate with increasing cycle lengths.

    For cycle i, the half-cycle length is:
        step_size_i = step_size_1 * 2^i

    Each full cycle has length:
        2 * step_size_i

    """
    steps_before_cycle = 0

    for cycle_idx in range(n_cycles):
        current_step_size = step_size_1 * (2 ** cycle_idx)
        current_cycle_length = 2 * current_step_size

        if t < steps_before_cycle + current_cycle_length:
            cycle_position = t - steps_before_cycle

            if cycle_position <= current_step_size:
                eta = eta_min + (cycle_position / current_step_size) * (eta_max - eta_min)
            else:
                eta = eta_max - (
                    (cycle_position - current_step_size) / current_step_size
                ) * (eta_max - eta_min)

            return eta, cycle_idx, cycle_position, current_step_size

        steps_before_cycle += current_cycle_length

    # If t is outside the planned schedule, return eta_min safely.
    return eta_min, n_cycles - 1, 2 * step_size_1 * (2 ** (n_cycles - 1)), step_size_1 * (2 ** (n_cycles - 1))

def compute_total_updates_increasing(step_size_1, n_cycles):
    # Compute total number of updates for increasing cycle length CLR.
    total_updates = 0

    for cycle_idx in range(n_cycles):
        current_step_size = step_size_1 * (2 ** cycle_idx)
        total_updates += 2 * current_step_size

    return total_updates

def update_parameters(params, grads, eta):
    # Update network parameters using gradient descent.
    for key in params:
        params[key] -= eta * grads[key]

def init_history(track_test=False):
    # Initialize training history dictionary.
    history = {
        "update_steps": [],
        "learning_rates": [],
        "train_loss": [],
        "train_cost": [],
        "train_acc": [],
        "val_loss": [],
        "val_cost": [],
        "val_acc": [],
    }

    if track_test:
        history.update({
            "test_loss": [],
            "test_cost": [],
            "test_acc": [],
        })

    return history

def evaluate_metrics(MX, Y, y, params, lam=0.0):
    # Evaluate loss, cost, and accuracy on a dataset.
    P, _ = evaluate_network(MX, params)

    loss = compute_loss(P, Y)
    cost = compute_cost(P, Y, params, lam=lam)
    acc = compute_accuracy(P, y)

    return loss, cost, acc

def check_parameter_shapes(params, f, nf, nh, K=10, img_size=32, channels=3):
    # Check whether parameter shapes match the network architecture.
    patch_dim = f * f * channels
    n_p = (img_size // f) ** 2
    d0 = n_p * nf

    expected_shapes = {
        "Fs_flat": (patch_dim, nf),
        "b0": (nf, 1),
        "W1": (nh, d0),
        "b1": (nh, 1),
        "W2": (K, nh),
        "b2": (K, 1),
    }

    for key, expected_shape in expected_shapes.items():
        actual_shape = params[key].shape
        assert actual_shape == expected_shape, (
            f"{key} has shape {actual_shape}, expected {expected_shape}"
        )

    return expected_shapes

def load_cifar_batch(filename):
    # Load one CIFAR-10 batch file.
    with open(filename, "rb") as fo:
        data_dict = pickle.load(fo, encoding="bytes")

    X = data_dict[b"data"].astype(np.float32) / 255.0
    X = X.T

    y = np.array(data_dict[b"labels"], dtype=np.int64)

    Y = np.zeros((10, y.size), dtype=np.float32)
    Y[y, np.arange(y.size)] = 1.0

    return X, Y, y


def load_all_cifar10(data_dir, val_size=1000):
    # Load all CIFAR-10 training batches and split off a validation set.
    X_list = []
    Y_list = []
    y_list = []

    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        X_batch, Y_batch, y_batch = load_cifar_batch(batch_path)

        X_list.append(X_batch)
        Y_list.append(Y_batch)
        y_list.append(y_batch)

    X_all = np.concatenate(X_list, axis=1)
    Y_all = np.concatenate(Y_list, axis=1)
    y_all = np.concatenate(y_list, axis=0)

    X_train = X_all[:, :-val_size]
    Y_train = Y_all[:, :-val_size]
    y_train = y_all[:-val_size]

    X_val = X_all[:, -val_size:]
    Y_val = Y_all[:, -val_size:]
    y_val = y_all[-val_size:]

    X_test, Y_test, y_test = load_cifar_batch(os.path.join(data_dir, "test_batch"))

    return X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test

def flat_cifar_to_images(X, img_size=32, channels=3):
    # Convert flattened CIFAR-10 data to image tensor.
    n = X.shape[1]

    X_ims = np.transpose(
        X.reshape((img_size, img_size, channels, n), order="F"),
        (1, 0, 2, 3),
    )

    return X_ims

def preprocess_data(X_train, X_val, X_test):
    # Normalize train, validation, and test data using training-set statistics.
    X_mean = np.mean(X_train, axis=1, keepdims=True)
    X_std = np.std(X_train, axis=1, keepdims=True)

    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    return (
        X_train_norm.astype(np.float32),
        X_val_norm.astype(np.float32),
        X_test_norm.astype(np.float32),
        X_mean.astype(np.float32),
        X_std.astype(np.float32),
    )

def random_horizontal_flip_flat(X_batch, img_size=32, channels=3, p=0.5, rng=None):
    # Apply random horizontal flipping to a mini-batch of flattened CIFAR-10 images.
    if rng is None:
        rng = np.random.default_rng()

    n = X_batch.shape[1]
    flip_mask = rng.random(n) < p

    X_ims = flat_cifar_to_images(X_batch, img_size=img_size, channels=channels)

    if np.any(flip_mask):
        X_ims[:, :, :, flip_mask] = X_ims[:, ::-1, :, flip_mask]

    X_aug = np.transpose(X_ims, (1, 0, 2, 3)).reshape(
        (img_size * img_size * channels, n),
        order="F",
    )

    return X_aug.astype(X_batch.dtype), flip_mask

def build_augmented_MX_batch(X_batch, f, rng=None, flip_probability=0.5):
    # Apply random horizontal flip to a mini-batch and build its MX representation.
    X_aug, flip_mask = random_horizontal_flip_flat(
        X_batch,
        p=flip_probability,
        rng=rng,
    )

    X_ims_aug = flat_cifar_to_images(X_aug)
    MX_batch = build_MX(X_ims_aug, f).astype(np.float32)

    return MX_batch, flip_mask

def run_data_loading_check(data_dir="./Datasets/cifar-10-batches-py", val_size=1000):
    # Check CIFAR-10 loading and preprocessing.
    (
        X_train,
        Y_train,
        y_train,
        X_val,
        Y_val,
        y_val,
        X_test,
        Y_test,
        y_test,
    ) = load_all_cifar10(data_dir, val_size=val_size)

    X_train, X_val, X_test, X_mean, X_std = preprocess_data(X_train, X_val, X_test)

    print("\n[CIFAR-10] Data loading and preprocessing check")
    print(f"  X_train: {X_train.shape}, {X_train.dtype}")
    print(f"  Y_train: {Y_train.shape}, {Y_train.dtype}")
    print(f"  y_train: {y_train.shape}, {y_train.dtype}")
    print(f"  X_val: {X_val.shape}, {X_val.dtype}")
    print(f"  Y_val: {Y_val.shape}, {Y_val.dtype}")
    print(f"  y_val: {y_val.shape}, {y_val.dtype}")
    print(f"  X_test: {X_test.shape}, {X_test.dtype}")
    print(f"  Y_test: {Y_test.shape}, {Y_test.dtype}")
    print(f"  y_test: {y_test.shape}, {y_test.dtype}")
    print(f"  train mean after normalization: {np.mean(X_train):.6f}")
    print(f"  train std after normalization: {np.std(X_train):.6f}")

    assert X_train.shape == (3072, 50000 - val_size)
    assert Y_train.shape == (10, 50000 - val_size)
    assert y_train.shape == (50000 - val_size,)
    assert X_val.shape == (3072, val_size)
    assert Y_val.shape == (10, val_size)
    assert y_val.shape == (val_size,)
    assert X_test.shape == (3072, 10000)
    assert Y_test.shape == (10, 10000)
    assert y_test.shape == (10000,)

    print("  PASSED")

    return X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test

def precompute_or_load_MX(X, f, split_name, cache_dir=CACHE_DIR, force_recompute=False):
    # Precompute or load the MX representation for a dataset split.
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"MX_f{f}_{split_name}.npz")

    if os.path.exists(cache_path) and not force_recompute:
        print(f"  Loading cached MX from {cache_path}")
        return np.load(cache_path)["MX"]

    print(f"  Computing MX for split='{split_name}', f={f}, X shape={X.shape}")

    start_time = time.time()

    X_ims = flat_cifar_to_images(X)
    MX = build_MX(X_ims, f).astype(np.float32)

    elapsed = time.time() - start_time

    np.savez_compressed(cache_path, MX=MX)

    print(f"  Saved MX to {cache_path}")
    print(f"  MX shape: {MX.shape}, dtype: {MX.dtype}")
    print(f"  Time: {elapsed:.2f} seconds")

    return MX

def run_mx_cache_check(data_dir="./Datasets/cifar-10-batches-py", val_size=1000, f=4):
    # Load CIFAR-10 data and precompute/load MX for train, val, and test splits.
    (
        X_train,
        Y_train,
        y_train,
        X_val,
        Y_val,
        y_val,
        X_test,
        Y_test,
        y_test,
    ) = load_all_cifar10(data_dir, val_size=val_size)

    X_train, X_val, X_test, X_mean, X_std = preprocess_data(X_train, X_val, X_test)

    print(f"\n[MX cache] Precompute/load MX for f={f}")

    MX_train = precompute_or_load_MX(X_train, f=f, split_name="train")
    MX_val = precompute_or_load_MX(X_val, f=f, split_name="val")
    MX_test = precompute_or_load_MX(X_test, f=f, split_name="test")

    n_p = (32 // f) ** 2
    patch_dim = f * f * 3

    assert MX_train.shape == (n_p, patch_dim, X_train.shape[1])
    assert MX_val.shape == (n_p, patch_dim, X_val.shape[1])
    assert MX_test.shape == (n_p, patch_dim, X_test.shape[1])

    print("  MX cache check PASSED")

    return (
        MX_train,
        Y_train,
        y_train,
        MX_val,
        Y_val,
        y_val,
        MX_test,
        Y_test,
        y_test,
    )
    
def train_model(
    MX_train,
    Y_train,
    y_train,
    MX_val,
    Y_val,
    y_val,
    params,
    lam=0.003,
    n_batch=100,
    eta_min=1e-5,
    eta_max=1e-1,
    step_size=800,
    n_cycles=3,
    eval_every=None,
    train_eval_size=5000,
    max_updates=None,
    seed=42,
    label_smoothing=0.0,
):
    # Train the network using mini-batch gradient descent and cyclical learning rates.
    rng = np.random.default_rng(seed)

    n_train = Y_train.shape[1]
    total_updates = 2 * step_size * n_cycles

    if max_updates is not None:
        total_updates = min(total_updates, max_updates)

    if eval_every is None:
        eval_every = step_size // 2

    history = init_history()

    start_time = time.time()
    update_step = 0

    # Fixed subset for less noisy but still efficient training metrics
    train_eval_size = min(train_eval_size, n_train)
    train_eval_indices = rng.choice(n_train, size=train_eval_size, replace=False)

    while update_step < total_updates:
        permutation = rng.permutation(n_train)

        for batch_start in range(0, n_train, n_batch):
            if update_step >= total_updates:
                break

            batch_indices = permutation[batch_start:batch_start + n_batch]

            MX_batch = MX_train[:, :, batch_indices]
            Y_batch = Y_train[:, batch_indices]

            eta = compute_cyclic_learning_rate(
                t=update_step,
                step_size=step_size,
                eta_min=eta_min,
                eta_max=eta_max,
            )

            P_batch, cache = evaluate_network(MX_batch, params)
            grads = compute_gradients(
                MX_batch,
                Y_batch,
                params,
                cache,
                lam=lam,
                label_smoothing=label_smoothing,
            )
            update_parameters(params, grads, eta)

            if update_step % eval_every == 0 or update_step == total_updates - 1:
                MX_train_eval = MX_train[:, :, train_eval_indices]
                Y_train_eval = Y_train[:, train_eval_indices]
                y_train_eval = y_train[train_eval_indices]

                train_loss, train_cost, train_acc = evaluate_metrics(
                    MX_train_eval,
                    Y_train_eval,
                    y_train_eval,
                    params,
                    lam=lam,
                )

                val_loss, val_cost, val_acc = evaluate_metrics(
                    MX_val,
                    Y_val,
                    y_val,
                    params,
                    lam=lam,
                )

                history["update_steps"].append(update_step)
                history["learning_rates"].append(eta)
                history["train_loss"].append(train_loss)
                history["train_cost"].append(train_cost)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_cost"].append(val_cost)
                history["val_acc"].append(val_acc)

                print(
                    f"  step={update_step:5d}/{total_updates}, "
                    f"eta={eta:.6f}, "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"train_acc={train_acc:.4f}, "
                    f"val_acc={val_acc:.4f}"
                )

            update_step += 1

    training_time = time.time() - start_time

    return params, history, training_time

def run_training_sanity_check(data_dir="./Datasets/cifar-10-batches-py", val_size=1000):
    # Run a short training sanity check for the initial Exercise 3 architecture.
    (
        X_train,
        Y_train,
        y_train,
        X_val,
        Y_val,
        y_val,
        X_test,
        Y_test,
        y_test,
    ) = load_all_cifar10(data_dir, val_size=val_size)

    X_train, X_val, X_test, X_mean, X_std = preprocess_data(X_train, X_val, X_test)

    f = 4
    nf = 10
    nh = 50

    print("\n[Training sanity check]")
    print("  Loading cached MX matrices...")

    MX_train = precompute_or_load_MX(X_train, f=f, split_name="train")
    MX_val = precompute_or_load_MX(X_val, f=f, split_name="val")

    params = initialize_parameters(f=f, nf=nf, nh=nh, seed=42, dtype=np.float32)

    params, history, training_time = train_model(
        MX_train=MX_train,
        Y_train=Y_train,
        y_train=y_train,
        MX_val=MX_val,
        Y_val=Y_val,
        y_val=y_val,
        params=params,
        lam=0.003,
        n_batch=100,
        eta_min=1e-5,
        eta_max=1e-1,
        step_size=800,
        n_cycles=3,
        eval_every=5,
        train_eval_size=1000,
        max_updates=20,
        seed=42,
    )

    print(f"  Sanity training time: {training_time:.2f} seconds")

    assert len(history["update_steps"]) > 0
    assert not np.isnan(history["train_loss"][-1])
    assert not np.isnan(history["val_loss"][-1])

    print("  Training sanity check PASSED")

def run_bonus_flip_sanity_check(data_dir="./Datasets/cifar-10-batches-py", val_size=1000):
    # Run a short sanity check for random horizontal flip augmentation.
    data = get_preprocessed_cifar10(data_dir=data_dir, val_size=val_size)

    f = 4
    nf = 40
    nh = 300

    print("\n[Bonus] Random horizontal flip sanity check")
    print(f"  Architecture: f={f}, nf={nf}, nh={nh}")

    mx_splits = get_mx_splits(data, f=f)

    params = initialize_parameters(f=f, nf=nf, nh=nh, seed=42, dtype=np.float32)

    params, history, training_time = train_model_increasing_cycles_with_flip(
        X_train=data["X_train"],
        Y_train=data["Y_train"],
        y_train=data["y_train"],
        MX_val=mx_splits["MX_val"],
        Y_val=data["Y_val"],
        y_val=data["y_val"],
        params=params,
        f=f,
        lam=0.0025,
        n_batch=100,
        eta_min=1e-5,
        eta_max=1e-1,
        step_size_1=800,
        n_cycles=4,
        eval_every=5,
        train_eval_size=1000,
        max_updates=30,
        seed=42,
        MX_test=mx_splits["MX_test"],
        Y_test=data["Y_test"],
        y_test=data["y_test"],
        label_smoothing=0.0,
        flip_probability=0.5,
    )

    print(f"  Sanity training time: {training_time:.2f} seconds")

    assert len(history["update_steps"]) > 0
    assert not np.isnan(history["train_loss"][-1])
    assert not np.isnan(history["val_loss"][-1])
    assert not np.isnan(history["test_loss"][-1])
    assert 0.0 <= history["flip_fraction"][-1] <= 1.0

    print("  Bonus flip sanity check PASSED")

def run_bonus_flip_experiment(
    data_dir="./Datasets/cifar-10-batches-py",
    val_size=1000,
    lam=0.0025,
    n_cycles=4,
    eta_max=1e-1,
    result_name=None,
):
    """
    Run the full bonus experiment with random horizontal flip augmentation.

    Architecture:
        f=4, nf=40, nh=300

    Training:
        n_cycles=4, step_size_1=800, lambda=0.0025

    This experiment uses dynamic random horizontal flipping on each training mini-batch.
    """
    if result_name is None:
        result_name = f"bonus_flip_f4_nf40_nh300_lam{lam:g}".replace(".", "p")

    data = get_preprocessed_cifar10(data_dir=data_dir, val_size=val_size)

    f = 4
    nf = 40
    nh = 300
    n_batch = 100
    eta_min = 1e-5
    step_size_1 = 800
    seed = 42
    label_smoothing = 0.0
    flip_probability = 0.5

    print(f"\n[Bonus] Full random horizontal flip experiment: {result_name}")
    print(f"  Architecture: f={f}, nf={nf}, nh={nh}")
    print(f"  lambda={lam}, n_cycles={n_cycles}, step_size_1={step_size_1}")
    print(f"  flip_probability={flip_probability}")
    print(f"  label_smoothing={label_smoothing}")

    mx_splits = get_mx_splits(data, f=f)

    total_updates = compute_total_updates_increasing(step_size_1, n_cycles)
    print(f"  Total updates: {total_updates}")

    params = initialize_parameters(f=f, nf=nf, nh=nh, seed=seed, dtype=np.float32)

    params, history, training_time = train_model_increasing_cycles_with_flip(
        X_train=data["X_train"],
        Y_train=data["Y_train"],
        y_train=data["y_train"],
        MX_val=mx_splits["MX_val"],
        Y_val=data["Y_val"],
        y_val=data["y_val"],
        params=params,
        f=f,
        lam=lam,
        n_batch=n_batch,
        eta_min=eta_min,
        eta_max=eta_max,
        step_size_1=step_size_1,
        n_cycles=n_cycles,
        eval_every=step_size_1 // 2,
        train_eval_size=5000,
        max_updates=None,
        seed=seed,
        MX_test=mx_splits["MX_test"],
        Y_test=data["Y_test"],
        y_test=data["y_test"],
        label_smoothing=label_smoothing,
        flip_probability=flip_probability,
    )

    # Final evaluation on non-augmented train subset, validation set, and test set.
    train_eval_indices = np.arange(5000)
    X_train_eval = data["X_train"][:, train_eval_indices]
    MX_train_eval = build_MX(flat_cifar_to_images(X_train_eval), f).astype(np.float32)

    train_loss, train_cost, train_acc = evaluate_metrics(
        MX_train_eval,
        data["Y_train"][:, train_eval_indices],
        data["y_train"][train_eval_indices],
        params,
        lam=lam,
    )

    val_loss, val_cost, val_acc = evaluate_metrics(
        mx_splits["MX_val"],
        data["Y_val"],
        data["y_val"],
        params,
        lam=lam,
    )

    test_loss, test_cost, test_acc = evaluate_metrics(
        mx_splits["MX_test"],
        data["Y_test"],
        data["y_test"],
        params,
        lam=lam,
    )

    final_metrics = {
        "f": f,
        "nf": nf,
        "nh": nh,
        "lam": lam,
        "label_smoothing": label_smoothing,
        "flip_probability": flip_probability,
        "n_batch": n_batch,
        "eta_min": eta_min,
        "eta_max": eta_max,
        "step_size_1": step_size_1,
        "n_cycles": n_cycles,
        "total_updates": total_updates,
        "train_loss_subset": train_loss,
        "train_cost_subset": train_cost,
        "train_acc_subset": train_acc,
        "val_loss": val_loss,
        "val_cost": val_cost,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_cost": test_cost,
        "test_acc": test_acc,
    }

    print(f"\n[Bonus] Final results: {result_name}")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Train subset accuracy: {train_acc:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test cost: {test_cost:.4f}")

    save_experiment_results(
        result_name=result_name,
        params=params,
        history=history,
        training_time=training_time,
        final_metrics=final_metrics,
    )

    return params, history, training_time, final_metrics

def run_bonus_flip_label_smoothing_experiment(
    data_dir="./Datasets/cifar-10-batches-py",
    val_size=1000,
):
    """
    Run the full bonus experiment with random horizontal flip augmentation
    and label smoothing.

    Architecture:
        f=4, nf=40, nh=300

    Training:
        n_cycles=4, step_size_1=800, lambda=0.0025

    This experiment combines dynamic random horizontal flipping with
    label smoothing epsilon=0.1.
    """
    result_name = "bonus_flip_label_smoothing_f4_nf40_nh300"

    data = get_preprocessed_cifar10(data_dir=data_dir, val_size=val_size)

    f = 4
    nf = 40
    nh = 300

    lam = 0.0025
    n_batch = 100
    eta_min = 1e-5
    eta_max = 1e-1
    step_size_1 = 800
    n_cycles = 4
    seed = 42
    label_smoothing = 0.1
    flip_probability = 0.5

    print(f"\n[Bonus] Full flip + label smoothing experiment: {result_name}")
    print(f"  Architecture: f={f}, nf={nf}, nh={nh}")
    print(f"  lambda={lam}, n_cycles={n_cycles}, step_size_1={step_size_1}")
    print(f"  flip_probability={flip_probability}")
    print(f"  label_smoothing={label_smoothing}")

    mx_splits = get_mx_splits(data, f=f)

    total_updates = compute_total_updates_increasing(step_size_1, n_cycles)
    print(f"  Total updates: {total_updates}")

    params = initialize_parameters(f=f, nf=nf, nh=nh, seed=seed, dtype=np.float32)

    params, history, training_time = train_model_increasing_cycles_with_flip(
        X_train=data["X_train"],
        Y_train=data["Y_train"],
        y_train=data["y_train"],
        MX_val=mx_splits["MX_val"],
        Y_val=data["Y_val"],
        y_val=data["y_val"],
        params=params,
        f=f,
        lam=lam,
        n_batch=n_batch,
        eta_min=eta_min,
        eta_max=eta_max,
        step_size_1=step_size_1,
        n_cycles=n_cycles,
        eval_every=step_size_1 // 2,
        train_eval_size=5000,
        max_updates=None,
        seed=seed,
        MX_test=mx_splits["MX_test"],
        Y_test=data["Y_test"],
        y_test=data["y_test"],
        label_smoothing=label_smoothing,
        flip_probability=flip_probability,
    )

    train_eval_indices = np.arange(5000)
    X_train_eval = data["X_train"][:, train_eval_indices]
    MX_train_eval = build_MX(flat_cifar_to_images(X_train_eval), f).astype(np.float32)

    train_loss, train_cost, train_acc = evaluate_metrics(
        MX_train_eval,
        data["Y_train"][:, train_eval_indices],
        data["y_train"][train_eval_indices],
        params,
        lam=lam,
    )

    val_loss, val_cost, val_acc = evaluate_metrics(
        mx_splits["MX_val"],
        data["Y_val"],
        data["y_val"],
        params,
        lam=lam,
    )

    test_loss, test_cost, test_acc = evaluate_metrics(
        mx_splits["MX_test"],
        data["Y_test"],
        data["y_test"],
        params,
        lam=lam,
    )

    final_metrics = {
        "f": f,
        "nf": nf,
        "nh": nh,
        "lam": lam,
        "label_smoothing": label_smoothing,
        "flip_probability": flip_probability,
        "n_batch": n_batch,
        "eta_min": eta_min,
        "eta_max": eta_max,
        "step_size_1": step_size_1,
        "n_cycles": n_cycles,
        "total_updates": total_updates,
        "train_loss_subset": train_loss,
        "train_cost_subset": train_cost,
        "train_acc_subset": train_acc,
        "val_loss": val_loss,
        "val_cost": val_cost,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_cost": test_cost,
        "test_acc": test_acc,
    }

    print(f"\n[Bonus] Final results: {result_name}")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Train subset accuracy: {train_acc:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test cost: {test_cost:.4f}")

    save_experiment_results(
        result_name=result_name,
        params=params,
        history=history,
        training_time=training_time,
        final_metrics=final_metrics,
    )

    return params, history, training_time, final_metrics

def convert_history_to_serializable(history):
    """
    Convert numpy values in history to Python floats/ints for JSON saving.
    """
    serializable = {}

    for key, values in history.items():
        serializable[key] = []
        for value in values:
            if isinstance(value, (np.integer,)):
                serializable[key].append(int(value))
            elif isinstance(value, (np.floating,)):
                serializable[key].append(float(value))
            else:
                serializable[key].append(value)

    return serializable


def save_experiment_results(result_name, params, history, training_time, final_metrics):
    """
    Save experiment history, final metrics, and trained parameters.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    history_json = convert_history_to_serializable(history)

    summary = {
        "result_name": result_name,
        "training_time": float(training_time),
        "final_metrics": {
            key: float(value) for key, value in final_metrics.items()
        },
        "history": history_json,
    }

    summary_path = os.path.join(OUTPUT_DIR, f"{result_name}_summary.json")
    params_path = os.path.join(OUTPUT_DIR, f"{result_name}_params.npz")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(params_path, **params)

    print(f"  Saved summary to {summary_path}")
    print(f"  Saved parameters to {params_path}")

def train_model_increasing_cycles(
    MX_train,
    Y_train,
    y_train,
    MX_val,
    Y_val,
    y_val,
    params,
    lam=0.003,
    n_batch=100,
    eta_min=1e-5,
    eta_max=1e-1,
    step_size_1=800,
    n_cycles=3,
    eval_every=None,
    train_eval_size=5000,
    max_updates=None,
    seed=42,
    MX_test=None,
    Y_test=None,
    y_test=None,
    label_smoothing=0.0,
):
    """
    Train the network using mini-batch gradient descent and cyclical learning rates
    with increasing cycle lengths.

    The half-cycle step size doubles after each cycle.

    If MX_test, Y_test, and y_test are provided, test metrics are also recorded
    during training.
    """
    rng = np.random.default_rng(seed)

    n_train = Y_train.shape[1]
    total_updates = compute_total_updates_increasing(step_size_1, n_cycles)

    if max_updates is not None:
        total_updates = min(total_updates, max_updates)

    if eval_every is None:
        eval_every = step_size_1 // 2

    track_test = MX_test is not None and Y_test is not None and y_test is not None

    history = init_history(track_test=track_test)
    history["cycle_indices"] = []
    history["cycle_step_sizes"] = []

    start_time = time.time()
    update_step = 0

    train_eval_size = min(train_eval_size, n_train)
    train_eval_indices = rng.choice(n_train, size=train_eval_size, replace=False)

    while update_step < total_updates:
        permutation = rng.permutation(n_train)

        for batch_start in range(0, n_train, n_batch):
            if update_step >= total_updates:
                break

            batch_indices = permutation[batch_start:batch_start + n_batch]

            MX_batch = MX_train[:, :, batch_indices]
            Y_batch = Y_train[:, batch_indices]

            eta, cycle_idx, cycle_position, current_step_size = (
                compute_cyclic_learning_rate_increasing(
                    t=update_step,
                    step_size_1=step_size_1,
                    eta_min=eta_min,
                    eta_max=eta_max,
                    n_cycles=n_cycles,
                )
            )

            P_batch, cache = evaluate_network(MX_batch, params)
            grads = compute_gradients(
                MX_batch,
                Y_batch,
                params,
                cache,
                lam=lam,
                label_smoothing=label_smoothing,
            )
            update_parameters(params, grads, eta)

            if update_step % eval_every == 0 or update_step == total_updates - 1:
                MX_train_eval = MX_train[:, :, train_eval_indices]
                Y_train_eval = Y_train[:, train_eval_indices]
                y_train_eval = y_train[train_eval_indices]

                train_loss, train_cost, train_acc = evaluate_metrics(
                    MX_train_eval,
                    Y_train_eval,
                    y_train_eval,
                    params,
                    lam=lam,
                )

                val_loss, val_cost, val_acc = evaluate_metrics(
                    MX_val,
                    Y_val,
                    y_val,
                    params,
                    lam=lam,
                )

                if track_test:
                    test_loss, test_cost, test_acc = evaluate_metrics(
                        MX_test,
                        Y_test,
                        y_test,
                        params,
                        lam=lam,
                    )

                history["update_steps"].append(update_step)
                history["learning_rates"].append(eta)
                history["cycle_indices"].append(cycle_idx)
                history["cycle_step_sizes"].append(current_step_size)
                history["train_loss"].append(train_loss)
                history["train_cost"].append(train_cost)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_cost"].append(val_cost)
                history["val_acc"].append(val_acc)

                if track_test:
                    history["test_loss"].append(test_loss)
                    history["test_cost"].append(test_cost)
                    history["test_acc"].append(test_acc)

                message = (
                    f"  step={update_step:5d}/{total_updates}, "
                    f"cycle={cycle_idx + 1}, "
                    f"cycle_step={current_step_size}, "
                    f"eta={eta:.6f}, "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"train_acc={train_acc:.4f}, "
                    f"val_acc={val_acc:.4f}"
                )

                if track_test:
                    message += f", test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"

                print(message)

            update_step += 1

    training_time = time.time() - start_time

    return params, history, training_time

def train_model_increasing_cycles_with_flip(
    X_train,
    Y_train,
    y_train,
    MX_val,
    Y_val,
    y_val,
    params,
    f,
    lam=0.0025,
    n_batch=100,
    eta_min=1e-5,
    eta_max=1e-1,
    step_size_1=800,
    n_cycles=4,
    eval_every=None,
    train_eval_size=5000,
    max_updates=None,
    seed=42,
    MX_test=None,
    Y_test=None,
    y_test=None,
    label_smoothing=0.0,
    flip_probability=0.5,
):
    """
    Train the network with increasing-cycle CLR and random horizontal flip augmentation.

    Training mini-batches are dynamically augmented from X_train, then converted to MX.
    Validation and test metrics use fixed precomputed MX matrices.
    """
    rng = np.random.default_rng(seed)

    n_train = Y_train.shape[1]
    total_updates = compute_total_updates_increasing(step_size_1, n_cycles)

    if max_updates is not None:
        total_updates = min(total_updates, max_updates)

    if eval_every is None:
        eval_every = step_size_1 // 2

    track_test = MX_test is not None and Y_test is not None and y_test is not None

    history = init_history(track_test=track_test)
    history["cycle_indices"] = []
    history["cycle_step_sizes"] = []
    history["flip_fraction"] = []

    start_time = time.time()
    update_step = 0

    train_eval_size = min(train_eval_size, n_train)
    train_eval_indices = rng.choice(n_train, size=train_eval_size, replace=False)

    # For training metrics, use non-augmented fixed MX from a subset.
    # This makes curves comparable to validation/test metrics.
    X_train_eval = X_train[:, train_eval_indices]
    X_train_eval_ims = flat_cifar_to_images(X_train_eval)
    MX_train_eval = build_MX(X_train_eval_ims, f).astype(np.float32)
    Y_train_eval = Y_train[:, train_eval_indices]
    y_train_eval = y_train[train_eval_indices]

    while update_step < total_updates:
        permutation = rng.permutation(n_train)

        for batch_start in range(0, n_train, n_batch):
            if update_step >= total_updates:
                break

            batch_indices = permutation[batch_start:batch_start + n_batch]

            X_batch = X_train[:, batch_indices]
            Y_batch = Y_train[:, batch_indices]

            MX_batch, flip_mask = build_augmented_MX_batch(
                X_batch,
                f=f,
                rng=rng,
                flip_probability=flip_probability,
            )

            eta, cycle_idx, cycle_position, current_step_size = (
                compute_cyclic_learning_rate_increasing(
                    t=update_step,
                    step_size_1=step_size_1,
                    eta_min=eta_min,
                    eta_max=eta_max,
                    n_cycles=n_cycles,
                )
            )

            P_batch, cache = evaluate_network(MX_batch, params)
            grads = compute_gradients(
                MX_batch,
                Y_batch,
                params,
                cache,
                lam=lam,
                label_smoothing=label_smoothing,
            )
            update_parameters(params, grads, eta)

            if update_step % eval_every == 0 or update_step == total_updates - 1:
                train_loss, train_cost, train_acc = evaluate_metrics(
                    MX_train_eval,
                    Y_train_eval,
                    y_train_eval,
                    params,
                    lam=lam,
                )

                val_loss, val_cost, val_acc = evaluate_metrics(
                    MX_val,
                    Y_val,
                    y_val,
                    params,
                    lam=lam,
                )

                if track_test:
                    test_loss, test_cost, test_acc = evaluate_metrics(
                        MX_test,
                        Y_test,
                        y_test,
                        params,
                        lam=lam,
                    )

                history["update_steps"].append(update_step)
                history["learning_rates"].append(eta)
                history["cycle_indices"].append(cycle_idx)
                history["cycle_step_sizes"].append(current_step_size)
                history["flip_fraction"].append(float(np.mean(flip_mask)))
                history["train_loss"].append(train_loss)
                history["train_cost"].append(train_cost)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_cost"].append(val_cost)
                history["val_acc"].append(val_acc)

                if track_test:
                    history["test_loss"].append(test_loss)
                    history["test_cost"].append(test_cost)
                    history["test_acc"].append(test_acc)

                message = (
                    f"  step={update_step:5d}/{total_updates}, "
                    f"cycle={cycle_idx + 1}, "
                    f"cycle_step={current_step_size}, "
                    f"eta={eta:.6f}, "
                    f"flip={np.mean(flip_mask):.2f}, "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"train_acc={train_acc:.4f}, "
                    f"val_acc={val_acc:.4f}"
                )

                if track_test:
                    message += f", test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"

                print(message)

            update_step += 1

    training_time = time.time() - start_time

    return params, history, training_time

def run_increasing_cycle_sanity_check(data_dir="./Datasets/cifar-10-batches-py", val_size=1000):
    # Run a short sanity check for increasing-cycle CLR training.
    (
        X_train,
        Y_train,
        y_train,
        X_val,
        Y_val,
        y_val,
        X_test,
        Y_test,
        y_test,
    ) = load_all_cifar10(data_dir, val_size=val_size)

    X_train, X_val, X_test, X_mean, X_std = preprocess_data(X_train, X_val, X_test)

    f = 4
    nf = 10
    nh = 50

    print("\n[Increasing-cycle sanity check]")
    print("  Loading cached MX matrices...")

    MX_train = precompute_or_load_MX(X_train, f=f, split_name="train")
    MX_val = precompute_or_load_MX(X_val, f=f, split_name="val")

    total_updates = compute_total_updates_increasing(step_size_1=800, n_cycles=3)
    print(f"  Total updates for full schedule: {total_updates}")

    assert total_updates == 11200

    params = initialize_parameters(f=f, nf=nf, nh=nh, seed=42, dtype=np.float32)

    params, history, training_time = train_model_increasing_cycles(
        MX_train=MX_train,
        Y_train=Y_train,
        y_train=y_train,
        MX_val=MX_val,
        Y_val=Y_val,
        y_val=y_val,
        params=params,
        lam=0.003,
        n_batch=100,
        eta_min=1e-5,
        eta_max=1e-1,
        step_size_1=800,
        n_cycles=3,
        eval_every=5,
        train_eval_size=1000,
        max_updates=30,
        seed=42,
        label_smoothing=0.0,
    )

    print(f"  Sanity training time: {training_time:.2f} seconds")

    assert len(history["update_steps"]) > 0
    assert not np.isnan(history["train_loss"][-1])
    assert not np.isnan(history["val_loss"][-1])

    print("  Increasing-cycle sanity check PASSED")

def run_long_training_experiment(
    result_name,
    f,
    nf,
    nh,
    data_dir="./Datasets/cifar-10-batches-py",
    val_size=1000,
    lam=0.003,
    n_batch=100,
    eta_min=1e-5,
    eta_max=1e-1,
    step_size_1=800,
    n_cycles=3,
    seed=42,
    label_smoothing=0.0,
):
    """
    Run a long training experiment using increasing cycle lengths.

    The half-cycle step size doubles after each cycle.
    """
    (
        X_train,
        Y_train,
        y_train,
        X_val,
        Y_val,
        y_val,
        X_test,
        Y_test,
        y_test,
    ) = load_all_cifar10(data_dir, val_size=val_size)

    X_train, X_val, X_test, X_mean, X_std = preprocess_data(X_train, X_val, X_test)

    print(f"\n[Exercise 3/4] Long training experiment: {result_name}")
    print(f"  Architecture: f={f}, nf={nf}, nh={nh}")
    print(f"  lambda={lam}, n_cycles={n_cycles}, step_size_1={step_size_1}")
    print(f"  label_smoothing={label_smoothing}")
    print("  Loading or computing cached MX matrices...")

    MX_train = precompute_or_load_MX(X_train, f=f, split_name="train")
    MX_val = precompute_or_load_MX(X_val, f=f, split_name="val")
    MX_test = precompute_or_load_MX(X_test, f=f, split_name="test")

    total_updates = compute_total_updates_increasing(step_size_1, n_cycles)
    print(f"  Total updates: {total_updates}")

    params = initialize_parameters(f=f, nf=nf, nh=nh, seed=seed, dtype=np.float32)

    params, history, training_time = train_model_increasing_cycles(
        MX_train=MX_train,
        Y_train=Y_train,
        y_train=y_train,
        MX_val=MX_val,
        Y_val=Y_val,
        y_val=y_val,
        params=params,
        lam=lam,
        n_batch=n_batch,
        eta_min=eta_min,
        eta_max=eta_max,
        step_size_1=step_size_1,
        n_cycles=n_cycles,
        eval_every=step_size_1 // 2,
        train_eval_size=5000,
        max_updates=None,
        seed=seed,
        MX_test=MX_test,
        Y_test=Y_test,
        y_test=y_test,
        label_smoothing=label_smoothing,
    )

    train_loss, train_cost, train_acc = evaluate_metrics(
        MX_train[:, :, :5000],
        Y_train[:, :5000],
        y_train[:5000],
        params,
        lam=lam,
    )

    val_loss, val_cost, val_acc = evaluate_metrics(
        MX_val,
        Y_val,
        y_val,
        params,
        lam=lam,
    )

    test_loss, test_cost, test_acc = evaluate_metrics(
        MX_test,
        Y_test,
        y_test,
        params,
        lam=lam,
    )

    final_metrics = {
        "f": f,
        "nf": nf,
        "nh": nh,
        "lam": lam,
        "label_smoothing": label_smoothing,
        "n_batch": n_batch,
        "eta_min": eta_min,
        "eta_max": eta_max,
        "step_size_1": step_size_1,
        "n_cycles": n_cycles,
        "total_updates": total_updates,
        "train_loss_subset": train_loss,
        "train_cost_subset": train_cost,
        "train_acc_subset": train_acc,
        "val_loss": val_loss,
        "val_cost": val_cost,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_cost": test_cost,
        "test_acc": test_acc,
    }

    print(f"\n[Exercise 3/4] Long training final results: {result_name}")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Train subset accuracy: {train_acc:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test cost: {test_cost:.4f}")

    save_experiment_results(
        result_name=result_name,
        params=params,
        history=history,
        training_time=training_time,
        final_metrics=final_metrics,
    )

    return params, history, training_time, final_metrics

def run_exercise4_experiment(label_smoothing=0.0):
    """
    Run Exercise 4 large network experiment.

    Architecture:
        f=4, nf=40, nh=300

    Training:
        n_cycles=4, step_size_1=800, lambda=0.0025

    If label_smoothing > 0, label smoothing is applied in the backward pass.
    """
    if label_smoothing > 0.0:
        result_name = "ex4_large_label_smoothing"
    else:
        result_name = "ex4_large_no_label_smoothing"

    return run_long_training_experiment(
        result_name=result_name,
        f=4,
        nf=40,
        nh=300,
        data_dir="./Datasets/cifar-10-batches-py",
        val_size=1000,
        lam=0.0025,
        n_batch=100,
        eta_min=1e-5,
        eta_max=1e-1,
        step_size_1=800,
        n_cycles=4,
        seed=42,
        label_smoothing=label_smoothing,
    )

def run_initial_convnet_experiment(data_dir="./Datasets/cifar-10-batches-py", val_size=1000):
    """
    Run the first full ConvNet experiment from Exercise 3.

    Architecture:
        f=4, nf=10, nh=50

    Training:
        n_cycles=3, step_size=800, eta_min=1e-5, eta_max=1e-1,
        n_batch=100, lambda=0.003
    """
    (
        X_train,
        Y_train,
        y_train,
        X_val,
        Y_val,
        y_val,
        X_test,
        Y_test,
        y_test,
    ) = load_all_cifar10(data_dir, val_size=val_size)

    X_train, X_val, X_test, X_mean, X_std = preprocess_data(X_train, X_val, X_test)

    f = 4
    nf = 10
    nh = 50

    print("\n[Exercise 3] Initial ConvNet experiment")
    print(f"  Architecture: f={f}, nf={nf}, nh={nh}")
    print("  Loading cached MX matrices...")

    MX_train = precompute_or_load_MX(X_train, f=f, split_name="train")
    MX_val = precompute_or_load_MX(X_val, f=f, split_name="val")
    MX_test = precompute_or_load_MX(X_test, f=f, split_name="test")

    params = initialize_parameters(f=f, nf=nf, nh=nh, seed=42, dtype=np.float32)

    params, history, training_time = train_model(
        MX_train=MX_train,
        Y_train=Y_train,
        y_train=y_train,
        MX_val=MX_val,
        Y_val=Y_val,
        y_val=y_val,
        params=params,
        lam=0.003,
        n_batch=100,
        eta_min=1e-5,
        eta_max=1e-1,
        step_size=800,
        n_cycles=3,
        eval_every=400,
        train_eval_size=5000,
        max_updates=None,
        seed=42,
    )

    train_loss, train_cost, train_acc = evaluate_metrics(
        MX_train[:, :, :5000],
        Y_train[:, :5000],
        y_train[:5000],
        params,
        lam=0.003,
    )

    val_loss, val_cost, val_acc = evaluate_metrics(
        MX_val,
        Y_val,
        y_val,
        params,
        lam=0.003,
    )

    test_loss, test_cost, test_acc = evaluate_metrics(
        MX_test,
        Y_test,
        y_test,
        params,
        lam=0.003,
    )

    final_metrics = {
        "train_loss_subset": train_loss,
        "train_cost_subset": train_cost,
        "train_acc_subset": train_acc,
        "val_loss": val_loss,
        "val_cost": val_cost,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_cost": test_cost,
        "test_acc": test_acc,
    }

    print("\n[Exercise 3] Initial ConvNet final results")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Train subset accuracy: {train_acc:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test cost: {test_cost:.4f}")

    save_experiment_results(
        result_name="ex3_initial_f4_nf10_nh50",
        params=params,
        history=history,
        training_time=training_time,
        final_metrics=final_metrics,
    )

    return params, history, training_time, final_metrics

def run_architecture_experiment(
    result_name,
    f,
    nf,
    nh,
    data_dir="./Datasets/cifar-10-batches-py",
    val_size=1000,
    lam=0.003,
    n_batch=100,
    eta_min=1e-5,
    eta_max=1e-1,
    step_size=800,
    n_cycles=3,
    seed=42,
):
    """
    Run one architecture experiment for Exercise 3.

    This function trains a ConvNet with the given architecture and saves
    the summary, history, final metrics, and trained parameters.
    """
    (
        X_train,
        Y_train,
        y_train,
        X_val,
        Y_val,
        y_val,
        X_test,
        Y_test,
        y_test,
    ) = load_all_cifar10(data_dir, val_size=val_size)

    X_train, X_val, X_test, X_mean, X_std = preprocess_data(X_train, X_val, X_test)

    print(f"\n[Exercise 3] Architecture experiment: {result_name}")
    print(f"  Architecture: f={f}, nf={nf}, nh={nh}")
    print("  Loading or computing cached MX matrices...")

    MX_train = precompute_or_load_MX(X_train, f=f, split_name="train")
    MX_val = precompute_or_load_MX(X_val, f=f, split_name="val")
    MX_test = precompute_or_load_MX(X_test, f=f, split_name="test")

    params = initialize_parameters(f=f, nf=nf, nh=nh, seed=seed, dtype=np.float32)

    params, history, training_time = train_model(
        MX_train=MX_train,
        Y_train=Y_train,
        y_train=y_train,
        MX_val=MX_val,
        Y_val=Y_val,
        y_val=y_val,
        params=params,
        lam=lam,
        n_batch=n_batch,
        eta_min=eta_min,
        eta_max=eta_max,
        step_size=step_size,
        n_cycles=n_cycles,
        eval_every=step_size // 2,
        train_eval_size=5000,
        max_updates=None,
        seed=seed,
    )

    train_loss, train_cost, train_acc = evaluate_metrics(
        MX_train[:, :, :5000],
        Y_train[:, :5000],
        y_train[:5000],
        params,
        lam=lam,
    )

    val_loss, val_cost, val_acc = evaluate_metrics(
        MX_val,
        Y_val,
        y_val,
        params,
        lam=lam,
    )

    test_loss, test_cost, test_acc = evaluate_metrics(
        MX_test,
        Y_test,
        y_test,
        params,
        lam=lam,
    )

    final_metrics = {
        "f": f,
        "nf": nf,
        "nh": nh,
        "lam": lam,
        "n_batch": n_batch,
        "eta_min": eta_min,
        "eta_max": eta_max,
        "step_size": step_size,
        "n_cycles": n_cycles,
        "train_loss_subset": train_loss,
        "train_cost_subset": train_cost,
        "train_acc_subset": train_acc,
        "val_loss": val_loss,
        "val_cost": val_cost,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_cost": test_cost,
        "test_acc": test_acc,
    }

    print(f"\n[Exercise 3] Final results: {result_name}")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Train subset accuracy: {train_acc:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test cost: {test_cost:.4f}")

    save_experiment_results(
        result_name=result_name,
        params=params,
        history=history,
        training_time=training_time,
        final_metrics=final_metrics,
    )

    return params, history, training_time, final_metrics

def get_preprocessed_cifar10(data_dir="./Datasets/cifar-10-batches-py", val_size=1000):
    """Load CIFAR-10 and apply train-set normalization."""
    (
        X_train,
        Y_train,
        y_train,
        X_val,
        Y_val,
        y_val,
        X_test,
        Y_test,
        y_test,
    ) = load_all_cifar10(data_dir, val_size=val_size)

    X_train, X_val, X_test, X_mean, X_std = preprocess_data(X_train, X_val, X_test)

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "y_train": y_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "y_val": y_val,
        "X_test": X_test,
        "Y_test": Y_test,
        "y_test": y_test,
        "X_mean": X_mean,
        "X_std": X_std,
    }


def get_mx_splits(data, f):
    """Load or compute MX matrices for train/validation/test splits."""
    return {
        "MX_train": precompute_or_load_MX(data["X_train"], f=f, split_name="train"),
        "MX_val": precompute_or_load_MX(data["X_val"], f=f, split_name="val"),
        "MX_test": precompute_or_load_MX(data["X_test"], f=f, split_name="test"),
    }


def run_debug_checks():
    ensure_dirs()
    debug_data = load_debug_info(verbose=False)

    X = debug_data["X"]
    Fs = debug_data["Fs"]

    n = X.shape[1]
    f = Fs.shape[0]
    nf = Fs.shape[3]

    X_ims = flat_cifar_to_images(X)

    print("Assignment 3 debug checks")
    print("-------------------------")
    print(f"Images: n={n}, X_ims={X_ims.shape}")
    print(f"Filters: f={f}, nf={nf}, Fs={Fs.shape}")

    conv_outputs_slow = conv_slow(X_ims, Fs)
    conv_outputs_ref = debug_data["conv_outputs"]

    max_abs_diff = np.max(np.abs(conv_outputs_slow - conv_outputs_ref))
    rel_err = relative_error(conv_outputs_slow, conv_outputs_ref)

    print("\n[Exercise 1] Slow convolution")
    print(f"  max absolute difference: {max_abs_diff:.12e}")
    print(f"  relative error: {rel_err:.12e}")

    assert conv_outputs_slow.shape == conv_outputs_ref.shape
    assert max_abs_diff < 1e-10, "Slow convolution does not match debug conv_outputs."

    print("  PASSED")

    MX = build_MX(X_ims, f)
    MX_ref = debug_data["MX"]

    mx_max_abs_diff = np.max(np.abs(MX - MX_ref))
    mx_rel_err = relative_error(MX, MX_ref)

    print("\n[Exercise 1] MX construction")
    print(f"  MX shape: {MX.shape}")
    print(f"  reference MX shape: {MX_ref.shape}")
    print(f"  max absolute difference: {mx_max_abs_diff:.12e}")
    print(f"  relative error: {mx_rel_err:.12e}")

    assert MX.shape == MX_ref.shape
    assert mx_max_abs_diff < 1e-10, "Constructed MX does not match debug MX."

    print("  PASSED")

    Fs_flat = flatten_filters(Fs)
    Fs_flat_ref = debug_data["Fs_flat"]

    fs_flat_max_abs_diff = np.max(np.abs(Fs_flat - Fs_flat_ref))
    fs_flat_rel_err = relative_error(Fs_flat, Fs_flat_ref)

    print("\n[Exercise 1] Filter flattening")
    print(f"  Fs_flat shape: {Fs_flat.shape}")
    print(f"  reference Fs_flat shape: {Fs_flat_ref.shape}")
    print(f"  max absolute difference: {fs_flat_max_abs_diff:.12e}")
    print(f"  relative error: {fs_flat_rel_err:.12e}")

    assert Fs_flat.shape == Fs_flat_ref.shape
    assert fs_flat_max_abs_diff < 1e-10, "Flattened filters do not match debug Fs_flat."

    print("  PASSED")

    conv_outputs_mat = conv_matrix_multiplication(MX, Fs_flat)
    conv_outputs_mat_ref = debug_data["conv_outputs_mat"]

    conv_mat_max_abs_diff = np.max(np.abs(conv_outputs_mat - conv_outputs_mat_ref))
    conv_mat_rel_err = relative_error(conv_outputs_mat, conv_outputs_mat_ref)

    print("\n[Exercise 1] Matrix multiplication convolution")
    print(f"  conv_outputs_mat shape: {conv_outputs_mat.shape}")
    print(f"  reference conv_outputs_mat shape: {conv_outputs_mat_ref.shape}")
    print(f"  max absolute difference: {conv_mat_max_abs_diff:.12e}")
    print(f"  relative error: {conv_mat_rel_err:.12e}")

    assert conv_outputs_mat.shape == conv_outputs_mat_ref.shape
    assert conv_mat_max_abs_diff < 1e-10, (
        "Matrix multiplication convolution does not match debug conv_outputs_mat."
    )

    print("  PASSED")

    conv_outputs_einsum = conv_einsum(MX, Fs_flat)

    einsum_vs_ref_max_abs_diff = np.max(np.abs(conv_outputs_einsum - conv_outputs_mat_ref))
    einsum_vs_ref_rel_err = relative_error(conv_outputs_einsum, conv_outputs_mat_ref)

    einsum_vs_loop_max_abs_diff = np.max(np.abs(conv_outputs_einsum - conv_outputs_mat))
    einsum_vs_loop_rel_err = relative_error(conv_outputs_einsum, conv_outputs_mat)

    print("\n[Exercise 1] Einsum convolution")
    print(f"  conv_outputs_einsum shape: {conv_outputs_einsum.shape}")
    print(f"  reference conv_outputs_mat shape: {conv_outputs_mat_ref.shape}")
    print(f"  max absolute difference vs reference: {einsum_vs_ref_max_abs_diff:.12e}")
    print(f"  relative error vs reference: {einsum_vs_ref_rel_err:.12e}")
    print(f"  max absolute difference vs loop matmul: {einsum_vs_loop_max_abs_diff:.12e}")
    print(f"  relative error vs loop matmul: {einsum_vs_loop_rel_err:.12e}")

    assert conv_outputs_einsum.shape == conv_outputs_mat_ref.shape
    assert einsum_vs_ref_max_abs_diff < 1e-10, (
        "Einsum convolution does not match debug conv_outputs_mat."
    )
    assert einsum_vs_loop_max_abs_diff < 1e-10, (
        "Einsum convolution does not match loop matrix multiplication convolution."
    )

    print("  PASSED")

    params_debug = {
        "Fs_flat": Fs_flat,
        "b0": np.zeros((Fs_flat.shape[1], 1)),
        "W1": debug_data["W1"],
        "W2": debug_data["W2"],
        "b1": debug_data["b1"],
        "b2": debug_data["b2"],
    }

    P, cache = evaluate_network(MX, params_debug)

    conv_flat_ref = debug_data["conv_flat"]
    X1_ref = np.squeeze(debug_data["X1"])
    P_ref = debug_data["P"]

    conv_flat_max_abs_diff = np.max(np.abs(cache["conv_flat"] - conv_flat_ref))
    conv_flat_rel_err = relative_error(cache["conv_flat"], conv_flat_ref)

    X1_max_abs_diff = np.max(np.abs(cache["X1"] - X1_ref))
    X1_rel_err = relative_error(cache["X1"], X1_ref)

    P_max_abs_diff = np.max(np.abs(P - P_ref))
    P_rel_err = relative_error(P, P_ref)

    print("\n[Exercise 2] Forward pass")
    print(f"  conv_flat shape: {cache['conv_flat'].shape}")
    print(f"  reference conv_flat shape: {conv_flat_ref.shape}")
    print(f"  conv_flat max absolute difference: {conv_flat_max_abs_diff:.12e}")
    print(f"  conv_flat relative error: {conv_flat_rel_err:.12e}")

    print(f"  X1 shape: {cache['X1'].shape}")
    print(f"  reference X1 shape after squeeze: {X1_ref.shape}")
    print(f"  X1 max absolute difference: {X1_max_abs_diff:.12e}")
    print(f"  X1 relative error: {X1_rel_err:.12e}")

    print(f"  P shape: {P.shape}")
    print(f"  reference P shape: {P_ref.shape}")
    print(f"  P max absolute difference: {P_max_abs_diff:.12e}")
    print(f"  P relative error: {P_rel_err:.12e}")

    assert cache["conv_flat"].shape == conv_flat_ref.shape
    assert cache["X1"].shape == X1_ref.shape
    assert P.shape == P_ref.shape

    assert conv_flat_max_abs_diff < 1e-10, "conv_flat does not match debug conv_flat."
    assert X1_max_abs_diff < 1e-10, "X1 does not match debug X1."
    assert P_max_abs_diff < 1e-10, "P does not match debug P."

    print("  PASSED")

    grads = compute_gradients(MX, debug_data["Y"], params_debug, cache)

    grad_checks = [
        ("Fs_flat", grads["Fs_flat"], debug_data["grad_Fs_flat"]),
        ("W1", grads["W1"], debug_data["grad_W1"]),
        ("W2", grads["W2"], debug_data["grad_W2"]),
        ("b1", grads["b1"], debug_data["grad_b1"]),
        ("b2", grads["b2"], debug_data["grad_b2"]),
    ]

    print("\n[Exercise 2] Backward pass")

    for name, grad, grad_ref in grad_checks:
        max_abs_diff = np.max(np.abs(grad - grad_ref))
        rel_err = relative_error(grad, grad_ref)

        print(f"  {name}:")
        print(f"    shape: {grad.shape}")
        print(f"    reference shape: {grad_ref.shape}")
        print(f"    max absolute difference: {max_abs_diff:.12e}")
        print(f"    relative error: {rel_err:.12e}")

        assert grad.shape == grad_ref.shape
        assert max_abs_diff < 1e-10, f"Gradient check failed for {name}."

    print("  b0:")
    print(f"    shape: {grads['b0'].shape}")
    print(f"    norm: {np.linalg.norm(grads['b0']):.12e}")

    assert grads["b0"].shape == params_debug["b0"].shape

    print("  PASSED")

    debug_loss = compute_loss(P, debug_data["Y"])
    debug_cost_no_reg = compute_cost(P, debug_data["Y"], params_debug, lam=0.0)
    debug_cost_with_reg = compute_cost(P, debug_data["Y"], params_debug, lam=0.003)
    debug_accuracy = compute_accuracy(P, debug_data["y"])

    print("\n[Exercise 2] Loss, cost, and accuracy sanity check")
    print(f"  loss: {debug_loss:.12f}")
    print(f"  cost with lam=0: {debug_cost_no_reg:.12f}")
    print(f"  cost with lam=0.003: {debug_cost_with_reg:.12f}")
    print(f"  accuracy: {debug_accuracy:.4f}")

    assert abs(debug_loss - debug_cost_no_reg) < 1e-12
    assert debug_cost_with_reg > debug_loss
    assert 0.0 <= debug_accuracy <= 1.0

    print("  PASSED")

    init_params = initialize_parameters(f=4, nf=10, nh=50, seed=42)
    expected_shapes = check_parameter_shapes(init_params, f=4, nf=10, nh=50)

    print("\n[Exercise 2] Parameter initialization")
    for key, shape in expected_shapes.items():
        print(f"  {key}: {init_params[key].shape}")

    print("  PASSED")

    Y_smooth = smooth_labels(debug_data["Y"], epsilon=0.1)

    print("\n[Exercise 4] Label smoothing sanity check")
    print(f"  Y_smooth shape: {Y_smooth.shape}")
    print(f"  column sums min: {np.min(np.sum(Y_smooth, axis=0)):.12f}")
    print(f"  column sums max: {np.max(np.sum(Y_smooth, axis=0)):.12f}")
    print(f"  max target value: {np.max(Y_smooth):.12f}")
    print(f"  min target value: {np.min(Y_smooth):.12f}")

    assert Y_smooth.shape == debug_data["Y"].shape
    assert np.allclose(np.sum(Y_smooth, axis=0), 1.0)
    assert np.isclose(np.max(Y_smooth), 0.9)
    assert np.isclose(np.min(Y_smooth), 0.1 / 9)

    print("  PASSED")



def main():
    parser = argparse.ArgumentParser(description="DD2424 Assignment 3 experiments")
    parser.add_argument(
        "--action",
        default="debug",
        choices=[
            "debug",
            "bonus-sanity",
            "bonus-flip",
            "plot-bonus",
            "bonus-flip-ls",
            "bonus-flip-lam0015",
            "bonus-flip-lam0010",
            "bonus-flip-lam0030",
            "bonus-flip-5cycles",
            "bonus-flip-5cycles-etamax005",
            "bonus-flip-5cycles-etamax0075",
        ],
        help="Which task to run.",
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.action == "debug":
        run_debug_checks()
    elif args.action == "bonus-flip":
        run_bonus_flip_experiment(data_dir="./Datasets/cifar-10-batches-py", val_size=1000)
    elif args.action == "plot-bonus":
        plot_bonus_results()
    elif args.action == "bonus-flip-ls":
        run_bonus_flip_label_smoothing_experiment(
            data_dir="./Datasets/cifar-10-batches-py",
            val_size=1000,
        )
    elif args.action == "bonus-flip-lam0015":
        run_bonus_flip_experiment(
            data_dir="./Datasets/cifar-10-batches-py",
            val_size=1000,
            lam=0.0015,
            result_name="bonus_flip_f4_nf40_nh300_lam0015",
        )
    elif args.action == "bonus-flip-lam0010":
        run_bonus_flip_experiment(
            data_dir="./Datasets/cifar-10-batches-py",
            val_size=1000,
            lam=0.0010,
            result_name="bonus_flip_f4_nf40_nh300_lam0010",
        )   
    elif args.action == "bonus-flip-lam0030":
        run_bonus_flip_experiment(
            data_dir="./Datasets/cifar-10-batches-py",
            val_size=1000,
            lam=0.0030,
            result_name="bonus_flip_f4_nf40_nh300_lam0030",
        )   
    elif args.action == "bonus-flip-5cycles":
        run_bonus_flip_experiment(
            data_dir="./Datasets/cifar-10-batches-py",
            val_size=1000,
            lam=0.0025,
            n_cycles=5,
            result_name="bonus_flip_f4_nf40_nh300_5cycles",
        )
    elif args.action == "bonus-flip-5cycles-etamax005":
        run_bonus_flip_experiment(
            data_dir="./Datasets/cifar-10-batches-py",
            val_size=1000,
            lam=0.0025,
            n_cycles=5,
            eta_max=5e-2,
            result_name="bonus_flip_f4_nf40_nh300_5cycles_etamax005",
        )
    elif args.action == "bonus-flip-5cycles-etamax0075":
        run_bonus_flip_experiment(
            data_dir="./Datasets/cifar-10-batches-py",
            val_size=1000,
            lam=0.0025,
            n_cycles=5,
            eta_max=7.5e-2,
            result_name="bonus_flip_f4_nf40_nh300_5cycles_etamax0075",
        )
        
if __name__ == "__main__":
    main()