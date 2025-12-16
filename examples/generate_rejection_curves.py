import os
import sys

# Add the 'build' directory to sys.path
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(os.getcwd()), "build"))
)
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

import pytagi
from examples.tagi_resnet_model import resnet18_cifar10
from examples.calibration_utils import apply_remax

# Configuration
CHECKPOINT_PATH = "./checkpoints/calibration_run/best_model_1_10.bin"
BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_BINS = 15
NORMALIZATION_MEAN = [0.4914, 0.4822, 0.4465]
NORMALIZATION_STD = [0.2470, 0.2435, 0.2616]


def collect_logits_separate(net, data_loader, batch_size, num_classes=10, scale=1.0, shift=0.0, 
                            logit_min=-40.0, logit_max=40.0, var_max=1000.0):
    """
    Collect logits with Centering, Scaling, and Clipping.
    Returns both epistemic-only variance and total variance (aleatoric + epistemic).
    
    Args:
        net: The neural network model
        data_loader: DataLoader for the dataset
        batch_size: Batch size
        num_classes: Number of classes (default: 10 for CIFAR-10)
        scale: Scaling factor for logits
        shift: Shift value for logits
        logit_min: Minimum logit value for clipping
        logit_max: Maximum logit value for clipping
        var_max: Maximum variance value for clipping
    
    Returns:
        all_logits_mu: Centered and scaled epistemic means [N, num_classes]
        all_var_epistemic: Epistemic variance only [N, num_classes]
        all_var_total: Total variance (epistemic + aleatoric) [N, num_classes]
        all_labels: Labels [N]
    """
    net.eval()
    
    all_logits_mu = []
    all_var_epistemic = []
    all_var_total = []
    all_labels = []
    
    for data, target in data_loader:
        m_pred_raw, v_pred_raw = net(data)
        
        # --- RESHAPE 1D -> 2D ---
        current_batch_size = len(target)
        m_pred_2d = m_pred_raw.reshape(current_batch_size, -1)
        v_pred_2d = v_pred_raw.reshape(current_batch_size, -1)
        
        # --- SEPARATE COMPONENTS ---
        m_epistemic = m_pred_2d[:, ::2]   # [Batch, num_classes]
        v_epistemic = v_pred_2d[:, ::2]   # [Batch, num_classes]
        m_aleatoric = m_pred_2d[:, 1::2]  # [Batch, num_classes] - this is actually aleatoric variance
        
        # --- STEP 2: SCALING & SHIFTING ---
        m_epistemic_final = (m_epistemic * scale) + shift
        
        # --- STEP 3: VARIANCE SCALING ---
        v_epistemic_scaled = v_epistemic * (scale ** 2)
        m_aleatoric_scaled = m_aleatoric * (scale ** 2)
        
        # Combine for total variance
        v_total = m_aleatoric_scaled + v_epistemic_scaled
        
        # CRITICAL FIX: Clip Variances to prevent overflow in remax_analytical
        v_epistemic_scaled = np.clip(v_epistemic_scaled, 1e-6, var_max)
        v_total = np.clip(v_total, 1e-6, var_max)
        
        all_logits_mu.append(m_epistemic_final)
        all_var_epistemic.append(v_epistemic_scaled)
        all_var_total.append(v_total)
        all_labels.append(target.numpy())
    
    return (np.concatenate(all_logits_mu, axis=0), 
            np.concatenate(all_var_epistemic, axis=0),
            np.concatenate(all_var_total, axis=0),
            np.concatenate(all_labels, axis=0))


def calculate_rejection_metrics(uncertainties, predictions, labels):
    """
    Calculate accuracy vs rejection rate.
    
    Args:
        uncertainties: Array of uncertainty scores (higher = more uncertain)
        predictions: Array of predicted class indices
        labels: Array of true labels
        
    Returns:
        rejection_rates: Array of rejection rates [0, 1]
        accuracies: Array of accuracies at each rejection rate
    """
    # Sort samples by uncertainty (descending)
    sorted_indices = np.argsort(-uncertainties)
    sorted_predictions = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Calculate accuracy at each rejection point
    n_samples = len(uncertainties)
    rejection_rates = np.linspace(0, 0.95, 20) # 0% to 95% rejection
    accuracies = []
    
    for rate in rejection_rates:
        # Number of samples to reject
        n_rejected = int(n_samples * rate)
        
        # Keep the samples with lowest uncertainty (at the end of sorted array)
        if n_rejected == 0:
            kept_predictions = sorted_predictions
            kept_labels = sorted_labels
        else:
            kept_predictions = sorted_predictions[n_rejected:]
            kept_labels = sorted_labels[n_rejected:]
            
        # Calculate accuracy
        if len(kept_labels) > 0:
            acc = np.mean(kept_predictions == kept_labels)
        else:
            acc = 1.0 # Edge case
            
        accuracies.append(acc)
        
    return rejection_rates, np.array(accuracies)


def plot_rejection_curves(probs, var_probs, predictions, labels, title_suffix='', output_file='rejection_curves.png'):
    """
    Plot rejection curves for different uncertainty metrics.
    """
    print(f"Calculating rejection metrics{title_suffix}...")
    
    # 1. Calculate Metrics
    
    # A. Confidence (Max Probability) - Uncertainty is 1 - confidence
    confidences = np.max(probs, axis=1)
    uncertainty_conf = 1.0 - confidences
    
    # B. Entropy
    epsilon = 1e-10
    entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)
    
    # C. Total Variance (Sum of variances)
    total_variance = np.sum(var_probs, axis=1)
    
    # 2. Calculate Curves
    rates_conf, acc_conf = calculate_rejection_metrics(uncertainty_conf, predictions, labels)
    rates_ent, acc_ent = calculate_rejection_metrics(entropy, predictions, labels)
    rates_var, acc_var = calculate_rejection_metrics(total_variance, predictions, labels)
    
    # 3. Plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(rates_conf * 100, acc_conf * 100, marker='o', label='Confidence', linewidth=2)
    plt.plot(rates_ent * 100, acc_ent * 100, marker='s', label='Entropy', linewidth=2)
    plt.plot(rates_var * 100, acc_var * 100, marker='^', label='Total Variance', linewidth=2)
    
    plt.xlabel('Rejection Rate (%)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy vs Rejection Rate{title_suffix}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set y limit to zoom in a bit, but ensure at least 0-100 is visible if needed
    # Usually accuracy goes up, so min is at 0% rejection
    min_acc = np.min([acc_conf.min(), acc_ent.min(), acc_var.min()])
    plt.ylim(bottom=max(min_acc * 90, 0), top=102)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Rejection curves plot saved to {output_file}")


def custom_collate_fn(batch):
    batch_images, batch_labels = zip(*batch)
    batch_images = torch.stack(batch_images)
    batch_labels = torch.tensor(batch_labels)
    batch_images = batch_images.reshape(-1)
    batch_images = batch_images.numpy()
    return batch_images, batch_labels


def main():
    print(f"PyTAGI CUDA available: {pytagi.cuda.is_available()}")
    
    # 1. Data Loading
    transform_test = transforms.Compose([
        transforms.ToImage(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
    ])
    
    test_set = torchvision.datasets.CIFAR10(
        root="../data/cifar", train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, 
        drop_last=True, num_workers=1, collate_fn=custom_collate_fn
    )
    
    # 2. Model Loading
    net = resnet18_cifar10(is_remax=False, gain_w=0.083, gain_b=0.083)
    device = "cuda" if pytagi.cuda.is_available() else "cpu"
    net.to_device(device)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return
        
    net.load(CHECKPOINT_PATH)
    net.eval()
    print(f"Model loaded from {CHECKPOINT_PATH}")
    
    # 3. Get Predictions - now collecting both epistemic-only and total variance
    scale = 1.0
    shift = 0.0 # Using fixed params from notebook for consistency
    
    print("Collecting logits and metrics on test set...")
    mu_logits, var_epistemic, var_total, labels = collect_logits_separate(
        net, test_loader, BATCH_SIZE, scale=scale, shift=shift
    )
    
    # =====================================================================
    # GRAPH 1: Using Total Uncertainty (Aleatoric + Epistemic) for Remax
    # =====================================================================
    print("\n=== Remax with Total Uncertainty (Aleatoric + Epistemic) ===")
    probs_total, var_probs_total = apply_remax(mu_logits, var_total)
    predictions_total = np.argmax(probs_total, axis=1)
    accuracy_total = np.mean(predictions_total == labels)
    print(f"Accuracy (Total Uncertainty): {accuracy_total*100:.2f}%")
    
    plot_rejection_curves(
        probs_total, var_probs_total, predictions_total, labels,
        title_suffix=' (Aleatoric + Epistemic)',
        output_file='rejection_curves_total.png'
    )
    
    # =====================================================================
    # GRAPH 2: Using Only Epistemic Uncertainty for Remax
    # =====================================================================
    print("\n=== Remax with Epistemic Uncertainty Only ===")
    probs_epistemic, var_probs_epistemic = apply_remax(mu_logits, var_epistemic)
    predictions_epistemic = np.argmax(probs_epistemic, axis=1)
    accuracy_epistemic = np.mean(predictions_epistemic == labels)
    print(f"Accuracy (Epistemic Only): {accuracy_epistemic*100:.2f}%")
    
    plot_rejection_curves(
        probs_epistemic, var_probs_epistemic, predictions_epistemic, labels,
        title_suffix=' (Epistemic Only)',
        output_file='rejection_curves_epistemic.png'
    )
    
    print("\n=== Summary ===")
    print(f"Accuracy with Total Uncertainty:    {accuracy_total*100:.2f}%")
    print(f"Accuracy with Epistemic Only:       {accuracy_epistemic*100:.2f}%")


if __name__ == "__main__":
    main()
