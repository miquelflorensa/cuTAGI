"""
Calibrate Pre-trained TAGI Model (Global Search + Visualization)

Features:
1. Uses Differential Evolution to find global optimum for Scale/Shift.
2. Generates NLL Landscape Heatmaps to visualize the solution space.
3. Optimizes in-memory (pre-fetches logits) for high speed.
4. Handles Remax (Scale+Shift) and Softmax (Scale only, Shift=0).
"""

import os
import sys
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from scipy.optimize import differential_evolution
from scipy.stats import norm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import fire

# --- TAGI Imports ---
# Adjust these paths if your project structure is different
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import pytagi
from examples.tagi_resnet_model import resnet18_cifar10
from examples.calibration_utils import softmax_analytical

# --- Constants & Setup ---
torch.manual_seed(42)
np.random.seed(42)

NORMALIZATION_MEAN = [0.4914, 0.4822, 0.4465]
NORMALIZATION_STD = [0.2470, 0.2435, 0.2616]
NUM_BINS = 15

# =============================================================================
# 1. DATA LOADING
# =============================================================================

def custom_collate_fn(batch):
    batch_images, batch_labels = zip(*batch)
    batch_images = torch.stack(batch_images)
    batch_labels = torch.tensor(batch_labels)
    batch_images = batch_images.reshape(-1)
    batch_images = batch_images.numpy()
    return batch_images, batch_labels

def load_validation_set(batch_size: int, val_split: float = 0.1):
    from torch.utils.data import Subset
    
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
    ])
    
    full_train_set = torchvision.datasets.CIFAR10(
        root="./data/cifar", train=True, download=True, transform=transform,
    )
    
    num_train = len(full_train_set)
    num_val = int(num_train * val_split)
    
    # Deterministic split for reproducibility
    g = torch.Generator()
    g.manual_seed(42)
    indices = torch.randperm(num_train, generator=g).tolist()
    val_indices = indices[-num_val:]
    
    val_set = Subset(full_train_set, val_indices)
    
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=True,
        num_workers=0, collate_fn=custom_collate_fn,
    )
    return val_loader

def load_test_set(batch_size: int):
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
    ])
    
    test_set = torchvision.datasets.CIFAR10(
        root="./data/cifar", train=False, download=True, transform=transform,
    )
    
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, drop_last=True,
        num_workers=0, collate_fn=custom_collate_fn,
    )
    return test_loader

# =============================================================================
# 2. MATH & ANALYTICAL FUNCTIONS
# =============================================================================

def phi(x):
    return norm.pdf(x)

def Phi(x):
    return norm.cdf(x)

def remax_analytical(mu_z, sigma_z_sq):
    """Analytical transformation through ReLU and normalization with NaN protection."""
    epsilon = 1e-20
    
    # Ensure inputs are valid
    mu_z = np.nan_to_num(mu_z, nan=0.0, posinf=100.0, neginf=-100.0)
    sigma_z_sq = np.nan_to_num(sigma_z_sq, nan=1.0, posinf=1000.0, neginf=epsilon)
    sigma_z_sq = np.maximum(sigma_z_sq, epsilon)
    
    sigma_z = np.sqrt(sigma_z_sq)
    alpha = np.divide(mu_z, sigma_z)
    
    # Moments of ReLU(Z)
    mu_m = np.maximum(sigma_z * phi(alpha) + mu_z * Phi(alpha), epsilon)
    
    term1 = (mu_z**2 + sigma_z_sq) * Phi(alpha)
    term2 = mu_z * sigma_z * phi(alpha)
    sigma_m_sq = np.maximum(term1 + term2 - mu_m**2, epsilon)
    
    # Log-Normal approximation
    sigma_ln_m_sq = np.log(1 + sigma_m_sq / (mu_m**2))
    mu_ln_m = np.log(mu_m) - 0.5 * sigma_ln_m_sq
    
    # Denominator (Sum)
    mu_m_tilde = np.sum(mu_m, axis=1, keepdims=True)
    sigma_m_tilde_sq = np.sum(sigma_m_sq, axis=1, keepdims=True)
    
    sigma_ln_m_tilde_sq = np.log(1 + sigma_m_tilde_sq / (mu_m_tilde**2))
    mu_ln_m_tilde = np.log(mu_m_tilde) - 0.5 * sigma_ln_m_tilde_sq
    
    # Covariance (Assuming independence between numerator and sum for tractability)
    cov_m_m_tilde = sigma_m_sq 
    cov_ln_m_ln_m_tilde = np.log(1 + cov_m_m_tilde / (mu_m * mu_m_tilde))
    
    # Output Log-Normal
    mu_ln_a = mu_ln_m - mu_ln_m_tilde
    sigma_ln_a_sq = sigma_ln_m_sq + sigma_ln_m_tilde_sq - 2 * cov_ln_m_ln_m_tilde
    
    # Final conversion back to Normal/Real space
    sigma_ln_a_sq = np.clip(sigma_ln_a_sq, 0, 50.0) # Avoid exp overflow
    
    mu_a = np.maximum(np.exp(mu_ln_a + 0.5 * sigma_ln_a_sq), epsilon)
    sigma_a_sq = mu_a**2 * (np.exp(sigma_ln_a_sq) - 1)
    
    # Normalize Probabilities (Heuristic)
    s = np.sum(mu_a, axis=1, keepdims=True)
    mu_a = mu_a / s
    sigma_a_sq = sigma_a_sq / (s**2)
    
    return {
        "mu_a": mu_a, 
        "sigma_a_sq": sigma_a_sq,
    }

def apply_activation(mu_z, sigma_z_sq, use_remax=True):
    if use_remax:
        result = remax_analytical(mu_z, sigma_z_sq)
    else:
        result = softmax_analytical(mu_z, sigma_z_sq)
    return result["mu_a"], result["sigma_a_sq"]

def calculate_metrics(probs, predictions, labels, num_bins=NUM_BINS):
    # NaN protection
    probs = np.nan_to_num(probs, nan=1.0/probs.shape[1])
    confidences = np.max(probs, axis=1)
    
    # NLL
    epsilon = 1e-15
    probs_clipped = np.clip(probs, epsilon, 1.0)
    correct_confidences = probs_clipped[np.arange(len(probs)), labels]
    nll = -np.sum(np.log(correct_confidences))

    print(f"Labels: {labels}")
    print(f"Confidences: {confidences}")
    # ECE/ACE/MCE
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = (predictions == labels)
    
    ece = 0.0
    ace = 0.0
    mce = 0.0
    active_bins = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.astype(float).mean()
        
        if prop_in_bin > 0:
            active_bins += 1
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            absolute_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
            ece += absolute_error * prop_in_bin
            ace += absolute_error
            mce = max(mce, absolute_error)
            
    if active_bins > 0:
        ace /= active_bins
    
    return {"ece": ece, "ace": ace, "mce": mce, "nll": nll}

# =============================================================================
# 3. LOGIT COLLECTION & PROCESSING
# =============================================================================

def collect_logits(net, data_loader, batch_size, scale=1.0, shift=0.0, 
                   logit_min=-10.0, logit_max=10.0, var_max=100.0):
    """
    Collects logits and applies Scale/Shift/Clip. 
    Can be used for raw collection if logit_min/max are set wide.
    """
    net.eval()
    
    all_logits_mu = []
    all_logits_var = []
    all_labels = []
    
    for data, target in data_loader:
        # Run forward pass (GPU)
        m_pred_raw, v_pred_raw = net(data)
        
        # Reshape: [Batch*10] -> [Batch, 10]
        current_batch_size = len(target)
        m_pred_2d = m_pred_raw.reshape(current_batch_size, -1)
        v_pred_2d = v_pred_raw.reshape(current_batch_size, -1)
        
        # Separate Aleatoric/Epistemic
        m_epistemic = m_pred_2d[:, ::2]   
        v_epistemic = v_pred_2d[:, ::2]   
        m_aleatoric = m_pred_2d[:, 1::2]  
        
        # Apply Transformation
        m_epistemic_final = (m_epistemic * scale) + shift
        v_epistemic_final = v_epistemic * (scale ** 2)
        m_aleatoric_final = m_aleatoric * (scale ** 2)
        
        # Clipping
        m_epistemic_final = np.clip(m_epistemic_final, logit_min, logit_max)
        v_total = m_aleatoric_final + v_epistemic_final
        v_total = np.clip(v_total, 1e-6, var_max)
        
        all_logits_mu.append(m_epistemic_final)
        all_logits_var.append(v_total)
        all_labels.append(target.numpy())
    
    return np.concatenate(all_logits_mu, axis=0), \
           np.concatenate(all_logits_var, axis=0), \
           np.concatenate(all_labels, axis=0)

def evaluate_with_params(mu_logits, var_logits, labels, use_remax=True):
    probs, var_probs = apply_activation(mu_logits, var_logits, use_remax=use_remax)
    predictions = np.argmax(probs, axis=1)
    accuracy = np.mean(predictions == labels)
    metrics = calculate_metrics(probs, predictions, labels)
    return metrics, accuracy

# =============================================================================
# 4. OPTIMIZATION & VISUALIZATION
# =============================================================================

def plot_loss_landscape(mu_raw, var_raw, labels_raw,
                       scale_bounds, shift_bounds,
                       logit_min, logit_max, var_max,
                       use_remax, output_path):
    """Generates NLL Heatmap using pre-fetched logits."""
    print("\nGenerating Loss Landscape Heatmap...")
    
    resolution = 20
    scales = np.linspace(scale_bounds[0], scale_bounds[1], resolution)
    shifts = np.linspace(shift_bounds[0], shift_bounds[1], resolution)
    nll_grid = np.zeros((resolution, resolution))
    
    for i, s_val in enumerate(scales):
        for j, sh_val in enumerate(shifts):
            # Transform In-Memory
            mu_t = mu_raw * s_val + sh_val
            var_t = var_raw * (s_val ** 2)
            
            mu_t = np.clip(mu_t, logit_min, logit_max)
            var_t = np.clip(var_t, 1e-6, var_max)
            
            metrics, _ = evaluate_with_params(mu_t, var_t, labels_raw, use_remax=use_remax)
            nll_grid[i, j] = metrics['nll']
            
    plt.figure(figsize=(10, 8))
    # Note: imshow puts 0,0 at top-left, we want bottom-left or correct coords
    sns.heatmap(nll_grid, xticklabels=np.round(shifts, 1), yticklabels=np.round(scales, 1),
                cmap="viridis_r", annot=False)
    
    plt.xlabel("Shift")
    plt.ylabel("Scale")
    plt.title(f"NLL Landscape ({'Remax' if use_remax else 'Softmax'})")
    
    save_path = str(output_path).replace(".txt", "_landscape.png")
    plt.savefig(save_path)
    plt.close()
    print(f" Saved to {save_path}")


def optimize_calibration(net, data_loader, batch_size,
                        logit_min, logit_max, var_max,
                        scale_bounds, shift_bounds,
                        use_remax, acc_tolerance):
    
    activation_name = "Remax" if use_remax else "Softmax"
    print(f"\nOptimization Mode: {activation_name}")
    
    if not use_remax:
        print("  -> Softmax detected: Freezing Shift to 0.0")
        shift_bounds = (0.0, 0.0) 

    # 1. Pre-fetch RAW data (Infinite bounds to get pure values)
    print("  -> Pre-fetching validation logits...")
    mu_raw, var_raw, labels_raw = collect_logits(
        net, data_loader, batch_size, scale=1.0, shift=0.0,
        logit_min=-1000, logit_max=1000, var_max=10000 
    )
    
    # Baseline
    mu_base = np.clip(mu_raw, logit_min, logit_max)
    var_base = np.clip(var_raw, 1e-6, var_max)
    base_metrics, base_acc = evaluate_with_params(mu_base, var_base, labels_raw, use_remax=use_remax)
    min_acceptable_acc = base_acc - acc_tolerance
    
    print(f"  Baseline NLL: {base_metrics['nll']:.4f} | Acc: {base_acc*100:.2f}%")
    print(f"  Min Acceptable Acc: {min_acceptable_acc*100:.2f}%")

    # 2. Define Objective Function
    def objective(params):
        s_val, sh_val = params
        
        # Apply transform in-memory
        mu_t = mu_raw * s_val + sh_val
        var_t = var_raw * (s_val ** 2)
        
        mu_t = np.clip(mu_t, logit_min, logit_max)
        var_t = np.clip(var_t, 1e-6, var_max)
        
        metrics, accuracy = evaluate_with_params(mu_t, var_t, labels_raw, use_remax=use_remax)
        
        # Constraint Penalty
        if accuracy < min_acceptable_acc:
            return metrics['nll'] + 100.0 * (min_acceptable_acc - accuracy)
        return metrics['nll']

    # 3. Run Global Optimization
    print(f"  -> Running Differential Evolution...")
    bounds = [scale_bounds, shift_bounds]
    
    result = differential_evolution(
        objective, 
        bounds, 
        strategy='best1bin', 
        maxiter=50, 
        popsize=15, 
        tol=0.01,
        mutation=(0.5, 1), 
        recombination=0.7,
        disp=True
    )

    opt_scale, opt_shift = result.x
    
    # Final Eval
    mu_fin = np.clip(mu_raw * opt_scale + opt_shift, logit_min, logit_max)
    var_fin = np.clip(var_raw * (opt_scale**2), 1e-6, var_max)
    final_metrics, _ = evaluate_with_params(mu_fin, var_fin, labels_raw, use_remax=use_remax)
    
    print(f"\nOptimization results:")
    print(f"  Optimal Scale: {opt_scale:.4f}")
    print(f"  Optimal Shift: {opt_shift:.4f}")
    print(f"  Final NLL: {final_metrics['nll']:.4f}")

    return opt_scale, opt_shift, final_metrics, (mu_raw, var_raw, labels_raw)

# =============================================================================
# 5. MAIN
# =============================================================================

def main(
    model_path: str = "./checkpoints/calibration_run/best_model_1_1.bin",
    batch_size: int = 128,
    val_split: float = 0.1,
    logit_min: float = -100.0,
    logit_max: float = 100.0,
    var_max: float = 1000.0,
    scale_min: float = 0.1,
    scale_max: float = 20.0,
    shift_min: float = -10.0, 
    shift_max: float = 10.0,  # Increased max shift based on discussion
    use_remax: bool = True,
    acc_tolerance: float = 0.005,
    output_file: str = "./checkpoints/calibration_run/calibration_results.txt"
):
    print("=" * 80)
    print(f"TAGI Model Calibration (Global Search)")
    print("=" * 80)
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    print(f"Loading model: {model_path}")
    net = resnet18_cifar10(is_remax=False, gain_w=0.083, gain_b=0.083)
    net.to_device("cuda" if pytagi.cuda.is_available() else "cpu")
    net.load(model_path)
    net.eval()
    
    # Load Data
    val_loader = load_validation_set(batch_size, val_split=val_split)
    test_loader = load_test_set(batch_size)
    
    # Optimize
    opt_scale, opt_shift, best_metrics, raw_val_data = optimize_calibration(
        net, val_loader, batch_size,
        logit_min=logit_min, logit_max=logit_max, var_max=var_max,
        scale_bounds=(scale_min, scale_max),
        shift_bounds=(shift_min, shift_max),
        use_remax=use_remax,
        acc_tolerance=acc_tolerance
    )
    
    # Plot Landscape using the pre-fetched raw data
    plot_loss_landscape(
        raw_val_data[0], raw_val_data[1], raw_val_data[2],
        scale_bounds=(scale_min, scale_max),
        shift_bounds=(shift_min, shift_max),
        logit_min=logit_min, logit_max=logit_max, var_max=var_max,
        use_remax=use_remax,
        output_path=output_file
    )

    print("\n" + "=" * 80)
    print("Test Set Evaluation")
    print("=" * 80)
    
    # Collect Test Data
    test_mu, test_var, test_lbl = collect_logits(
        net, test_loader, batch_size, scale=1.0, shift=0.0,
        logit_min=logit_min, logit_max=logit_max, var_max=var_max
    )
    test_metrics_base, test_acc_base = evaluate_with_params(test_mu, test_var, test_lbl, use_remax=use_remax)
    
    # Apply Optimal Parameters
    test_mu_cal, test_var_cal, _ = collect_logits(
        net, test_loader, batch_size, scale=opt_scale, shift=opt_shift,
        logit_min=logit_min, logit_max=logit_max, var_max=var_max
    )
    test_metrics_cal, test_acc_cal = evaluate_with_params(test_mu_cal, test_var_cal, test_lbl, use_remax=use_remax)
    
    print(f"\nTest Set Results (Scale={opt_scale:.4f}, Shift={opt_shift:.4f}):")
    print(f"Accuracy: {test_acc_base*100:.2f}% -> {test_acc_cal*100:.2f}%")
    print(f"ECE:      {test_metrics_base['ece']:.4f} -> {test_metrics_cal['ece']:.4f}")
    print(f"ACE:      {test_metrics_base['ace']:.4f} -> {test_metrics_cal['ace']:.4f}")
    print(f"NLL:      {test_metrics_base['nll']:.4f} -> {test_metrics_cal['nll']:.4f}")
    
    # Save Results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"Activation: {'Remax' if use_remax else 'Softmax'}\n")
        f.write(f"Scale: {opt_scale:.6f}, Shift: {opt_shift:.6f}\n")
        f.write(f"Test ACE: {test_metrics_cal['ace']:.4f}\n")
        f.write(f"Test ECE: {test_metrics_cal['ece']:.4f}\n")
        f.write(f"Test NLL: {test_metrics_cal['nll']:.4f}\n")
        
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    fire.Fire(main)