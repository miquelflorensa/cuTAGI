"""
Calibration Utilities for Test Model Notebook

This module provides functions that match calibrate_pretrained_model.py
for consistent calibration results across the notebook and script.
"""

import numpy as np
from scipy.stats import norm


# Constants
LOGIT_MIN = -20.0
LOGIT_MAX = 30.0
VAR_MAX = 1000.0
SCALE = 1.0  # Default scale (will be optimized)
SHIFT = 0.0  # Default shift (will be optimized)


def phi(x):
    """Probability Density Function (PDF) of the standard normal distribution."""
    return norm.pdf(x)


def Phi(x):
    """Cumulative Distribution Function (CDF) of the standard normal distribution."""
    return norm.cdf(x)


def remax_analytical(mu_z, sigma_z_sq):
    """Analytical transformation through ReLU and normalization with NaN protection.
    
    This function matches the implementation in calibrate_pretrained_model.py.
    """
    epsilon = 1e-20
    
    # Ensure inputs are valid
    mu_z = np.nan_to_num(mu_z, nan=0.0, posinf=100.0, neginf=-100.0)
    sigma_z_sq = np.nan_to_num(sigma_z_sq, nan=1.0, posinf=1000.0, neginf=epsilon)
    sigma_z_sq = np.maximum(sigma_z_sq, epsilon)
    
    sigma_z = np.sqrt(sigma_z_sq)
    alpha = np.divide(mu_z, sigma_z)
    
    mu_m = np.maximum(sigma_z * phi(alpha) + mu_z * Phi(alpha), epsilon)
    
    # Variance of ReLU output
    term1 = (mu_z**2 + sigma_z_sq) * Phi(alpha)
    term2 = mu_z * sigma_z * phi(alpha)
    sigma_m_sq = np.maximum(term1 + term2 - mu_m**2, epsilon)
    
    # Log-Normal approximation
    sigma_ln_m_sq = np.log(1 + sigma_m_sq / (mu_m**2))
    mu_ln_m = np.log(mu_m) - 0.5 * sigma_ln_m_sq
    
    # Denominator (Sum) - handle both 1D (single sample) and 2D (batch) cases
    if mu_m.ndim == 1:
        mu_m_tilde = np.sum(mu_m)
        sigma_m_tilde_sq = np.sum(sigma_m_sq)
    else:
        mu_m_tilde = np.sum(mu_m, axis=1, keepdims=True)
        sigma_m_tilde_sq = np.sum(sigma_m_sq, axis=1, keepdims=True)
    
    sigma_ln_m_tilde_sq = np.log(1 + sigma_m_tilde_sq / (mu_m_tilde**2))
    mu_ln_m_tilde = np.log(mu_m_tilde) - 0.5 * sigma_ln_m_tilde_sq
    
    # Covariance
    cov_m_m_tilde = sigma_m_sq  # Since noise is independent, cov(Mi, Msum) = Var(Mi)
    cov_ln_m_ln_m_tilde = np.log(1 + cov_m_m_tilde / (mu_m * mu_m_tilde))
    
    # Output Log-Normal
    mu_ln_a = mu_ln_m - mu_ln_m_tilde
    sigma_ln_a_sq = sigma_ln_m_sq + sigma_ln_m_tilde_sq - 2 * cov_ln_m_ln_m_tilde
    
    # Final conversion back to Normal/Real space
    # Clamp sigma_ln_a_sq to avoid exp() overflow
    # sigma_ln_a_sq = np.clip(sigma_ln_a_sq, 0, 50.0)
    
    mu_a = np.maximum(np.exp(mu_ln_a + 0.5 * sigma_ln_a_sq), epsilon)
    sigma_a_sq = mu_a**2 * (np.exp(sigma_ln_a_sq) - 1)
    
    return {
        "mu_a": mu_a, 
        "sigma_a_sq": sigma_a_sq,
    }


def softmax_analytical(mu_z, sigma_z_sq):
    """
    Calculates the moments and correlation coefficients of the probabilistic softmax output analytically.
    
    Uses numerical stability techniques including log-sum-exp centering and log-space variance computation.

    Args:
        mu_z (np.ndarray): Mean of the initial Gaussian vector Z.
        sigma_z_sq (np.ndarray): Variance of the initial Gaussian vector Z.

    Returns:
        dict: A dictionary containing the analytical results.
    """
    epsilon = 1e-20
    # Use conservative max to prevent exp(x)^2 overflow: exp(20)^2 = exp(40) ~ 2e17 (safe)
    max_exp_arg = 20.0
    max_log_arg = 40.0  # For log-space computations
    
    # Sanitize inputs
    mu_z = np.nan_to_num(mu_z, nan=0.0, posinf=50.0, neginf=-50.0)
    sigma_z_sq = np.nan_to_num(sigma_z_sq, nan=1.0, posinf=50.0, neginf=epsilon)
    sigma_z_sq = np.clip(sigma_z_sq, epsilon, 50.0)
    
    # --- LOG-SUM-EXP TRICK: Center logits to prevent exp() overflow ---
    # Subtract max along class axis before any exp() operations
    if mu_z.ndim == 1:
        mu_z_max = np.max(mu_z)
    else:
        mu_z_max = np.max(mu_z, axis=1, keepdims=True)
    
    mu_z_centered = mu_z - mu_z_max
    
    # Clamp the centered logits + variance term conservatively
    exp_arg = mu_z_centered + 0.5 * sigma_z_sq
    exp_arg = np.clip(exp_arg, -max_exp_arg, max_exp_arg)
    
    # Clamp variance for all computations
    sigma_z_sq_clamped = np.clip(sigma_z_sq, epsilon, max_exp_arg)
    
    # 1. Moments of the exponentiated variables E_i = exp(Z_i)
    mu_e = np.exp(exp_arg)
    mu_e = np.clip(mu_e, epsilon, 1e15)  # Conservative upper bound
    
    # Compute sigma_e_sq in log-space to avoid overflow:
    # sigma_e_sq = mu_e^2 * (exp(sigma_z_sq) - 1)
    # log(sigma_e_sq) = 2*log(mu_e) + log(exp(sigma_z_sq) - 1)
    # For large sigma_z_sq: log(exp(s) - 1) ≈ s
    # For small sigma_z_sq: log(exp(s) - 1) ≈ log(s)
    log_mu_e = np.log(mu_e + epsilon)
    exp_sigma_minus_1 = np.expm1(sigma_z_sq_clamped)  # exp(x) - 1, more stable for small x
    exp_sigma_minus_1 = np.clip(exp_sigma_minus_1, epsilon, 1e15)
    log_sigma_e_sq = 2 * log_mu_e + np.log(exp_sigma_minus_1 + epsilon)
    log_sigma_e_sq = np.clip(log_sigma_e_sq, -max_log_arg, max_log_arg)
    sigma_e_sq = np.exp(log_sigma_e_sq)
    sigma_e_sq = np.clip(sigma_e_sq, epsilon, 1e30)
    
    # 2. Moments of the sum E_tilde = sum(E_j)
    if mu_e.ndim == 1:
        mu_e_tilde = np.sum(mu_e)
        sigma_e_tilde_sq = np.sum(sigma_e_sq)
    else:
        mu_e_tilde = np.sum(mu_e, axis=1, keepdims=True)
        sigma_e_tilde_sq = np.sum(sigma_e_sq, axis=1, keepdims=True)
    
    # Ensure denominators are safe
    mu_e_tilde = np.clip(mu_e_tilde, epsilon, 1e30)
    sigma_e_tilde_sq = np.clip(sigma_e_tilde_sq, epsilon, 1e30)
    
    cov_e_e_tilde = sigma_e_sq  # Simplified due to independence assumption

    # 3. Moments of ln(E_tilde) assuming E_tilde is Log-Normal
    # Compute ratio in log-space: log(sigma/mu^2) = log(sigma) - 2*log(mu)
    log_mu_e_tilde = np.log(mu_e_tilde + epsilon)
    log_sigma_e_tilde_sq = np.log(sigma_e_tilde_sq + epsilon)
    log_ratio_tilde = log_sigma_e_tilde_sq - 2 * log_mu_e_tilde
    log_ratio_tilde = np.clip(log_ratio_tilde, -max_log_arg, max_log_arg)
    ratio_tilde = np.exp(log_ratio_tilde)
    ratio_tilde = np.clip(ratio_tilde, 0, 1e10)
    
    sigma_ln_e_tilde_sq = np.log1p(ratio_tilde)
    sigma_ln_e_tilde_sq = np.clip(sigma_ln_e_tilde_sq, 0, max_exp_arg)
    mu_ln_e_tilde = log_mu_e_tilde - 0.5 * sigma_ln_e_tilde_sq

    # 4. Covariance between Z_i and ln(E_tilde)
    # Compute ratio in log-space: log(cov / (mu_e * mu_e_tilde))
    log_cov = np.log(cov_e_e_tilde + epsilon)
    log_denom = log_mu_e + log_mu_e_tilde
    log_ratio_cov = log_cov - log_denom
    log_ratio_cov = np.clip(log_ratio_cov, -max_log_arg, max_log_arg)
    ratio_cov = np.exp(log_ratio_cov)
    ratio_cov = np.clip(ratio_cov, 0, 1e10)
    
    cov_z_ln_e_tilde = np.log1p(ratio_cov)
    cov_z_ln_e_tilde = np.clip(cov_z_ln_e_tilde, 0, max_exp_arg)

    # 5. Moments of the Log-Space Output ln(A_i)
    # The derivation shows ln(A_i) = Z_i - ln(E_tilde)
    # Use centered values for numerical stability
    mu_ln_a = mu_z_centered - mu_ln_e_tilde
    sigma_ln_a_sq = sigma_z_sq_clamped + sigma_ln_e_tilde_sq - 2 * cov_z_ln_e_tilde
    
    # Ensure variance is non-negative and bounded
    sigma_ln_a_sq = np.clip(sigma_ln_a_sq, 0, max_exp_arg)

    # 6. Moments of the Final Output A_i (approximated as Log-Normal)
    final_exp_arg = np.clip(mu_ln_a + 0.5 * sigma_ln_a_sq, -max_exp_arg, max_exp_arg)
    mu_a = np.exp(final_exp_arg)
    mu_a = np.nan_to_num(mu_a, nan=epsilon, posinf=1.0, neginf=epsilon)
    
    # Normalize to ensure sum(A_i) = 1
    if mu_a.ndim == 1:
        mu_a_sum = np.sum(mu_a)
    else:
        mu_a_sum = np.sum(mu_a, axis=1, keepdims=True)
    mu_a_sum = np.maximum(mu_a_sum, epsilon)
    mu_a = mu_a / mu_a_sum
    
    # Compute output variance
    var_exp_arg = np.clip(sigma_ln_a_sq, 0, max_exp_arg)
    sigma_a_sq = mu_a**2 * np.expm1(var_exp_arg)  # expm1 = exp(x) - 1
    sigma_a_sq = np.nan_to_num(sigma_a_sq, nan=epsilon, posinf=1.0, neginf=epsilon)
    sigma_a_sq = np.clip(sigma_a_sq, 0, 1.0)  # Probabilities have variance <= 0.25

    return {
        "mu_a": mu_a,
        "sigma_a_sq": sigma_a_sq,
    }


def apply_remax(mu_z, sigma_z_sq):
    """Apply remax and return only the mean probabilities."""
    result = remax_analytical(mu_z, sigma_z_sq)
    return result['mu_a'], result['sigma_a_sq']

def apply_softmax(mu_z, sigma_z_sq):
    """Apply softmax and return only the mean probabilities."""
    result = softmax_analytical(mu_z, sigma_z_sq)
    return result['mu_a'], result['sigma_a_sq']


def collect_logits(net, data_loader, batch_size, num_classes=10, scale=1.0, shift=0.0, 
                   logit_min=-40.0, logit_max=40.0, var_max=1000.0, total_uncertainty=True):
    """
    Collect logits with Centering, Scaling, and Clipping.
    
    This function matches the implementation in calibrate_pretrained_model.py.
    
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
        all_logits_var: Total variance (epistemic + aleatoric) [N, num_classes]
        all_labels: Labels [N]
    """
    net.eval()
    
    all_logits_mu = []
    all_logits_var = []
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
        m_aleatoric = m_pred_2d[:, 1::2]  # [Batch, num_classes]
        
        # --- STEP 1: LOGIT CENTERING ---
        # row_means = np.mean(m_epistemic, axis=1, keepdims=True)
        # m_epistemic = m_epistemic - row_means
        
        # --- STEP 2: SCALING & SHIFTING ---
        m_epistemic_final = (m_epistemic * scale) + shift
        
        # --- STEP 3: VARIANCE SCALING ---
        v_epistemic_final = v_epistemic * (scale ** 2)
        m_aleatoric_final = m_aleatoric * (scale ** 2)
        
        # --- STEP 4: CLIPPING ---
        # Clip Means to prevent huge inputs
        # m_epistemic_final = np.clip(m_epistemic_final, logit_min, logit_max)
        
        # Combine Variances
        if total_uncertainty:
            v_total = m_aleatoric_final + v_epistemic_final
        else:
            v_total = v_epistemic_final
        
        # CRITICAL FIX: Clip Variance to prevent overflow in remax_analytical
        v_total = np.clip(v_total, 1e-6, var_max)
        
        all_logits_mu.append(m_epistemic_final)
        all_logits_var.append(v_total)
        all_labels.append(target.numpy())
    
    return np.concatenate(all_logits_mu, axis=0), \
           np.concatenate(all_logits_var, axis=0), \
           np.concatenate(all_labels, axis=0)


def calculate_metrics(probs, predictions, labels, num_bins=15):
    """
    Calculate calibration metrics: ECE, ACE, MCE, and NLL.
    
    This function matches the implementation in calibrate_pretrained_model.py.
    
    Args:
        probs: Probability array [N, num_classes]
        predictions: Predicted class indices [N]
        labels: True labels [N]
        num_bins: Number of bins for calibration
    
    Returns:
        dict with keys: 'ece', 'ace', 'mce', 'nll'
    """
    # Handle NaNs in probs just in case
    probs = np.nan_to_num(probs, nan=1.0/probs.shape[1])
    
    confidences = np.max(probs, axis=1)
    
    # Calculate NLL
    epsilon = 1e-15
    probs_clipped = np.clip(probs, epsilon, 1.0)
    correct_confidences = probs_clipped[np.arange(len(probs)), labels]
    nll = -np.mean(np.log(correct_confidences))

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = (predictions == labels)
    
    ece = 0.0
    ace = 0.0
    mce = 0.0
    active_bins = 0
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
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
            
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)
            
    if active_bins > 0:
        ace /= active_bins
    
    return {
        "ece": ece, 
        "ace": ace, 
        "mce": mce, 
        "nll": nll,
        "bin_accuracies": np.array(bin_accuracies),
        "bin_confidences": np.array(bin_confidences),
        "bin_counts": np.array(bin_counts),

    }


def evaluate_with_params(mu_logits, var_logits, labels, num_bins=15, remax=True):
    """
    Evaluate model with given logits and compute metrics.
    
    Args:
        mu_logits: Centered and scaled epistemic means [N, num_classes]
        var_logits: Total variance [N, num_classes]
        labels: True labels [N]
        num_bins: Number of bins for calibration
    
    Returns:
        metrics: dict with calibration metrics
        accuracy: Model accuracy
    """
    if remax:
        probs, var_probs = apply_remax(mu_logits, var_logits)
    else:
        probs, var_probs = apply_softmax(mu_logits, var_logits)

    predictions = np.argmax(probs, axis=1)
    
    accuracy = np.mean(predictions == labels)
    metrics = calculate_metrics(probs, predictions, labels, num_bins=num_bins)

    

    metrics['probs'] = probs
    metrics['var_probs'] = var_probs

    return metrics, accuracy


def load_calibration_params(filepath="./checkpoints/calibration_run/calibration_results.txt"):
    """
    Load optimized scale and shift parameters from calibration results file.
    
    Args:
        filepath: Path to the calibration results file
    
    Returns:
        scale: Optimal scale value
        shift: Optimal shift value
    """
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            # Parse "Scale: X.XXXXXX, Shift: Y.YYYYYY"
            parts = first_line.split(',')
            scale = float(parts[0].split(':')[1].strip())
            shift = float(parts[1].split(':')[1].strip())
        return scale, shift
    except (FileNotFoundError, IndexError, ValueError) as e:
        print(f"Warning: Could not load calibration params from {filepath}: {e}")
        print("Using default scale=1.0, shift=0.0")
        return 1.0, 0.0
