"""
CIFAR-10 Training with Temperature Scaling and Early Stopping

This script implements:
1. Training ResNet-18 on CIFAR-10 with train/validation/test splits
2. Early stopping with patience of 15 epochs
3. Training in the logits space
4. Temperature scaling on validation set for calibration
5. Final evaluation on test set
"""

import os
import sys

# Add the 'build' directory to sys.path
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import fire
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
import shutil
from scipy.optimize import minimize
from scipy.stats import norm

import pytagi
from examples.tagi_resnet_model import resnet18_cifar10
from pytagi.nn import OutputUpdater

torch.manual_seed(42)
np.random.seed(42)

# Constants for dataset normalization
NORMALIZATION_MEAN = [0.4914, 0.4822, 0.4465]
NORMALIZATION_STD = [0.2470, 0.2435, 0.2616]

# ECE calculation constants
NUM_BINS = 15


def one_hot_encode(labels, num_classes=10):
    """Convert labels to one-hot encoding in [-3, 3] range"""
    labels = labels.clone().detach()
    labels = F.one_hot(labels, num_classes=num_classes).numpy().flatten()
    # Convert to -10 / 10
    labels = labels * 20 - 10
    return labels


def custom_collate_fn(batch):
    """Custom collate function for CIFAR-10 dataloader"""
    batch_images, batch_labels = zip(*batch)
    
    # Convert to a single tensor
    batch_images = torch.stack(batch_images)
    batch_labels = torch.tensor(batch_labels)
    
    # Flatten images to shape (B*C*H*W,)
    batch_images = batch_images.reshape(-1)
    
    # Convert to numpy arrays
    batch_images = batch_images.numpy()
    
    return batch_images, batch_labels


def load_datasets(batch_size: int, val_split: float = 0.1):
    """
    Load and split CIFAR-10 into train, validation, and test sets.
    
    Args:
        batch_size: Batch size for dataloaders
        val_split: Fraction of training data to use for validation (default: 0.1)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(
            mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD
        ),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToImage(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(
            mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD
        ),
    ])
    
    # Load full training set
    full_train_set = torchvision.datasets.CIFAR10(
        root="./data/cifar",
        train=True,
        download=True,
        transform=transform_train,
    )
    
    # Split into train and validation
    num_train = len(full_train_set)
    num_val = int(num_train * val_split)
    num_train_actual = num_train - num_val
    
    indices = torch.randperm(num_train).tolist()
    train_indices = indices[:num_train_actual]
    val_indices = indices[num_train_actual:]
    
    train_set = Subset(full_train_set, train_indices)
    
    # Validation set with test transform (no augmentation)
    val_full_set = torchvision.datasets.CIFAR10(
        root="./data/cifar",
        train=True,
        download=True,
        transform=transform_test,
    )
    val_set = Subset(val_full_set, val_indices)
    
    # Test set
    test_set = torchvision.datasets.CIFAR10(
        root="./data/cifar",
        train=False,
        download=True,
        transform=transform_test,
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # Changed from 1 to avoid GPU memory issues
        collate_fn=custom_collate_fn,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,  # Changed from 1 to avoid GPU memory issues
        collate_fn=custom_collate_fn,
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,  # Changed from 1 to avoid GPU memory issues
        collate_fn=custom_collate_fn,
    )
    
    print(f"Dataset splits:")
    print(f"  Training samples: {num_train_actual}")
    print(f"  Validation samples: {num_val}")
    print(f"  Test samples: {len(test_set)}")
    
    return train_loader, val_loader, test_loader


def calculate_ece(confidences, predictions, labels, num_bins=NUM_BINS):
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        confidences: Array of confidence scores (max probability for each prediction)
        predictions: Array of predicted class indices
        labels: Array of true labels
        num_bins: Number of bins to use for calibration
    
    Returns:
        ece: Expected Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = (predictions == labels)
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.astype(float).mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def phi(x):
    """Probability Density Function (PDF) of the standard normal distribution."""
    return norm.pdf(x)


def Phi(x):
    """Cumulative Distribution Function (CDF) of the standard normal distribution."""
    return norm.cdf(x)


def remax_analytical(mu_z, sigma_z_sq):
    """
    Calculates the moments and correlation coefficients of the Remax output analytically.
    
    This implements the analytical transformation of Gaussian distributions through
    ReLU and normalization to get probability distributions.
    
    Args:
        mu_z: Mean of the logits (shape: [batch_size, num_classes])
        sigma_z_sq: Variance of the logits (shape: [batch_size, num_classes])
    
    Returns:
        Dictionary with:
            mu_a: Mean of the output probabilities
            sigma_a_sq: Variance of the output probabilities
            mu_m: Mean after ReLU
            sigma_m_sq: Variance after ReLU
    """
    epsilon = 1e-20
    sigma_z_sq = np.maximum(sigma_z_sq, epsilon)
    sigma_z = np.sqrt(sigma_z_sq)
    alpha = np.divide(mu_z, sigma_z)
    mu_m = np.maximum(sigma_z * phi(alpha) + mu_z * Phi(alpha), epsilon)
    sigma_m_sq = np.maximum((mu_z**2 + sigma_z_sq) * Phi(alpha) + mu_z * sigma_z * phi(alpha) - mu_m**2, epsilon)
    cov_z_m = np.maximum(sigma_z_sq * Phi(alpha), epsilon)
    sigma_ln_m_sq = np.log(1 + sigma_m_sq / (mu_m**2))
    mu_ln_m = np.log(mu_m) - 0.5 * sigma_ln_m_sq
    mu_m_tilde = np.sum(mu_m, axis=1, keepdims=True)
    sigma_m_tilde_sq = np.sum(sigma_m_sq, axis=1, keepdims=True)
    cov_m_m_tilde = sigma_m_sq
    sigma_ln_m_tilde_sq = np.log(1 + sigma_m_tilde_sq / (mu_m_tilde**2))
    mu_ln_m_tilde = np.log(mu_m_tilde) - 0.5 * sigma_ln_m_tilde_sq
    cov_ln_m_ln_m_tilde = np.log(1 + cov_m_m_tilde / (mu_m * mu_m_tilde))
    mu_ln_a = mu_ln_m - mu_ln_m_tilde
    sigma_ln_a_sq = sigma_ln_m_sq + sigma_ln_m_tilde_sq - 2 * cov_ln_m_ln_m_tilde
    mu_a_lognormal = np.maximum(np.exp(mu_ln_a + 0.5 * sigma_ln_a_sq), epsilon)
    sigma_a_lognormal_sq = mu_a_lognormal**2 * (np.exp(sigma_ln_a_sq) - 1)
    mu_a = mu_a_lognormal
    sigma_a_sq = sigma_a_lognormal_sq
    cov_ln_a_ln_m = sigma_ln_m_sq - cov_ln_m_ln_m_tilde
    cov_z_a_direct = mu_a * cov_z_m * (1/(mu_m) - 1/(mu_m_tilde))
    return {
        "mu_a": mu_a, 
        "sigma_a_sq": sigma_a_sq,
        "mu_m": mu_m, 
        "sigma_m_sq": sigma_m_sq,
    }


def apply_softmax(logits):
    """Apply softmax to logits"""
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def apply_remax(mu_z, sigma_z_sq):
    """
    Apply analytical remax to get probabilities from logit distributions.
    
    Args:
        mu_z: Mean of logits (shape: [batch_size, num_classes])
        sigma_z_sq: Variance of logits (shape: [batch_size, num_classes])
    
    Returns:
        mu_a: Mean probabilities (shape: [batch_size, num_classes])
    """
    result = remax_analytical(mu_z, sigma_z_sq)
    return result["mu_a"]


def scale_logits(logits, temperature):
    """
    Scale logits by temperature.
    
    Note: When using analytical remax, we need to scale both mean and variance.
    For Gaussian distributions, scaling the mean by T scales the variance by T^2.
    
    Args:
        logits: Logits to scale (can be tuple of (mu, var) or just mu)
        temperature: Temperature parameter
    
    Returns:
        Scaled logits (same format as input)
    """
    if isinstance(logits, tuple):
        mu, var = logits
        return mu / temperature, var / (temperature ** 2)
    else:
        return logits / temperature


def evaluate_model(net, data_loader, batch_size, temperature=1.0):
    """
    Evaluate model on a dataset using analytical remax.
    
    Args:
        net: Network to evaluate
        data_loader: DataLoader for the dataset
        batch_size: Batch size
        temperature: Temperature for scaling logits
    
    Returns:
        error_rate: Classification error rate
        ece: Expected Calibration Error
        all_logits_mu: All mean logits from the model
        all_logits_var: All variance logits from the model
        all_labels: All true labels
    """
    net.eval()
    
    error_count = 0
    num_samples = 0
    
    all_logits_mu = []
    all_logits_var = []
    all_labels = []
    
    for data, target in data_loader:
        # Get predictions
        m_pred, v_pred = net(data)
        
        # Extract epistemic mean (even indices) and total variance
        m_epistemic = m_pred[::2]
        v_epistemic = v_pred[::2]
        m_aleatoric = m_pred[1::2]
        
        # Total variance is aleatoric mean + epistemic variance
        v_total = m_aleatoric + v_epistemic
        
        # Reshape to (batch_size, num_classes)
        mu_z = m_epistemic.reshape(batch_size, 10)
        sigma_z_sq = v_total.reshape(batch_size, 10)
        
        all_logits_mu.append(mu_z)
        all_logits_var.append(sigma_z_sq)
        all_labels.append(target.numpy())
        
        # For error calculation, just use mean
        predictions = np.argmax(mu_z, axis=1)
        error_count += np.sum(predictions != target.numpy())
        num_samples += len(target)
    
    # Concatenate all batches
    all_logits_mu = np.concatenate(all_logits_mu, axis=0)
    all_logits_var = np.concatenate(all_logits_var, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Apply temperature scaling to both mean and variance
    scaled_mu = all_logits_mu / temperature
    scaled_var = all_logits_var / (temperature ** 2)
    
    # Apply analytical remax to get probabilities
    probs = apply_remax(scaled_mu, scaled_var)
    
    # Get predictions and confidences
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    # Calculate metrics
    error_rate = error_count / num_samples
    ece = calculate_ece(confidences, predictions, all_labels)
    
    return error_rate, ece, all_logits_mu, all_logits_var, all_labels


def optimize_temperature(val_logits_mu, val_logits_var, val_labels):
    """
    Find optimal temperature using validation set to minimize ECE with analytical remax.
    
    Args:
        val_logits_mu: Mean logits on validation set
        val_logits_var: Variance logits on validation set
        val_labels: True labels on validation set
    
    Returns:
        optimal_temperature: Best temperature value
        best_ece: ECE with optimal temperature
    """
    def temperature_ece(temperature):
        """Objective function: ECE as a function of temperature"""
        # Scale both mean and variance
        scaled_mu = val_logits_mu / temperature[0]
        scaled_var = val_logits_var / (temperature[0] ** 2)
        
        # Apply analytical remax
        probs = apply_remax(scaled_mu, scaled_var)
        
        predictions = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        return calculate_ece(confidences, predictions, val_labels)
    
    # Search for optimal temperature
    print("\nOptimizing temperature on validation set...")
    result = minimize(
        temperature_ece,
        x0=np.array([1.0]),
        method='Nelder-Mead',
        bounds=[(0.1, 10.0)],
        options={'maxiter': 100}
    )
    
    optimal_temperature = result.x[0]
    best_ece = result.fun
    
    print(f"Optimal temperature: {optimal_temperature:.4f}")
    print(f"ECE with optimal temperature: {best_ece:.4f}")
    
    return optimal_temperature, best_ece


def main(
    num_epochs: int = 100,
    batch_size: int = 128,
    sigma_v: float = 0.05,
    early_stopping_patience: int = 15,
    val_split: float = 0.1,
    checkpoint_dir: str = "./checkpoints/calibration_run"
):
    """
    Train ResNet-18 on CIFAR-10 with early stopping and temperature scaling.
    
    Args:
        num_epochs: Maximum number of training epochs
        batch_size: Batch size for training
        sigma_v: Observation noise variance
        early_stopping_patience: Number of epochs to wait before early stopping
        val_split: Fraction of training data to use for validation
        checkpoint_dir: Directory to save checkpoints
    """
    print("=" * 80)
    print("CIFAR-10 Training with Temperature Scaling")
    print("=" * 80)
    
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    train_loader, val_loader, test_loader = load_datasets(batch_size, val_split)
    
    # Initialize network
    print("\nInitializing ResNet-18...")
    net = resnet18_cifar10(is_remax=False, gain_w=0.083, gain_b=0.083)
    net.to_device("cuda" if pytagi.cuda.is_available() else "cpu")
    
    out_updater = OutputUpdater(net.device)
    
    # Training setup
    var_y = np.full((batch_size * 10,), sigma_v**2, dtype=np.float32)
    
    # Early stopping variables
    best_val_error = float('inf')
    epochs_without_improvement = 0
    best_epoch = -1
    best_model_path = checkpoint_path / "best_model.bin"
    
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    # Training loop
    for epoch in range(num_epochs):
        net.train()
        train_error = 0
        num_train_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for _, (data, target) in enumerate(pbar):
            # Feedforward and backward pass
            m_pred, v_pred = net(data)
            
            # Extract epistemic mean for loss calculation
            m_pred_loss = m_pred[::2]
            
            # Convert labels to one-hot encoding
            y = one_hot_encode(target)
            
            # Update output layers
            out_updater.update_heteros(
                output_states=net.output_z_buffer,
                mu_obs=y,
                delta_states=net.input_delta_z_buffer,
            )
            
            # Update parameters
            net.backward()
            net.step()
            
            # Calculate error rate
            pred = np.reshape(m_pred_loss, (batch_size, 10))
            label = np.argmax(pred, axis=1)
            train_error += np.sum(label != target.numpy())
            num_train_samples += len(target)
            
            # Update progress bar
            pbar.set_postfix(
                {"train_error": f"{train_error/num_train_samples*100:.2f}%"}
            )
        
        # Validation
        val_error, val_ece, _, _, _ = evaluate_model(net, val_loader, batch_size)
        
        train_error_rate = (train_error / num_train_samples) * 100
        val_error_rate = val_error * 100
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Error: {train_error_rate:.2f}%")
        print(f"  Val Error: {val_error_rate:.2f}%")
        print(f"  Val ECE: {val_ece:.4f}")
        
        # Early stopping check
        if val_error < best_val_error:
            best_val_error = val_error
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # Save best model
            net.save(str(best_model_path))
            print(f"  ✓ New best model saved (Val Error: {val_error_rate:.2f}%)")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
        
        # Check early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation error: {best_val_error*100:.2f}% at epoch {best_epoch+1}")
            break
    
    # Load best model
    print("\n" + "=" * 80)
    print("Loading Best Model for Temperature Scaling")
    print("=" * 80)
    
    net.load(str(best_model_path))
    
    # Get validation set logits for temperature scaling
    print("\nEvaluating on validation set before temperature scaling...")
    val_error_before, val_ece_before, val_logits_mu, val_logits_var, val_labels = evaluate_model(
        net, val_loader, batch_size
    )
    print(f"Validation Error (before scaling): {val_error_before*100:.2f}%")
    print(f"Validation ECE (before scaling): {val_ece_before:.4f}")
    
    # Optimize temperature
    optimal_temp, best_ece = optimize_temperature(val_logits_mu, val_logits_var, val_labels)
    
    # Evaluate on validation set with optimal temperature
    val_error_after, val_ece_after, _, _, _ = evaluate_model(
        net, val_loader, batch_size, temperature=optimal_temp
    )
    print(f"\nValidation Error (after scaling): {val_error_after*100:.2f}%")
    print(f"Validation ECE (after scaling): {val_ece_after:.4f}")
    
    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("Final Evaluation on Test Set")
    print("=" * 80)
    
    # Without temperature scaling
    print("\nWithout temperature scaling:")
    test_error_before, test_ece_before, test_logits_mu, test_logits_var, test_labels = evaluate_model(
        net, test_loader, batch_size
    )
    print(f"  Test Error: {test_error_before*100:.2f}%")
    print(f"  Test ECE: {test_ece_before:.4f}")
    
    # With optimal temperature scaling
    print(f"\nWith temperature scaling (T={optimal_temp:.4f}):")
    test_error_after, test_ece_after, _, _, _ = evaluate_model(
        net, test_loader, batch_size, temperature=optimal_temp
    )
    print(f"  Test Error: {test_error_after*100:.2f}%")
    print(f"  Test ECE: {test_ece_after:.4f}")
    
    # Save results
    results = {
        'best_epoch': best_epoch + 1,
        'best_val_error': best_val_error * 100,
        'optimal_temperature': optimal_temp,
        'test_error_before_scaling': test_error_before * 100,
        'test_ece_before_scaling': test_ece_before,
        'test_error_after_scaling': test_error_after * 100,
        'test_ece_after_scaling': test_ece_after,
    }
    
    results_file = checkpoint_path / "results.txt"
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nResults saved to {results_file}")
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
