# CIFAR-100 Training Script with PyTAGI ResNet-18
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
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

import pytagi
from examples.tagi_resnet_model import resnet18_cifar100
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    ClosedFormSoftmax,
    Conv2d,
    Linear,
    MixtureReLU,
    OutputUpdater,
    ReLU,
    Remax,
    Sequential,
)

torch.manual_seed(17)

# Constants for dataset normalization (CIFAR-100 uses same normalization as CIFAR-10)
NORMALIZATION_MEAN = [0.5071, 0.4867, 0.4408]
NORMALIZATION_STD = [0.2675, 0.2565, 0.2761]

NUM_CLASSES = 100


def one_hot_encode(labels, num_classes=NUM_CLASSES):
    """Convert labels to one-hot encoding"""
    labels = labels.clone().detach()

    labels = F.one_hot(labels, num_classes=num_classes).numpy().flatten()

    # Convert to -0.5 / 16.5 (same as CIFAR-10 training)
    labels = labels * 17 - 0.5

    return labels


def custom_collate_fn(batch):
    # batch is a list of tuples (image, label)
    batch_images, batch_labels = zip(*batch)

    # Convert to a single tensor
    batch_images = torch.stack(batch_images)
    batch_labels = torch.tensor(batch_labels)

    # Flatten images to shape (B*C*H*W,)
    batch_images = batch_images.reshape(-1)

    # Convert to numpy arrays
    batch_images = batch_images.numpy()

    return batch_images, batch_labels


def load_datasets(batch_size: int):
    """Load and transform CIFAR-100 training and test datasets."""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD
            ),
        ]
    )

    train_set = torchvision.datasets.CIFAR100(
        root="./data/cifar100",
        train=True,
        download=True,
        transform=transform_train,
    )
    test_set = torchvision.datasets.CIFAR100(
        root="./data/cifar100",
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )
    return train_loader, test_loader


def print_statistics(m_pred, v_pred):
    print("============================================")
    print("LOGITS STATISTICS")
    print("============================================")
    print("Even Stream (Epistemic):")
    print("mean: ", np.mean(m_pred[::2]))
    print("std: ", np.std(m_pred[::2]))
    print("max: ", np.max(m_pred[::2]))
    print("min: ", np.min(m_pred[::2]))
    print("Odd Stream (Aleatoric):")
    print("mean: ", np.mean(m_pred[1::2]))
    print("std: ", np.std(m_pred[1::2]))
    print("max: ", np.max(m_pred[1::2]))
    print("min: ", np.min(m_pred[1::2]))
    print("============================================")


def main(num_epochs: int = 100, batch_size: int = 128, sigma_v: float = 0.05):
    """
    Run classification training on the CIFAR-100 dataset using PyTAGI.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        sigma_v: Observation noise standard deviation
    """
    train_loader, test_loader = load_datasets(batch_size)

    # Initialize network (ResNet-18 for CIFAR-100)
    net = resnet18_cifar100(is_remax=False, gain_w=0.083, gain_b=0.083)
    net.to_device("cuda" if pytagi.cuda.is_available() else "cpu")

    out_updater = OutputUpdater(net.device)

    # Training loop
    var_y = np.full((batch_size * NUM_CLASSES,), sigma_v**2, dtype=np.float32)

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/cifar100/")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_test_error = float('inf')

    for epoch in range(num_epochs):
        net.train()
        train_error = 0
        num_train_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            # Feedforward and backward pass
            m_pred, v_pred = net(data)

            # Print statistics only for first batch of each epoch
            if batch_idx == 0 and epoch % 10 == 0:
                print_statistics(m_pred, v_pred)

            m_pred_epistemic = m_pred[::2]

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
            pred = np.reshape(m_pred_epistemic, (batch_size, NUM_CLASSES))
            label = np.argmax(pred, axis=1)
            train_error += np.sum(label != target.numpy())
            num_train_samples += len(target)

            # Update progress bar
            pbar.set_postfix(
                {"train_error": f"{train_error/num_train_samples * 100:.2f}%"}
            )

        # Testing
        net.eval()
        test_error = 0
        num_test_samples = 0

        for data, target in test_loader:
            m_pred, v_pred = net(data)
            v_total = m_pred[1::2] + v_pred[::2]
            m_pred_epistemic = m_pred[::2]

            # Calculate test error
            pred = np.reshape(m_pred_epistemic, (batch_size, NUM_CLASSES))
            label = np.argmax(pred, axis=1)
            test_error += np.sum(label != target.numpy())
            num_test_samples += len(target)

        test_error_rate = (test_error / num_test_samples) * 100
        train_error_rate = (train_error / num_train_samples) * 100
        
        print(
            f"\nEpoch {epoch+1}/{num_epochs}: "
            f"Train Error: {train_error_rate:.2f}% | "
            f"Test Error: {test_error_rate:.2f}%"
        )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            model_path = checkpoint_dir / f"model_epoch{epoch+1}.bin"
            net.save(str(model_path))
            print(f"Saved checkpoint: {model_path}")

        # Save best model
        if test_error_rate < best_test_error:
            best_test_error = test_error_rate
            best_model_path = checkpoint_dir / "best_model.bin"
            net.save(str(best_model_path))
            print(f"New best model saved with test error: {test_error_rate:.2f}%")

    print(f"\nTraining complete! Best test error: {best_test_error:.2f}%")


if __name__ == "__main__":
    fire.Fire(main)
