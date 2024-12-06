# Temporary import. It will be removed in the final vserion
import os
import sys

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from pytagi import HRCSoftmaxMetric, Utils, exponential_scheduler
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    OutputUpdater,
    ReLU,
    Sequential,
    Remax,
    MixtureReLU,
    MixtureSigmoid,
    Softmax,
)


DATA_FOLDER = "./data/mnist"


# TORCH
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class TorchFNN(nn.Module):
    def __init__(self):
        super(TorchFNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
        )
        self.model.apply(initialize_weights)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.model(x)
        return x


class TorchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        self.model.apply(initialize_weights)

    def forward(self, x):
        return self.model(x)


class TorchCNNBatchNorm(nn.Module):
    def __init__(self):
        super(TorchCNNBatchNorm, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        self.model.apply(initialize_weights)

    def forward(self, x):
        return self.model(x)


import numpy as np
from scipy.stats import norm

def mixture_relu_mean_var(m_pred, v_pred):
    """
    Python implementation of mixture ReLU function to calculate mean and variance.

    Parameters:
        m_pred (np.ndarray): Mean predictions (1D array of shape batch_size * num_classes).
        v_pred (np.ndarray): Variance predictions (1D array of shape batch_size * num_classes).

    Returns:
        M_m_pred (np.ndarray): Adjusted means after applying mixture ReLU.
        M_v_pred (np.ndarray): Adjusted variances after applying mixture ReLU.
    """
    # Ensure inputs are numpy arrays
    m_pred = np.array(m_pred)
    v_pred = np.array(v_pred)

    # Initialize outputs
    M_m_pred = np.zeros_like(m_pred)
    M_v_pred = np.zeros_like(v_pred)

    # Calculate the moments for each element
    for i in range(len(m_pred)):
        std_z = np.sqrt(v_pred[i])
        alpha = m_pred[i] / std_z
        pdf_alpha = norm.pdf(alpha)  # Standard normal PDF
        cdf_alpha = norm.cdf(alpha)  # Standard normal CDF

        # Mean adjustment
        M_m_pred[i] = m_pred[i] * cdf_alpha + std_z * pdf_alpha

        # Variance adjustment
        M_v_pred[i] = -M_m_pred[i] ** 2 + 2 * M_m_pred[i] * m_pred[i] - \
                      m_pred[i] * std_z * pdf_alpha + \
                      (v_pred[i] - m_pred[i] ** 2) * cdf_alpha

    return M_m_pred, M_v_pred

def compute_remax_predictions(m_pred, v_pred, batch_size):
    """
    Compute the transformed means (A_m_pred) and variances (A_v_pred) for remax probabilities.

    Args:
        m_pred (np.ndarray): 1D array of predicted means (shape: [batch_size * num_classes]).
        v_pred (np.ndarray): 1D array of predicted variances (shape: [batch_size * num_classes]).
        batch_size (int): Number of batches.

    Returns:
        A_m_pred (np.ndarray): 1D array of transformed means (shape: [batch_size * num_classes]).
        A_v_pred (np.ndarray): 1D array of transformed variances (shape: [batch_size * num_classes]).
    """
    no = len(m_pred) // batch_size  # Number of classes per batch

    # Reshape to 2D arrays for batch processing
    m_pred = m_pred.reshape(batch_size, no)
    v_pred = v_pred.reshape(batch_size, no)

    # Step 1: Convert to log space
    v_log = np.log(1.0 + v_pred / (m_pred ** 2))
    m_log = np.log(m_pred) - 0.5 * v_log

    # Step 2: Compute log-sum-exp means and variances
    m_logsum = np.log(np.sum(np.exp(m_log + 0.5 * v_log), axis=1))  # Shape: [batch_size]
    v_logsum = np.log(1.0 + np.sum(v_pred / m_pred**2, axis=1))     # Shape: [batch_size]

    # Step 3: Compute covariance between log and log-sum
    cov_log_logsum = np.log(1.0 + v_pred / (np.expand_dims(np.sum(m_pred, axis=1), axis=1) * m_pred))

    # Step 4: Compute final means and variances
    tmp_mu = m_log - m_logsum[:, None]
    tmp_var = v_log + v_logsum[:, None] - 2 * cov_log_logsum
    A_m_pred = np.exp(tmp_mu + 0.5 * tmp_var)  # Shape: [batch_size, no]
    A_v_pred = m_pred**2 * (np.exp(tmp_var) - 1.0)  # Shape: [batch_size, no]

    # Flatten the results back to 1D arrays
    return A_m_pred.flatten(), A_v_pred.flatten()

def custom_collate_fn(batch):
    # batch is a list of tuples (image, label)
    batch_images, batch_labels = zip(*batch)

    # Convert to a single tensor and then to numpy
    batch_images = torch.stack(batch_images)
    batch_labels = torch.tensor(batch_labels)

    # Flatten images and labels to 1D
    batch_images = batch_images.numpy().reshape(len(batch_images), -1).flatten()
    batch_labels = batch_labels.numpy().flatten()

    return batch_images, batch_labels


def tagi_trainer(
    batch_size: int, num_epochs: int, device: str = "cpu", sigma_v: float = 0.2
):
    # Data loading and preprocessing
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root=DATA_FOLDER, train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root=DATA_FOLDER, train=False, transform=transform, download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=custom_collate_fn,
    )

    utils = Utils()

    # Hierachical Softmax
    # metric = HRCSoftmaxMetric(num_classes=10)

    import pytagi
    pytagi.manual_seed(42)

    TAGI_FNN = Sequential(
    Linear(784, 4096),
    ReLU(),
    Linear(4096, 4096),
    ReLU(),
    Linear(4096, 10),
    # MixtureSigmoid(),
    # Remax(),
    )

    TAGI_CNN_BATCHNORM = Sequential(
    Conv2d(1, 32, 4, padding=1, in_width=28, in_height=28, bias=False),
    ReLU(),
    BatchNorm2d(32),
    AvgPool2d(3, 2),
    Conv2d(32, 64, 5, bias=False),
    ReLU(),
    BatchNorm2d(64),
    AvgPool2d(3, 2),
    Linear(64 * 4 * 4, 256),
    ReLU(),
    Linear(256, 10),
    # MixtureSigmoid(),
    # Remax(),
    )


    net = TAGI_CNN_BATCHNORM
    net.to_device(device)
    # net.set_threads(16)
    out_updater = OutputUpdater(net.device)

    # Training
    error_rates = []
    pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in pbar:
        # count = 0

        # Decaying observation's variance
        # sigma_v = exponential_scheduler(
        #     curr_v=sigma_v, min_v=0.1, decaying_factor=0.99, curr_iter=epoch
        # )
        var_y = np.full(
            (batch_size * 10,), sigma_v**2, dtype=np.float32
        )
        net.train()
        for x, labels in train_loader:
            # Feedforward and backward pass
            m_pred, v_pred = net(x)
            # print(f"m_pred: {m_pred}")

            y = np.full((batch_size * 10,), 0.0, dtype=np.float32)
            # print(f"y: {y}")
            # print(f"labels: {labels}")

            for i in range(len(labels)):
                y[i * 10 + labels[i]] = 1.0

            # print(f"y: {y}")

            out_updater.update(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                delta_states=net.input_delta_z_buffer,
            )

            # Update parameters
            net.backward()
            net.step()

            # Take m_pred with highest value as prediction for each batch

            # M_m_pred, M_v_pred = mixture_relu_mean_var(m_pred, v_pred)
            # A_m_pred, A_v_pred = compute_remax_predictions(M_m_pred, M_v_pred, len(labels))

            error = 0
            for i in range(len(labels)):
                pred = np.argmax(m_pred[i * 10 : (i + 1) * 10])
                if pred != labels[i]:
                    error += 1
                # print(f"Predicted: {pred} | Actual: {labels[i]}")


            error_rates.append(error / len(labels))

            # Training metric
            # error_rate = metric.error_rate(m_pred, v_pred, labels)
            # error_rates.append(error_rate)

        # Averaged error
        avg_error_rate = sum(error_rates[-100:])

        # Testing
        test_error_rates = []
        net.eval()
        for x, labels in test_loader:
            m_pred, v_pred = net(x)

            # Training metric
            # error_rate = metric.error_rate(m_pred, v_pred, labels)
            # test_error_rates.append(error_rate)

            # M_m_pred, M_v_pred = mixture_relu_mean_var(m_pred, v_pred)
            # A_m_pred, A_v_pred = compute_remax_predictions(M_m_pred, M_v_pred, len(labels))

            for i in range(len(labels)):
                pred = np.argmax(m_pred[i * 10 : (i + 1) * 10])
                if pred != labels[i]:
                    test_error_rates.append(1)
                else:
                    test_error_rates.append(0)
                # print(f"Predicted: {pred} | Actual: {labels[i]}")

        test_error_rate = sum(test_error_rates) / len(test_error_rates)
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} | training error: {avg_error_rate:.2f}% | test error: {test_error_rate * 100:.2f}%",
            refresh=True,
        )
    print("Training complete.")


def torch_trainer(batch_size: int, num_epochs: int, device: str = "cpu"):
    # Hyperparameters
    learning_rate = 0.01

    # torch.set_float32_matmul_precision("high")

    # Data loading and preprocessing
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root=DATA_FOLDER, train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root=DATA_FOLDER, train=False, transform=transform, download=True
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    torch_device = torch.device(device)
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check your CUDA installation."
        )
    model = TorchFNN().to(torch_device)
    # model = torch.compile(model, mode="reduce-overhead")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    error_rates = []
    for epoch in pbar:
        model.train()
        for _, (data, target) in enumerate(train_loader):
            data = data.to(torch_device)
            target = target.to(torch_device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            train_correct = pred.eq(target.view_as(pred)).sum().item()
            error_rates.append((1.0 - (train_correct / data.shape[0])))

        # Averaged error
        avg_error_rate = sum(error_rates[-100:])

        # Evaluate the model on the test set
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(torch_device)
                target = target.to(torch_device)
                output = model(data)

                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_error_rate = (1.0 - correct / len(test_loader.dataset)) * 100
        pbar.set_description(
            f"Epoch# {epoch + 1}/{num_epochs}| training error: {avg_error_rate:.2f}% | Test error: {test_error_rate: .2f}%"
        )


def main(
    framework: str = "tagi",
    batch_size: int = 128,
    epochs: int = 10,
    device: str = "cuda",
):
    if framework == "torch":
        torch_trainer(batch_size=batch_size, num_epochs=epochs, device=device)
    elif framework == "tagi":
        tagi_trainer(batch_size=batch_size, num_epochs=epochs, device=device)
    else:
        raise RuntimeError(f"Invalid Framework: {framework}")


if __name__ == "__main__":
    fire.Fire(main)
