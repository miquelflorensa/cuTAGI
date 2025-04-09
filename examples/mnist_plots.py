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
    MixtureTanh,
    MixtureReLU,
    MixtureSigmoid,
    Tanh,
    EvenExp,
)

import pytagi

pytagi.manual_seed(42)
# Set manual seed for reproducibility
torch.manual_seed(42)

TAGI_FNN = Sequential(
    Linear(784, 4096),
    ReLU(),
    Linear(4096, 4096),
    ReLU(),
    Linear(4096, 11),
)

TAGI_CNN = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28),
    ReLU(),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5),
    ReLU(),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 256),
    ReLU(),
    Linear(256, 11),
)

TAGI_CNN_BATCHNORM = Sequential(
    Conv2d(1, 32, 4, padding=1, in_width=28, in_height=28, bias=False, gain_weight=0.1, gain_bias=0.1),
    ReLU(),
    BatchNorm2d(32),
    AvgPool2d(3, 2),
    Conv2d(32, 64, 5, bias=False, gain_weight=0.1, gain_bias=0.1),
    ReLU(),
    BatchNorm2d(64),
    AvgPool2d(3, 2),
    Linear(64 * 4 * 4, 256, gain_weight=0.1, gain_bias=0.1),
    ReLU(),
    Linear(256, 20, gain_weight=0.1, gain_bias=0.1),
    EvenExp(),
)


DATA_FOLDER = "./data/mnist"

import matplotlib.pyplot as plt
import numpy as np

def plot_class_uncertainty(image_idx, images, m_preds, v_preds, delta=0.2):
    """
    Plots class probabilities with uncertainty bars similar to your reference example.
    """
    img = images[image_idx].reshape(28, 28)
    means = m_preds[image_idx]
    stds = np.sqrt(v_preds[image_idx])  # Convert variance to std deviation
    classes = np.arange(10)

    fig, (ax_img, ax_probs) = plt.subplots(1, 2, figsize=(14, 6))

    # Image plot
    ax_img.imshow(img, cmap='gray')
    ax_img.set_title(f'Img {image_idx}, Predicted: {np.argmax(means)}, Sum of Probabilities: {np.sum(means):.2f}')
    ax_img.axis('off')

    # Probability plot
    for i in classes:
        # Uncertainty bars (μ ± σ)
        ax_probs.plot([i, i], [means[i] - stds[i], means[i] + stds[i]],
                     'b', linewidth=2, label='μ ± σ' if i == 0 else "")
        ax_probs.plot([i - delta, i + delta], [means[i] + stds[i], means[i] + stds[i]],
                     'b', linewidth=2)
        ax_probs.plot([i - delta, i + delta], [means[i] - stds[i], means[i] - stds[i]],
                     'b', linewidth=2)

        # Scatter points for probabilities
        ax_probs.scatter(i, means[i], color='r', marker='p', s=60,
                        label='Probability' if i == 0 else "")

    ax_probs.set_xticks(classes)
    ax_probs.set_xlabel('Class #')
    ax_probs.set_ylabel('Probability')
    ax_probs.set_ylim(-0.1, 1.1)
    ax_probs.grid(True, linestyle='--', alpha=0.6)
    ax_probs.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('./images/mnist_image_probs_variances_{}.png'.format(image_idx))


def find_and_plot_ambiguous_images(images, m_preds, v_preds, threshold=0.1, min_classes=2, max_images_to_plot=5):
    """
    Finds images with ambiguous predictions and plots them.

    Args:
        threshold: Minimum probability to consider a class as "active"
        min_classes: Minimum number of classes needed to trigger ambiguity (use 3 for ">2 classes")
        max_images_to_plot: Maximum number of ambiguous images to display
    """
    # Find ambiguous images
    ambiguous_indices = []
    for idx in m_preds:
        prob_mask = m_preds[idx] > threshold
        if np.sum(prob_mask) >= min_classes:
            ambiguous_indices.append(idx)


    print(f"Found {len(ambiguous_indices)} ambiguous images")

    # Plot first few ambiguous cases
    # for i, idx in enumerate(ambiguous_indices[:max_images_to_plot]):
    #     plot_image_probs_and_variances(idx, images, m_preds, v_preds, i)
    #     plt.suptitle(f"Ambiguous Image #{i+1} (Index {idx})", y=1.05)

    return ambiguous_indices

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

def probs_from_logits(m_pred, v_pred, nb_classes):
    muZ = m_pred
    varZ = v_pred

    muE = np.exp(muZ + 0.5 * varZ)
    varE = np.exp(2 * muZ + varZ) * (np.exp(varZ) - 1)

    # Sum of muE and varE over mini-batch
    muE_sum = []
    varE_sum = []
    for i in range(0, len(muE), nb_classes):
        muE_sum.append(np.sum(muE[i : i + nb_classes]))
        varE_sum.append(np.sum(varE[i : i + nb_classes]))

    muE_sum = np.array(muE_sum)
    varE_sum = np.array(varE_sum)

    # Log of the sum of muE and varE
    log_muE_sum = []
    log_varE_sum = []
    for i in range(0, len(muE_sum)):
        tmp = np.log(1 + varE_sum[i] / (muE_sum[i] ** 2))
        log_varE_sum.append(tmp)
        log_muE_sum.append(np.log(muE_sum[i]) - 0.5 * tmp)

    # cov_Z_log_sumE = ln(1 + varE / muE_sum / muE)
    cov_Z_log_sumE = []
    for i in range(0, len(muE)):
        cov_Z_log_sumE.append(np.log(1 + varE[i] / muE_sum[i // nb_classes] / muE[i]))

    log_muE_sum = np.array(log_muE_sum)
    log_varE_sum = np.array(log_varE_sum)
    cov_Z_log_sumE = np.array(cov_Z_log_sumE)

    # Do division as subtraction
    muA_cap = muZ - log_muE_sum.repeat(nb_classes)
    varA_cap = varZ + log_varE_sum.repeat(nb_classes) - 2 * cov_Z_log_sumE

    muA = np.exp(muA_cap + 0.5 * varA_cap)
    varA = muA ** 2 * (np.exp(varA_cap) - 1)

    return muA, varA


def tagi_trainer(
    batch_size: int, num_epochs: int, device: str = "cpu", sigma_v: float = 0.05
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
    metric = HRCSoftmaxMetric(num_classes=10)
    nb_classes = 10
    net = TAGI_CNN_BATCHNORM
    net.to_device(device)
    # net.set_threads(16)
    out_updater = OutputUpdater(net.device)

    # Training
    error_rates = []
    # pbar = tqdm(range(num_epochs), desc="Training Progress")

    # Load model
    net.load("models_bin/mnist_cnn.bin")

    # Testing
    test_error_rates = []
    net.eval()

    images = {}
    m_preds = {}
    v_preds = {}

    count = 0

    for x, labels in test_loader:
        m_pred, v_pred = net(x)

        # m_pred, v_pred = net.get_outputs()
        v_pred = v_pred[::2] + m_pred[1::2]
        m_pred = m_pred[::2]

        if (count == 3807):
            print("m_pred_z: ", m_pred)
            print("v_pred_z: ", v_pred)

        # Use proabilistic softmax to convert m_pred logits to probabilities
        m_pred, v_pred = probs_from_logits(m_pred, v_pred, nb_classes)

        if (count == 3807):
            print("m_pred_a: ", m_pred)
            print("v_pred_a: ", v_pred)

        # Save each prediction with each image, m_pred is falened array of (nb_classes * batch_size, )
        for i in range(len(labels)):
            images[count] = x[i * 784 : (i + 1) * 784]
            m_preds[count] = m_pred[i * 10 : (i + 1) * 10]
            v_preds[count] = v_pred[i * 10 : (i + 1) * 10]
            count += 1



        # ma, va = net.get_outputs()
        # m_pred = np.exp(ma + 0.5 * va)
        # v_pred = m_pred**2 * (np.exp(va) - 1)

        # Training metric
        # error_rate = metric.error_rate(m_pred, v_pred, labels)
        # test_error_rates.append(error_rate)

        for i in range(len(labels)):
            pred = np.argmax(m_pred[i * 10 : (i + 1) * 10])
            if pred != labels[i]:
                test_error_rates.append(1)
            else:
                test_error_rates.append(0)
            print(f"Predicted: {pred} | Actual: {labels[i]}")

    test_error_rate = sum(test_error_rates) / len(test_error_rates)
    print(f"Test Error Rate: {test_error_rate * 100:.2f}%")
    print(f"Sum of m_pred: {np.sum(m_pred)}")

    ambiguous_list = find_and_plot_ambiguous_images(images, m_preds, v_preds)

    # ambiguous_list = range(0, 5)

    for i in range(len(ambiguous_list)):
        plot_class_uncertainty(ambiguous_list[i], images, m_preds, v_preds)




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
    batch_size: int = 1,
    epochs: int = 20,
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
