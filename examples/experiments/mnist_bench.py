import os
import sys
import gc
import time
import fire
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb

from pytagi import HRCSoftmaxMetric, Utils, exponential_scheduler
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    OutputUpdater,
    ReLU,
    Sequential,
)

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

# -------------------- TAGI MODEL BUILDERS --------------------

def build_tagi_fnn(num_layers, neurons_per_layer, nb_classes):
    layers = []
    input_dim = 28 * 28  # MNIST images are 28x28
    for _ in range(num_layers - 1):
        layers.append(Linear(input_dim, neurons_per_layer))
        layers.append(ReLU())
        input_dim = neurons_per_layer
    layers.append(Linear(input_dim, nb_classes))  # 10 classes + 1 dummy for HRCSoftmax
    return Sequential(*layers)

def build_tagi_cnn(num_layers, channels_per_layer, nb_classes, in_height=28, in_width=28):
    # A generic CNN builder for TAGI given number of layers and channels.
    # For simplicity, we apply pooling on the first two layers only.
    layers = []
    input_channels = 1
    current_height, current_width = in_height, in_width
    conv_channels = channels_per_layer

    for i in range(num_layers):
        layers.append(Conv2d(input_channels, conv_channels, kernel_size=3, padding=1, in_width=current_width, in_height=current_height))
        layers.append(ReLU())

        input_channels = conv_channels
        conv_channels *= 2  # Double channels each layer for complexity

        # Apply pooling on first two layers only to reduce spatial dimensions
        if i < 2:
            pool_kernel = 2
            pool_stride = 2
            layers.append(AvgPool2d(pool_kernel, pool_stride))
            # Update dimensions after pooling
            current_height = (current_height // pool_stride)
            current_width = (current_width // pool_stride)

    flatten_dim = input_channels * current_height * current_width
    layers.append(Linear(flatten_dim, 256))
    layers.append(ReLU())
    layers.append(Linear(256, nb_classes))
    return Sequential(*layers)

def build_tagi_cnn_batchnorm(num_layers, channels_per_layer, nb_classes, in_height=28, in_width=28):
    layers = []
    input_channels = 1
    current_height, current_width = in_height, in_width
    conv_channels = channels_per_layer

    for i in range(num_layers):
        layers.append(Conv2d(input_channels, conv_channels, kernel_size=3, padding=1, bias=False, in_width=current_width, in_height=current_height))
        layers.append(BatchNorm2d(conv_channels))
        layers.append(ReLU())

        input_channels = conv_channels
        conv_channels *= 2

        # Apply pooling on first two layers
        if i < 2:
            pool_kernel = 2
            pool_stride = 2
            layers.append(AvgPool2d(pool_kernel, pool_stride))
            current_height = (current_height // pool_stride)
            current_width = (current_width // pool_stride)

    flatten_dim = input_channels * current_height * current_width
    layers.append(Linear(flatten_dim, 256))
    layers.append(ReLU())
    layers.append(Linear(256, nb_classes))
    return Sequential(*layers)

# -------------------- TORCH MODELS --------------------

class TorchFNN(nn.Module):
    def __init__(self, num_layers, neurons_per_layer):
        super(TorchFNN, self).__init__()
        layers = []
        input_dim = 28 * 28
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, neurons_per_layer))
            layers.append(nn.ReLU())
            input_dim = neurons_per_layer
        layers.append(nn.Linear(input_dim, 10))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.model(x)

class TorchCNN(nn.Module):
    def __init__(self, num_layers, channels_per_layer, in_height=28, in_width=28):
        super(TorchCNN, self).__init__()
        layers = []
        input_channels = 1
        current_height, current_width = in_height, in_width
        conv_channels = channels_per_layer

        for i in range(num_layers):
            layers.append(nn.Conv2d(input_channels, conv_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            input_channels = conv_channels
            conv_channels *= 2

            # Pooling in first two layers
            if i < 2:
                pool_kernel = 2
                pool_stride = 2
                layers.append(nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride))
                current_height = current_height // pool_stride
                current_width = current_width // pool_stride

        flatten_dim = input_channels * current_height * current_width
        layers.append(nn.Flatten())
        layers.append(nn.Linear(flatten_dim, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 10))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class TorchCNNBatchNorm(nn.Module):
    def __init__(self, num_layers, channels_per_layer, in_height=28, in_width=28):
        super(TorchCNNBatchNorm, self).__init__()
        layers = []
        input_channels = 1
        current_height, current_width = in_height, in_width
        conv_channels = channels_per_layer

        for i in range(num_layers):
            layers.append(nn.Conv2d(input_channels, conv_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(conv_channels))
            layers.append(nn.ReLU())
            input_channels = conv_channels
            conv_channels *= 2

            if i < 2:
                pool_kernel = 2
                pool_stride = 2
                layers.append(nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride))
                current_height = current_height // pool_stride
                current_width = current_width // pool_stride

        flatten_dim = input_channels * current_height * current_width
        layers.append(nn.Flatten())
        layers.append(nn.Linear(flatten_dim, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 10))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


DATA_FOLDER = "./data/mnist"

def custom_collate_fn(batch):
    batch_images, batch_labels = zip(*batch)
    batch_images = torch.stack(batch_images)
    batch_labels = torch.tensor(batch_labels)
    batch_images = batch_images.numpy().reshape(len(batch_images), -1).flatten()
    batch_labels = batch_labels.numpy().flatten()
    return batch_images, batch_labels

# -------------------- TAGI TRAINER HRC --------------------

def tagi_trainer_hrc(
    framework: str,
    model: str,
    num_layers: int,
    neurons_per_layer: int,
    channels_per_layer: int,
    batch_size: int,
    num_epochs: int,
    device: str = "cpu",
    sigma_v: float = 2.0
):
    from pytagi import HRCSoftmaxMetric, Utils, exponential_scheduler

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

    # Validation set
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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

    # Hierarchical Softmax
    metric = HRCSoftmaxMetric(num_classes=10)
    nb_classes = 11

    # Build TAGI Model
    if model == "FNN":
        net = build_tagi_fnn(num_layers=num_layers, neurons_per_layer=neurons_per_layer, nb_classes=nb_classes)
    elif model == "CNN":
        net = build_tagi_cnn(num_layers=num_layers, channels_per_layer=channels_per_layer, nb_classes=nb_classes)
    elif model == "CNNBatchNorm":
        net = build_tagi_cnn_batchnorm(num_layers=num_layers, channels_per_layer=channels_per_layer, nb_classes=nb_classes)
    else:
        raise ValueError(f"Unknown TAGI model: {model}")

    net.to_device(device)
    if device != "cuda":
        net.set_threads(8)
        wandb.config.update({"device": "cpu", "num_threads": 8})

    out_updater = OutputUpdater(net.device)
    var_y = np.full((batch_size * metric.hrc_softmax.num_obs,), sigma_v**2, dtype=np.float32)

    error_rates = []
    best_val_error_rate = float('inf')
    no_improvement_epochs = 0

    pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in pbar:
        var_y = np.full((batch_size * metric.hrc_softmax.num_obs,), sigma_v**2, dtype=np.float32)
        net.train()
        for x, labels in train_loader:
            m_pred, v_pred = net(x)
            y, y_idx, _ = utils.label_to_obs(labels=labels, num_classes=10)
            out_updater.update_using_indices(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                selected_idx=y_idx,
                delta_states=net.input_delta_z_buffer,
            )

            net.backward()
            net.step()

            error_rate = metric.error_rate(m_pred, v_pred, labels)
            error_rates.append(error_rate)

        avg_error_rate = sum(error_rates[-100:])

        # Evaluate on validation and test sets every 2 epochs
        if (epoch + 1) % 2 == 0 or (epoch + 1) == num_epochs:
            net.eval()

            # Compute validation error rate
            val_error_rates = []
            for x_val, labels_val in val_loader:  # Assuming val_loader is your validation DataLoader
                m_pred_val, v_pred_val = net(x_val)
                error_rate_val = metric.error_rate(m_pred_val, v_pred_val, labels_val)
                val_error_rates.append(error_rate_val)
            val_error_rate = sum(val_error_rates) / len(val_error_rates)

            # Compute test error rate (optional for monitoring, but not for stopping)
            test_error_rates = []
            for x_test, labels_test in test_loader:
                m_pred_test, v_pred_test = net(x_test)
                error_rate_test = metric.error_rate(m_pred_test, v_pred_test, labels_test)
                test_error_rates.append(error_rate_test)
            test_error_rate = sum(test_error_rates) / len(test_error_rates)

            # Update progress bar
            pbar.set_description(
                f"Epoch {epoch + 1}/{num_epochs} | Train Error: {avg_error_rate:.2f}% | "
                f"Validation Error: {val_error_rate*100:.2f}% | Test Error: {test_error_rate*100:.2f}%",
                refresh=True,
            )

            # Log metrics
            wandb.log({
                "epoch": epoch + 1,
                "training_error_rate": avg_error_rate,
                "validation_error_rate": val_error_rate * 100,
                "test_error_rate": test_error_rate * 100,
            })

            # Early stopping based on validation error
            if val_error_rate < best_val_error_rate:
                best_val_error_rate = val_error_rate
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs > 10:
                    print("Early stopping triggered.")
                    break
        else:
            # Log only training error if not evaluating
            wandb.log({
                "epoch": epoch + 1,
                "training_error_rate": avg_error_rate
            })

    print("Training complete.")

    # Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# -------------------- TAGI TRAINER REMAX --------------------

def tagi_trainer_remax(
    framework: str,
    model: str,
    num_layers: int,
    neurons_per_layer: int,
    channels_per_layer: int,
    batch_size: int,
    num_epochs: int,
    device: str = "cpu",
    sigma_v: float = 2.0
):
    from pytagi import Utils, exponential_scheduler

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

    # Validation set
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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
    nb_classes = 10

    # Build TAGI Model
    if model == "FNN":
        net = build_tagi_fnn(num_layers=num_layers, neurons_per_layer=neurons_per_layer, nb_classes=nb_classes)
    elif model == "CNN":
        net = build_tagi_cnn(num_layers=num_layers, channels_per_layer=channels_per_layer, nb_classes=nb_classes)
    elif model == "CNNBatchNorm":
        net = build_tagi_cnn_batchnorm(num_layers=num_layers, channels_per_layer=channels_per_layer, nb_classes=nb_classes)
    else:
        raise ValueError(f"Unknown TAGI model: {model}")

    net.to_device(device)
    if device != "cuda":
        net.set_threads(8)
        wandb.config.update({"device": "cpu", "num_threads": 8})

    out_updater = OutputUpdater(net.device)
    var_y = np.full((batch_size * nb_classes,), sigma_v**2, dtype=np.float32)

    error_rates = []
    best_val_error_rate = float('inf')
    no_improvement_epochs = 0

    pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in pbar:
        var_y = np.full((batch_size * nb_classes,), sigma_v**2, dtype=np.float32)
        net.train()
        for x, labels in train_loader:
            m_pred, v_pred = net(x)

            y = np.full((batch_size * nb_classes,), 0.0, dtype=np.float32)
            for i in range(len(labels)):
                y[i * nb_classes + labels[i]] = 1.0

            out_updater.update(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                delta_states=net.input_delta_z_buffer,
            )

            net.backward()
            net.step()

            error = 0
            for i in range(len(labels)):
                pred = np.argmax(m_pred[i * nb_classes : (i + 1) * nb_classes])
                if pred != labels[i]:
                    error += 1

            error_rates.append(error / len(labels))

        avg_error_rate = sum(error_rates[-100:])

        # Evaluate on validation and test sets every 2 epochs
        if (epoch + 1) % 2 == 0 or (epoch + 1) == num_epochs:
            net.eval()

            # Compute validation error rate
            val_error_rates = []
            for x_val, labels_val in val_loader:  # Assuming val_loader is your validation DataLoader
                m_pred_val, v_pred_val = net(x_val)
                for i in range(len(labels_val)):
                    pred = np.argmax(m_pred_val[i * nb_classes : (i + 1) * nb_classes])
                    if pred != labels_val[i]:
                        val_error_rates.append(1)
                    else:
                        val_error_rates.append(0)
            val_error_rate = sum(val_error_rates) / len(val_error_rates)

            # Compute test error rate (optional for monitoring, but not for stopping)
            test_error_rates = []
            for x_test, labels_test in test_loader:
                m_pred_test, v_pred_test = net(x_test)
                for i in range(len(labels_test)):
                    pred = np.argmax(m_pred_test[i * nb_classes : (i + 1) * nb_classes])
                    if pred != labels_test[i]:
                        test_error_rates.append(1)
                    else:
                        test_error_rates.append(0)
            test_error_rate = sum(test_error_rates) / len(test_error_rates)

            # Update progress bar
            pbar.set_description(
                f"Epoch {epoch + 1}/{num_epochs} | Train Error: {avg_error_rate:.2f}% | "
                f"Validation Error: {val_error_rate*100:.2f}% | Test Error: {test_error_rate*100:.2f}%",
                refresh=True,
            )

            # Log metrics
            wandb.log({
                "epoch": epoch + 1,
                "training_error_rate": avg_error_rate,
                "validation_error_rate": val_error_rate * 100,
                "test_error_rate": test_error_rate * 100,
            })

            # Early stopping based on validation error
            if val_error_rate < best_val_error_rate:
                best_val_error_rate = val_error_rate
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs > 10:
                    print("Early stopping triggered.")
                    break
        else:
            # Log only training error if not evaluating
            wandb.log({
                "epoch": epoch + 1,
                "training_error_rate": avg_error_rate
            })

    print("Training complete.")

    # Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# -------------------- TORCH TRAINER --------------------

def torch_trainer(
    framework: str,
    model: str,
    num_layers: int,
    neurons_per_layer: int,
    channels_per_layer: int,
    batch_size: int,
    num_epochs: int,
    device: str = "cpu",
    sigma_v: float = 2.0
):
    # sigma_v not used for Torch

    learning_rate = 0.01

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root=DATA_FOLDER, train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root=DATA_FOLDER, train=False, transform=transform, download=True
    )

    # Validation set
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])


    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    torch_device = torch.device(device)
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    # Select Torch model
    if model == "FNN":
        model_instance = TorchFNN(num_layers=num_layers, neurons_per_layer=neurons_per_layer).to(torch_device)
    elif model == "CNN":
        model_instance = TorchCNN(num_layers=num_layers, channels_per_layer=channels_per_layer).to(torch_device)
    elif model == "CNNBatchNorm":
        model_instance = TorchCNNBatchNorm(num_layers=num_layers, channels_per_layer=channels_per_layer).to(torch_device)
    else:
        raise ValueError(f"Unknown Torch model: {model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_instance.parameters(), lr=learning_rate)

    pbar = tqdm(range(num_epochs), desc="Training Progress")
    error_rates = []
    best_val_error_rate = float('inf')
    no_improvement_epochs = 0

    for epoch in pbar:
        model_instance.train()
        for data, target in train_loader:
            data = data.to(torch_device)
            target = target.to(torch_device)

            optimizer.zero_grad()
            output = model_instance(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            train_correct = pred.eq(target.view_as(pred)).sum().item()
            error_rates.append((1.0 - (train_correct / data.shape[0])) * 100)

        avg_error_rate = sum(error_rates[-100:]) / len(error_rates[-100:]) if len(error_rates) >= 100 else sum(error_rates)/len(error_rates)

        # Test and validation evaluation every epoch
        model_instance.eval()
        test_loss = 0
        test_correct = 0
        val_loss = 0
        val_correct = 0

        with torch.no_grad():
            # Evaluate on test set
            for data, target in test_loader:
                data = data.to(torch_device)
                target = target.to(torch_device)
                output = model_instance(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            test_error_rate = (1.0 - test_correct / len(test_loader.dataset)) * 100

            # Evaluate on validation set
            for data, target in val_loader:  # Assuming val_loader is your validation DataLoader
                data = data.to(torch_device)
                target = target.to(torch_device)
                output = model_instance(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()

            val_loss /= len(val_loader.dataset)
            val_error_rate = (1.0 - val_correct / len(val_loader.dataset)) * 100

        # Update progress bar
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} | Train Err: {avg_error_rate:.2f}% | "
            f"Val Err: {val_error_rate:.2f}% | Test Err: {test_error_rate:.2f}%"
        )

        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "training_error_rate": avg_error_rate,
            "validation_error_rate": val_error_rate,
            "test_error_rate": test_error_rate,
        })

        # Early stopping based on validation error
        if val_error_rate < best_val_error_rate:
            best_val_error_rate = val_error_rate
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs > 10:
                print("Early stopping triggered.")
                break

    print("Training complete.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# -------------------- MAIN FUNCTION --------------------

def main(
    framework: str = "tagi",
    model: str = "FNN",
    num_layers: int = 2,
    neurons_per_layer: int = 50,
    channels_per_layer: int = 16,
    batch_size: int = 128,
    epochs: int = 100,
    device: str = "cuda",
    sigma_v: float = 0.1,
    learning_rate: float = 0.001
):
    if model == "FNN":
        run_name = f"{framework}_{model}_layers{num_layers}_neurons{neurons_per_layer}_batch{batch_size}"
    else:
        run_name = f"{framework}_{model}_layers{num_layers}_channels{channels_per_layer}_batch{batch_size}"

    if framework == "tagi":
        run_name += f"_sigma{sigma_v}"
    run_name += f"_lr{learning_rate}"

    wandb.init(project="Remax",
            config={
                "framework": framework,
                "model": model,
                "num_layers": num_layers,
                "neurons_per_layer": neurons_per_layer if model == "FNN" else 0,
                "channels_per_layer": channels_per_layer if model in ["CNN", "CNNBatchNorm"] else 0,
                "batch_size": batch_size,
                "num_epochs": epochs,
                "sigma_v": sigma_v if framework == "tagi" else 0,
                "learning_rate": learning_rate
            },
            name=run_name,
            reinit=True)

    if model == "FNN":
        effective_neurons = neurons_per_layer
        effective_channels = 0
    else:
        effective_neurons = 0
        effective_channels = channels_per_layer

    if framework.lower() == "torch":
        torch_trainer(
            framework=framework,
            model=model,
            num_layers=num_layers,
            neurons_per_layer=effective_neurons,
            channels_per_layer=effective_channels,
            batch_size=batch_size,
            num_epochs=epochs,
            device=device,
            sigma_v=sigma_v
        )
    elif framework.lower() == "tagi_hrc":
        tagi_trainer_hrc(
            framework=framework,
            model=model,
            num_layers=num_layers,
            neurons_per_layer=effective_neurons,
            channels_per_layer=effective_channels,
            batch_size=batch_size,
            num_epochs=epochs,
            device=device,
            sigma_v=sigma_v
        )
    elif framework.lower() == "tagi_remax":
        tagi_trainer_remax(
            framework=framework,
            model=model,
            num_layers=num_layers,
            neurons_per_layer=effective_neurons,
            channels_per_layer=effective_channels,
            batch_size=batch_size,
            num_epochs=epochs,
            device=device,
            sigma_v=sigma_v
        )
    else:
        raise ValueError(f"Unknown framework: {framework}")

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)