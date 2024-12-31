# Temporary import. It will be removed in the final vserion
import os
import sys

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
import uuid


from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    OutputUpdater,
    ReLU,
    Sequential,
)


def plot_image_and_class_distributions(x, m, S, num_points=200, label=None):
    """
    Plot the MNIST image corresponding to x alongside the normal distributions
    for each class given their means and variances.

    Parameters
    ----------
    x : array-like
        Flattened MNIST image (784 values).
    m : array-like
        A list or array of mean values for each class.
    S : array-like
        A list or array of variances for each class.
    num_points : int, optional
        Number of points used to plot the distribution curves.
    """
    # Convert inputs to arrays
    x = np.array(x)
    m = np.array(m)
    S = np.array(S)

    # Ensure variances are not too close to zero to avoid division by zero errors
    S = np.where(S < 1e-3, 1e-3, S)

    # Reshape the flattened image into 28x28
    img = x.reshape(28, 28)

    # Compute a suitable range for the x-axis
    x_min = np.min(m) - 3 * np.sqrt(np.max(S))
    x_max = np.max(m) + 3 * np.sqrt(np.max(S))
    X = np.linspace(x_min, x_max, num_points)

    # Create subplots: one for the image, one for the distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the image on the left
    ax_img = axes[0]
    ax_img.imshow(img, cmap="gray", interpolation="nearest")
    ax_img.set_title("MNIST Image")
    if label is not None:
        ax_img.text(0.5, -0.1, f"Label: {label}", size=12, ha="center", transform=ax_img.transAxes)
    ax_img.axis("off")  # Hide axis ticks for image

    # Plot the distributions on the right
    ax_dist = axes[1]
    for i, (mean, var) in enumerate(zip(m, S)):
        std = np.sqrt(var)
        y = norm.pdf(X, mean, std)
        ax_dist.plot(X, y, label=f"Digit {i})", alpha=0.6)

    predicted_label = np.argmax(m)
    ax_dist.set_title(f"Class Probability Distributions (Predicted: {predicted_label})")
    ax_dist.set_xlabel("Value")
    ax_dist.set_ylabel("Density")
    ax_dist.legend()
    ax_dist.grid(True)

    plt.tight_layout()

    # save the plot
    # Ensure the output directory exists
    output_dir = "./out/plots"
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique filename to avoid overwriting
    unique_filename = f"output_{uuid.uuid4().hex}.png"

    # Save the plot
    plt.savefig(os.path.join(output_dir, unique_filename))


DATA_FOLDER = "./data/mnist"


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


def main(
    batch_size= 128,
    num_epochs= 20,
    device: str = "cuda",
    sigma_v: float = 0,
    train: bool = True,
    test: bool = True,
):

    # Data loading and preprocessing
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root=DATA_FOLDER, train=True, transform=transform, download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=custom_collate_fn,
    )
    test_dataset = datasets.MNIST(
        root=DATA_FOLDER, train=False, transform=transform, download=True
    )

    # Create a subset of the test set containing 10 random images
    test_indices = np.random.choice(len(test_dataset), size=20, replace=False)
    test_subset = torch.utils.data.Subset(test_dataset, indices=test_indices)

    test_loader = DataLoader(
        test_subset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=custom_collate_fn,
    )
    print("Data loaded")

    net= Sequential(
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
    )

    print("network created")
    net.to_device(device)
    out_updater = OutputUpdater(net.device)
    S_preds = {idx: [] for idx in range(10)}

    if train:
        error_rates = []
        pbar = tqdm(range(num_epochs), desc="Training Progress")

        for epoch in pbar:

            var_y = np.full((batch_size * 10,), sigma_v**2, dtype=np.float32)
            net.train()
            for x, labels in train_loader:
                # Feedforward and backward pass
                m_pred, v_pred = net(x)
                # print(f"m_pred: {m_pred}")

                y = np.full((len(labels) * 10,), 0.0, dtype=np.float32)
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

                # m_pred, v_pred = net.get_outputs()
                # print(f"m_pred: {m_pred}")
                # if epoch == num_epochs - 1:
                #     for i in range(len(labels)):
                #         plot_image_and_class_distributions(
                #             x[i * 784 : (i + 1) * 784],
                #             m_pred[i * 10 : (i + 1) * 10],
                #             v_pred[i * 10 : (i + 1) * 10]
                #         )

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


            if test:
                test_error_rates = []
                img_idx = 0
                for x, labels in test_loader:

                    net.eval()
                    m_pred, v_pred = net(x)


                    y = np.full((len(labels) * 10,), 0.0, dtype=np.float32)

                    for i in range(len(labels)):
                        y[i * 10 + labels[i]] = 1.0

                    out_updater.update(
                        output_states=net.output_z_buffer,
                        mu_obs=y,
                        var_obs=np.full((len(labels) * 10,), 0, dtype=np.float32),
                        delta_states=net.input_delta_z_buffer,
                    )

                    m_pred, v_pred = net.get_outputs()

                    # Append the variance corresponding to the true label
                    # S_preds[img_idx].append(np.mean(v_pred))
                    # S_preds[img_idx].append(v_pred[labels[0]])
                    img_idx += 1


                    if epoch > 5:
                        for i in range(10):
                            # check if m_pred are not very close to 1 and are distributed between 0 and 1
                            if m_pred.size > 0 and np.max(m_pred) < 0.8:
                                plot_image_and_class_distributions(
                                    x,
                                    m_pred,
                                    v_pred,
                                    label=labels[0],
                                )
                        # stop running the whole code
                        # sys.exit()

                    for i in range(len(labels)):
                        pred = np.argmax(m_pred[i * 10 : (i + 1) * 10])
                        if pred != labels[i]:
                            test_error_rates.append(1)
                        else:
                            test_error_rates.append(0)


                    # plot variances
                    # if epoch > 5:
                    #     print(S_preds)
                    #     for i in range(10):
                    #         plt.plot(S_preds[i], label=f"Image {i}")
                    #     plt.xlabel("Epochs")
                    #     plt.ylabel("S_pred")
                    #     plt.legend()
                    #     plt.savefig(f"out/plots/S_pred_{epoch}.png")
                    #     plt.close()

if __name__ == "__main__":
    main()