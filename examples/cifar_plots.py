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
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim


import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import confusion_matrix


from pytagi import HRCSoftmaxMetric, Utils, exponential_scheduler
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    OutputUpdater,
    ReLU,
    MixtureReLU,
    Sequential,
)
from examples.tagi_resnet_model import resnet18_cifar10
from examples.torch_resnet_model import ResNet18

torch.manual_seed(17)

# Constants for dataset normalization
NORMALIZATION_MEAN = [0.4914, 0.4822, 0.4465]
NORMALIZATION_STD = [0.2470, 0.2435, 0.2616]

CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def plot_class_uncertainty(image_idx, images, m_preds, v_preds, v_preds_epistemic, v_preds_aleatoric, delta=0.3):
    """
    Plots CIFAR-10 images with class probabilities and uncertainties.
    """
    # 1. Fix image preprocessing for CIFAR-10
    img = images[image_idx]
    img = img.reshape(3, 32, 32).transpose(1, 2, 0)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Denormalize if using standard CIFAR normalization
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    img = std * img + mean  # Reverse normalization
    img = np.clip(img, 0, 1)  # Ensure valid pixel range

    # 2. Handle probabilistic outputs
    means = m_preds[image_idx]
    stds = np.sqrt(v_preds[image_idx])
    stds_epistemic = np.sqrt(v_preds_epistemic[image_idx])
    stds_aleatoric = np.sqrt(v_preds_aleatoric[image_idx])
    classes = np.arange(10)

    # 3. Create figure
    fig, (ax_img, ax_probs) = plt.subplots(1, 2, figsize=(14, 6))

    # Image plot (CIFAR-10 specific)
    ax_img.imshow(img)
    ax_img.set_title(f'Img {image_idx}\nPred: {np.argmax(means)} | Σprob: {np.sum(means):.2f}')
    ax_img.axis('off')

    # Probability plot with error bars
    for i in classes:
        delta = 0.3
        # Vertical uncertainty bars
        ax_probs.errorbar(i, means[i], yerr=stds[i],
                         fmt='none', ecolor='b', elinewidth=1, capsize=5)

        # Horizontal caps
        ax_probs.plot([i-delta, i+delta], [means[i]+stds[i], means[i]+stds[i]],
                     'b', linewidth=1)
        ax_probs.plot([i-delta, i+delta], [means[i]-stds[i], means[i]-stds[i]],
                     'b', linewidth=1)

        delta = 0.2
        # Aleatoric uncertainty
        ax_probs.errorbar(i, means[i], yerr=stds_aleatoric[i],
                         fmt='none', ecolor='y', elinewidth=1, capsize=5)

        # Horizontal caps
        ax_probs.plot([i-delta, i+delta], [means[i]+stds_aleatoric[i], means[i]+stds_aleatoric[i]],
                     'y', linewidth=1)
        ax_probs.plot([i-delta, i+delta], [means[i]-stds_aleatoric[i], means[i]-stds_aleatoric[i]],
                        'y', linewidth=1)

        # delta = 0.1
        # Epistemic uncertainty
        ax_probs.errorbar(i, means[i], yerr=stds_epistemic[i],
                         fmt='none', ecolor='g', elinewidth=1, capsize=5)


        # Horizontal caps
        ax_probs.plot([i-delta, i+delta], [means[i]+stds_epistemic[i], means[i]+stds_epistemic[i]],
                     'g', linewidth=1)
        ax_probs.plot([i-delta, i+delta], [means[i]-stds_epistemic[i], means[i]-stds_epistemic[i]],
                        'g', linewidth=1)


        # Scatter points
        ax_probs.scatter(i, means[i], color='r', marker='p', s=80,
                        zorder=3, edgecolor='k', linewidth=0.5)

     # Replace numeric x-axis with class names
    ax_probs.set_xticks(classes)
    ax_probs.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)

    # Adjust spacing to accommodate rotated labels
    plt.subplots_adjust(bottom=0.15)

    ax_probs.set_xlim(-0.5, 9.5)
    # ax_probs.set_ylim(0, 1.1)  # Adjusted for probability range
    ax_probs.grid(True, axis='y', linestyle=':')
    ax_probs.set_xlabel('Class')
    ax_probs.set_ylabel('Probability')

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='p', color='w', label='Probability',
              markerfacecolor='r', markersize=10),
        Line2D([0], [0], color='b', lw=2, label='μ ± σ'),
        Line2D([0], [0], color='g', lw=2, label='μ ± σ_epistemic'),
        Line2D([0], [0], color='y', lw=2, label='μ ± σ_aleatoric')
    ]
    ax_probs.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(f'./images_cifar_LL/cifar_{image_idx}_probs.png', bbox_inches='tight')
    plt.close()  # Prevent memory leaks


def find_and_plot_ambiguous_images(images, m_preds, v_preds, threshold=0.5, min_classes=2, max_images_to_plot=5):
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
        prob_mask = v_preds[idx] > threshold
        if np.sum(prob_mask) >= min_classes:
            ambiguous_indices.append(idx)


    print(f"Found {len(ambiguous_indices)} ambiguous images")

    # Plot first few ambiguous cases
    # for i, idx in enumerate(ambiguous_indices[:max_images_to_plot]):
    #     plot_image_probs_and_variances(idx, images, m_preds, v_preds, i)
    #     plt.suptitle(f"Ambiguous Image #{i+1} (Index {idx})", y=1.05)

    return ambiguous_indices

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
    batch_labels = batch_labels.numpy()

    return batch_images, batch_labels


def load_datasets(batch_size: int, framework: str = "tagi"):
    """Load and transform CIFAR10 training and test datasets."""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data/cifar", train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data/cifar", train=False, download=True, transform=transform_test
    )

    if framework == "torch":
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=1
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=1
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=custom_collate_fn,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=custom_collate_fn,
        )
    return train_loader, test_loader


def tagi_trainer(
    num_epochs: int,
    batch_size: int,
    device: str,
    sigma_v: float,
):
    """
    Run classification training on the Cifar dataset using a custom neural model.

    Parameters:
    - num_epochs: int, number of epochs for training
    - batch_size: int, size of the batch for training
    """
    utils = Utils()
    train_loader, test_loader = load_datasets(batch_size, "tagi")

    # Hierachical Softmax
    metric = HRCSoftmaxMetric(num_classes=10)
    nb_classes = 10

    # Resnet18
    # net = TAGI_CNN_NET
    net = resnet18_cifar10(gain_w=0.083, gain_b=0.083)
    net.to_device(device)
    # net.set_threads(10)
    out_updater = OutputUpdater(net.device)



    net.load("models_bin/cifar_resnet_50.bin")

    # Testing
    test_error_rates = []
    net.eval()

    images = {}
    m_preds = {}
    v_preds = {}
    v_preds_epistemic = {}
    v_preds_aleatoric = {}

    count = 0

    for x, labels in test_loader:
        m_pred, v_pred = net(x)

        v_pred_epi = v_pred[::2]
        v_pred_ale = m_pred[1::2]
        v_pred = v_pred[::2] + m_pred[1::2]
        m_pred = m_pred[::2]

        for i in range(len(labels)):
            images[count] = x[i * 1024 * 3 : (i + 1) * 1024 * 3]
            m_preds[count] = m_pred[i * 10 : (i + 1) * 10]
            v_preds[count] = v_pred[i * 10 : (i + 1) * 10]
            v_preds_epistemic[count] = v_pred_epi[i * 10 : (i + 1) * 10]
            v_preds_aleatoric[count] = v_pred_ale[i * 10 : (i + 1) * 10]
            count += 1


        for i in range(len(labels)):
            pred = np.argmax(m_pred[i * 10 : (i + 1) * 10])
            if pred != labels[i]:
                test_error_rates.append(1)
            else:
                test_error_rates.append(0)
            # print(f"Predicted: {pred} | Actual: {labels[i]}")

    test_error_rate = sum(test_error_rates) / len(test_error_rates)
    print(f"Test Error Rate: {test_error_rate * 100:.2f}%")
    print(f"Sum of m_pred: {np.sum(m_pred)}")

    # assume m_preds, v_preds, v_preds_aleatoric, v_preds_epistemic are already populated
    # Number of classes
    n_classes = 10

    # 1. Gather (sample_key, uncertainty) for each sample,
    #    where uncertainty = total var at the predicted class
    keys = sorted(m_preds.keys())
    uncert_list = []
    for k in keys:
        m_vec = m_preds[k]
        total_v = v_preds[k]
        pred = np.argmax(m_vec)
        uncert_list.append((k, total_v[pred]))

    # 2. Sort by descending uncertainty and take top 20%
    uncert_list.sort(key=lambda x: x[1], reverse=True)
    top_n = int(len(uncert_list) * 0.05)
    top_keys = [k for k, _ in uncert_list[:top_n]]

    # 3. For each class, compute corr(mean, aleatoric_var) and corr(mean, epistemic_var)
    print(f"Using top {top_n} ({100 * 0.05:.0f}%) most‐uncertain samples:\n")
    print(f"{'Class':>5} | {'r(mean, aleatoric)':>18} | {'r(mean, epistemic)':>18}")
    print("-" * 50)
    for cls in range(n_classes):
        # build arrays only over the filtered samples
        mean_cls = np.array([m_preds[k][cls] for k in top_keys])
        alea_cls = np.array([v_preds_aleatoric[k][cls] for k in top_keys])
        epi_cls  = np.array([v_preds_epistemic[k][cls] for k in top_keys])

        r_alea = np.corrcoef(mean_cls, alea_cls)[0, 1]
        r_epi  = np.corrcoef(mean_cls, epi_cls)[0, 1]

        print(f"{cls:5d} | {r_alea:18.4f} | {r_epi:18.4f}")


    # # Build arrays
    # N = len(m_preds)
    # accuracy_array = np.array([1 - err for err in test_error_rates])

    # # Aggregate uncertainties
    # epistemic_uncertainty = np.array([
    #     np.mean(v_preds_epistemic[i]) for i in range(N)
    # ])
    # aleatoric_uncertainty = np.array([
    #     np.mean(v_preds_aleatoric[i]) for i in range(N)
    # ])

    # # Classification results
    # correct_unc_epistemic = [np.mean(v_preds_epistemic[i]) for i in range(N) if accuracy_array[i] == 1]
    # wrong_unc_epistemic   = [np.mean(v_preds_epistemic[i]) for i in range(N) if accuracy_array[i] == 0]

    # correct_unc_aleatoric = [np.mean(v_preds_aleatoric[i]) for i in range(N) if accuracy_array[i] == 1]
    # wrong_unc_aleatoric   = [np.mean(v_preds_aleatoric[i]) for i in range(N) if accuracy_array[i] == 0]

    # # Optional: remove extreme outliers
    # def filter_percentile(data, percentile=99):
    #     thresh = np.percentile(data, percentile)
    #     return [v for v in data if v <= thresh]

    # correct_unc_epistemic = filter_percentile(correct_unc_epistemic)
    # wrong_unc_epistemic   = filter_percentile(wrong_unc_epistemic)
    # correct_unc_aleatoric = filter_percentile(correct_unc_aleatoric)
    # wrong_unc_aleatoric   = filter_percentile(wrong_unc_aleatoric)

    # # Aggregate statistics
    # def summarize(data):
    #     data = np.array(data)
    #     return np.mean(data), np.std(data)

    # # Epistemic stats
    # mean_epi_correct, std_epi_correct = summarize(correct_unc_epistemic)
    # mean_epi_wrong, std_epi_wrong = summarize(wrong_unc_epistemic)

    # # Aleatoric stats
    # mean_ale_correct, std_ale_correct = summarize(correct_unc_aleatoric)
    # mean_ale_wrong, std_ale_wrong = summarize(wrong_unc_aleatoric)

    # # Data for plots
    # labels = ['Correct', 'Incorrect']

    # means_epistemic = [mean_epi_correct, mean_epi_wrong]
    # errors_epistemic = [std_epi_correct, std_epi_wrong]

    # means_aleatoric = [mean_ale_correct, mean_ale_wrong]
    # errors_aleatoric = [std_ale_correct, std_ale_wrong]

    # x = np.arange(len(labels))
    # width = 0.6

    # # Plot
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # # Epistemic
    # axs[0].bar(x, means_epistemic, yerr=errors_epistemic, capsize=10, alpha=0.7)
    # axs[0].set_xticks(x)
    # axs[0].set_xticklabels(labels)
    # axs[0].set_title("Epistemic Uncertainty")
    # axs[0].set_ylabel("Mean Uncertainty")

    # # Aleatoric
    # axs[1].bar(x, means_aleatoric, yerr=errors_aleatoric, capsize=10, alpha=0.7, color='orange')
    # axs[1].set_xticks(x)
    # axs[1].set_xticklabels(labels)
    # axs[1].set_title("Aleatoric Uncertainty")

    # plt.tight_layout()
    # plt.savefig("uncertainty_vs_accuracy.png")
    # plt.close()


    # # Step 1: Compute predicted class, uncertainty, and true label
    # preds = []
    # trues = []
    # uncertainties = []

    # original_labels = np.array([int(label) for label in test_loader.dataset.targets])

    # for i in range(N):
    #     pred_class = np.argmax(m_preds[i])
    #     unc = v_preds_aleatoric[i][pred_class]
    #     preds.append(pred_class)
    #     true_label = int(original_labels[i])  # use a safe source of ground-truth labels
    #     trues.append(true_label)
    #     uncertainties.append(unc)

    # # Step 2: Filter high uncertainty predictions
    # uncertainties = np.array(uncertainties)
    # threshold = np.percentile(uncertainties, 80)

    # filtered_preds = []
    # filtered_trues = []

    # for i in range(N):
    #     if uncertainties[i] > threshold:
    #         filtered_preds.append(preds[i])
    #         filtered_trues.append(trues[i])

    # # Step 3: Confusion matrix
    # cm = confusion_matrix(filtered_trues, filtered_preds, labels=range(10))

    # # Step 4: Plot
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CIFAR10_LABELS, yticklabels=CIFAR10_LABELS)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix (Top 20% Aleatoric Uncertainty)')
    # plt.tight_layout()
    # plt.savefig("class_uncertainty_correlation.png")
    # plt.close()

    # # Step 1: Compute predicted class + its uncertainty
    # predicted_classes = []
    # predicted_uncertainties = []

    # for i in range(N):
    #     pred = np.argmax(m_preds[i])
    #     unc = v_preds_aleatoric[i][pred]
    #     predicted_classes.append(pred)
    #     predicted_uncertainties.append(unc)

    # # Step 2: Define the range of "high but not extreme" uncertainty
    # lower_thresh = np.percentile(predicted_uncertainties, 90)
    # upper_thresh = np.percentile(predicted_uncertainties, 99)

    # # Step 3: Collect uncertainty vectors for high-but-not-extreme samples
    # class_unc_spillover = {c: [] for c in range(10)}  # c = predicted class

    # for i in range(N):
    #     pred = predicted_classes[i]
    #     unc = predicted_uncertainties[i]

    #     if lower_thresh < unc < upper_thresh:
    #         class_unc_spillover[pred].append(np.sqrt(v_preds_aleatoric[i]))

    # # Step 4: Compute mean uncertainty vector per class
    # spillover_matrix = np.zeros((10, 10))

    # for c in range(10):
    #     vecs = class_unc_spillover[c]
    #     if vecs:
    #         spillover_matrix[c] = np.mean(np.stack(vecs), axis=0)


    # plt.figure(figsize=(8, 6))
    # sns.heatmap(spillover_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
    #             xticklabels=CIFAR10_LABELS, yticklabels=CIFAR10_LABELS)
    # plt.xlabel("Uncertainty On Class")
    # plt.ylabel("Predicted Class (High-but-Not-Extreme Uncertainty)")
    # plt.title("Class-wise Aleatoric Uncertainty (std) Spread (Outliers Removed)")
    # plt.tight_layout()
    # plt.savefig("uncertainty_spillover.png")
    # plt.close()


    list = [i for i in range(20)]

    ambiguous_list = find_and_plot_ambiguous_images(images, m_preds, v_preds)

    ambiguous_list = list + ambiguous_list

    # if len(ambiguous_list) > 20:
    #     ambiguous_list = ambiguous_list[:20]

    for i in range(len(ambiguous_list)):
        plot_class_uncertainty(ambiguous_list[i], images, m_preds, v_preds, v_preds_epistemic, v_preds_aleatoric)





def torch_trainer(batch_size: int, num_epochs: int, device: str = "cuda"):
    # Hyperparameters
    learning_rate = 0.001

    # torch.set_float32_matmul_precision("high")
    train_loader, test_loader = load_datasets(batch_size, "torch")

    # Initialize the model, loss function, and optimizer
    torch_device = torch.device(device)
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check your CUDA installation."
        )
    model = ResNet18()
    # model = TorchCNN()
    # model = torch.compile(model)
    model.to(torch_device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        model.train()
        error_rates = []
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
        sample_count = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(torch_device)
                target = target.to(torch_device)
                output = model(data)

                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                sample_count += data.shape[0]

        test_loss /= len(test_loader.dataset)
        test_error_rate = (1.0 - correct / len(test_loader.dataset)) * 100
        pbar.set_description(
            f"Epoch# {epoch +1}/{num_epochs}| training error: {avg_error_rate:.2f}% | Test error: {test_error_rate: .2f}%\n",
            refresh=False,
        )


def main(
    framework: str = "tagi",
    batch_size: int = 128,
    epochs: int = 9,
    device: str = "cuda",
    sigma_v: float = 0.01,
):
    if framework == "torch":
        torch_trainer(batch_size=batch_size, num_epochs=epochs, device=device)
    elif framework == "tagi":
        tagi_trainer(
            batch_size=batch_size, num_epochs=epochs, device=device, sigma_v=sigma_v
        )
    else:
        raise RuntimeError(f"Invalid Framework: {framework}")


if __name__ == "__main__":
    fire.Fire(main)
