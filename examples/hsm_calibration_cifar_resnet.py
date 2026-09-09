"""CIFAR-10 / ResNet-18 with the hierarchical softmax calibration of Section 4.

Standard TAGI training (the loop is untouched, +/-1 targets in logit space).
After each epoch, Algorithm 1 of the note *Hierarchical Softmax
Calibration* (Goulet, Nguyen & Florensa-Montilla) runs on a held-out
validation split and updates the Gaussian gains :math:`G_h` only. The test set
is then scored with TAGI's fixed scale :math:`\\alpha` of Eq. (1) and with the
learned gains of (E:class), for the three gain-sharing schemes of Section 4.2.

Usage::

    python -m examples.hsm_calibration_cifar_resnet
    python -m examples.hsm_calibration_cifar_resnet --num_epochs 50 --alpha 3
"""

import os
import sys

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import json

import fire
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from examples.tagi_resnet_model import resnet18_cifar10
from pytagi import HRCSoftmaxMetric, Utils, exponential_scheduler
from pytagi.hsm_calibration import HSMGainCalibrator
from pytagi.nn import OutputUpdater

NORMALIZATION_MEAN = (0.4914, 0.4822, 0.4465)
NORMALIZATION_STD = (0.2023, 0.1994, 0.2010)


def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images).reshape(-1).numpy()
    return images, torch.tensor(labels).numpy()


def load_datasets(batch_size: int, num_val: int, seed: int = 0):
    """CIFAR-10 with the last `num_val` training images held out for calibration.

    The validation split uses the test-time transform: Algorithm 1 needs the
    forward-pass moments of genuine out-of-sample predictions, not of augmented
    images.
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ]
    )

    train_full = torchvision.datasets.CIFAR10(
        root="./data/cifar",
        train=True,
        download=True,
        transform=transform_train,
    )
    val_full = torchvision.datasets.CIFAR10(
        root="./data/cifar", train=True, download=True, transform=transform_test
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data/cifar",
        train=False,
        download=True,
        transform=transform_test,
    )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_full))
    train_idx, val_idx = perm[: len(train_full) - num_val], perm[-num_val:]

    kw = dict(num_workers=2, collate_fn=custom_collate_fn)
    train_loader = DataLoader(
        Subset(train_full, train_idx), batch_size=batch_size, shuffle=True, **kw
    )
    val_loader = DataLoader(
        Subset(val_full, val_idx), batch_size=batch_size, shuffle=False, **kw
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, **kw
    )
    return train_loader, val_loader, test_loader


def forward_pass(net, loader, tree_len: int):
    """Forward-pass moments of the output layer over a split; no network update."""
    mu, var, lab = [], [], []
    net.eval()
    for x, labels in loader:
        m_pred, v_pred = net(x)
        n = len(labels)
        mu.append(np.asarray(m_pred, dtype=np.float64).reshape(n, tree_len))
        var.append(np.asarray(v_pred, dtype=np.float64).reshape(n, tree_len))
        lab.append(np.asarray(labels).reshape(n))
    return np.concatenate(mu), np.concatenate(var), np.concatenate(lab)


def expected_calibration_error(prob, labels, n_bins: int = 15):
    conf = prob.max(axis=1)
    correct = (prob.argmax(axis=1) == labels).astype(float)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def negative_log_likelihood(prob, labels):
    return float(
        -np.mean(
            np.log(np.clip(prob[np.arange(len(labels)), labels], 1e-12, 1.0))
        )
    )


def brier_score(prob, labels):
    onehot = np.zeros_like(prob)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))


def score(cal, mu, var, labels):
    prob, prob_var = cal.class_probabilities(mu, var)
    pred = prob.argmax(axis=1)
    return {
        "error": 100.0 * float(np.mean(pred != labels)),
        "ece": expected_calibration_error(prob, labels),
        "nll": negative_log_likelihood(prob, labels),
        "brier": brier_score(prob, labels),
        "mean_conf": float(prob.max(axis=1).mean()),
        "sigma_pr": float(
            np.sqrt(prob_var[np.arange(len(labels)), pred]).mean()
        ),
    }


def reliability_figure(curves, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (conf, acc) in curves.items():
        ax.plot(conf, acc, "o-", label=name)
    ax.plot([0, 1], [0, 1], ":", c="k")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("CIFAR-10 / ResNet-18 reliability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def reliability_curve(cal, mu, var, labels, n_bins: int = 15):
    prob, _ = cal.class_probabilities(mu, var, compute_var=False)
    conf = prob.max(axis=1)
    correct = (prob.argmax(axis=1) == labels).astype(float)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = bins == b
        if m.sum() > 20:
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
    return xs, ys


def main(
    num_epochs: int = 50,
    batch_size: int = 128,
    sigma_v: float = 0.05,
    sigma_v_min: float = 0.05,
    sigma_v_decay: float = 0.95,
    num_val: int = 5000,
    gain_w: float = 0.1,
    gain_b: float = 0.1,
    alpha: float = 3.0,
    mu_gain_init: float = 0.3,
    sigma_gain_init: float = 1.0,
    out_json: str = "saved_results/hsm_cifar_resnet18.json",
    ckpt_dir: str = "saved_results/ckpt_hsm_cifar_resnet18",
    ckpt_every: int = 5,
):
    """
    :param sigma_v: initial logit-channel observation noise. Set
        ``sigma_v_min == sigma_v`` to hold it constant.
    :param num_val: training images held out for the calibration pass.
    :param gain_w: weight-initialization gain of the ResNet-18.
    :param gain_b: bias-initialization gain of the ResNet-18.
    :param alpha: the ``alpha`` of cuTAGI's ``obs_to_class`` (src/cost.cpp).
        NOTE the C++ computes ``Phi(mu / sqrt((1/alpha)^2 + var))``, i.e. an
        effective gain of ``alpha`` itself, whereas Eq. (1) of the note reads
        ``Phi(s mu / sqrt(alpha^2 + var))``, an effective gain of ``1/alpha``.
        Both are scored, as ``tagi_code`` (gain = alpha, what cuTAGI does) and
        ``tagi_note`` (gain = 1/alpha, Eq. (1) read literally).
    :param ckpt_dir: directory for the per-epoch network checkpoints.
    :param ckpt_every: save a network checkpoint every N epochs (the last
        epoch is always saved). The forward-pass moments, which are what
        re-scoring needs, are saved every epoch regardless.
    """
    utils = Utils()
    metric = HRCSoftmaxMetric(num_classes=10)
    tree_len = metric.hrc_softmax.len
    n_obs = metric.hrc_softmax.num_obs

    train_loader, val_loader, test_loader = load_datasets(batch_size, num_val)

    net = resnet18_cifar10(gain_w=gain_w, gain_b=gain_b)
    net.to_device("cuda")
    out_updater = OutputUpdater(net.device)

    # The gains, initialized once with a weakly informative prior (Algorithm 1).
    cals = {
        s: HSMGainCalibrator(
            num_classes=10,
            sharing=s,
            mu_gain_init=mu_gain_init,
            sigma_gain_init=sigma_gain_init,
        )
        for s in ("global", "level", "node")
    }
    # Both fixed-scale baselines are the same machinery with a deterministic
    # gain. Verified against the C++ obs_to_class to 3e-8: cuTAGI's effective
    # gain is `alpha`, not `1/alpha`.
    baselines = {
        "tagi_code": HSMGainCalibrator(
            num_classes=10,
            sharing="global",
            mu_gain_init=alpha,
            sigma_gain_init=0.0,
        ),
        "tagi_note": HSMGainCalibrator(
            num_classes=10,
            sharing="global",
            mu_gain_init=1.0 / alpha,
            sigma_gain_init=0.0,
        ),
    }

    history = []
    var_y = np.full((batch_size * n_obs,), sigma_v**2, dtype=np.float32)
    for epoch in range(num_epochs):
        if epoch > 1:
            sigma_v = exponential_scheduler(
                curr_v=sigma_v,
                min_v=sigma_v_min,
                decaying_factor=sigma_v_decay,
                curr_iter=epoch,
            )
            var_y = np.full((batch_size * n_obs,), sigma_v**2, dtype=np.float32)

        # ---- standard TAGI training, unmodified -------------------------
        net.train()
        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{num_epochs}",
            leave=False,
            disable=not sys.stdout.isatty(),
        )
        for x, labels in pbar:
            m_pred, v_pred = net(x)
            y, y_idx, _ = utils.label_to_obs(labels=labels, num_classes=10)
            out_updater.update_using_indices(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y[: len(y)],
                selected_idx=y_idx,
                delta_states=net.input_delta_z_buffer,
            )
            net.backward()
            net.step()

        # ---- Algorithm 1 on the validation split ------------------------
        mu_v, var_v, lab_v = forward_pass(net, val_loader, tree_len)
        for cal in cals.values():
            cal.calibrate(mu_v, var_v, lab_v)  # keeps mu_G, resets sigma_G^2

        # ---- test -------------------------------------------------------
        mu_t, var_t, lab_t = forward_pass(net, test_loader, tree_len)
        row = {
            "epoch": epoch + 1,
            "sigma_v": float(sigma_v),
            "gain_w": gain_w,
            "gain_b": gain_b,
        }
        for name, base in baselines.items():
            row[name] = score(base, mu_t, var_t, lab_t)
        for s, cal in cals.items():
            row[s] = score(cal, mu_t, var_t, lab_t)
            row[f"{s}_mu_gain"] = [float(v) for v in cal.mu_gain]
            row[f"{s}_sigma_gain"] = [float(v) for v in cal.sigma_gain]
        history.append(row)

        print(f"epoch {epoch + 1:3d} | sigma_v {sigma_v:.3f}")
        for name in ("tagi_code", "tagi_note"):
            b = row[name]
            g = alpha if name == "tagi_code" else 1.0 / alpha
            print(
                f"            | {name} (g={g:.3g}): err {b['error']:5.2f}% "
                f"ECE {b['ece']:.4f} NLL {b['nll']:.4f} Brier {b['brier']:.4f}"
            )
        for s in ("global", "level", "node"):
            r = row[s]
            print(
                f"            | {s:6s} gains: err {r['error']:5.2f}% "
                f"ECE {r['ece']:.4f} NLL {r['nll']:.4f} Brier {r['brier']:.4f} "
                f"| mu_G "
                + ", ".join(f"{v:.2f}" for v in cals[s].mu_gain[:6])
                + ("..." if cals[s].num_gains > 6 else "")
            )
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(history, f, indent=1)

        # Checkpoint the network and the forward-pass moments, so any scoring
        # question can be revisited without retraining.
        os.makedirs(ckpt_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(ckpt_dir, f"moments_epoch{epoch + 1:03d}.npz"),
            mu_val=mu_v.astype(np.float32),
            var_val=var_v.astype(np.float32),
            lab_val=lab_v.astype(np.int16),
            mu_test=mu_t.astype(np.float32),
            var_test=var_t.astype(np.float32),
            lab_test=lab_t.astype(np.int16),
        )
        if (epoch + 1) % ckpt_every == 0 or epoch + 1 == num_epochs:
            net.save(os.path.join(ckpt_dir, f"epoch{epoch + 1:03d}.bin"))

    # ---- final reliability diagram --------------------------------------
    curves = {
        f"cuTAGI code (gain {alpha:g})": reliability_curve(
            baselines["tagi_code"], mu_t, var_t, lab_t
        ),
        f"note Eq.(1) (gain {1 / alpha:.3g})": reliability_curve(
            baselines["tagi_note"], mu_t, var_t, lab_t
        ),
    }
    for s, cal in cals.items():
        curves[f"calibrated ({s})"] = reliability_curve(cal, mu_t, var_t, lab_t)
    fig_path = out_json.replace(".json", "_reliability.pdf")
    reliability_figure(curves, fig_path)
    print("saved", fig_path, "and", out_json)


if __name__ == "__main__":
    fire.Fire(main)
