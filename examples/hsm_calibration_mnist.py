"""MNIST with the hierarchical softmax calibration of Section 4.

Standard TAGI training (the loop is untouched, +/-1 targets in logit space),
plus the auxiliary channel of the note: after each epoch, Algorithm 1 runs on a
held-out validation split and updates the Gaussian gains :math:`G_h` only. The
test set is then scored twice: with TAGI's fixed scale :math:`\\alpha` of
Eq. (1) and with the learned gains of (E:class).

Usage::

    python -m examples.hsm_calibration_mnist
    python -m examples.hsm_calibration_mnist --sharing node --num_epochs 5
"""

import os
import sys

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import fire
import numpy as np
from tqdm import tqdm

import pytagi
from examples.data_loader import MnistDataLoader
from pytagi import HRCSoftmaxMetric, Utils
from pytagi.hsm_calibration import HSMGainCalibrator
from pytagi.nn import Linear, MixtureReLU, OutputUpdater, Sequential

FNN = Sequential(
    Linear(784, 128),
    MixtureReLU(),
    Linear(128, 128),
    MixtureReLU(),
    Linear(128, 11),
)


def expected_calibration_error(
    prob: np.ndarray, labels: np.ndarray, n_bins: int = 15
):
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def negative_log_likelihood(prob: np.ndarray, labels: np.ndarray):
    p = np.clip(prob[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def forward_pass(net, dtl, batch_size: int, tree_len: int):
    """Collect the forward-pass moments of the output layer, no update."""
    mu, var, lab = [], [], []
    net.eval()
    for x, _, _, label in dtl.create_data_loader(batch_size, shuffle=False):
        m_pred, v_pred = net(x)
        n = len(label)
        mu.append(np.asarray(m_pred, dtype=np.float64).reshape(n, tree_len))
        var.append(np.asarray(v_pred, dtype=np.float64).reshape(n, tree_len))
        lab.append(np.asarray(label).reshape(n))
    return np.concatenate(mu), np.concatenate(var), np.concatenate(lab)


def score(cal: HSMGainCalibrator, mu, var, labels, name: str):
    prob, prob_var = cal.class_probabilities(mu, var)
    pred = prob.argmax(axis=1)
    err = 100.0 * float(np.mean(pred != labels))
    print(
        f"  {name:28s} error {err:5.2f}%  ECE {expected_calibration_error(prob, labels):.4f}"
        f"  NLL {negative_log_likelihood(prob, labels):.4f}"
        f"  mean sigma_Pr {float(np.sqrt(prob_var[np.arange(len(labels)), pred]).mean()):.4f}"
    )
    return err


def main(
    num_epochs: int = 5,
    batch_size: int = 64,
    sigma_v: float = 0.05,
    num_val: int = 5000,
    sharing: str = "level",
    mu_gain_init: float = 0.3,
    sigma_gain_init: float = 1.0,
    alpha: float = 3.0,
):
    """
    :param num_val: size of the validation split held out of the 60 000
        training images; the calibration pass runs there.
    :param sharing: gain sharing, ``"node"``, ``"level"`` or ``"global"``.
    :param alpha: the fixed scale of TAGI's Eq. (1), used as the baseline.
    """
    utils = Utils()
    metric = HRCSoftmaxMetric(num_classes=10)
    tree_len = metric.hrc_softmax.len

    full_dtl = MnistDataLoader(
        x_file="data/mnist/train-images-idx3-ubyte",
        y_file="data/mnist/train-labels-idx1-ubyte",
        num_images=60000,
    )
    test_dtl = MnistDataLoader(
        x_file="data/mnist/t10k-images-idx3-ubyte",
        y_file="data/mnist/t10k-labels-idx1-ubyte",
        num_images=10000,
    )

    # Hold out the last `num_val` images as the validation split.
    x, y, y_idx, labels = full_dtl.dataset["value"]
    n_train = labels.shape[0] - num_val
    train_dtl = MnistDataLoader.__new__(MnistDataLoader)
    val_dtl = MnistDataLoader.__new__(MnistDataLoader)
    train_dtl.dataset = {
        "value": (x[:n_train], y[:n_train], y_idx[:n_train], labels[:n_train])
    }
    val_dtl.dataset = {
        "value": (x[n_train:], y[n_train:], y_idx[n_train:], labels[n_train:])
    }
    n_obs = metric.hrc_softmax.num_obs

    net = FNN
    if pytagi.cuda.is_available():
        net.to_device("cuda")
    else:
        net.set_threads(8)
    out_updater = OutputUpdater(net.device)

    # The gains, initialized once with a weakly informative prior (Algorithm 1).
    cal = HSMGainCalibrator(
        num_classes=10,
        sharing=sharing,
        mu_gain_init=mu_gain_init,
        sigma_gain_init=sigma_gain_init,
    )
    # TAGI's Eq. (1) is the same machinery with a deterministic gain 1/alpha.
    baseline = HSMGainCalibrator(
        num_classes=10,
        sharing="global",
        mu_gain_init=1.0 / alpha,
        sigma_gain_init=0.0,
    )

    var_y = np.full((batch_size * n_obs,), sigma_v**2, dtype=np.float32)
    for epoch in range(num_epochs):
        # ---- standard TAGI training, unmodified -------------------------
        net.train()
        pbar = tqdm(
            train_dtl.create_data_loader(batch_size=batch_size),
            desc=f"epoch {epoch + 1}/{num_epochs}",
        )
        for x, y, y_idx, label in pbar:
            m_pred, v_pred = net(x)
            out_updater.update_using_indices(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y[: len(y)],
                selected_idx=y_idx,
                delta_states=net.input_delta_z_buffer,
            )
            net.backward()
            net.step()

        # ---- Algorithm 1: gain calibration pass on the validation split --
        mu_v, var_v, lab_v = forward_pass(net, val_dtl, batch_size, tree_len)
        cal.calibrate(mu_v, var_v, lab_v)  # keeps mu_G, resets sigma_G^2

        # ---- test -------------------------------------------------------
        mu_t, var_t, lab_t = forward_pass(net, test_dtl, batch_size, tree_len)
        print(f"epoch {epoch + 1}:")
        score(baseline, mu_t, var_t, lab_t, f"TAGI Eq.(1), alpha={alpha}")
        score(cal, mu_t, var_t, lab_t, f"calibrated ({sharing} gains)")
        print(
            "  gains mu_G = "
            + ", ".join(f"{m:.3f}" for m in cal.mu_gain)
            + " | sigma_G = "
            + ", ".join(f"{s:.3f}" for s in cal.sigma_gain)
        )


if __name__ == "__main__":
    fire.Fire(main)
