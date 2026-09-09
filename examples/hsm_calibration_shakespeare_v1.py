"""Hierarchical softmax calibration on the TAGI attention model that learns.

Built on ``examples.next_token_predictor`` (v1). The v2 predictor was tried
first and rejected: its attention collapses to uniform and its CE plateaus
barely below the unigram baseline, so its calibration numbers say nothing
about attention. v1 reaches ~1.9 nats at 80 epochs and keeps improving.

v1 also matches the note's assumptions more closely than v2 did: it uses the
sparse path-only logit update (``update_using_indices`` with +/-1 targets on the
H nodes of the true path), which is the logit channel of Section 4.1, whereas v2
pushed a dense target onto all 70 tree nodes.

``obs_scale`` scales the logit targets to +/-2; predictions are divided by it
(and the variances by its square) before any probability is formed, exactly as
v1's own scoring does.

Usage::

    python -m examples.hsm_calibration_shakespeare_v1
    python -m examples.hsm_calibration_shakespeare_v1 --num_epochs 500 --eval_every 20
"""

import os
import sys

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import json

import fire
import numpy as np
from tqdm import tqdm

import pytagi
from examples.next_token_predictor import CharDataset, build_mingpt
from pytagi import Utils
from pytagi.hsm_calibration import (
    HSMGainCalibrator,
    log_sum_exp,
    standard_normal_log_cdf,
)
from pytagi.nn import OutputUpdater

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "shakespeare", "input.txt"
)


def log_class_probs(cal: HSMGainCalibrator, mu, var):
    """(E:class) mean in log space; identical to prod_h mu_{P_h}, underflow-free."""
    nm = cal.unit_moments(mu, var, compute_var=False)
    lp_plus = standard_normal_log_cdf(nm.a)
    lp_minus = standard_normal_log_cdf(-nm.a)
    signs = cal.tree.obs[None, :, :]
    lp = np.where(
        signs > 0, lp_plus[:, cal.tree.idx], lp_minus[:, cal.tree.idx]
    ).sum(-1)
    return lp - log_sum_exp(lp, axis=1, keepdims=True)


def metrics(cal, mu, var, labels, n_bins: int = 15):
    log_prob = log_class_probs(cal, mu, var)
    pred = log_prob.argmax(axis=1)
    conf = np.exp(log_prob.max(axis=1))
    correct = (pred == labels).astype(float)
    b = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    lp_true = np.maximum(log_prob[np.arange(len(labels)), labels], -50.0)
    nll = float(-np.mean(lp_true))
    prob = np.exp(log_prob)
    onehot = np.zeros_like(prob)
    onehot[np.arange(len(labels)), labels] = 1.0
    return {
        "acc": float(100 * correct.mean()),
        "ece": float(e),
        "nll": nll,
        "ppl": float(np.exp(nll)),
        "brier": float(np.mean(np.sum((prob - onehot) ** 2, axis=1))),
        "mean_conf": float(conf.mean()),
    }


def make_eval_set(data, seq_len, n_seq, seed):
    """A fixed set of sequences, so every evaluation sees the same tokens."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - seq_len - 1, size=n_seq)
    x = np.stack([data[s : s + seq_len] for s in starts])
    y = np.stack([data[s + 1 : s + 1 + seq_len] for s in starts])
    return (
        x.reshape(n_seq, seq_len, 1).astype(np.float32),
        y.reshape(-1).astype(np.int64),
    )


def forward_eval(net, x, tree_len, batch_size, obs_scale):
    """Forward-pass moments of the output layer, rescaled by ``obs_scale``."""
    mu, var = [], []
    net.eval()
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size]
        m, v = net(xb)
        n = xb.shape[0] * xb.shape[1]
        mu.append(
            np.asarray(m, dtype=np.float64).reshape(n, tree_len) / obs_scale
        )
        var.append(
            np.asarray(v, dtype=np.float64).reshape(n, tree_len) / obs_scale**2
        )
    return np.concatenate(mu), np.concatenate(var)


def main(
    num_epochs: int = 500,
    eval_every: int = 20,
    batch_size: int = 32,
    seq_len: int = 64,
    embed_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 1,
    ffn_hidden: int = 2048,
    steps_per_epoch: int = 200,
    sigma_v: float = 10.0,
    sigma_v_min: float = 2.0,
    decay_factor: float = 0.96,
    qkv_gain: float = 0.25,
    gain_w_rms: float = 5.0,
    embed_scale: float = 0.15,
    obs_scale: float = 2.0,
    alpha: float = 3.0,
    n_val_seq: int = 640,
    n_test_seq: int = 1280,
    mu_gain_init: float = 0.3,
    sigma_gain_init: float = 1.0,
    out_json: str = "saved_results/hsm_shakespeare_v1.json",
    ckpt_dir: str = "saved_results/ckpt_hsm_shakespeare_v1",
    resume: str = "",
    resume_epoch: int = 0,
    seed: int = 1384,
):
    """Defaults are v1's own, except the 90/5/5 split and the eval cadence.

    :param resume: checkpoint to continue from. ``num_epochs`` is then the
        number of ADDITIONAL epochs, and history is appended to ``out_json``.
    :param resume_epoch: epoch number that checkpoint corresponds to; sigma_v
        is replayed through its schedule to that point. The data-sampling RNG
        stream is not restored, so batches differ from an uninterrupted run.
    :param seed: seeds the weight init and the training-batch sampling. The
        val/test sets keep their own fixed seeds (11, 22) so runs at different
        seeds are scored on exactly the same tokens.
    """
    np.random.seed(seed)
    pytagi.manual_seed(seed)

    text = open(DATA_PATH, "r").read()
    dataset = CharDataset(text, seq_len)
    vocab_size = dataset.vocab_size
    utils = Utils()
    hrc = utils.get_hierarchical_softmax(vocab_size)
    tree_len = hrc.len
    print(f"vocab {vocab_size}, H {hrc.num_obs}, tree nodes {tree_len}")

    n = len(dataset.data)
    n_tr, n_va = int(0.90 * n), int(0.05 * n)
    train_data = dataset.data[:n_tr]
    val_data = dataset.data[n_tr : n_tr + n_va]
    test_data = dataset.data[n_tr + n_va :]
    dataset.data = train_data

    x_val, y_val = make_eval_set(val_data, seq_len, n_val_seq, seed=11)
    x_test, y_test = make_eval_set(test_data, seq_len, n_test_seq, seed=22)
    print(f"calibration tokens {y_val.size}, test tokens {y_test.size}")

    net = build_mingpt(
        vocab_size=vocab_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_hidden=ffn_hidden,
        output_size=tree_len,
        qkv_gain=qkv_gain,
        gain_w_rms=gain_w_rms,
        embed_scale=embed_scale,
    )
    net.to_device("cuda" if pytagi.cuda.is_available() else "cpu")
    if resume:
        net.load(resume)
        print(f"resumed from {resume} at epoch {resume_epoch}")
    out_updater = OutputUpdater(net.device)

    SHARINGS = ("global", "level", "node")

    def new_cal(sharing):
        return HSMGainCalibrator(
            vocab_size,
            sharing=sharing,
            mu_gain_init=mu_gain_init,
            sigma_gain_init=sigma_gain_init,
        )

    cals = {s: new_cal(s) for s in SHARINGS}
    baseline = HSMGainCalibrator(
        vocab_size, sharing="global", mu_gain_init=alpha, sigma_gain_init=0.0
    )

    history, train_log = [], []
    current_sigma_v = sigma_v
    if resume:
        for _ in range(resume_epoch):  # replay the sigma_v schedule
            current_sigma_v = max(sigma_v_min, current_sigma_v * decay_factor)
        if os.path.exists(out_json):
            prev = json.load(open(out_json))
            history, train_log = prev["evals"], prev["train"]
            print(
                f"appending to {len(history)} evals / {len(train_log)} train rows"
            )
            if history:  # carry the warm gains, as the note prescribes
                for sh in SHARINGS:
                    key = f"{sh}_mu_gain"
                    if key in history[-1]:
                        cals[sh].mu_gain[:] = np.asarray(history[-1][key])
                print(
                    "carried warm gains:",
                    [f"{v:.2f}" for v in cals["level"].mu_gain],
                )
    num_samples = batch_size * seq_len
    epoch0 = resume_epoch

    for epoch in tqdm(
        range(epoch0, epoch0 + num_epochs),
        desc="training",
        disable=not sys.stdout.isatty(),
    ):
        var_y = np.full(
            (batch_size * seq_len * hrc.num_obs,),
            current_sigma_v**2,
            dtype=np.float32,
        )
        net.train()
        errs, ces = [], []
        for _ in range(steps_per_epoch):
            x, labels = dataset.next_batch(batch_size)
            m_pred, v_pred = net(x)
            y_obs, y_idx, _ = utils.label_to_obs(
                labels=labels, num_classes=vocab_size
            )
            out_updater.update_using_indices(
                output_states=net.output_z_buffer,
                mu_obs=np.asarray(y_obs, dtype=np.float32) * obs_scale,
                var_obs=var_y,
                selected_idx=y_idx,
                delta_states=net.input_delta_z_buffer,
            )
            net.backward()
            net.step()

            # v1's own training metrics: rescale, then C++ gain-3 probabilities
            m_m = (np.asarray(m_pred) / obs_scale).tolist()
            v_m = (np.asarray(v_pred) / obs_scale**2).tolist()
            er, prob = utils.get_errors(
                m_m, v_m, labels, vocab_size, num_samples
            )
            errs.append(float(np.mean(er)))
            p = np.asarray(prob).reshape(num_samples, vocab_size)
            p = p / p.sum(axis=1, keepdims=True)
            ces.append(
                float(
                    -np.log(
                        np.clip(p[np.arange(num_samples), labels], 1e-9, 1.0)
                    ).mean()
                )
            )

        train_ce = float(np.mean(ces[-100:]))
        train_err = float(np.mean(errs[-100:]))
        train_log.append(
            {
                "epoch": epoch + 1,
                "sigma_v": float(current_sigma_v),
                "train_ce": train_ce,
                "train_err": train_err,
            }
        )
        current_sigma_v = max(sigma_v_min, current_sigma_v * decay_factor)

        if (epoch + 1) % eval_every == 0 or epoch + 1 == num_epochs:
            mu_v, var_v = forward_eval(
                net, x_val, tree_len, batch_size, obs_scale
            )
            cold = {s: new_cal(s) for s in SHARINGS}
            for c in list(cals.values()) + list(cold.values()):
                c.calibrate(mu_v, var_v, y_val)
            mu_t, var_t = forward_eval(
                net, x_test, tree_len, batch_size, obs_scale
            )

            snr = float(np.mean(np.abs(mu_v) / np.sqrt(var_v)))
            row = {
                "epoch": epoch + 1,
                "sigma_v": float(current_sigma_v),
                "snr": snr,
                "train_ce": train_ce,
                "train_err": train_err,
                "tagi": metrics(baseline, mu_t, var_t, y_test),
            }
            for s in SHARINGS:
                row[s] = metrics(cals[s], mu_t, var_t, y_test)
                row[f"{s}_mu_gain"] = [float(v) for v in cals[s].mu_gain]
                row[s + "_cold"] = metrics(cold[s], mu_t, var_t, y_test)
            history.append(row)

            print(
                f"\nepoch {epoch + 1:4d} | sigma_v {current_sigma_v:.3f} | "
                f"snr {snr:.3f} | train ce {train_ce:.4f} err {train_err:.4f}"
            )
            for k in (
                "tagi",
                "global",
                "level",
                "node",
                "global_cold",
                "level_cold",
                "node_cold",
            ):
                m = row[k]
                print(
                    f"  {k:12s} acc {m['acc']:5.2f}%  ECE {m['ece']:.4f}  "
                    f"CE {m['nll']:.4f}  bits {m['nll'] / np.log(2):.4f}  "
                    f"Brier {m['brier']:.4f}  conf {m['mean_conf']:.3f}"
                )
            print(
                "  gains  global %.2f  level " % cals["global"].mu_gain[0]
                + " ".join(f"{v:.2f}" for v in cals["level"].mu_gain)
            )
            sys.stdout.flush()

            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            with open(out_json, "w") as f:
                json.dump({"evals": history, "train": train_log}, f, indent=1)
            os.makedirs(ckpt_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(ckpt_dir, f"moments_epoch{epoch + 1:04d}.npz"),
                mu_val=mu_v.astype(np.float32),
                var_val=var_v.astype(np.float32),
                lab_val=y_val.astype(np.int16),
                mu_test=mu_t.astype(np.float32),
                var_test=var_t.astype(np.float32),
                lab_test=y_test.astype(np.int16),
            )
            net.save(os.path.join(ckpt_dir, f"epoch{epoch + 1:04d}.bin"))

    print("done")


if __name__ == "__main__":
    fire.Fire(main)
