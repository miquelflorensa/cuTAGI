"""Diagnostics for hierarchical softmax probability normalization.

This script mirrors the C++ implementation in src/cost.cpp and checks whether
the probabilities returned by the current hierarchical-softmax mapping sum to
one. If the optional cutagi extension is importable, it also compares the
mirrored NumPy implementation against the backend.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HRCSoftmaxMirror:
    obs: np.ndarray
    idx: np.ndarray
    path_len: np.ndarray
    num_obs: int
    length: int


def class_to_obs(num_classes: int) -> HRCSoftmaxMirror:
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")

    obs_by_class: list[list[float]] = [[] for _ in range(num_classes)]
    idx_by_class: list[list[int]] = [[] for _ in range(num_classes)]
    next_node_idx = 2

    def assign_paths(start_class: int, end_class: int, node_idx: int) -> None:
        nonlocal next_node_idx
        span = end_class - start_class
        if span <= 1:
            return

        left_size = (span + 1) // 2
        split_class = start_class + left_size

        for label in range(start_class, split_class):
            obs_by_class[label].append(1.0)
            idx_by_class[label].append(node_idx)
        for label in range(split_class, end_class):
            obs_by_class[label].append(-1.0)
            idx_by_class[label].append(node_idx)

        if left_size > 1:
            child_idx = next_node_idx
            next_node_idx += 1
            assign_paths(start_class, split_class, child_idx)

        right_size = end_class - split_class
        if right_size > 1:
            child_idx = next_node_idx
            next_node_idx += 1
            assign_paths(split_class, end_class, child_idx)

    assign_paths(0, num_classes, 1)

    path_len = np.array([len(path) for path in idx_by_class], dtype=np.int32)
    num_obs = int(path_len.max())
    obs = np.zeros((num_classes, num_obs), dtype=np.float32)
    idx = np.zeros((num_classes, num_obs), dtype=np.int32)
    for label in range(num_classes):
        obs[label, : path_len[label]] = obs_by_class[label]
        idx[label, : path_len[label]] = idx_by_class[label]

    return HRCSoftmaxMirror(
        obs=obs.reshape(-1),
        idx=idx.reshape(-1),
        path_len=path_len,
        num_obs=num_obs,
        length=num_classes - 1,
    )


def normcdf(values: np.ndarray) -> np.ndarray:
    return np.array(
        [0.5 * math.erfc(-float(value) / math.sqrt(2.0)) for value in values],
        dtype=np.float32,
    )


def obs_to_class(
    mz: np.ndarray,
    sz: np.ndarray,
    hrc: HRCSoftmaxMirror,
    num_classes: int,
    alpha: float = 3.0,
) -> np.ndarray:
    denom = np.sqrt((1.0 / alpha) ** 2 + sz[: hrc.length])
    p_z = normcdf(mz[: hrc.length] / denom)
    probs = np.zeros(num_classes, dtype=np.float32)

    for row in range(num_classes):
        prob = 1.0
        for col in range(int(hrc.path_len[row])):
            flat_idx = row * hrc.num_obs + col
            node_prob = float(p_z[hrc.idx[flat_idx] - 1])
            if hrc.obs[flat_idx] == -1.0:
                prob *= abs(node_prob - 1.0)
            else:
                prob *= node_prob
        probs[row] = prob

    return probs


def random_inputs(
    rng: np.random.Generator, length: int, mean_scale: float, var_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    mz = rng.normal(0.0, mean_scale, size=length).astype(np.float32)
    sz = rng.uniform(0.0, var_scale, size=length).astype(np.float32)
    return mz, sz


def summarize_case(
    num_classes: int,
    trials: int,
    seed: int,
    mean_scale: float,
    var_scale: float,
) -> dict[str, float]:
    hrc = class_to_obs(num_classes)
    rng = np.random.default_rng(seed)

    zero_mz = np.zeros(hrc.length, dtype=np.float32)
    zero_sz = np.zeros(hrc.length, dtype=np.float32)
    zero_probs = obs_to_class(zero_mz, zero_sz, hrc, num_classes)

    sums = np.empty(trials, dtype=np.float32)
    for trial in range(trials):
        mz, sz = random_inputs(rng, hrc.length, mean_scale, var_scale)
        sums[trial] = float(obs_to_class(mz, sz, hrc, num_classes).sum())

    return {
        "num_classes": float(num_classes),
        "num_obs": float(hrc.num_obs),
        "length": float(hrc.length),
        "zero_sum": float(zero_probs.sum()),
        "min_sum": float(sums.min()),
        "mean_sum": float(sums.mean()),
        "max_sum": float(sums.max()),
        "max_abs_sum_error": float(np.max(np.abs(sums - 1.0))),
    }


def summarize_model_outputs(
    mz_file: str,
    sz_file: str,
    num_classes: int,
) -> None:
    hrc = class_to_obs(num_classes)
    mz = np.load(mz_file).astype(np.float32).reshape(-1)
    sz = np.load(sz_file).astype(np.float32).reshape(-1)

    if mz.size != sz.size:
        raise ValueError(
            f"mz and sz must have the same size, got {mz.size} and {sz.size}"
        )
    if mz.size % hrc.length != 0:
        raise ValueError(
            f"array size {mz.size} is not divisible by HRC length {hrc.length}"
        )

    batch_size = mz.size // hrc.length
    sums = np.empty(batch_size, dtype=np.float32)
    for row in range(batch_size):
        start = row * hrc.length
        stop = start + hrc.length
        probs = obs_to_class(mz[start:stop], sz[start:stop], hrc, num_classes)
        sums[row] = float(probs.sum())

    worst_low = int(np.argmin(sums))
    worst_high = int(np.argmax(sums))
    print(
        "model_outputs "
        f"batch_size={batch_size} "
        f"min_sum={sums[worst_low]:.8f}@{worst_low} "
        f"mean_sum={sums.mean():.8f} "
        f"max_sum={sums[worst_high]:.8f}@{worst_high} "
        f"max_abs_sum_error={np.max(np.abs(sums - 1.0)):.8f}"
    )


def compare_cutagi(num_classes: int) -> None:
    try:
        from pytagi import Utils
    except ImportError:
        print("cutagi comparison: skipped (pytagi/cutagi is not importable)")
        return

    utils = Utils()
    backend_hrc = utils.get_hierarchical_softmax(num_classes)
    mirror_hrc = class_to_obs(num_classes)

    mz = np.linspace(-1.0, 1.0, mirror_hrc.length, dtype=np.float32)
    sz = np.full(mirror_hrc.length, 0.05, dtype=np.float32)
    backend_prob = utils.obs_to_label_prob(mz, sz, backend_hrc, num_classes)
    mirror_prob = obs_to_class(mz, sz, mirror_hrc, num_classes)

    print(
        "cutagi comparison: max_abs_diff="
        f"{np.max(np.abs(backend_prob - mirror_prob)):.8g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 8, 10, 16, 100, 1000],
    )
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mean-scale", type=float, default=2.0)
    parser.add_argument("--var-scale", type=float, default=0.5)
    parser.add_argument("--mz-file")
    parser.add_argument("--sz-file")
    parser.add_argument("--compare-cutagi", action="store_true")
    args = parser.parse_args()

    header = (
        "classes obs len zero_sum min_sum mean_sum max_sum max_abs_sum_error"
    )
    print(header)
    for num_classes in args.classes:
        stats = summarize_case(
            num_classes=num_classes,
            trials=args.trials,
            seed=args.seed,
            mean_scale=args.mean_scale,
            var_scale=args.var_scale,
        )
        print(
            f"{int(stats['num_classes']):7d} "
            f"{int(stats['num_obs']):3d} "
            f"{int(stats['length']):3d} "
            f"{stats['zero_sum']:.8f} "
            f"{stats['min_sum']:.8f} "
            f"{stats['mean_sum']:.8f} "
            f"{stats['max_sum']:.8f} "
            f"{stats['max_abs_sum_error']:.8f}"
        )

    if args.compare_cutagi:
        compare_cutagi(args.classes[0])

    if args.mz_file or args.sz_file:
        if not args.mz_file or not args.sz_file:
            raise ValueError("--mz-file and --sz-file must be provided together")
        summarize_model_outputs(args.mz_file, args.sz_file, args.classes[0])


if __name__ == "__main__":
    main()
