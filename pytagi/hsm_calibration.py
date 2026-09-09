"""Hierarchical softmax calibration for TAGI.

TAGI's hierarchical binary decomposition collapses each factor of the class
probability to its expected value with a fixed scale alpha. This module instead
treats each factor P_h = Phi(s_h G_h Z_h) as a random variable with closed-form
mean, variance and covariances, and replaces the fixed 1/alpha by a Gaussian
gain G_h learned from the class labels through an auxiliary observation model.

The gains are fitted on a held-out validation split by a scalar recursion per
node, from forward-pass moments only. The training loop and the C++/CUDA
backend are not involved.

Docstrings below tag each quantity with the equation label it implements, from
the reference note "Hierarchical Softmax Calibration" (Goulet, Nguyen &
Florensa-Montilla).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "standard_normal_cdf",
    "standard_normal_pdf",
    "standard_normal_log_cdf",
    "log_sum_exp",
    "phi_mean",
    "phi_variance",
    "phi_expected_bernoulli_variance",
    "phi_covariance",
    "NodeMoments",
    "node_moments",
    "conditional_branch_prob",
    "conditional_branch_prob_derivative",
    "class_moments",
    "class_covariance",
    "gain_update",
    "HSMTree",
    "HSMGainCalibrator",
    "HSMCalibratedMetric",
]


_GL8_X, _GL8_W = np.polynomial.legendre.leggauss(8)
_GL8_X = _GL8_X.astype(np.float64)
_GL8_W = _GL8_W.astype(np.float64)

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

# numpy ships no erfc, so the libm one is vectorized elementwise. It is exact
# to the last bit, and the calibration recursion is scalar anyway; only the
# prediction path evaluates arrays, once per epoch.
_erfc = np.vectorize(math.erfc, otypes=[np.float64])

# Below this argument erfc(-x / sqrt(2)) underflows float64 and log(Phi) is
# continued with its asymptotic series instead.
_LOG_CDF_TAIL = -37.5


def standard_normal_cdf(x):
    """Standard normal CDF Phi(x)."""
    return 0.5 * _erfc(-np.asarray(x, dtype=np.float64) / _SQRT2)


def standard_normal_pdf(x):
    """Standard normal PDF phi(x)."""
    x = np.asarray(x, dtype=np.float64)
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def standard_normal_log_cdf(x):
    """Log of the standard normal CDF, without underflow in the left tail.

    Used to build the class probability in log space, where the product over the
    path nodes would underflow. Past x = -37.5 the erfc itself underflows and an
    asymptotic expansion is continued instead.

    :param x: The argument.
    :type x: np.ndarray
    :return: log Phi(x), same shape as the input.
    :rtype: np.ndarray
    """
    x = np.asarray(x, dtype=np.float64)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    out = np.empty_like(x)

    # x > 0: log1p of the complementary tail, so log(Phi) -> 0 keeps its
    # relative accuracy instead of rounding to exactly zero.
    right = x > 0.0
    if np.any(right):
        out[right] = np.log1p(-0.5 * _erfc(x[right] / _SQRT2))
    head = (x <= 0.0) & (x > _LOG_CDF_TAIL)
    if np.any(head):
        out[head] = np.log(0.5 * _erfc(-x[head] / _SQRT2))
    tail = x <= _LOG_CDF_TAIL
    if np.any(tail):
        z = x[tail]
        u = 1.0 / (z * z)
        series = 1.0 - u * (1.0 - u * (3.0 - u * (15.0 - 105.0 * u)))
        out[tail] = -0.5 * z * z - np.log(-z) - _LOG_SQRT_2PI + np.log(series)
    return out[0] if scalar else out


def log_sum_exp(x, axis=None, keepdims=False):
    """Log of a sum of exponentials, shifted by the maximum to avoid overflow.

    :param x: The values to reduce.
    :type x: np.ndarray
    :param axis: Axis to reduce over, or None for all axes.
    :type axis: int
    :param keepdims: Keep the reduced axis as a singleton.
    :type keepdims: bool
    :return: The reduced log-sum-exp.
    :rtype: np.ndarray
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    if keepdims:
        return out
    if axis is None:
        return out.reshape(())[()]
    return np.squeeze(out, axis=axis)


def phi_mean(mu, var):
    """Mean of Phi(X) for a Gaussian X, equal to Phi(a) (E:mean)."""
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    return standard_normal_cdf(mu / np.sqrt(1.0 + var))


def phi_variance(a, var):
    """Variance of Phi(X) for a Gaussian X, from the integral form (E:varint).

    An 8-point Gauss-Legendre rule on [b, 1] with b = 1 / sqrt(1 + 2 sigma^2).
    Positive by construction, and free of the cancellation that (E:var) suffers
    for small sigma, so Owen's T is never needed.

    :param a: The standardized mean, mu / sqrt(1 + sigma^2).
    :type a: np.ndarray
    :param var: The variance sigma^2 of the Gaussian argument.
    :type var: np.ndarray
    :return: Var[Phi(X)].
    :rtype: np.ndarray
    """
    a = np.asarray(a, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    b = 1.0 / np.sqrt(1.0 + 2.0 * var)

    half = 0.5 * (1.0 - b)
    mid = 0.5 * (1.0 + b)
    t = mid[..., None] + half[..., None] * _GL8_X
    integrand = np.exp(-0.5 * a[..., None] ** 2 * (1.0 + t * t)) / (1.0 + t * t)
    return (half / math.pi) * np.sum(integrand * _GL8_W, axis=-1)


def phi_expected_bernoulli_variance(a, var):
    """Expected Bernoulli variance E[Phi(X) (1 - Phi(X))] (E:var).

    This is the noise variance of the auxiliary observation model (E:obs). It is
    obtained as Phi(a) (1 - Phi(a)) - Var[Phi(X)].
    """
    a = np.asarray(a, dtype=np.float64)
    m = standard_normal_cdf(a)
    return m * (1.0 - m) - phi_variance(a, var)


def phi_covariance(mu, var):
    """Covariance between a Gaussian X and Phi(X), by Stein's identity (E:cov)."""
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    denom = np.sqrt(1.0 + var)
    return var * standard_normal_pdf(mu / denom) / denom


@dataclass
class NodeMoments:
    """Moments of the branch probability P_h = Phi(s_h G_h Z_h) (E:node).

    :ivar mu_p: Mean of the branch probability.
    :ivar var_p: Variance of the branch probability, None when not requested.
    :ivar cov_zp: Covariance between the hidden state and the branch probability.
    :ivar cov_gp: Covariance between the gain and the branch probability.
    :ivar mu_s: Mean of the scaled logit S_h = G_h Z_h.
    :ivar var_s: Variance of the scaled logit.
    :ivar a: Standardized mean of the scaled logit (E:a).
    :ivar b: Variance-form argument of (E:a).
    """

    mu_p: np.ndarray
    var_p: Optional[np.ndarray]
    cov_zp: np.ndarray
    cov_gp: np.ndarray
    mu_s: np.ndarray
    var_s: np.ndarray
    a: np.ndarray
    b: np.ndarray


def node_moments(
    mu_z,
    var_z,
    mu_g,
    var_g,
    s=1.0,
    compute_var: bool = True,
) -> NodeMoments:
    """Node-level moments of the branch probability P_h = Phi(s_h G_h Z_h).

    The scaled logit S_h = G_h Z_h is treated as Gaussian with its exact moments
    (E:S), which gives the mean and covariances of (E:node) in closed form. The
    approximation is exact when either the hidden state or the gain is
    deterministic.

    :param mu_z: Forward-pass mean of the output unit.
    :type mu_z: np.ndarray
    :param var_z: Forward-pass variance of the output unit.
    :type var_z: np.ndarray
    :param mu_g: Mean of the gain.
    :type mu_g: np.ndarray
    :param var_g: Variance of the gain.
    :type var_g: np.ndarray
    :param s: Branch sign, +1 or -1.
    :type s: float
    :param compute_var: Also evaluate the branch variance (E:cond). The
        calibration pass does not need it; prediction does.
    :type compute_var: bool
    :return: The node moments.
    :rtype: NodeMoments
    """
    mu_z = np.asarray(mu_z, dtype=np.float64)
    var_z = np.asarray(var_z, dtype=np.float64)
    mu_g = np.asarray(mu_g, dtype=np.float64)
    var_g = np.asarray(var_g, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    mu_s = mu_g * mu_z  # (E:S)
    var_s = var_g * var_z + var_g * mu_z**2 + mu_g**2 * var_z  # (E:S)

    one_p = 1.0 + var_s
    a = s * mu_s / np.sqrt(one_p)  # (E:a)
    b = 1.0 / np.sqrt(1.0 + 2.0 * var_s)  # (E:a)

    mu_p = standard_normal_cdf(a)  # (E:node)
    pdf_a = standard_normal_pdf(a)
    denom = one_p * np.sqrt(one_p)  # (1+var_s)^{3/2}

    cov_zp = s * var_z * mu_g * (one_p - var_g * mu_z**2) / denom * pdf_a
    cov_gp = s * var_g * mu_z * (one_p - mu_g**2 * var_z) / denom * pdf_a

    var_p = phi_variance(a, var_s) if compute_var else None

    return NodeMoments(
        mu_p=mu_p,
        var_p=var_p,
        cov_zp=cov_zp,
        cov_gp=cov_gp,
        mu_s=mu_s,
        var_s=var_s,
        a=a,
        b=b,
    )


def conditional_branch_prob(g, mu_z, var_z, s=1.0):
    """Branch probability conditional on a fixed gain g (E:mg)."""
    g = np.asarray(g, dtype=np.float64)
    mu_z = np.asarray(mu_z, dtype=np.float64)
    var_z = np.asarray(var_z, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    return standard_normal_cdf(s * g * mu_z / np.sqrt(1.0 + g**2 * var_z))


def conditional_branch_prob_derivative(g, mu_z, var_z, s=1.0):
    """Derivative with respect to g of the conditional branch probability (E:mg).

    For a small gain variance, Cov(G, P) tends to var_g times this derivative
    evaluated at the gain mean.
    """
    g = np.asarray(g, dtype=np.float64)
    mu_z = np.asarray(mu_z, dtype=np.float64)
    var_z = np.asarray(var_z, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    d = 1.0 + g**2 * var_z
    return (
        s
        * mu_z
        * standard_normal_pdf(s * g * mu_z / np.sqrt(d))
        / (d * np.sqrt(d))
    )


def class_moments(mu_p, var_p, axis: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and variance of the class probability, the product over path nodes.

    The mean is the product of the node means and the variance follows from
    (E:class). With independent nodes these are the exact moments of the product.

    :param mu_p: Node means along ``axis``.
    :type mu_p: np.ndarray
    :param var_p: Node variances along ``axis``.
    :type var_p: np.ndarray
    :param axis: Axis holding the nodes of the path.
    :type axis: int
    :return: The mean and variance of the class probability.
    :rtype: tuple
    """
    mu_p = np.asarray(mu_p, dtype=np.float64)
    var_p = np.asarray(var_p, dtype=np.float64)
    mu_pr = np.prod(mu_p, axis=axis)
    var_pr = np.prod(var_p + mu_p**2, axis=axis) - np.prod(mu_p**2, axis=axis)
    return mu_pr, var_pr


def class_covariance(mu_p, cov_px, axis: int = -1) -> np.ndarray:
    """Covariance between the class probability and an on-path variable (E:class).

    Equal to the node covariance times the product of the other node means. The
    leave-one-out product is accumulated rather than divided out, so a saturated
    node mean cannot blow it up.

    :param mu_p: Node means along ``axis``.
    :type mu_p: np.ndarray
    :param cov_px: Node covariance for each node, same shape as ``mu_p``.
    :type cov_px: np.ndarray
    :param axis: Axis holding the nodes of the path.
    :type axis: int
    :return: The covariance for each on-path variable. It is zero for output
        units off the path.
    :rtype: np.ndarray
    """
    mu_p = np.asarray(mu_p, dtype=np.float64)
    cov_px = np.asarray(cov_px, dtype=np.float64)
    mu_moved = np.moveaxis(mu_p, axis, -1)
    fwd = np.cumprod(
        np.concatenate(
            [np.ones_like(mu_moved[..., :1]), mu_moved[..., :-1]], axis=-1
        ),
        axis=-1,
    )
    bwd = np.cumprod(
        np.concatenate(
            [np.ones_like(mu_moved[..., :1]), mu_moved[..., :0:-1]], axis=-1
        ),
        axis=-1,
    )[..., ::-1]
    leave_one_out = np.moveaxis(fwd * bwd, -1, axis)
    return cov_px * leave_one_out


def gain_update(
    mu_g,
    var_g,
    mu_p,
    cov_gp,
    y: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gain update of the auxiliary observation channel (E:update).

    Conditions the joint Gaussian of the gain and the observation, whose variance
    is the Bernoulli variance mu_P (1 - mu_P) of (E:obs). The mean step is exact
    under the Bernoulli likelihood; the variance step is exact on average over the
    label. The mean of the branch probability is clipped away from 0 and 1 and the
    posterior variance is floored at zero, as Section 4.2 prescribes.

    :param mu_g: Mean of the gain.
    :type mu_g: np.ndarray
    :param var_g: Variance of the gain.
    :type var_g: np.ndarray
    :param mu_p: Mean of the branch probability.
    :type mu_p: np.ndarray
    :param cov_gp: Covariance between the gain and the branch probability.
    :type cov_gp: np.ndarray
    :param y: The observation, 1 when the branch was taken and 0 otherwise.
    :type y: float
    :param eps: Clipping bound applied to ``mu_p``.
    :type eps: float
    :return: The posterior mean and variance of the gain.
    :rtype: tuple
    """
    mu_g = np.asarray(mu_g, dtype=np.float64)
    var_g = np.asarray(var_g, dtype=np.float64)
    cov_gp = np.asarray(cov_gp, dtype=np.float64)
    mu_p = np.clip(np.asarray(mu_p, dtype=np.float64), eps, 1.0 - eps)

    var_y = mu_p * (1.0 - mu_p)
    mu_post = mu_g + cov_gp * (y - mu_p) / var_y
    var_post = np.maximum(var_g - cov_gp**2 / var_y, 0.0)
    return mu_post, var_post


class HSMTree:
    """The TAGI hierarchical binary decomposition, as numpy arrays.

    Wraps :func:`pytagi.Utils.get_hierarchical_softmax` so the calibration code
    does not have to deal with the flat C++ layout.

    :ivar num_classes: The number of classes.
    :ivar n_obs: The path length, ceil(log2(num_classes)).
    :ivar len: The number of output units of the tree.
    :ivar obs: Branch signs of every path, shape (num_classes, n_obs).
    :ivar idx: Zero-based output-unit indices of every path, same shape.
    :ivar unit_level: Tree depth of each output unit, shape (len,).
    """

    def __init__(self, num_classes: int):
        from pytagi.tagi_utils import Utils

        hs = Utils().get_hierarchical_softmax(num_classes)
        self.num_classes = int(num_classes)
        self.n_obs = int(hs.num_obs)
        self.len = int(hs.len)
        self.obs = np.asarray(hs.obs, dtype=np.float64).reshape(
            self.num_classes, self.n_obs
        )
        self.idx = (
            np.asarray(hs.idx, dtype=np.int64).reshape(
                self.num_classes, self.n_obs
            )
            - 1
        )
        self.unit_level = np.zeros(self.len, dtype=np.int64)
        for h in range(self.n_obs):
            self.unit_level[self.idx[:, h]] = h


class HSMGainCalibrator:
    """Gaussian gains for the hierarchical softmax.

    Implements Algorithm 1 of the note: a scalar Bayesian recursion per node, run
    sample by sample, that learns the gains from the class labels through the
    auxiliary observation model (E:obs) and (E:update). The network's training loop
    is untouched; the gains form a separate calibration channel.

    :param num_classes: The number of classes.
    :type num_classes: int
    :param sharing: One gain per tree node ("node"), one per tree level
        ("level", the recommended default) or a single gain ("global", the
        closest analogue of temperature scaling).
    :type sharing: str
    :param mu_gain_init: Prior mean of the gains.
    :type mu_gain_init: float
    :param sigma_gain_init: Prior standard deviation of the gains. A weakly
        informative prior is recommended, at least as wide as the mean.
    :type sigma_gain_init: float
    :param eps: Clipping bound applied to the branch probability before the
        update.
    :type eps: float
    :param min_mu_gain: Positivity clip on the gain mean. A negative gain would
        silently swap the two branches of a node. The note asks for "a small
        positive value" without fixing it, so it is exposed here.
    :type min_mu_gain: float
    :param process_noise: Per-sample process noise added to the gain variance
        before each update, for streams without epochs. Zero disables it.
    :type process_noise: float
    """

    def __init__(
        self,
        num_classes: int,
        sharing: str = "level",
        mu_gain_init: float = 0.3,
        sigma_gain_init: float = 1.0,
        eps: float = 1e-6,
        min_mu_gain: float = 1e-3,
        process_noise: float = 0.0,
    ):
        if sharing not in ("node", "level", "global"):
            raise ValueError(
                f"sharing must be 'node', 'level' or 'global', got {sharing!r}"
            )
        self.tree = HSMTree(num_classes)
        self.num_classes = self.tree.num_classes
        self.sharing = sharing
        self.mu_gain_init = float(mu_gain_init)
        self.var_gain_init = float(sigma_gain_init) ** 2
        self.eps = float(eps)
        self.min_mu_gain = float(min_mu_gain)
        self.process_noise = float(process_noise)

        if sharing == "node":
            self.num_gains = self.tree.len
            self._unit_to_gain = np.arange(self.tree.len, dtype=np.int64)
        elif sharing == "level":
            self.num_gains = self.tree.n_obs
            self._unit_to_gain = self.tree.unit_level.copy()
        else:
            self.num_gains = 1
            self._unit_to_gain = np.zeros(self.tree.len, dtype=np.int64)

        self.gain_idx = self._unit_to_gain[self.tree.idx]

        self.mu_gain = np.full(
            self.num_gains, self.mu_gain_init, dtype=np.float64
        )
        self.var_gain = np.full(
            self.num_gains, self.var_gain_init, dtype=np.float64
        )

    def reset(self) -> None:
        """Reset both gain moments to the prior."""
        self.mu_gain[:] = self.mu_gain_init
        self.var_gain[:] = self.var_gain_init

    def reset_variance(self) -> None:
        """Epoch reset: keep the gain means, restore the prior gain variance.

        Across epochs the gains follow a random walk, because a gain calibrates the
        network as it currently is. The precision accumulated during one epoch
        describes a network that no longer exists once the next one has started.
        """
        self.var_gain[:] = self.var_gain_init

    @property
    def sigma_gain(self) -> np.ndarray:
        return np.sqrt(self.var_gain)

    def gains_per_unit(self) -> Tuple[np.ndarray, np.ndarray]:
        """The gain moments broadcast to every output unit of the tree."""
        return (
            self.mu_gain[self._unit_to_gain],
            self.var_gain[self._unit_to_gain],
        )

    def update_sample(
        self, mu_z: np.ndarray, var_z: np.ndarray, label: int
    ) -> None:
        """Apply the inner loop of Algorithm 1 for one sample.

        The recursion is sequential: each node recomputes the branch probability and
        its covariance with the gain from the moments left by the previous node, which
        is what makes level-shared and global gains work.

        :param mu_z: Forward-pass means of the output units of one sample.
        :type mu_z: np.ndarray
        :param var_z: Forward-pass variances of the same units.
        :type var_z: np.ndarray
        :param label: The true class.
        :type label: int
        """
        idx = self.tree.idx[label]
        signs = self.tree.obs[label]
        gidx = self.gain_idx[label]
        eps = self.eps

        for h in range(self.tree.n_obs):
            j = int(idx[h])
            g = int(gidx[h])
            s = float(signs[h])
            m_z = float(mu_z[j])
            v_z = float(var_z[j])

            if self.process_noise > 0.0:
                self.var_gain[g] += self.process_noise

            m_g = float(self.mu_gain[g])
            v_g = float(self.var_gain[g])

            var_s = v_g * v_z + v_g * m_z * m_z + m_g * m_g * v_z
            one_p = 1.0 + var_s
            sqrt_one_p = math.sqrt(one_p)
            a = s * m_g * m_z / sqrt_one_p  # (E:a)

            mu_p = 0.5 * math.erfc(-a / _SQRT2)
            pdf_a = _INV_SQRT_2PI * math.exp(-0.5 * a * a)
            cov_gp = (
                s
                * v_g
                * m_z
                * (one_p - m_g * m_g * v_z)
                / (one_p * sqrt_one_p)
                * pdf_a
            )

            mu_p = min(max(mu_p, eps), 1.0 - eps)
            var_y = mu_p * (1.0 - mu_p)
            m_g_post = m_g + cov_gp / mu_p
            v_g_post = max(v_g - cov_gp * cov_gp / var_y, 0.0)

            self.mu_gain[g] = max(m_g_post, self.min_mu_gain)
            self.var_gain[g] = v_g_post

    def calibrate(
        self,
        mu_z: np.ndarray,
        var_z: np.ndarray,
        labels: Sequence[int],
        reset_variance: bool = True,
    ) -> None:
        """Run the gain calibration pass of Algorithm 1 over a validation split.

        :param mu_z: Forward-pass means, shape (N, len) or flat.
        :type mu_z: np.ndarray
        :param var_z: Forward-pass variances, same shape.
        :type var_z: np.ndarray
        :param labels: The true classes of the N samples.
        :type labels: Sequence[int]
        :param reset_variance: Perform the epoch reset before the pass. Set to False
            for a stream without epochs, where the process noise plays that role.
        :type reset_variance: bool
        """
        mu_z = np.asarray(mu_z, dtype=np.float64).reshape(-1, self.tree.len)
        var_z = np.asarray(var_z, dtype=np.float64).reshape(-1, self.tree.len)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if mu_z.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{mu_z.shape[0]} samples of hidden states for "
                f"{labels.shape[0]} labels"
            )
        if reset_variance:
            self.reset_variance()
        for i in range(labels.shape[0]):
            self.update_sample(mu_z[i], var_z[i], int(labels[i]))

    def unit_moments(
        self, mu_z: np.ndarray, var_z: np.ndarray, compute_var: bool = True
    ) -> NodeMoments:
        """Node moments of every output unit, oriented at s = +1.

        The sibling branch follows by symmetry: its mean is one minus this one, its
        variance is the same and its covariances have the opposite sign.

        :param mu_z: Forward-pass means, shape (N, len) or (len,).
        :type mu_z: np.ndarray
        :param var_z: Matching forward-pass variances.
        :type var_z: np.ndarray
        :return: The node moments of every unit.
        :rtype: NodeMoments
        """
        mu_z = np.asarray(mu_z, dtype=np.float64).reshape(-1, self.tree.len)
        var_z = np.asarray(var_z, dtype=np.float64).reshape(-1, self.tree.len)
        mu_g, var_g = self.gains_per_unit()
        return node_moments(
            mu_z,
            var_z,
            mu_g[None, :],
            var_g[None, :],
            s=1.0,
            compute_var=compute_var,
        )

    def class_probabilities(
        self,
        mu_z: np.ndarray,
        var_z: np.ndarray,
        compute_var: bool = True,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Class probabilities and their variances (E:class).

        The means sum to one exactly when the number of classes is a power of two;
        otherwise they are normalized as in TAGI.

        :param mu_z: Forward-pass means, shape (N, len) or (len,).
        :type mu_z: np.ndarray
        :param var_z: Matching forward-pass variances.
        :type var_z: np.ndarray
        :param compute_var: Also return the variance of the class probability.
        :type compute_var: bool
        :param normalize: Normalize the means when the number of classes is not a
            power of two.
        :type normalize: bool
        :return: The means and variances, of shape (N, num_classes). The variance is
            None when ``compute_var`` is False.
        :rtype: tuple
        """
        nm = self.unit_moments(mu_z, var_z, compute_var=compute_var)
        mu_plus = nm.mu_p[:, self.tree.idx]  # (N, K, H)
        signs = self.tree.obs[None, :, :]
        mu_p = np.where(signs > 0.0, mu_plus, 1.0 - mu_plus)

        mu_pr = np.prod(mu_p, axis=-1)
        var_pr = None
        if compute_var:
            var_p = nm.var_p[:, self.tree.idx]  # same for both branches
            var_pr = np.prod(var_p + mu_p**2, axis=-1) - np.prod(
                mu_p**2, axis=-1
            )

        is_power_of_two = (self.num_classes & (self.num_classes - 1)) == 0
        if normalize and not is_power_of_two:
            mu_pr = mu_pr / np.sum(mu_pr, axis=-1, keepdims=True)
        return mu_pr, var_pr

    def class_covariances(
        self, mu_z: np.ndarray, var_z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Covariances of the class probability with the hidden states and the gains.

        :param mu_z: Forward-pass means, shape (N, len) or (len,).
        :type mu_z: np.ndarray
        :param var_z: Matching forward-pass variances.
        :type var_z: np.ndarray
        :return: Two arrays of shape (N, num_classes, len). Entries of output units
            off the path of a class are zero.
        :rtype: tuple
        """
        nm = self.unit_moments(mu_z, var_z, compute_var=False)
        mu_plus = nm.mu_p[:, self.tree.idx]
        signs = self.tree.obs[None, :, :]
        mu_p = np.where(signs > 0.0, mu_plus, 1.0 - mu_plus)
        cov_zp = signs * nm.cov_zp[:, self.tree.idx]
        cov_gp = signs * nm.cov_gp[:, self.tree.idx]

        cov_pr_z_path = class_covariance(mu_p, cov_zp, axis=-1)  # (N, K, H)
        cov_pr_g_path = class_covariance(mu_p, cov_gp, axis=-1)

        n = mu_p.shape[0]
        cov_pr_z = np.zeros((n, self.num_classes, self.tree.len))
        cov_pr_g = np.zeros((n, self.num_classes, self.tree.len))
        k_grid, h_grid = np.meshgrid(
            np.arange(self.num_classes),
            np.arange(self.tree.n_obs),
            indexing="ij",
        )
        unit = self.tree.idx[k_grid, h_grid]
        cov_pr_z[:, k_grid, unit] = cov_pr_z_path
        cov_pr_g[:, k_grid, unit] = cov_pr_g_path
        return cov_pr_z, cov_pr_g

    def predict(
        self, mu_z: np.ndarray, var_z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicted labels and their class probabilities."""
        mu_pr, _ = self.class_probabilities(mu_z, var_z, compute_var=False)
        pred = np.argmax(mu_pr, axis=-1)
        return pred, mu_pr[np.arange(mu_pr.shape[0]), pred]


class HSMCalibratedMetric:
    """Drop-in companion of :class:`pytagi.metric.HRCSoftmaxMetric` using gains.

    Same interface, but the class probabilities come from (E:class) with the
    learned gains instead of the fixed scale alpha, and their variances are
    available.

    :param num_classes: The number of classes.
    :type num_classes: int
    """

    def __init__(self, num_classes: int, **calibrator_kwargs):
        self.num_classes = num_classes
        self.calibrator = HSMGainCalibrator(num_classes, **calibrator_kwargs)
        self.len = self.calibrator.tree.len

    def _reshape(self, m_pred: np.ndarray, v_pred: np.ndarray):
        batch_size = m_pred.shape[0] // self.len
        return (
            np.asarray(m_pred, dtype=np.float64).reshape(batch_size, self.len),
            np.asarray(v_pred, dtype=np.float64).reshape(batch_size, self.len),
        )

    def calibrate(
        self,
        m_pred: np.ndarray,
        v_pred: np.ndarray,
        labels: Sequence[int],
        reset_variance: bool = True,
    ) -> None:
        """Run the calibration pass of Algorithm 1 on a validation split."""
        mu_z, var_z = self._reshape(m_pred, v_pred)
        self.calibrator.calibrate(
            mu_z, var_z, labels, reset_variance=reset_variance
        )

    def get_predicted_labels(
        self, m_pred: np.ndarray, v_pred: np.ndarray
    ) -> np.ndarray:
        mu_z, var_z = self._reshape(m_pred, v_pred)
        pred, _ = self.calibrator.predict(mu_z, var_z)
        return pred

    def get_class_probabilities(
        self, m_pred: np.ndarray, v_pred: np.ndarray, compute_var: bool = True
    ):
        mu_z, var_z = self._reshape(m_pred, v_pred)
        return self.calibrator.class_probabilities(
            mu_z, var_z, compute_var=compute_var
        )

    def error_rate(
        self, m_pred: np.ndarray, v_pred: np.ndarray, label: np.ndarray
    ) -> float:
        pred = self.get_predicted_labels(m_pred, v_pred)
        label = np.asarray(label).reshape(-1)
        return float(np.mean(pred != label))
