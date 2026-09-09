"""Verification and illustration of the hierarchical softmax calibration note.

Reproduces, one by one, the numbers of Sections 5 and 6 of the note
*Hierarchical Softmax Calibration* (Goulet, Nguyen & Florensa-Montilla)
and the checks listed
under "Verification": the node moments (E:node) against Monte Carlo and
against a 20-point Gauss-Hermite integration over the gain, the class moments
(E:class), and the exact posterior means E[G P]/E[P], E[G(1-P)]/E[1-P]
against the gain update (E:update).

Usage::

    python -m examples.hsm_calibration                # run all checks
    python -m examples.hsm_calibration --figures      # also write the 3 PDFs
"""

import os
import sys

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import math

import numpy as np

from pytagi.hsm_calibration import (
    HSMGainCalibrator,
    class_moments,
    conditional_branch_prob,
    conditional_branch_prob_derivative,
    gain_update,
    node_moments,
    phi_expected_bernoulli_variance,
    phi_variance,
    standard_normal_cdf,
    standard_normal_pdf,
)

# The worked example of Section 5.
MU_Z, SIGMA_Z = 0.5, 0.5
MU_G, SIGMA_G = 1.5, 0.3


def _line(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


# ---------------------------------------------------------------------------
# Reference implementations used only for verification
# ---------------------------------------------------------------------------


def owens_t_reference(h, b, n: int = 64):
    """Owen's :math:`T`, by direct quadrature of its definition.

    .. math::
        T(h,b)=\\frac{1}{2\\pi}\\int_0^b
        \\frac{e^{-h^2(1+t^2)/2}}{1+t^2}\\,\\mathrm dt

    An ``n``-point Gauss-Legendre rule on :math:`[0,b]`. This exists only as
    an independent reference for the 8-point rule that (E:varint) runs on
    :math:`[b,1]`: a different interval and a different order, so agreement
    between the two is a real check. The library never needs Owen's T, which
    is the point of the integral form -- see Section 6, "Cost".
    """
    x, w = np.polynomial.legendre.leggauss(n)
    h = np.asarray(h, dtype=np.float64)[..., None]
    half = 0.5 * np.asarray(b, dtype=np.float64)[..., None]
    t = half * (x + 1.0)
    integrand = np.exp(-0.5 * h**2 * (1.0 + t * t)) / (1.0 + t * t)
    return (half[..., 0] / (2.0 * np.pi)) * np.sum(integrand * w, axis=-1)


def exact_node_moments_gh(mu_z, var_z, mu_g, var_g, s=1.0, n: int = 20):
    """Node moments obtained by conditioning on G and integrating with a
    20-point Gauss-Hermite rule (Section 6, "Verification").

    Given ``G = g`` the product ``S = g Z`` is exactly Gaussian, so every
    conditional moment is closed-form; only the Gaussian treatment of ``S``
    of (E:S) is bypassed.
    """
    x, w = np.polynomial.hermite_e.hermegauss(n)
    g = mu_g + math.sqrt(var_g) * x
    w = w / np.sum(w)

    d = 1.0 + g**2 * var_z
    a_g = s * g * mu_z / np.sqrt(d)
    m = standard_normal_cdf(a_g)  # E[P | g]
    rho = g**2 * var_z / d
    # E[P^2 | g] = Phi_2(a, a; rho) = Phi(a) - 2 T(a, sqrt((1-rho)/(1+rho)))
    m2 = m - 2.0 * owens_t_reference(a_g, np.sqrt((1.0 - rho) / (1.0 + rho)))
    cov_zp_g = s * g * var_z * standard_normal_pdf(a_g) / np.sqrt(d)

    mu_p = float(np.sum(w * m))
    var_p = float(np.sum(w * m2) - mu_p**2)
    cov_gp = float(np.sum(w * (g - mu_g) * m))
    cov_zp = float(np.sum(w * cov_zp_g))
    return mu_p, var_p, cov_zp, cov_gp


def mc_node_moments(mu_z, var_z, mu_g, var_g, s=1.0, n=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    z = mu_z + math.sqrt(var_z) * rng.standard_normal(n)
    g = mu_g + math.sqrt(var_g) * rng.standard_normal(n)
    p = standard_normal_cdf(s * g * z)
    return (
        float(p.mean()),
        float(p.var()),
        float(np.cov(z, p)[0, 1]),
        float(np.cov(g, p)[0, 1]),
    )


def projection_cov_gp(mu_z, var_z, mu_g, var_g, s=1.0):
    """The TAGI projection through ``S_h`` discussed (and rejected) in Section 3.1:
    ``Cov(G,S) Cov(S,P) / var_S = s var_G mu_z phi(a) / sqrt(1 + var_S)``."""
    nm = node_moments(mu_z, var_z, mu_g, var_g, s, compute_var=False)
    return float(
        s * var_g * mu_z * standard_normal_pdf(nm.a) / math.sqrt(1.0 + nm.var_s)
    )


def exact_gain_posterior_1obs(mu_z, var_z, mu_g, var_g, s, y, grid=200_001):
    """Exact posterior of G after one Bernoulli observation, on a grid over g.

    ``p(g | y) propto N(g; mu_G, var_G) m(g)^y (1-m(g))^(1-y)`` with ``m(g)``
    from (E:mg).
    """
    lo = mu_g - 12.0 * math.sqrt(var_g)
    hi = mu_g + 12.0 * math.sqrt(var_g)
    g = np.linspace(lo, hi, grid)
    prior = np.exp(-0.5 * (g - mu_g) ** 2 / var_g)
    m = conditional_branch_prob(g, mu_z, var_z, s)
    lik = m if y == 1 else 1.0 - m
    post = prior * lik
    post /= np.trapezoid(post, g)
    mean = float(np.trapezoid(g * post, g))
    var = float(np.trapezoid((g - mean) ** 2 * post, g))
    return mean, math.sqrt(var)


def exact_gain_posterior_stream(mu, sigma, y, mu_g, sigma_g, grid=20_001):
    """Exact posterior of the scalar gain over a whole stream, on a grid."""
    g = np.linspace(-1.0, 8.0, grid)
    logp = -0.5 * (g - mu_g) ** 2 / sigma_g**2
    m = conditional_branch_prob(
        g[None, :], mu[:, None], sigma[:, None] ** 2, 1.0
    )
    m = np.clip(m, 1e-300, 1.0 - 1e-16)
    logp = logp + np.sum(
        y[:, None] * np.log(m) + (1 - y)[:, None] * np.log1p(-m), axis=0
    )
    logp -= logp.max()
    post = np.exp(logp)
    post /= np.trapezoid(post, g)
    mean = float(np.trapezoid(g * post, g))
    var = float(np.trapezoid((g - mean) ** 2 * post, g))
    return mean, math.sqrt(var), g, post


# ---------------------------------------------------------------------------
# Section 5 - Node moments
# ---------------------------------------------------------------------------


def check_node_moments():
    _line("Section 5 - Node moments (mu_Z=0.5, sigma_Z=0.5, G ~ N(1.5, 0.3^2))")
    nm = node_moments(MU_Z, SIGMA_Z**2, MU_G, SIGMA_G**2, s=1.0)
    closed = (
        float(nm.mu_p),
        float(np.sqrt(nm.var_p)),
        float(nm.cov_zp),
        float(nm.cov_gp),
    )
    mc = mc_node_moments(MU_Z, SIGMA_Z**2, MU_G, SIGMA_G**2)
    mc = (mc[0], math.sqrt(mc[1]), mc[2], mc[3])
    gh = exact_node_moments_gh(MU_Z, SIGMA_Z**2, MU_G, SIGMA_G**2)
    gh = (gh[0], math.sqrt(gh[1]), gh[2], gh[3])

    names = ["mu_P", "sigma_P", "Cov(Z,P)", "Cov(G,P)"]
    note_closed = [0.7229, 0.2142, 0.0977, 0.00773]
    note_mc = [0.7226, 0.2076, 0.0982, 0.00786]
    print(
        f"{'':10s} {'(E:node)':>12s} {'note':>10s} {'MC 2e6':>12s} "
        f"{'note':>10s} {'Gauss-Hermite':>15s}"
    )
    for k, name in enumerate(names):
        print(
            f"{name:10s} {closed[k]:12.5f} {note_closed[k]:10.4f} "
            f"{mc[k]:12.5f} {note_mc[k]:10.4f} {gh[k]:15.5f}"
        )

    # "with a wide prior on the gain (sigma_G = 1) Cov(G,P) stays within 10%
    #  of the exact value"
    nm_wide = node_moments(MU_Z, SIGMA_Z**2, MU_G, 1.0**2, s=1.0)
    gh_wide = exact_node_moments_gh(MU_Z, SIGMA_Z**2, MU_G, 1.0**2)
    rel = abs(float(nm_wide.cov_gp) / gh_wide[3] - 1.0)
    print(
        f"\nwide prior sigma_G=1: Cov(G,P) = {float(nm_wide.cov_gp):.5f} vs exact "
        f"{gh_wide[3]:.5f}  ->  {100 * rel:.1f}% (note: within 10%)"
    )

    # Exactness at sigma_Z = 0 and at sigma_G = 0.
    nm0 = node_moments(MU_Z, 0.0, MU_G, SIGMA_G**2)
    gh0 = exact_node_moments_gh(MU_Z, 0.0, MU_G, SIGMA_G**2)
    print(
        f"sigma_Z = 0 : closed {float(nm0.mu_p):.8f} exact {gh0[0]:.8f}  "
        f"Cov(G,P) {float(nm0.cov_gp):.8f} vs {gh0[3]:.8f}"
    )
    nm1 = node_moments(MU_Z, SIGMA_Z**2, MU_G, 0.0)
    print(
        f"sigma_G = 0 : mu_P {float(nm1.mu_p):.8f} = m(mu_G) "
        f"{float(conditional_branch_prob(MU_G, MU_Z, SIGMA_Z**2)):.8f}, "
        f"Cov(G,P) = {float(nm1.cov_gp):.1e}"
    )


def check_projection():
    _line("Section 3.1 - the projection through S_h vs the Stein form (E:node)")
    for mu_g, sigma_z, expected in [(MU_G, SIGMA_Z, 1.54), (MU_G, 1.0, 3.0)]:
        stein = float(
            node_moments(
                MU_Z, sigma_z**2, mu_g, SIGMA_G**2, compute_var=False
            ).cov_gp
        )
        proj = projection_cov_gp(MU_Z, sigma_z**2, mu_g, SIGMA_G**2)
        print(
            f"mu_G={mu_g}, sigma_Z={sigma_z}: projection/Stein = {proj / stein:.3f} "
            f"(note: {expected}); 1 + mu_G^2 sigma_Z^2 = "
            f"{1 + mu_g**2 * sigma_z**2:.3f}"
        )


def check_bound(n: int = 2_000_000, seed: int = 1):
    _line("Section 3.1 - bound Cov(G,P)^2 <= var_G mu_P (1 - mu_P)")
    rng = np.random.default_rng(seed)
    mu_z = rng.uniform(-8.0, 8.0, n)
    sigma_z = rng.uniform(0.0, 4.0, n)
    mu_g = rng.uniform(-1.0, 6.0, n)
    sigma_g = rng.uniform(0.0, 5.0, n)
    s = rng.choice([-1.0, 1.0], n)
    nm = node_moments(mu_z, sigma_z**2, mu_g, sigma_g**2, s, compute_var=False)
    denom = sigma_g**2 * nm.mu_p * (1.0 - nm.mu_p)
    ratio = np.where(denom > 0.0, nm.cov_gp**2 / np.maximum(denom, 1e-300), 0.0)
    print(
        f"max ratio over {n:.0e} random combinations: {ratio.max():.4f} "
        f"(note: never exceeded 0.64)"
    )

    # sign Cov(G,P) = sign(s mu_Z), wherever Cov(G,P) does not vanish
    # (it does when sigma_G = 0, mu_Z = 0, or phi(a) underflows).
    nz = nm.cov_gp != 0.0
    sign_ok = bool(np.all(np.sign(nm.cov_gp[nz]) == np.sign(s * mu_z)[nz]))
    print(
        f"sign Cov(G,P) == sign(s mu_Z) on the {nz.mean():.1%} of cases with "
        f"Cov(G,P) != 0: {sign_ok}"
    )
    print(
        "1 + var_S - mu_G^2 var_Z == 1 + var_G (var_Z + mu_Z^2): max err "
        f"{np.max(np.abs((1 + nm.var_s - mu_g**2 * sigma_z**2) - (1 + sigma_g**2 * (sigma_z**2 + mu_z**2)))):.2e}"
    )


def check_phi_variance():
    _line(
        "Section 2 - Var[Phi(X)]: 8-point Gauss-Legendre (E:varint) vs Owen's T"
    )
    rng = np.random.default_rng(2)
    a = rng.uniform(-8.0, 8.0, 200_000)
    var = rng.uniform(0.0, 25.0, 200_000)
    gl = phi_variance(a, var)
    b = 1.0 / np.sqrt(1.0 + 2.0 * var)
    m = standard_normal_cdf(a)
    owen = m * (1.0 - m) - 2.0 * owens_t_reference(a, b)
    print(
        f"max |GL8 - Owen| over |a|<=8, sigma^2<=25: {np.max(np.abs(gl - owen)):.2e} "
        f"(note: 1e-12)"
    )
    print(
        f"min Var[Phi(X)] from (E:varint): {gl.min():.2e} (positive by construction)"
    )
    print(
        "E[Phi(1-Phi)] = 2T(a,b) check: max err "
        f"{np.max(np.abs(phi_expected_bernoulli_variance(a, var) - 2.0 * owens_t_reference(a, b))):.2e}"
    )


def check_limits():
    _line("Section 6 - Limits")
    # sigma_h -> 0, mu_h = 1, G ~ N(1.5, 0.3^2)
    nm = node_moments(1.0, 0.0, 1.5, 0.3**2)
    print(
        f"sigma_h -> 0, mu_h = 1: var_P = {float(nm.var_p):.4e} (note 1.8e-3), "
        f"Cov(G,P) = {float(nm.cov_gp):.4f} (note 0.012), "
        f"Cov(Z,P) = {float(nm.cov_zp):.1e} (note 0)"
    )
    # mu_h = 0: var_P = 1/4 - arctan(b)/pi
    for sigma_z in [0.5, 2.0, 50.0]:
        nm0 = node_moments(0.0, sigma_z**2, 1.5, 0.3**2)
        closed = 0.25 - math.atan(float(nm0.b)) / math.pi
        print(
            f"mu_h = 0, sigma_h = {sigma_z:5.1f}: var_P = {float(nm0.var_p):.6f}, "
            f"1/4 - arctan(b)/pi = {closed:.6f}"
        )
    # With a fixed gain 1/alpha, mu_Pr is exactly TAGI's Eq. (1).
    alpha = 3.0
    nm_fixed = node_moments(MU_Z, SIGMA_Z**2, 1.0 / alpha, 0.0)
    tagi = standard_normal_cdf(MU_Z / math.sqrt(alpha**2 + SIGMA_Z**2))
    print(
        f"fixed gain 1/alpha (alpha={alpha}): mu_P = {float(nm_fixed.mu_p):.10f}, "
        f"TAGI Eq.(1) = {float(tagi):.10f}"
    )


# ---------------------------------------------------------------------------
# Section 5 - One observation
# ---------------------------------------------------------------------------


def check_one_observation():
    _line("Section 5 - One observation")
    nm = node_moments(MU_Z, SIGMA_Z**2, MU_G, SIGMA_G**2, s=1.0)
    mu_p, cov_gp = float(nm.mu_p), float(nm.cov_gp)

    mu1, var1 = gain_update(MU_G, SIGMA_G**2, mu_p, cov_gp, y=1.0)
    mu0, var0 = gain_update(MU_G, SIGMA_G**2, mu_p, cov_gp, y=0.0)
    ex1 = exact_gain_posterior_1obs(MU_Z, SIGMA_Z**2, MU_G, SIGMA_G**2, 1.0, 1)
    ex0 = exact_gain_posterior_1obs(MU_Z, SIGMA_Z**2, MU_G, SIGMA_G**2, 1.0, 0)
    print(
        f"label class 1: d mu_G = {float(mu1) - MU_G:+.4f} (note +0.0107), "
        f"exact {ex1[0] - MU_G:+.4f} (note +0.0109)"
    )
    print(
        f"label class 0: d mu_G = {float(mu0) - MU_G:+.4f} (note -0.0279), "
        f"exact {ex0[0] - MU_G:+.4f} (note -0.0284)"
    )
    print(
        f"recursion sigma_G -> {math.sqrt(float(var1)):.4f} (note 0.2995); "
        f"exact widths {ex1[1]:.4f} / {ex0[1]:.4f} (note 0.2984 / 0.3022)"
    )

    # sigma_h -> 0 the order reverses (note: 0.2985 / 0.2975)
    ex1s = exact_gain_posterior_1obs(MU_Z, 1e-12, MU_G, SIGMA_G**2, 1.0, 1)
    ex0s = exact_gain_posterior_1obs(MU_Z, 1e-12, MU_G, SIGMA_G**2, 1.0, 0)
    print(
        f"sigma_h -> 0: exact widths {ex1s[1]:.4f} / {ex0s[1]:.4f} "
        f"(note 0.2985 / 0.2975)"
    )

    # The mean update is exact under the Bernoulli likelihood:
    # E[G|y=1] = E[G P]/E[P] = mu_G + Cov(G,P)/mu_P.
    print(
        f"E[G P]/E[P] identity: {MU_G + cov_gp / mu_p:.6f} vs recursion "
        f"{float(mu1):.6f}"
    )

    # Small-sigma_G form: Cov(G,P) -> var_G m'(mu_G)  (E:mg)
    small = 1e-3
    nm_s = node_moments(MU_Z, SIGMA_Z**2, MU_G, small**2, compute_var=False)
    approx = small**2 * float(
        conditional_branch_prob_derivative(MU_G, MU_Z, SIGMA_Z**2, 1.0)
    )
    print(
        f"small sigma_G: Cov(G,P) = {float(nm_s.cov_gp):.6e} vs "
        f"var_G m'(mu_G) = {approx:.6e}"
    )


# ---------------------------------------------------------------------------
# Section 5 - Learning the gain from a validation pass
# ---------------------------------------------------------------------------


def make_stream(n=3000, g_star=2.0, seed=0):
    """The synthetic validation split of Section 5."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 1.5, n)
    sigma = rng.uniform(0.3, 1.2, n)
    z = rng.normal(mu, sigma)
    y = (rng.uniform(size=n) < standard_normal_cdf(g_star * z)).astype(int)
    return mu, sigma, y


def run_recursion(
    mu, sigma, y, mu_g0, sigma_g0, q=0.0, post_update_sigma_v=None
):
    """Algorithm 1 on the K = 2 stream, returning the trajectory of (mu_G, sigma_G).

    ``post_update_sigma_v`` reproduces the failure mode of Section 4.1: feeding
    the channel with the moments *after* the logit update instead of the
    forward-pass moments.
    """
    mu_g, var_g = float(mu_g0), float(sigma_g0) ** 2
    traj_m = np.empty(len(y))
    traj_s = np.empty(len(y))
    for i in range(len(y)):
        s = 1.0 if y[i] == 1 else -1.0
        m_z, v_z = float(mu[i]), float(sigma[i]) ** 2
        if post_update_sigma_v is not None:
            # Logit channel: observation y_h = s with noise variance sigma_V^2.
            sv2 = post_update_sigma_v**2
            k = v_z / (v_z + sv2)
            m_z, v_z = m_z + k * (s - m_z), v_z - k * v_z
        if q > 0.0:
            var_g += q
        nm = node_moments(m_z, v_z, mu_g, var_g, s, compute_var=False)
        mu_g, var_g = gain_update(mu_g, var_g, nm.mu_p, nm.cov_gp, y=1.0)
        mu_g, var_g = float(mu_g), float(var_g)
        traj_m[i], traj_s[i] = mu_g, math.sqrt(var_g)
    return traj_m, traj_s


def ece_nll(mu, sigma, y, g, n_bins=15):
    p = np.asarray(conditional_branch_prob(g, mu, sigma**2, 1.0))
    p_clip = np.clip(p, 1e-12, 1.0 - 1e-12)
    nll = float(-np.mean(y * np.log(p_clip) + (1 - y) * np.log1p(-p_clip)))
    conf = np.where(p >= 0.5, p, 1.0 - p)
    correct = (p >= 0.5).astype(int) == y
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece), nll


def check_validation_pass():
    _line("Section 5 - Learning the gain from a validation pass (N=3000, g*=2)")
    mu, sigma, y = make_stream(seed=0)

    for prior_mu, prior_sd, note in [
        (0.3, 1.0, "2.02 +/- 0.14 vs exact 2.14 +/- 0.20"),
        (0.3, 0.3, "1.49 +/- 0.07 vs exact 1.77 +/- 0.12"),
    ]:
        tm, ts = run_recursion(mu, sigma, y, prior_mu, prior_sd)
        em, es, _, _ = exact_gain_posterior_stream(
            mu, sigma, y, prior_mu, prior_sd
        )
        print(
            f"prior N({prior_mu}, {prior_sd}^2): recursion {tm[-1]:.2f} +/- {ts[-1]:.2f}"
            f" | exact {em:.2f} +/- {es:.2f} | width ratio {ts[-1] / es:.0%}"
            f"   [note: {note}]"
        )

    # Four independent streams with the wide prior.
    gaps, ratios = [], []
    for seed in range(4):
        m_s, s_s, y_s = make_stream(seed=seed)
        tm, ts = run_recursion(m_s, s_s, y_s, 0.3, 1.0)
        em, es, _, _ = exact_gain_posterior_stream(m_s, s_s, y_s, 0.3, 1.0)
        gaps.append(abs(tm[-1] - em))
        ratios.append(ts[-1] / es)
    print(
        f"4 streams: max |recursion - exact| mean gap = {max(gaps):.3f} "
        f"(note: within 0.15); width ratio "
        f"{min(ratios):.0%}-{max(ratios):.0%} (note: 70-95%)"
    )

    # Post-update moments: the gain grows without bound.
    tm_bad, _ = run_recursion(mu, sigma, y, 0.3, 1.0, post_update_sigma_v=0.3)
    print(
        f"channel fed with post-update moments: mu_G = {tm_bad[-1]:.1f} and rising "
        f"(note: 3.7 and rising)"
    )

    # Reliability on 40 000 fresh test samples.
    tm, _ = run_recursion(mu, sigma, y, 0.3, 1.0)
    mu_t, sigma_t, y_t = make_stream(n=40_000, seed=101)
    for name, g in [
        ("initial G=0.3", 0.3),
        (f"learned G={tm[-1]:.2f}", tm[-1]),
        ("true gain G=2", 2.0),
    ]:
        ece, nll = ece_nll(mu_t, sigma_t, y_t, g)
        print(f"{name:22s} ECE {ece:.3f}  NLL {nll:.3f}")
    print("note: ECE 0.197 / NLL 0.511 initial, ECE 0.005 / NLL 0.363 learned")


def check_epoch_reset():
    _line("Section 4.2 - Epochs: keep mu_G, re-initialize sigma_G^2")
    # A stream whose calibrated gain drifts from 2.0 to 1.0 over five epochs.
    g_epochs = np.linspace(2.0, 1.0, 5)
    finals = {"no reset": [], "reset": [], "process noise q=1e-4": []}
    for seed in range(8):
        state_no = (0.3, 1.0)
        state_re = (0.3, 1.0)
        state_q = (0.3, 1.0)
        for e, g_star in enumerate(g_epochs):
            mu, sigma, y = make_stream(
                n=3000, g_star=g_star, seed=100 * seed + e
            )
            tm, ts = run_recursion(mu, sigma, y, state_no[0], state_no[1])
            state_no = (tm[-1], ts[-1])
            tm, ts = run_recursion(
                mu, sigma, y, state_re[0], 1.0
            )  # variance reset
            state_re = (tm[-1], ts[-1])
            tm, ts = run_recursion(mu, sigma, y, state_q[0], state_q[1], q=1e-4)
            state_q = (tm[-1], ts[-1])
        finals["no reset"].append(state_no)
        finals["reset"].append(state_re)
        finals["process noise q=1e-4"].append(state_q)
    for k, v in finals.items():
        m = np.array([a[0] for a in v])
        s = np.array([a[1] for a in v])
        print(
            f"{k:24s} final mu_G = {m.mean():.2f} +/- {m.std():.2f}, "
            f"sigma_G = {s.mean():.2f}"
        )
    print(
        "note: no reset 1.19 +/- 0.04 | reset 1.00 +/- 0.05 | q=1e-4 tracks "
        "with sigma_G ~ 0.2"
    )


# ---------------------------------------------------------------------------
# Section 3.2 - Class level, on the cuTAGI binary tree
# ---------------------------------------------------------------------------


def check_class_level(num_classes: int = 8, seed: int = 3):
    _line(
        f"Section 3.2 - Class moments (E:class) on the cuTAGI tree, K={num_classes}"
    )
    rng = np.random.default_rng(seed)
    cal = HSMGainCalibrator(
        num_classes, sharing="node", mu_gain_init=1.5, sigma_gain_init=0.3
    )
    tree = cal.tree
    mu_z = rng.normal(0.0, 1.0, (1, tree.len))
    var_z = rng.uniform(0.2, 1.0, (1, tree.len))

    mu_pr, var_pr = cal.class_probabilities(mu_z, var_z)
    print(
        f"sum_C mu_Pr = {float(mu_pr.sum()):.10f} (exact 1 when K is a power of 2)"
    )

    # Monte Carlo over Z and G.
    n = 400_000
    z = rng.normal(mu_z, np.sqrt(var_z), (n, tree.len))
    mu_g, var_g = cal.gains_per_unit()
    g = rng.normal(mu_g, np.sqrt(var_g), (n, tree.len))
    p_plus = standard_normal_cdf(g * z)
    err_m, err_v, err_cz, err_cg = 0.0, 0.0, 0.0, 0.0
    cov_z, cov_g = cal.class_covariances(mu_z, var_z)
    for k in range(num_classes):
        idx, s = tree.idx[k], tree.obs[k]
        p = np.where(s > 0, p_plus[:, idx], 1.0 - p_plus[:, idx])
        prod = np.prod(p, axis=1)
        err_m = max(err_m, abs(prod.mean() - mu_pr[0, k]))
        err_v = max(err_v, abs(prod.var() - var_pr[0, k]))
        for h, j in enumerate(idx):
            err_cz = max(
                err_cz, abs(np.cov(z[:, j], prod)[0, 1] - cov_z[0, k, j])
            )
            err_cg = max(
                err_cg, abs(np.cov(g[:, j], prod)[0, 1] - cov_g[0, k, j])
            )
    print(
        f"max |MC - (E:class)| over the {num_classes} classes: mu {err_m:.2e}, "
        f"var {err_v:.2e}, Cov(Pr,Z) {err_cz:.2e}, Cov(Pr,G) {err_cg:.2e}"
    )

    # (E:class) is exact *given* the node moments; the residual above is the
    # node-level GMA error of (E:S) accumulated over the path. Feeding the
    # exact Gauss-Hermite node moments into (E:class) isolates it.
    err_m_gh, err_v_gh = 0.0, 0.0
    for k in range(num_classes):
        idx, s = tree.idx[k], tree.obs[k]
        p = np.where(s > 0, p_plus[:, idx], 1.0 - p_plus[:, idx])
        prod = np.prod(p, axis=1)
        m_gh, v_gh = [], []
        for h, j in enumerate(idx):
            m_, v_, _, _ = exact_node_moments_gh(
                float(mu_z[0, j]),
                float(var_z[0, j]),
                float(mu_g[j]),
                float(var_g[j]),
                float(s[h]),
                n=40,
            )
            m_gh.append(m_)
            v_gh.append(v_)
        m_c, v_c = class_moments(np.array(m_gh), np.array(v_gh))
        err_m_gh = max(err_m_gh, abs(prod.mean() - m_c))
        err_v_gh = max(err_v_gh, abs(prod.var() - v_c))
    print(
        f"same, with exact (Gauss-Hermite) node moments: mu {err_m_gh:.2e}, "
        f"var {err_v_gh:.2e}  (MC noise ~ {1 / math.sqrt(n):.1e})"
    )

    # Recursive form Q_h = Q_{h-1} P_h against the unrolled products.
    nm = cal.unit_moments(mu_z, var_z)
    k = 0
    idx, s = tree.idx[k], tree.obs[k]
    mu_p = np.where(s > 0, nm.mu_p[0, idx], 1.0 - nm.mu_p[0, idx])
    var_p = nm.var_p[0, idx]
    mq, vq = mu_p[0], var_p[0]
    for h in range(1, tree.n_obs):
        mq, vq = (
            mq * mu_p[h],
            vq * var_p[h] + mq**2 * var_p[h] + mu_p[h] ** 2 * vq,
        )
    m_un, v_un = class_moments(mu_p, var_p)
    print(
        f"recursion vs unrolled (E:class): mu {abs(mq - m_un):.2e}, "
        f"var {abs(vq - v_un):.2e}"
    )


def check_calibrator_matches_scalar():
    _line("HSMGainCalibrator (K=2) vs the scalar recursion of Section 5")
    mu, sigma, y = make_stream(n=500, seed=7)
    tm, ts = run_recursion(mu, sigma, y, 0.3, 1.0)

    cal = HSMGainCalibrator(
        2, sharing="global", mu_gain_init=0.3, sigma_gain_init=1.0
    )
    # cuTAGI orients the tree with s_h = (-1)^{C_h}: class 0 -> s = +1. The
    # note's illustration calls that class "1", hence the relabelling.
    labels = 1 - np.asarray(y)
    cal.calibrate(mu.reshape(-1, 1), (sigma**2).reshape(-1, 1), labels)
    print(f"scalar recursion mu_G = {tm[-1]:.6f}, sigma_G = {ts[-1]:.6f}")
    print(
        f"HSMGainCalibrator mu_G = {cal.mu_gain[0]:.6f}, "
        f"sigma_G = {cal.sigma_gain[0]:.6f}"
    )

    for sharing in ("node", "level", "global"):
        c = HSMGainCalibrator(10, sharing=sharing)
        print(
            f"K=10, sharing={sharing:6s}: {c.num_gains} gain(s) for "
            f"{c.tree.len} tree nodes, H={c.tree.n_obs}"
        )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_figures(outdir: str = "saved_results"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(0)

    # ---- Figure 1: node level ------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    n = 400_000
    z = rng.normal(MU_Z, SIGMA_Z, n)
    g = rng.normal(MU_G, SIGMA_G, n)
    p = standard_normal_cdf(g * z)
    nm = node_moments(MU_Z, SIGMA_Z**2, MU_G, SIGMA_G**2)
    ax[0].hist(p, bins=200, density=True, alpha=0.5, label="$P=\\Phi(GZ)$")
    xs = np.linspace(0, 1, 400)
    sd = math.sqrt(float(nm.var_p))
    ax[0].plot(
        xs,
        standard_normal_pdf((xs - float(nm.mu_p)) / sd) / sd,
        label="moment-matched Gaussian",
    )
    ax[0].axvline(float(nm.mu_p), ls=":", c="k")
    ax[0].set_title("(a) distribution of $P$")
    ax[0].set_xlabel("$p$")
    ax[0].legend(fontsize=8)

    mz = np.linspace(-4, 4, 400)
    for sz in [0.01, 0.5, 1.5]:
        nmb = node_moments(mz, sz**2, 1.5, 0.0)
        m, s = np.asarray(nmb.mu_p), np.sqrt(np.asarray(nmb.var_p))
        ax[1].plot(mz, m, label=f"$\\sigma_Z={sz}$")
        ax[1].fill_between(mz, m - s, m + s, alpha=0.2)
    ax[1].set_title("(b) $\\mu_P\\pm\\sigma_P$ at $G=1.5$")
    ax[1].set_xlabel("$\\mu_Z$")
    ax[1].legend(fontsize=8)

    nmc = node_moments(mz, 1.0, MU_G, SIGMA_G**2, compute_var=False)
    ax[2].plot(mz, nmc.cov_zp, label="$\\mathrm{Cov}(Z,P)$")
    ax[2].plot(
        mz,
        np.asarray(nmc.cov_gp) / SIGMA_G**2,
        label="$\\mathrm{Cov}(G,P)/\\sigma_G^2$",
    )
    ax[2].axhline(0, lw=0.5, c="k")
    ax[2].set_title("(c) covariances, $\\sigma_Z=1$")
    ax[2].set_xlabel("$\\mu_Z$")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_node.pdf"))
    plt.close(fig)

    # ---- Figure 2: one training sample ---------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    nmd = node_moments(mz, 1.0, 1.0, 0.3**2, s=1.0, compute_var=False)
    mu_p, cov_gp = np.asarray(nmd.mu_p), np.asarray(nmd.cov_gp)
    ax[0].plot(mz, cov_gp / np.clip(mu_p, 1e-6, 1), label="label = class 1")
    ax[0].plot(
        mz, -cov_gp / np.clip(1 - mu_p, 1e-6, 1), c="r", label="label = class 0"
    )
    ax[0].axhline(0, lw=0.5, c="k")
    ax[0].set_title("(a) $\\Delta\\mu_G$ vs $\\mu_Z$")
    ax[0].set_xlabel("$\\mu_Z$")
    ax[0].legend(fontsize=8)

    m_z, v_z, sv2 = 0.8, 0.8**2, 0.3**2
    k = v_z / (v_z + sv2)
    m_post, v_post = m_z + k * (1.0 - m_z), v_z - k * v_z
    zs = np.linspace(-2, 3, 400)
    ax[1].plot(
        zs,
        standard_normal_pdf((zs - m_z) / math.sqrt(v_z)) / math.sqrt(v_z),
        label="forward pass $Z$",
    )
    ax[1].plot(
        zs,
        standard_normal_pdf((zs - m_post) / math.sqrt(v_post))
        / math.sqrt(v_post),
        label="after logit update",
    )
    ax[1].axvline(1.0, ls=":", c="k", label="target $s=+1$")
    ax[1].set_title("(b) logit channel")
    ax[1].set_xlabel("$z$")
    ax[1].legend(fontsize=8)

    zz = rng.normal(m_z, math.sqrt(v_z), n)
    gg = rng.normal(1.0, 0.3, n)
    pp = standard_normal_cdf(gg * zz)
    nme = node_moments(m_z, v_z, 1.0, 0.3**2)
    sd = math.sqrt(float(nme.var_p))
    ax[2].hist(pp, bins=200, density=True, alpha=0.5, label="$P$")
    ax[2].plot(
        xs,
        standard_normal_pdf((xs - float(nme.mu_p)) / sd) / sd,
        label="moment-matched",
    )
    ax[2].axvline(1.0, ls=":", c="k", label="$y_h=1$")
    ax[2].set_title("(c) probability channel")
    ax[2].set_xlabel("$p$")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_update.pdf"))
    plt.close(fig)

    # ---- Figure 3: online calibration ----------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    mu, sigma, y = make_stream(seed=0)
    for j, (pm, ps) in enumerate([(0.3, 1.0), (0.3, 0.3)]):
        tm, ts = run_recursion(mu, sigma, y, pm, ps)
        it = np.arange(1, len(y) + 1)
        ax[j].plot(it, tm, label="recursion (E:update)")
        ax[j].fill_between(it, tm - 2 * ts, tm + 2 * ts, alpha=0.25)
        em, es, _, _ = exact_gain_posterior_stream(mu, sigma, y, pm, ps)
        ax[j].errorbar(
            [len(y)],
            [em],
            yerr=[2 * es],
            fmt="o",
            c="r",
            label="exact posterior",
        )
        ax[j].axhline(2.0, ls=":", c="k")
        ax[j].set_title(f"({'ab'[j]}) prior N({pm}, {ps}$^2$)")
        ax[j].set_xlabel("sample")
        ax[j].set_ylabel("$\\mu_G$")
    tm_bad, _ = run_recursion(mu, sigma, y, 0.3, 1.0, post_update_sigma_v=0.3)
    ax[1].plot(
        np.arange(1, len(y) + 1),
        tm_bad,
        "--",
        c="orange",
        label="post-update moments",
    )
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)

    tm, _ = run_recursion(mu, sigma, y, 0.3, 1.0)
    mu_t, sigma_t, y_t = make_stream(n=40_000, seed=101)
    for name, gval in [
        ("$G=0.3$", 0.3),
        (f"$G={tm[-1]:.2f}$ (learned)", tm[-1]),
    ]:
        p = np.asarray(conditional_branch_prob(gval, mu_t, sigma_t**2, 1.0))
        bins = np.linspace(0, 1, 16)
        which = np.clip(np.digitize(p, bins) - 1, 0, 14)
        xs_, ys_ = [], []
        for b in range(15):
            m_ = which == b
            if m_.sum() > 20:
                xs_.append(p[m_].mean())
                ys_.append(y_t[m_].mean())
        ax[2].plot(xs_, ys_, "o-", label=name)
    ax[2].plot([0, 1], [0, 1], ":", c="k")
    ax[2].set_title("(c) reliability diagram")
    ax[2].set_xlabel("predicted probability")
    ax[2].set_ylabel("empirical frequency")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_calibration.pdf"))
    plt.close(fig)
    print(
        f"\nfigures written to {outdir}/fig_node.pdf, fig_update.pdf, "
        f"fig_calibration.pdf"
    )


def main(figures: bool = False):
    check_phi_variance()
    check_node_moments()
    check_projection()
    check_bound()
    check_limits()
    check_one_observation()
    check_class_level()
    check_calibrator_matches_scalar()
    check_validation_pass()
    check_epoch_reset()
    if figures:
        make_figures()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
