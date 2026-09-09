import math
import os
import sys
import unittest

import numpy as np

sys.path.append(
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build")
    )
)

from pytagi.hsm_calibration import (
    HSMCalibratedMetric,
    HSMGainCalibrator,
    class_covariance,
    class_moments,
    conditional_branch_prob,
    conditional_branch_prob_derivative,
    gain_update,
    log_sum_exp,
    node_moments,
    phi_covariance,
    phi_expected_bernoulli_variance,
    phi_mean,
    phi_variance,
    standard_normal_cdf,
    standard_normal_log_cdf,
    standard_normal_pdf,
)


def owens_t_reference(h, b, n=64):
    """Owen's T by quadrature of its definition, T(h,b) =
    (1/2pi) int_0^b exp(-h^2(1+t^2)/2)/(1+t^2) dt, on an n-point
    Gauss-Legendre rule over [0, b].

    Independent reference for the 8-point rule that (E:varint) runs on
    [b, 1] -- different interval, different order. The library never needs
    Owen's T; this lives here so the test suite does not either.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    h = np.asarray(h, dtype=np.float64)[..., None]
    half = 0.5 * np.asarray(b, dtype=np.float64)[..., None]
    t = half * (x + 1.0)
    integrand = np.exp(-0.5 * h**2 * (1.0 + t * t)) / (1.0 + t * t)
    return (half[..., 0] / (2.0 * np.pi)) * np.sum(integrand * w, axis=-1)


# Worked example of Section 5 of the note (see pytagi/hsm_calibration.py)
MU_Z, SIGMA_Z = 0.5, 0.5
MU_G, SIGMA_G = 1.5, 0.3


class TestPhiMoments(unittest.TestCase):
    """Section 2 - Moments of Phi(X) for a Gaussian X."""

    def test_mean(self):
        # (E:mean) E[Phi(X)] = Phi(mu / sqrt(1 + sigma^2)). Exact, so the only
        # error is Monte Carlo noise; it has to average out over the batch.
        rng = np.random.default_rng(0)
        mu = rng.normal(0, 2, 20_000)
        var = rng.uniform(0, 5, 20_000)
        x = rng.normal(mu, np.sqrt(var), (400, 20_000))
        err = standard_normal_cdf(x).mean(axis=0) - phi_mean(mu, var)
        self.assertLess(abs(err.mean()), 3e-4)  # no bias
        self.assertLess(err.std(), 0.02)  # ~ 0.3 / sqrt(400)

    def test_variance_matches_owens_t(self):
        # (E:varint) 8-point Gauss-Legendre against (E:var) with Owen's T
        rng = np.random.default_rng(1)
        a = rng.uniform(-8, 8, 100_000)
        var = rng.uniform(0, 25, 100_000)
        b = 1.0 / np.sqrt(1.0 + 2.0 * var)
        m = standard_normal_cdf(a)
        owen = m * (1.0 - m) - 2.0 * owens_t_reference(a, b)
        self.assertLess(np.max(np.abs(phi_variance(a, var) - owen)), 1e-12)

    def test_variance_is_positive(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(-8, 8, 100_000)
        var = rng.uniform(0, 25, 100_000)
        self.assertTrue(np.all(phi_variance(a, var) >= 0.0))

    def test_expected_bernoulli_variance(self):
        # (E:var) E[Phi(1-Phi)] = mu_P(1-mu_P) - Var[Phi]
        a, var = 0.37, 1.4
        m = standard_normal_cdf(a)
        self.assertAlmostEqual(
            float(phi_expected_bernoulli_variance(a, var)),
            float(m * (1 - m) - phi_variance(a, var)),
            places=12,
        )

    def test_covariance_stein(self):
        # (E:cov) Cov(X, Phi(X)) = sigma^2 phi(a) / sqrt(1 + sigma^2)
        rng = np.random.default_rng(3)
        mu, var = 0.4, 0.8
        x = rng.normal(mu, math.sqrt(var), 4_000_000)
        mc = np.cov(x, standard_normal_cdf(x))[0, 1]
        self.assertAlmostEqual(mc, float(phi_covariance(mu, var)), places=3)


class TestNodeMoments(unittest.TestCase):
    """Section 3.1 - Node level."""

    def test_worked_example(self):
        # Section 5: mu_P = 0.7229, sigma_P = 0.2142, Cov(Z,P) = 0.0977,
        # Cov(G,P) = 0.00773
        nm = node_moments(MU_Z, SIGMA_Z**2, MU_G, SIGMA_G**2, s=1.0)
        self.assertAlmostEqual(float(nm.mu_p), 0.7229, places=4)
        self.assertAlmostEqual(float(np.sqrt(nm.var_p)), 0.2142, places=4)
        self.assertAlmostEqual(float(nm.cov_zp), 0.0977, places=4)
        self.assertAlmostEqual(float(nm.cov_gp), 0.00773, places=5)

    def test_monte_carlo(self):
        rng = np.random.default_rng(4)
        n = 2_000_000
        z = rng.normal(MU_Z, SIGMA_Z, n)
        g = rng.normal(MU_G, SIGMA_G, n)
        p = standard_normal_cdf(g * z)
        nm = node_moments(MU_Z, SIGMA_Z**2, MU_G, SIGMA_G**2, s=1.0)
        # Section 5 reports 0.7226 / 0.2076 / 0.0982 / 0.00786 for this MC:
        # the residual is the GMA on S_h of (E:S), largest on sigma_P.
        self.assertAlmostEqual(p.mean(), float(nm.mu_p), delta=1e-3)
        self.assertAlmostEqual(p.std(), float(np.sqrt(nm.var_p)), delta=8e-3)
        self.assertAlmostEqual(np.cov(z, p)[0, 1], float(nm.cov_zp), delta=1e-3)
        self.assertAlmostEqual(np.cov(g, p)[0, 1], float(nm.cov_gp), delta=2e-4)

    def test_exact_when_gain_is_deterministic(self):
        # sigma_G = 0: mu_P = m(mu_G) of (E:mg), Cov(G,P) = 0
        nm = node_moments(MU_Z, SIGMA_Z**2, MU_G, 0.0)
        self.assertAlmostEqual(
            float(nm.mu_p),
            float(conditional_branch_prob(MU_G, MU_Z, SIGMA_Z**2)),
            places=12,
        )
        self.assertEqual(float(nm.cov_gp), 0.0)

    def test_exact_when_logit_is_deterministic(self):
        # sigma_Z = 0: S = mu_Z G is Gaussian, so the GMA is exact
        nm = node_moments(MU_Z, 0.0, MU_G, SIGMA_G**2)
        self.assertAlmostEqual(
            float(nm.mu_p),
            float(phi_mean(MU_G * MU_Z, SIGMA_G**2 * MU_Z**2)),
            places=12,
        )
        self.assertEqual(float(nm.cov_zp), 0.0)

    def test_reduces_to_tagi_with_fixed_gain(self):
        # With a fixed gain 1/alpha, mu_P is exactly one factor of Eq. (1)
        alpha = 3.0
        for s in (-1.0, 1.0):
            nm = node_moments(MU_Z, SIGMA_Z**2, 1.0 / alpha, 0.0, s=s)
            tagi = standard_normal_cdf(
                s * MU_Z / math.sqrt(alpha**2 + SIGMA_Z**2)
            )
            self.assertAlmostEqual(float(nm.mu_p), float(tagi), places=12)

    def test_sibling_branch(self):
        # 1 - P_h: mean 1 - mu_P, same variance, covariances of opposite sign
        plus = node_moments(MU_Z, SIGMA_Z**2, MU_G, SIGMA_G**2, s=1.0)
        minus = node_moments(MU_Z, SIGMA_Z**2, MU_G, SIGMA_G**2, s=-1.0)
        self.assertAlmostEqual(
            float(minus.mu_p), 1.0 - float(plus.mu_p), places=12
        )
        self.assertAlmostEqual(float(minus.var_p), float(plus.var_p), places=14)
        self.assertAlmostEqual(
            float(minus.cov_zp), -float(plus.cov_zp), places=14
        )
        self.assertAlmostEqual(
            float(minus.cov_gp), -float(plus.cov_gp), places=14
        )

    def test_stein_identity_by_finite_difference(self):
        # (E:stein) Cov(Z,P) = var_Z d mu_P / d mu_Z, Cov(G,P) = var_G d mu_P / d mu_G
        h = 1e-6
        nm = node_moments(MU_Z, SIGMA_Z**2, MU_G, SIGMA_G**2, compute_var=False)
        d_mu_z = (
            float(
                node_moments(
                    MU_Z + h, SIGMA_Z**2, MU_G, SIGMA_G**2, compute_var=False
                ).mu_p
            )
            - float(
                node_moments(
                    MU_Z - h, SIGMA_Z**2, MU_G, SIGMA_G**2, compute_var=False
                ).mu_p
            )
        ) / (2 * h)
        d_mu_g = (
            float(
                node_moments(
                    MU_Z, SIGMA_Z**2, MU_G + h, SIGMA_G**2, compute_var=False
                ).mu_p
            )
            - float(
                node_moments(
                    MU_Z, SIGMA_Z**2, MU_G - h, SIGMA_G**2, compute_var=False
                ).mu_p
            )
        ) / (2 * h)
        self.assertAlmostEqual(float(nm.cov_zp), SIGMA_Z**2 * d_mu_z, places=8)
        self.assertAlmostEqual(float(nm.cov_gp), SIGMA_G**2 * d_mu_g, places=8)

    def test_small_gain_variance_limit(self):
        # Cov(G,P) -> var_G m'(mu_G)  (E:mg)
        small = 1e-3
        nm = node_moments(MU_Z, SIGMA_Z**2, MU_G, small**2, compute_var=False)
        approx = small**2 * float(
            conditional_branch_prob_derivative(MU_G, MU_Z, SIGMA_Z**2)
        )
        self.assertAlmostEqual(float(nm.cov_gp) / approx, 1.0, places=5)

    def test_zero_mean_variance(self):
        # Section 6: for mu_h = 0, var_P = 1/4 - arctan(b)/pi
        for sigma_z in (0.3, 1.0, 5.0):
            nm = node_moments(0.0, sigma_z**2, MU_G, SIGMA_G**2)
            self.assertAlmostEqual(
                float(nm.var_p),
                0.25 - math.atan(float(nm.b)) / math.pi,
                places=12,
            )

    def test_sign_and_bound(self):
        # sign Cov(G,P) = sign(s mu_Z), and Cov(G,P)^2 <= var_G mu_P (1 - mu_P)
        rng = np.random.default_rng(5)
        n = 200_000
        mu_z = rng.uniform(-8, 8, n)
        sigma_z = rng.uniform(0, 4, n)
        mu_g = rng.uniform(-1, 6, n)
        sigma_g = rng.uniform(0, 5, n)
        s = rng.choice([-1.0, 1.0], n)
        nm = node_moments(
            mu_z, sigma_z**2, mu_g, sigma_g**2, s, compute_var=False
        )
        nz = nm.cov_gp != 0.0
        self.assertTrue(np.all(np.sign(nm.cov_gp[nz]) == np.sign(s * mu_z)[nz]))
        denom = sigma_g**2 * nm.mu_p * (1.0 - nm.mu_p)
        ratio = np.where(
            denom > 0, nm.cov_gp**2 / np.maximum(denom, 1e-300), 0.0
        )
        self.assertLess(ratio.max(), 1.0)


class TestClassMoments(unittest.TestCase):
    """Section 3.2 - Class level."""

    def test_product_moments(self):
        rng = np.random.default_rng(6)
        mu_p = rng.uniform(0.05, 0.95, 4)
        var_p = rng.uniform(0.0, 0.05, 4)
        mu_pr, var_pr = class_moments(mu_p, var_p)
        # Recursion Q_h = Q_{h-1} P_h with the GMA
        mq, vq = mu_p[0], var_p[0]
        for h in range(1, 4):
            mq, vq = (
                mq * mu_p[h],
                vq * var_p[h] + mq**2 * var_p[h] + mu_p[h] ** 2 * vq,
            )
        self.assertAlmostEqual(float(mu_pr), float(mq), places=14)
        self.assertAlmostEqual(float(var_pr), float(vq), places=14)

    def test_leave_one_out_covariance(self):
        mu_p = np.array([0.6, 0.7, 0.8])
        cov_px = np.array([0.01, 0.02, 0.03])
        got = class_covariance(mu_p, cov_px)
        want = np.array(
            [cov_px[l] * np.prod(np.delete(mu_p, l)) for l in range(3)]
        )
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)

    def test_probabilities_sum_to_one_for_power_of_two(self):
        rng = np.random.default_rng(7)
        for k in (2, 4, 8, 16):
            cal = HSMGainCalibrator(k, sharing="node", mu_gain_init=1.5)
            mu_z = rng.normal(0, 1, (3, cal.tree.len))
            var_z = rng.uniform(0.1, 1.0, (3, cal.tree.len))
            mu_pr, var_pr = cal.class_probabilities(mu_z, var_z)
            np.testing.assert_allclose(mu_pr.sum(axis=-1), 1.0, atol=1e-12)
            self.assertTrue(np.all(var_pr >= 0.0))

    def test_class_moments_against_monte_carlo(self):
        rng = np.random.default_rng(8)
        cal = HSMGainCalibrator(
            4, sharing="node", mu_gain_init=1.5, sigma_gain_init=0.3
        )
        tree = cal.tree
        mu_z = rng.normal(0, 0.8, (1, tree.len))
        var_z = rng.uniform(0.1, 0.5, (1, tree.len))
        mu_pr, var_pr = cal.class_probabilities(mu_z, var_z)
        cov_z, cov_g = cal.class_covariances(mu_z, var_z)

        n = 1_000_000
        z = rng.normal(mu_z, np.sqrt(var_z), (n, tree.len))
        mu_g, var_g = cal.gains_per_unit()
        g = rng.normal(mu_g, np.sqrt(var_g), (n, tree.len))
        p_plus = standard_normal_cdf(g * z)
        for k in range(4):
            idx, s = tree.idx[k], tree.obs[k]
            p = np.where(s > 0, p_plus[:, idx], 1.0 - p_plus[:, idx])
            prod = np.prod(p, axis=1)
            # (E:class) is exact given the node moments; what is left is the
            # node-level GMA error of (E:S) accumulated along the path.
            self.assertAlmostEqual(prod.mean(), mu_pr[0, k], delta=1e-2)
            self.assertAlmostEqual(prod.var(), var_pr[0, k], delta=1e-2)
            for j in idx:
                self.assertAlmostEqual(
                    np.cov(z[:, j], prod)[0, 1], cov_z[0, k, j], delta=1e-2
                )
                self.assertAlmostEqual(
                    np.cov(g[:, j], prod)[0, 1], cov_g[0, k, j], delta=1e-2
                )
            # off-path units carry no covariance
            off = np.setdiff1d(np.arange(tree.len), idx)
            np.testing.assert_array_equal(cov_z[0, k, off], 0.0)
            np.testing.assert_array_equal(cov_g[0, k, off], 0.0)


class TestGainUpdate(unittest.TestCase):
    """Section 4.2 - Auxiliary observation model."""

    def test_worked_example(self):
        # Section 5: d mu_G = +0.0107 (y=1), -0.0279 (y=0), sigma_G -> 0.2995
        nm = node_moments(MU_Z, SIGMA_Z**2, MU_G, SIGMA_G**2, s=1.0)
        mu1, var1 = gain_update(MU_G, SIGMA_G**2, nm.mu_p, nm.cov_gp, y=1.0)
        mu0, _ = gain_update(MU_G, SIGMA_G**2, nm.mu_p, nm.cov_gp, y=0.0)
        self.assertAlmostEqual(float(mu1) - MU_G, +0.0107, places=4)
        self.assertAlmostEqual(float(mu0) - MU_G, -0.0279, places=4)
        self.assertAlmostEqual(math.sqrt(float(var1)), 0.2995, places=4)

    def test_mean_is_exact_bernoulli_posterior(self):
        # E[G | y=1] = E[G P] / E[P] on a grid over g
        nm = node_moments(MU_Z, SIGMA_Z**2, MU_G, SIGMA_G**2, s=1.0)
        g = np.linspace(MU_G - 12 * SIGMA_G, MU_G + 12 * SIGMA_G, 400_001)
        prior = np.exp(-0.5 * ((g - MU_G) / SIGMA_G) ** 2)
        m = conditional_branch_prob(g, MU_Z, SIGMA_Z**2, 1.0)
        for y, exact_note in ((1, 0.0109), (0, -0.0284)):
            post = prior * (m if y == 1 else 1 - m)
            post /= np.trapezoid(post, g)
            exact = float(np.trapezoid(g * post, g)) - MU_G
            step = float(
                gain_update(MU_G, SIGMA_G**2, nm.mu_p, nm.cov_gp, y=y)[0]
            )
            self.assertAlmostEqual(exact, exact_note, places=4)
            # The mean step is exact under the Bernoulli likelihood; the
            # residual is the GMA on S_h inside mu_P and Cov(G,P). Section 5:
            # +0.0107 vs +0.0109 and -0.0279 vs -0.0284.
            self.assertAlmostEqual(step - MU_G, exact, delta=1e-3)

    def test_variance_never_negative(self):
        rng = np.random.default_rng(9)
        n = 200_000
        mu_z = rng.uniform(-8, 8, n)
        sigma_z = rng.uniform(0, 4, n)
        mu_g = rng.uniform(-1, 6, n)
        sigma_g = rng.uniform(0, 5, n)
        s = rng.choice([-1.0, 1.0], n)
        nm = node_moments(
            mu_z, sigma_z**2, mu_g, sigma_g**2, s, compute_var=False
        )
        _, var_post = gain_update(mu_g, sigma_g**2, nm.mu_p, nm.cov_gp, y=1.0)
        self.assertTrue(np.all(var_post >= 0.0))

    def test_orientation_conventions_agree(self):
        # s-oriented "y = 1" equals the fixed-orientation "y+ = 0"
        nm_p = node_moments(
            MU_Z, SIGMA_Z**2, MU_G, SIGMA_G**2, s=+1.0, compute_var=False
        )
        nm_m = node_moments(
            MU_Z, SIGMA_Z**2, MU_G, SIGMA_G**2, s=-1.0, compute_var=False
        )
        a = gain_update(MU_G, SIGMA_G**2, nm_m.mu_p, nm_m.cov_gp, y=1.0)
        b = gain_update(MU_G, SIGMA_G**2, nm_p.mu_p, nm_p.cov_gp, y=0.0)
        self.assertAlmostEqual(float(a[0]), float(b[0]), places=14)
        self.assertAlmostEqual(float(a[1]), float(b[1]), places=14)

    def test_saturation_guard(self):
        # mu_P clipped to [eps, 1-eps] keeps the update finite
        mu, var = gain_update(1.0, 1.0, 0.0, 0.0, y=1.0)
        self.assertTrue(np.isfinite(mu) and np.isfinite(var))


class TestCalibrator(unittest.TestCase):
    """Section 4.3 - Algorithm 1."""

    @staticmethod
    def _stream(n=1500, g_star=2.0, seed=0):
        rng = np.random.default_rng(seed)
        mu = rng.normal(0.0, 1.5, n)
        sigma = rng.uniform(0.3, 1.2, n)
        z = rng.normal(mu, sigma)
        y = (rng.uniform(size=n) < standard_normal_cdf(g_star * z)).astype(int)
        return mu, sigma, y

    def test_recovers_the_true_gain(self):
        mu, sigma, y = self._stream(n=3000, g_star=2.0, seed=0)
        cal = HSMGainCalibrator(
            2, sharing="global", mu_gain_init=0.3, sigma_gain_init=1.0
        )
        # cuTAGI orients the tree with s_h = (-1)^{C_h}: class 0 -> s = +1
        cal.calibrate(mu.reshape(-1, 1), (sigma**2).reshape(-1, 1), 1 - y)
        self.assertAlmostEqual(float(cal.mu_gain[0]), 2.0, delta=0.35)
        self.assertLess(float(cal.sigma_gain[0]), 1.0)

    def test_variance_only_shrinks(self):
        mu, sigma, y = self._stream(seed=1)
        cal = HSMGainCalibrator(2, sharing="global")
        prev = float(cal.var_gain[0])
        for i in range(len(y)):
            cal.update_sample(
                mu[i : i + 1], sigma[i : i + 1] ** 2, int(1 - y[i])
            )
            self.assertLessEqual(float(cal.var_gain[0]), prev + 1e-15)
            prev = float(cal.var_gain[0])

    def test_epoch_reset(self):
        mu, sigma, y = self._stream(seed=2)
        cal = HSMGainCalibrator(2, sharing="global")
        cal.calibrate(mu.reshape(-1, 1), (sigma**2).reshape(-1, 1), 1 - y)
        mu_after = float(cal.mu_gain[0])
        cal.reset_variance()
        self.assertAlmostEqual(float(cal.mu_gain[0]), mu_after, places=14)
        self.assertAlmostEqual(
            float(cal.var_gain[0]), cal.var_gain_init, places=14
        )

    def test_process_noise_keeps_the_gain_adaptive(self):
        mu, sigma, y = self._stream(n=3000, seed=3)
        cal_q = HSMGainCalibrator(2, sharing="global", process_noise=1e-4)
        cal_0 = HSMGainCalibrator(2, sharing="global")
        for cal in (cal_q, cal_0):
            cal.calibrate(mu.reshape(-1, 1), (sigma**2).reshape(-1, 1), 1 - y)
        self.assertGreater(float(cal_q.var_gain[0]), float(cal_0.var_gain[0]))

    def test_positivity_clip(self):
        cal = HSMGainCalibrator(
            2,
            sharing="global",
            mu_gain_init=0.05,
            sigma_gain_init=1.0,
            min_mu_gain=1e-3,
        )
        # A long run of contradicted predictions pushes the gain down
        mu = np.full((400, 1), 2.0)
        var = np.full((400, 1), 0.5)
        cal.calibrate(mu, var, np.ones(400, dtype=int))  # class 1 -> s = -1
        self.assertGreaterEqual(float(cal.mu_gain[0]), 1e-3)

    def test_sharing_modes(self):
        for sharing, expected in (("node", 11), ("level", 4), ("global", 1)):
            cal = HSMGainCalibrator(10, sharing=sharing)
            self.assertEqual(cal.num_gains, expected)
            self.assertEqual(cal.gain_idx.shape, (10, cal.tree.n_obs))

    def test_only_path_gains_are_updated(self):
        cal = HSMGainCalibrator(8, sharing="node")
        before = cal.mu_gain.copy()
        rng = np.random.default_rng(10)
        mu_z = rng.normal(0, 1, cal.tree.len)
        var_z = rng.uniform(0.1, 1.0, cal.tree.len)
        cal.update_sample(mu_z, var_z, 3)
        moved = np.where(np.abs(cal.mu_gain - before) > 0)[0]
        np.testing.assert_array_equal(np.sort(moved), np.sort(cal.tree.idx[3]))

    def test_metric_interface(self):
        rng = np.random.default_rng(11)
        metric = HSMCalibratedMetric(10, sharing="level")
        n = 5
        m_pred = rng.normal(0, 1, n * metric.len)
        v_pred = rng.uniform(0.1, 1.0, n * metric.len)
        labels = rng.integers(0, 10, n)
        pred = metric.get_predicted_labels(m_pred, v_pred)
        self.assertEqual(pred.shape, (n,))
        mu_pr, var_pr = metric.get_class_probabilities(m_pred, v_pred)
        self.assertEqual(mu_pr.shape, (n, 10))
        np.testing.assert_allclose(mu_pr.sum(axis=1), 1.0, atol=1e-12)
        metric.calibrate(m_pred, v_pred, labels)
        self.assertTrue(np.all(np.isfinite(metric.calibrator.mu_gain)))
        self.assertIsInstance(metric.error_rate(m_pred, v_pred, labels), float)


class TestLogSpaceHelpers(unittest.TestCase):
    """log Phi and log-sum-exp, used to form (E:class) without underflow."""

    def test_log_cdf_matches_log_of_cdf_in_the_bulk(self):
        x = np.linspace(-30.0, 8.0, 100_001)
        self.assertLess(
            np.max(
                np.abs(
                    standard_normal_log_cdf(x) - np.log(standard_normal_cdf(x))
                )
            ),
            1e-12,
        )

    def test_log_cdf_right_tail_keeps_relative_accuracy(self):
        # log Phi(x) -> 0^- ; log(Phi(x)) rounds to exactly 0, log1p does not.
        x = np.array([5.0, 10.0, 20.0, 38.0])
        got = standard_normal_log_cdf(x)
        self.assertTrue(np.all(got < 0.0))
        # -log Phi(x) ~ Phi(-x) to leading order
        self.assertTrue(np.allclose(-got, standard_normal_cdf(-x), rtol=1e-6))

    def test_log_cdf_left_tail_beyond_erfc_underflow(self):
        # erfc underflows past x ~ -37.5; the asymptotic branch takes over and
        # must stay continuous and match the leading asymptotics.
        x = np.array([-37.4, -37.6, -100.0, -1000.0])
        got = standard_normal_log_cdf(x)
        self.assertTrue(np.all(np.isfinite(got)))
        lead = -0.5 * x**2 - np.log(-x) - 0.5 * math.log(2.0 * math.pi)
        self.assertTrue(np.allclose(got, lead, rtol=1e-3))
        # continuity across the switch
        self.assertAlmostEqual(
            float(standard_normal_log_cdf(-37.5 + 1e-9)),
            float(standard_normal_log_cdf(-37.5 - 1e-9)),
            places=6,
        )

    def test_log_cdf_scalar_and_shape(self):
        self.assertEqual(np.ndim(standard_normal_log_cdf(-1.0)), 0)
        self.assertEqual(
            standard_normal_log_cdf(np.zeros((3, 4))).shape, (3, 4)
        )

    def test_log_sum_exp(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0.0, 300.0, (200, 65))
        # overflow-free: exp(x) itself is inf here
        got = log_sum_exp(x, axis=1, keepdims=True)
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertEqual(got.shape, (200, 1))
        # normalizing with it gives probabilities summing to one
        lp = x - got
        self.assertTrue(np.allclose(np.exp(lp).sum(axis=1), 1.0, atol=1e-12))
        # against the direct computation where that is safe
        y = rng.normal(0.0, 2.0, (50, 9))
        self.assertTrue(
            np.allclose(
                log_sum_exp(y, axis=1),
                np.log(np.exp(y).sum(axis=1)),
                atol=1e-12,
            )
        )
        self.assertEqual(np.ndim(log_sum_exp(y)), 0)


if __name__ == "__main__":
    unittest.main()
