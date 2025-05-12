#include "../include/activation.h"
#include <algorithm>
#include <cmath>

#ifdef USE_CUDA
#include "activation_cuda.cuh"
#endif

void relu_mean_var(std::vector<float> const &mu_z,
                   std::vector<float> const &var_z, int start_chunk,
                   int end_chunk, std::vector<float> &mu_a,
                   std::vector<float> &jcb, std::vector<float> &var_a)
/*
 */
{
    float zero_pad = 0.0f;
    float one_pad = 1.0f;
    float tmp;
    int col;
    for (col = start_chunk; col < end_chunk; col++) {
        tmp = std::max(mu_z[col], zero_pad);
        mu_a[col] = tmp;
        if (tmp == 0) {
            jcb[col] = zero_pad;
            var_a[col] = zero_pad;
        } else {
            jcb[col] = one_pad;
            var_a[col] = var_z[col];
        }
    }
}

void relu_mean_var_mp(std::vector<float> const &mu_z,
                      std::vector<float> const &var_z, int n,
                      unsigned int num_threads, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a)
/*
 */
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_z, &var_z, &mu_a, &jcb, &var_a] {
            relu_mean_var(mu_z, var_z, start_chunk, end_chunk, mu_a, jcb,
                          var_a);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void sigmoid_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                      int start_chunk, int end_chunk, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a)
/*
 */
{
    float tmp;
    for (int col = start_chunk; col < end_chunk; col++) {
        tmp = 1 / (1 + expf(-mu_z[col]));
        mu_a[col] = tmp;
        jcb[col] = tmp * (1 - tmp);
        var_a[col] = tmp * (1 - tmp) * var_z[col] * tmp * (1 - tmp);
    }
}

void sigmoid_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                         int n, unsigned int num_threads,
                         std::vector<float> &mu_a, std::vector<float> &jcb,
                         std::vector<float> &var_a)
/*
 */
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_z, &var_z, &mu_a, &jcb, &var_a] {
            sigmoid_mean_var(mu_z, var_z, start_chunk, end_chunk, mu_a, jcb,
                             var_a);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void tanh_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                   int start_chunk, int end_chunk, std::vector<float> &mu_a,
                   std::vector<float> &jcb, std::vector<float> &var_a)
/*
 */
{
    float tmp = 0;
    for (int col = start_chunk; col < end_chunk; col++) {
        tmp = tanhf(mu_z[col]);
        mu_a[col] = tmp;
        jcb[col] = (1 - tmp * tmp);
        var_a[col] = (1 - tmp * tmp) * var_z[col] * (1 - tmp * tmp);
    }
}

void tanh_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                      int n, unsigned int num_threads, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_z, &var_z, &mu_a, &jcb, &var_a] {
            tanh_mean_var(mu_z, var_z, start_chunk, end_chunk, mu_a, jcb,
                          var_a);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void mixture_relu_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                           int start_chunk, int end_chunk,
                           std::vector<float> &mu_a, std::vector<float> &jcb,
                           std::vector<float> &var_a)
/*
 */
{
    float std_z, alpha, pdf_alpha, cdf_alpha;
    for (int i = start_chunk; i < end_chunk; i++) {
        // Reused components for moments calculations
        std_z = powf(var_z[i], 0.5);
        alpha = mu_z[i] / std_z;
        pdf_alpha = normpdf_cpu(alpha, 0.0f, 1.0f);
        cdf_alpha = normcdf_cpu(alpha);

        // Moments calculations (L. Alric, 2024)
        mu_a[i] = mu_z[i] * cdf_alpha + std_z * pdf_alpha;
        var_a[i] = -powf(mu_a[i], 2) + 2 * mu_a[i] * mu_z[i] -
                   mu_z[i] * std_z * pdf_alpha +
                   (var_z[i] - powf(mu_z[i], 2)) * cdf_alpha;
        jcb[i] = cdf_alpha;
    }
}

void mixture_relu_mean_var_mp(std::vector<float> &mu_z,
                              std::vector<float> &var_z, int n,
                              unsigned int num_threads,
                              std::vector<float> &mu_a, std::vector<float> &jcb,
                              std::vector<float> &var_a)
/*
 */
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_z, &var_z, &mu_a, &jcb, &var_a] {
            mixture_relu_mean_var(mu_z, var_z, start_chunk, end_chunk, mu_a,
                                  jcb, var_a);
        });
    }
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void mixture_sigmoid_mean_var(std::vector<float> &mu_z,
                              std::vector<float> &var_z, int start_chunk,
                              int end_chunk, std::vector<float> &mu_a,
                              std::vector<float> &jcb,
                              std::vector<float> &var_a)
/*
 */
{
    float std_z, alpha_l, alpha_u, pdf_l, pdf_u, cdf_l, cdf_u;
    for (int i = start_chunk; i < end_chunk; i++) {
        // cdf and pdf for truncated normal distribution
        std_z = powf(var_z[i], 0.5);
        alpha_l = (1.0f + mu_z[i]) / std_z;  // Lower truncation
        alpha_u = (1.0f - mu_z[i]) / std_z;  // Upper truncation
        cdf_l = normcdf_cpu(alpha_l);
        cdf_u = normcdf_cpu(alpha_u);
        pdf_l = normpdf_cpu(alpha_l, 0.0f, 1.0f);
        pdf_u = normpdf_cpu(alpha_u, 0.0f, 1.0f);

        // Moments calculations (L. Alric, 2024)
        mu_a[i] = (mu_z[i] + 1) * cdf_l + (mu_z[i] - 1) * cdf_u +
                  std_z * (pdf_l - pdf_u) - mu_z[i];
        var_a[i] =
            std::max(0.000001f,
                     (cdf_l * (var_z[i] - powf(mu_z[i], 2) - 2 * mu_z[i] - 1) +
                      cdf_u * (var_z[i] - powf(mu_z[i], 2) + 2 * mu_z[i] - 1) +
                      std_z * (pdf_u * (mu_z[i] - 1) - pdf_l * (mu_z[i] + 1)) -
                      powf(mu_a[i], 2) + 2 * mu_a[i] * mu_z[i] +
                      powf(mu_z[i], 2) - var_z[i] + 2) /
                         4.0f);
        mu_a[i] = mu_a[i] / 2.0f + 0.5f;
        jcb[i] = (cdf_u + cdf_l - 1) / 2.0f;
    }
}
void mixture_sigmoid_mean_var_mp(std::vector<float> &mu_z,
                                 std::vector<float> &var_z, int n,
                                 unsigned int num_threads,
                                 std::vector<float> &mu_a,
                                 std::vector<float> &jcb,
                                 std::vector<float> &var_a)
/*
 */
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_z, &var_z, &mu_a, &jcb, &var_a] {
            mixture_sigmoid_mean_var(mu_z, var_z, start_chunk, end_chunk, mu_a,
                                     jcb, var_a);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void mixture_tanh_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                           int start_chunk, int end_chunk,
                           std::vector<float> &mu_a, std::vector<float> &jcb,
                           std::vector<float> &var_a)
/*
 */
{
    float std_z, alpha_l, alpha_u, pdf_l, pdf_u, cdf_l, cdf_u;
    for (int i = start_chunk; i < end_chunk; i++) {
        // cdf and pdf for truncated normal distribution
        std_z = powf(var_z[i], 0.5);
        alpha_l = (1.0f + mu_z[i]) / std_z;  // Lower truncation
        alpha_u = (1.0f - mu_z[i]) / std_z;  // Upper truncation
        cdf_l = normcdf_cpu(alpha_l);
        cdf_u = normcdf_cpu(alpha_u);
        pdf_l = normpdf_cpu(alpha_l, 0.0f, 1.0f);
        pdf_u = normpdf_cpu(alpha_u, 0.0f, 1.0f);

        // Moments calculations (L. Alric, 2024)
        mu_a[i] = (mu_z[i] + 1) * cdf_l + (mu_z[i] - 1) * cdf_u +
                  std_z * (pdf_l - pdf_u) - mu_z[i];
        var_a[i] = std::max(
            0.000001f,
            cdf_l * (var_z[i] - powf(mu_z[i], 2) - 2 * mu_z[i] - 1) +
                cdf_u * (var_z[i] - powf(mu_z[i], 2) + 2 * mu_z[i] - 1) +
                std_z * (pdf_u * (mu_z[i] - 1) - pdf_l * (mu_z[i] + 1)) -
                powf(mu_a[i], 2) + 2 * mu_a[i] * mu_z[i] + powf(mu_z[i], 2) -
                var_z[i] + 2);
        jcb[i] = cdf_u + cdf_l - 1;
    }
}

void mixture_tanh_mean_var_mp(std::vector<float> &mu_z,
                              std::vector<float> &var_z, int n,
                              unsigned int num_threads,
                              std::vector<float> &mu_a, std::vector<float> &jcb,
                              std::vector<float> &var_a)
/*
 */
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_z, &var_z, &mu_a, &jcb, &var_a] {
            mixture_tanh_mean_var(mu_z, var_z, start_chunk, end_chunk, mu_a,
                                  jcb, var_a);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void softplus_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                       int start_chunk, int end_chunk, std::vector<float> &mu_a,
                       std::vector<float> &jcb, std::vector<float> &var_a)
/*
 */
{
    float tmp;
    for (int col = start_chunk; col < end_chunk; col++) {
        mu_a[col] = logf(1 + expf(mu_z[col]));
        tmp = 1 / (1 + expf(-mu_z[col]));
        jcb[col] = tmp;
        var_a[col] = tmp * var_z[col] * tmp;
    }
}

void softplus_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                          int n, unsigned int num_threads,
                          std::vector<float> &mu_a, std::vector<float> &jcb,
                          std::vector<float> &var_a)
/*
 */
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_z, &var_z, &mu_a, &jcb, &var_a] {
            softplus_mean_var(mu_z, var_z, start_chunk, end_chunk, mu_a, jcb,
                              var_a);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void leaky_relu_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                         float alpha, int start_chunk, int end_chunk,
                         std::vector<float> &mu_a, std::vector<float> &jcb,
                         std::vector<float> &var_a)
/*
 */
{
    float zeroPad = 0;
    float onePad = 1;
    float tmp;
    int col;
    for (col = start_chunk; col < end_chunk; col++) {
        tmp = std::max(mu_z[col], zeroPad);
        if (tmp == 0) {
            mu_a[col] = alpha * mu_z[col];
            jcb[col] = alpha;
            var_a[col] = alpha * var_z[col] * alpha;
        } else {
            mu_a[col] = tmp;
            jcb[col] = onePad;
            var_a[col] = var_z[col];
        }
    }
}

void leaky_relu_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                            float alpha, int n, unsigned int num_threads,
                            std::vector<float> &mu_a, std::vector<float> &jcb,
                            std::vector<float> &var_a)
/*
 */
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_z, &var_z, &alpha, &mu_a, &jcb, &var_a] {
            leaky_relu_mean_var(mu_z, var_z, alpha, start_chunk, end_chunk,
                                mu_a, jcb, var_a);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void softmax_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                      int no, int batch_size, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a)
/*
 */
{
    float sum, max_m, max_v;
    int idx;
    for (int i = 0; i < batch_size; i++) {
        sum = 0.0f;
        idx = i * no;
        auto max_idx =
            std::max_element(mu_z.begin() + idx, mu_z.begin() + idx + no) -
            mu_z.begin();
        max_m = mu_z[max_idx];
        max_v = var_z[max_idx];
        for (int j = 0; j < no; j++) {
            mu_a[idx + j] = expf(mu_z[idx + j] - max_m);
            sum += mu_a[idx + j];
        }
        for (int j = 0; j < no; j++) {
            mu_a[idx + j] = mu_a[idx + j] / sum;
            jcb[idx + j] = mu_a[idx + j] * (1 - mu_a[idx + j]);
            // TODO: double check on covariance formulation
            var_a[idx + j] =
                jcb[idx + j] * (var_z[idx + j] + max_v) * jcb[idx + j];
        }
    }
}

// CPU implementation of the standard normal CDF
inline double normcdf_cpu(double x) {
    // 0.5 * [1 + erf(x / sqrt(2))]
    static const double SQRT2 = 1.4142135623730951;
    return 0.5 * (1.0 + std::erf(x / SQRT2));
}

void remax_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
    size_t output_size, int batch_size, std::vector<float> &mu_a,
                    std::vector<float> &jcb, std::vector<float> &var_a) {
    // Stability constants
    constexpr double EPS           = 1e-8;
    constexpr double MIN_CDF       = 1e-20;
    constexpr double MAX_RATIO     = 1e3;
    constexpr double MAX_LOG       = 10.0;
    constexpr double MAX_INV_VARLN = 50.0;
    constexpr double SQRT_2PI      = 2.5066282746310002;

    const int K = static_cast<int>(output_size / 2);

    // Temp buffers per-element
    std::vector<double> muM(K), varM(K), cov_ZM(K), cov_M_M_sum(K);
    std::vector<double> mulnM(K), varlnM(K), cov_lnM_lnM_sum(K);
    std::vector<double> mulnA(K), varlnA(K), muA(K), cov_inv(K);

    for (int i = 0; i < batch_size; ++i) {
        // 1) Truncated-normal moments per component
        for (int k = 0; k < K; ++k) {
            int idx  = 2*k + i * output_size;
            double mz = static_cast<double>(mu_z[idx]);
            double vz = std::max(static_cast<double>(var_z[idx]), EPS);
            double sz = std::sqrt(vz);
            double α  = mz / sz;

            double cdfn = std::max(normcdf_cpu(α), MIN_CDF);
            double pdfn = std::max((1.0 / SQRT_2PI) * std::exp(-0.5 * α * α), MIN_CDF);

            double m = sz * pdfn + mz * cdfn;
            m = std::max(m, EPS);

            double e2 = (mz*mz + vz) * cdfn + mz * sz * pdfn;
            double v  = std::max(e2 - m*m, EPS);

            muM[k]         = m;
            varM[k]        = v;
            cov_ZM[k]      = std::max(cdfn * vz, EPS);
            cov_M_M_sum[k] = v;
        }

        // 2) Summations
        double muM_sum = EPS, varM_sum = EPS;
        for (int k = 0; k < K; ++k) {
            muM_sum  += muM[k];
            varM_sum += varM[k];
        }

        // 3) Log-domain transforms
        for (int k = 0; k < K; ++k) {
            double ratio = varM[k] / (muM[k]*muM[k]);
            ratio = std::min(std::max(ratio, MIN_CDF), MAX_RATIO);
            varlnM[k] = std::min(std::max(std::log(1.0 + ratio), MIN_CDF), MAX_LOG);

            double mln = std::log(muM[k]) - 0.5 * varlnM[k];
            mulnM[k]   = std::min(std::max(mln, -MAX_LOG), MAX_LOG);

            double corr = cov_M_M_sum[k] / (muM[k] * muM_sum);
            corr = std::min(std::max(corr, MIN_CDF), MAX_RATIO);
            cov_lnM_lnM_sum[k] = std::max(std::log(1.0 + corr), MIN_CDF);
        }

        double sum_ratio = varM_sum / (muM_sum * muM_sum);
        sum_ratio = std::min(std::max(sum_ratio, MIN_CDF), MAX_RATIO);
        double varlnM_sum = std::min(std::max(std::log(1.0 + sum_ratio), MIN_CDF), MAX_LOG);
        double mulnM_sum  = std::min(std::max(std::log(muM_sum) - 0.5 * varlnM_sum,
                                              -MAX_LOG), MAX_LOG);

        // 4) Inverse transforms
        double inv_varln = std::min(1.0 / varlnM_sum, MAX_INV_VARLN);
        double inv_mln   = 1.0 - mulnM_sum;
        double arg       = std::min(std::max(inv_mln + 0.5 * inv_varln, -MAX_LOG), MAX_LOG);
        double muM_inv   = std::exp(arg);
        double var_exp   = std::max(std::exp(inv_varln) - 1.0, MIN_CDF);
        double varM_inv  = muM_inv * muM_inv * var_exp;

        for (int k = 0; k < K; ++k) {
            double ce = std::max(std::exp(cov_lnM_lnM_sum[k]) - 1.0, EPS);
            cov_inv[k] = ce * muM_sum * muM_inv;
        }

        // 5) Compute A in log-domain
        for (int k = 0; k < K; ++k) {
            mulnA[k]  = std::min(std::max(mulnM[k] - mulnM_sum, -MAX_LOG), MAX_LOG);
            varlnA[k] = std::min(std::max(varlnM[k] + varlnM_sum - 2.0 * cov_lnM_lnM_sum[k],
                                          MIN_CDF), MAX_LOG);
        }

        // 6) Back to original domain & write outputs
        double muA_sum = 0.0;
        for (int k = 0; k < K; ++k) {
            double e = std::min(std::max(mulnA[k] + 0.5 * varlnA[k], -MAX_LOG), MAX_LOG);
            muA[k]   = std::max(std::exp(e), MIN_CDF);
            muA_sum += muA[k];
        }
        muA_sum = std::max(muA_sum, EPS);

        for (int k = 0; k < K; ++k) {
            int idx = 2*k + i * output_size;
            double norm_muA = muA[k] / muA_sum;
            norm_muA        = std::min(std::max(norm_muA, EPS), 1.0);

            double ve    = std::max(std::exp(varlnA[k]) - 1.0, EPS);
            double varA_l= norm_muA * norm_muA * ve;
            varA_l        = std::max(varA_l, norm_muA*(1.0 - norm_muA));

            double ce    = std::max(std::exp(varlnM[k] - cov_lnM_lnM_sum[k]) - 1.0, EPS);
            double covAM = ce * norm_muA * muM[k];
            double varZk = std::max(static_cast<double>(var_z[idx]), EPS);

            double covZA = covAM * cov_ZM[k] / varM[k];
            double maxC  = std::sqrt(varA_l) * std::sqrt(varZk);
            covZA        = std::min(std::max(covZA, EPS), maxC);

            mu_a[idx]  = static_cast<float>(norm_muA);
            var_a[idx] = static_cast<float>(varA_l);
            jcb[idx]   = static_cast<float>(covZA);
        }
    }
}


void even_exp_mean_var(std::vector<float> const &mu_z,
                       std::vector<float> const &var_z,
                       std::vector<float> &jcb_z, int start_chunk,
                       int end_chunk, std::vector<float> &mu_a,
                       std::vector<float> &var_a, std::vector<float> &jcb_a)

{
    for (int i = start_chunk; i < end_chunk; i++) {
        if (i % 2 == 0) {
            mu_a[i] = mu_z[i];
            var_a[i] = var_z[i];
            jcb_a[i] = jcb_z[i];
        } else {
            mu_a[i] = expf(mu_z[i] + 0.5 * var_z[i]);
            var_a[i] = expf(2 * mu_z[i] + var_z[i]) * (expf(var_z[i]) - 1);
            jcb_a[i] = var_z[i] * mu_a[i];
        }
    }
}

void even_exp_mean_var_mp(std::vector<float> const &mu_z,
                          std::vector<float> const &var_z,
                          std::vector<float> &jcb_z, int n,
                          unsigned int num_threads, std::vector<float> &mu_a,
                          std::vector<float> &var_a,
                          std::vector<float> &jcb_a) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        int start_chunk =
            i * n_per_thread + std::min(static_cast<int>(i), extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_z, &var_z, &jcb_z, &mu_a, &var_a, &jcb_a] {
            even_exp_mean_var(mu_z, var_z, jcb_z, start_chunk, end_chunk, mu_a,
                              var_a, jcb_a);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
/// ReLU
////////////////////////////////////////////////////////////////////////////////

ReLU::ReLU() {};
ReLU::~ReLU() {};

std::string ReLU::get_layer_info() const
/*
 */
{
    return "ReLU()";
}

std::string ReLU::get_layer_name() const
/*
 */
{
    return "ReLU";
}

LayerType ReLU::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void ReLU::forward(BaseHiddenStates &input_states,
                   BaseHiddenStates &output_states, BaseTempStates &temp_states)
/*
 */
{
    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    if (this->num_threads > 1) {
        relu_mean_var_mp(input_states.mu_a, input_states.var_a, end_chunk,
                         this->num_threads, output_states.mu_a,
                         output_states.jcb, output_states.var_a);
    } else {
        relu_mean_var(input_states.mu_a, input_states.var_a, start_chunk,
                      end_chunk, output_states.mu_a, output_states.jcb,
                      output_states.var_a);
    }

    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.size = input_states.size;
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> ReLU::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<ReLUCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Sigmoid
////////////////////////////////////////////////////////////////////////////////
Sigmoid::Sigmoid() {};
Sigmoid::~Sigmoid() {};

std::string Sigmoid::get_layer_info() const
/*
 */
{
    return "Sigmoid()";
}

std::string Sigmoid::get_layer_name() const
/*
 */
{
    return "Sigmoid";
}

LayerType Sigmoid::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void Sigmoid::forward(BaseHiddenStates &input_states,
                      BaseHiddenStates &output_states,
                      BaseTempStates &temp_states)
/*
 */
{
    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    sigmoid_mean_var(input_states.mu_a, input_states.var_a, start_chunk,
                     end_chunk, output_states.mu_a, output_states.jcb,
                     output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> Sigmoid::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<SigmoidCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Tanh
////////////////////////////////////////////////////////////////////////////////
Tanh::Tanh() {}
Tanh::~Tanh() {}

std::string Tanh::get_layer_info() const
/*
 */
{
    return "Tanh()";
}

std::string Tanh::get_layer_name() const
/*
 */

{
    return "Tanh";
}

LayerType Tanh::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void Tanh::forward(BaseHiddenStates &input_states,
                   BaseHiddenStates &output_states, BaseTempStates &temp_states)
/*
 */
{
    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    tanh_mean_var(input_states.mu_a, input_states.var_a, start_chunk, end_chunk,
                  output_states.mu_a, output_states.jcb, output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> Tanh::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<TanhCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Mixture ReLU
////////////////////////////////////////////////////////////////////////////////
MixtureReLU::MixtureReLU() {}
MixtureReLU::~MixtureReLU() {}

std::string MixtureReLU::get_layer_info() const
/*
 */
{
    return "MixtureReLU()";
}

std::string MixtureReLU::get_layer_name() const
/*
 */

{
    return "MixtureReLU";
}

LayerType MixtureReLU::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureReLU::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states)
/*
 */
{
    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    mixture_relu_mean_var(input_states.mu_a, input_states.var_a, start_chunk,
                          end_chunk, output_states.mu_a, output_states.jcb,
                          output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> MixtureReLU::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<MixtureReLUCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Mixture Sigmoid
////////////////////////////////////////////////////////////////////////////////
MixtureSigmoid::MixtureSigmoid() {};
MixtureSigmoid::~MixtureSigmoid() {};

std::string MixtureSigmoid::get_layer_info() const
/*
 */
{
    return "MixtureSigmoid()";
}

std::string MixtureSigmoid::get_layer_name() const
/*
 */

{
    return "MixtureSigmoid";
}

LayerType MixtureSigmoid::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureSigmoid::forward(BaseHiddenStates &input_states,
                             BaseHiddenStates &output_states,
                             BaseTempStates &temp_states)
/*
 */
{
    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    mixture_sigmoid_mean_var(input_states.mu_a, input_states.var_a, start_chunk,
                             end_chunk, output_states.mu_a, output_states.jcb,
                             output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> MixtureSigmoid::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<MixtureSigmoidCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Mixture Tanh
////////////////////////////////////////////////////////////////////////////////
MixtureTanh::MixtureTanh() {};
MixtureTanh::~MixtureTanh() {};

std::string MixtureTanh::get_layer_info() const
/*
 */
{
    return "MixtureTanh()";
}

std::string MixtureTanh::get_layer_name() const
/*
 */

{
    return "MixtureTanh";
}

LayerType MixtureTanh::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureTanh::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states)
/*
 */
{
    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    mixture_tanh_mean_var(input_states.mu_a, input_states.var_a, start_chunk,
                          end_chunk, output_states.mu_a, output_states.jcb,
                          output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> MixtureTanh::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<MixtureTanhCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Softplus
////////////////////////////////////////////////////////////////////////////////
Softplus::Softplus() {};
Softplus::~Softplus() {};
std::string Softplus::get_layer_info() const
/*
 */
{
    return "Softplus()";
}

std::string Softplus::get_layer_name() const
/*
 */

{
    return "Softplus";
}

LayerType Softplus::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void Softplus::forward(BaseHiddenStates &input_states,
                       BaseHiddenStates &output_states,
                       BaseTempStates &temp_states)
/*
 */
{
    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    softplus_mean_var(input_states.mu_a, input_states.var_a, start_chunk,
                      end_chunk, output_states.mu_a, output_states.jcb,
                      output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> Softplus::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<SoftplusCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Leaky ReLU
////////////////////////////////////////////////////////////////////////////////
LeakyReLU::LeakyReLU() {};
LeakyReLU::~LeakyReLU() {};

std::string LeakyReLU::get_layer_info() const
/*
 */
{
    return "leakyReLU()";
}

std::string LeakyReLU::get_layer_name() const
/*
 */

{
    return "leakReLU";
}

LayerType LeakyReLU::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void LeakyReLU::forward(BaseHiddenStates &input_states,
                        BaseHiddenStates &output_states,
                        BaseTempStates &temp_states)
/*
 */
{
    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    leaky_relu_mean_var(input_states.mu_a, input_states.var_a, start_chunk,
                        end_chunk, this->alpha, output_states.mu_a,
                        output_states.jcb, output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> LeakyReLU::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<LeakyReLUCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Stable Softmax
////////////////////////////////////////////////////////////////////////////////
Softmax::Softmax() {}
Softmax::~Softmax() {}
std::string Softmax::get_layer_info() const
/*
 */
{
    return "Softmax()";
}

std::string Softmax::get_layer_name() const
/*
 */

{
    return "Softmax";
}

LayerType Softmax::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void Softmax::forward(BaseHiddenStates &input_states,
                      BaseHiddenStates &output_states,
                      BaseTempStates &temp_states)
/*
 */
{
    // TODO: replace this function by the multiprocessing one
    int batch_size = input_states.size / input_states.block_size;
    softmax_mean_var(input_states.mu_a, input_states.var_a,
                     input_states.block_size, batch_size, output_states.mu_a,
                     output_states.jcb, output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.size = input_states.size;
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> Softmax::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<SoftmaxCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Remax
////////////////////////////////////////////////////////////////////////////////
Remax::Remax() {}
Remax::~Remax() {}

std::string Remax::get_layer_info() const
/*
 */
{
    return "Remax()";
}

std::string Remax::get_layer_name() const
/*
 */

{
    return "Remax";
}

LayerType Remax::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void Remax::forward(BaseHiddenStates &input_states,
                      BaseHiddenStates &output_states,
                      BaseTempStates &temp_states)
/*
 */
{
    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    remax_mean_var(input_states.mu_a, input_states.var_a,
                    input_states.actual_size, input_states.block_size,
                    output_states.mu_a, output_states.jcb,
                    output_states.var_a);
    

    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.size = input_states.size;
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> Remax::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<RemaxCuda>();
}
#endif


////////////////////////////////////////////////////////////////////////////////
/// EvenExp
////////////////////////////////////////////////////////////////////////////////
EvenExp::EvenExp() {}
EvenExp::~EvenExp() {}

std::string EvenExp::get_layer_info() const
/*
 */
{
    return "EvenExp()";
}

std::string EvenExp::get_layer_name() const
/*
 */

{
    return "EvenExp";
}

LayerType EvenExp::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void EvenExp::forward(BaseHiddenStates &input_states,
                      BaseHiddenStates &output_states,
                      BaseTempStates &temp_states)
/*
 */
{
    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    if (this->num_threads > 1) {
        even_exp_mean_var_mp(input_states.mu_a, input_states.var_a,
                             input_states.jcb, end_chunk, this->num_threads,
                             output_states.mu_a, output_states.var_a,
                             output_states.jcb);
    } else {
        even_exp_mean_var(input_states.mu_a, input_states.var_a,
                          input_states.jcb, start_chunk, end_chunk,
                          output_states.mu_a, output_states.var_a,
                          output_states.jcb);
    }

    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.size = input_states.size;
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> EvenExp::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<EvenExpCuda>();
}
#endif
