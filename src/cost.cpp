#include "../include/cost.h"

#include <cmath>

#include "../include/common.h"
#include "../include/custom_logger.h"

namespace {

// Gate activation slope inside obs_to_class. Keep in sync with the alpha used
// there: P_z = normcdf(mz / sqrt((1/alpha)^2 + Sz)).
constexpr float kHrcGateAlpha = 3.0f;

// Inverse standard-normal CDF (probit) via Acklam's rational approximation.
// Accurate to ~1e-9 over (0, 1). We only call it during tree construction.
float inverse_normcdf(float p) {
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                               -2.759285104469687e+02, 1.383577518672690e+02,
                               -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                               -1.556989798598866e+02, 6.680131188771972e+01,
                               -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                               -2.400758277161838e+00, -2.549732539343734e+00,
                               4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                               2.445134137142996e+00, 3.754408661907416e+00};
    const double p_low = 0.02425;
    const double p_high = 1.0 - p_low;
    double q, r, x;

    if (p < p_low) {
        q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
             c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= p_high) {
        q = p - 0.5;
        r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r +
             a[5]) *
            q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r +
             1.0);
    } else {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
              c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    return static_cast<float>(x);
}

// Bias that makes p_left = L/N at zero gate logit and zero gate variance.
float gate_prior_bias(int left_size, int span) {
    if (span <= 1 || left_size == span - left_size) {
        return 0.0f;
    }
    const float ratio = static_cast<float>(left_size) / static_cast<float>(span);
    return inverse_normcdf(ratio) / kHrcGateAlpha;
}

void assign_hrc_paths(int start_class, int end_class, int node_idx,
                      int &next_node_idx, std::vector<std::vector<float>> &obs,
                      std::vector<std::vector<int>> &idx,
                      std::vector<float> &bias) {
    const int num_classes = end_class - start_class;
    if (num_classes <= 1) {
        return;
    }

    const int left_size = (num_classes + 1) / 2;
    const int split_class = start_class + left_size;

    bias[node_idx - 1] = gate_prior_bias(left_size, num_classes);

    for (int label = start_class; label < split_class; label++) {
        obs[label].push_back(1.0f);
        idx[label].push_back(node_idx);
    }
    for (int label = split_class; label < end_class; label++) {
        obs[label].push_back(-1.0f);
        idx[label].push_back(node_idx);
    }

    if (left_size > 1) {
        const int child_node_idx = next_node_idx;
        next_node_idx++;
        assign_hrc_paths(start_class, split_class, child_node_idx,
                         next_node_idx, obs, idx, bias);
    }
    const int right_size = end_class - split_class;
    if (right_size > 1) {
        const int child_node_idx = next_node_idx;
        next_node_idx++;
        assign_hrc_paths(split_class, end_class, child_node_idx, next_node_idx,
                         obs, idx, bias);
    }
}

}  // namespace

void fliplr(std::vector<int> &v)
/* Flip an array from left to right
 *
 * Args:
 *    v: An array
 *
 * Returns:
 *    v: Flipped array
 *    */
{
    for (int i = 0; i < v.size() / 2; i++) {
        int tmp = v[i];
        v[i] = v[v.size() - i - 1];
        v[v.size() - i - 1] = tmp;
    }
}

std::vector<int> dec_to_bi(int base, int num, int n_c)
/*
 * Convert decimal to binary.
 *
 * Args:
 *    base: base number e.g. binary base = 2
 *    num: Number to be converted
 *    n_c: Number of columns to store base
 *
 * Returns:
 *    res: Base number
 *    */
{
    // Initialize pointers
    std::vector<int> res(n_c, 0);
    int index = 0;
    while (num > 0) {
        res[index++] = num % base;
        num /= base;
    }

    // Flip the left to right the base number
    fliplr(res);

    return res;
}

int bi_to_dec(std::vector<int> &v, int base)
/*
 * Convert binary to decimal.
 *
 * Args:
 *    v: vector of binary number
 *    base: base number
 *
 * Returns:
 *   num: Decimal number
 *   */
{
    int num = 0;
    int mlp = 1;
    for (int i = v.size() - 1; i >= 0; i--) {
        if (v[i] >= base) {
            printf("Invalid number");
            return -1;
        }

        num += v[i] * mlp;
        mlp = mlp * base;
    }

    return num;
}

HRCSoftmax class_to_obs(int n_classes, bool use_prior_bias)
/*
 * Convert class to hierarchical softmax for classification task.
 *
 * The tree has exactly one leaf per class and n_classes - 1 internal decision
 * nodes. Class paths can have different lengths, so obs/idx are padded to the
 * maximum path length with idx = 0. Updaters skip padded entries.
 *
 * When use_prior_bias is true (default), each gate carries a fixed prior bias
 * chosen so that with zero gate logits the per-class prior probability equals
 * 1/n_classes. When false, the bias vector is left empty and the forward and
 * backward paths fall back to the previous behavior (no prior shift).
 *
 * Args:
 *    n_classes: Number of classes.
 *    use_prior_bias: Whether to populate the per-gate uniform-prior bias.
 *
 * Returns:
 *    obs: Observation matrix for each class i.e. -1 and 1
 *    idx: Indices for each observation
 *    path_len: Number of real observations for each class
 *    bias: Prior offset per gate (length n_classes - 1, or empty if disabled)
 *    n_obs: Maximum number of observations
 *    len: Number of internal decision nodes
 **/
{
    if (n_classes < 2) {
        return {{}, {}, {}, {}, 0, 0};
    }

    std::vector<std::vector<float>> obs_by_class(n_classes);
    std::vector<std::vector<int>> idx_by_class(n_classes);
    std::vector<float> bias(n_classes - 1, 0.0f);
    int next_node_idx = 2;
    assign_hrc_paths(0, n_classes, 1, next_node_idx, obs_by_class,
                     idx_by_class, bias);
    if (!use_prior_bias) {
        bias.clear();
    }

    int max_path_len = 0;
    std::vector<int> path_len(n_classes, 0);
    for (int label = 0; label < n_classes; label++) {
        path_len[label] = idx_by_class[label].size();
        max_path_len = std::max(max_path_len, path_len[label]);
    }

    std::vector<float> obs(n_classes * max_path_len, 0.0f);
    std::vector<int> idx(n_classes * max_path_len, 0);
    for (int label = 0; label < n_classes; label++) {
        for (int col = 0; col < path_len[label]; col++) {
            obs[label * max_path_len + col] = obs_by_class[label][col];
            idx[label * max_path_len + col] = idx_by_class[label][col];
        }
    }

    return {obs, idx, path_len, bias, max_path_len, n_classes - 1};
}

//////////////////////////////////////////////////////////////////////////////
// CONVERT OBSERVATION TO CLASS
/////////////////////////////////////////////////////////////////////////////
// float normalCDF(float x)
// /* Normal cumulative distribution */
// {
//     return std::erfc(-x / std::sqrt(2)) / 2;
// }

std::vector<float> obs_to_class(std::vector<float> &mz, std::vector<float> &Sz,
                                HRCSoftmax &hs, int n_classes)
/*
 * Convert observation to classes.
 *
 * Args:
 *    mz: Mean of hidden states of the ouput layer
 *    Sz: Variance of hidden states of the output layer
 *    hs: Hierarchical softmax output
 *    n_classes: Number of classes
 *
 * Returns:
 *    P: Probability of the class
 **/
{
    // Initialization
    std::vector<float> P(n_classes);
    std::vector<float> P_z(hs.len);
    float alpha = kHrcGateAlpha;
    const bool has_bias = static_cast<int>(hs.bias.size()) == hs.len;

    // Compute probability for each observation
    for (int i = 0; i < hs.len; i++) {
        const float bias_i = has_bias ? hs.bias[i] : 0.0f;
        P_z[i] = normcdf_cpu((mz[i] + bias_i) /
                             pow(pow(1 / alpha, 2) + Sz[i], 0.5));
    }

    // Compute probability for the class
    for (int r = 0; r < n_classes; r++) {
        float tmp = 1.0f;
        const int path_len =
            hs.path_len.empty() ? hs.n_obs : hs.path_len[r];
        for (int c = 0; c < path_len; c++) {
            const int flat_idx = r * hs.n_obs + c;
            const int node_idx = hs.idx[flat_idx];
            if (node_idx <= 0) {
                continue;
            }
            if (hs.obs[flat_idx] == -1.0f) {
                tmp *= std::abs(P_z[node_idx - 1] - 1.0f);
            } else {
                tmp *= P_z[node_idx - 1];
            }
        }
        P[r] = tmp;
    }

    return P;
}

////////////////////////////////////////////////////////////////////////////
// ERROR RATE
////////////////////////////////////////////////////////////////////////////
std::tuple<std::vector<int>, std::vector<float>> get_error(
    std::vector<float> &mz, std::vector<float> &Sz, std::vector<int> &labels,
    int n_classes, int B)
/*
 * Compute error given an input image
 *
 * Args:
 *    mz: Mean of hidden states of the output layer
 *    Sz: Variance of hidden states of the output layer
 *    labels: Real label
 *    hs: Hierarchical softmax output
 *    n_classes: Number of classes
 *
 * Returns:
 *    er: error 1: wrong prediciton and 0: right one
 *    P: Probability for each class
 * */
{
    // Initialization
    auto hs = class_to_obs(n_classes);
    std::vector<int> er(B, 0);
    std::vector<float> P(B * n_classes);
    std::vector<float> mz_tmp(hs.len);
    std::vector<float> Sz_tmp(hs.len);

    // Compute probability for each class
    for (int r = 0; r < B; r++) {
        // Get sample
        for (int i = 0; i < hs.len; i++) {
            mz_tmp[i] = mz[r * hs.len + i];
            Sz_tmp[i] = Sz[r * hs.len + i];
        }

        // Compute probability
        std::vector<float> tmp(n_classes, 0);
        tmp = obs_to_class(mz_tmp, Sz_tmp, hs, n_classes);

        // Store in P matrix
        for (int c = 0; c < n_classes; c++) {
            P[r * n_classes + c] = tmp[c];
        }

        // Prediction
        int pred = std::distance(tmp.begin(),
                                 std::max_element(tmp.begin(), tmp.end()));

        // Get error
        if (pred != labels[r]) {
            er[r] = 1;
        }
    }

    return {er, P};
}

std::tuple<std::vector<int>, std::vector<float>, std::vector<int>> get_error_v2(
    std::vector<float> &mz, std::vector<float> &Sz, std::vector<int> &labels,
    int n_classes, int B)
/*
 * Compute error given an input image
 *
 * Args:
 *    mz: Mean of hidden states of the output layer
 *    Sz: Variance of hidden states of the output layer
 *    labels: Real label
 *    hs: Hierarchical softmax output
 *    n_classes: Number of classes
 *
 * Returns:
 *    er: error 1: wrong prediciton and 0: right one
 *    P: Probability for each class
 * */
{
    // Initialization
    auto hs = class_to_obs(n_classes);
    std::vector<int> er(B, 0);
    std::vector<int> preds(B);
    std::vector<float> P(B * n_classes);
    std::vector<float> mz_tmp(hs.len);
    std::vector<float> Sz_tmp(hs.len);

    // Compute probability for each class
    for (int r = 0; r < B; r++) {
        // Get sample
        for (int i = 0; i < hs.len; i++) {
            mz_tmp[i] = mz[r * hs.len + i];
            Sz_tmp[i] = Sz[r * hs.len + i];
        }

        // Compute probability
        auto tmp = obs_to_class(mz_tmp, Sz_tmp, hs, n_classes);

        // Store in P matrix
        for (int c = 0; c < n_classes; c++) {
            P[r * n_classes + c] = tmp[c];
        }

        // Prediction
        preds[r] = std::distance(tmp.begin(),
                                 std::max_element(tmp.begin(), tmp.end()));
        // Get error
        if (preds[r] != labels[r]) {
            er[r] = 1;
        }
    }

    return {er, P, preds};
}

std::vector<int> get_class_error(std::vector<float> &ma,
                                 std::vector<int> &labels, int n_classes,
                                 int B) {
    std::vector<int> er(B, 0);
    int idx;
    for (int i = 0; i < B; i++) {
        idx = i * n_classes;
        int pred =
            std::max_element(ma.begin() + idx, ma.begin() + idx + n_classes) -
            ma.begin() - idx;

        if (pred != labels[i]) {
            er[i] = 1;
        }
    }
    return er;
}

float mean_squared_error(std::vector<float> &pred, std::vector<float> &obs)
/* Compute mean squared error.
Args:
    pred: Prediction
    obs: Observation

Returns:
    mse: Mean squared error
*/
{
    if (pred.size() != obs.size()) {
        LOG(LogLevel::ERROR,
            "Prediciton and observation does not have the same.");
    }
    float sum = 0;
    for (int i = 0; i < pred.size(); i++) {
        sum += pow((obs[i] - pred[i]), 2);
    }

    return sum / obs.size();
}

float avg_univar_log_lik(std::vector<float> &x, std::vector<float> &mu,
                         std::vector<float> &sigma)
/* Compute the average of univariate log-likelihood.

Args:
    x: Prediction
    mu: Observation's mean
    sigma: Observation's standard deviation

Returns:
    avg_log_lik: Averaged log-likelihood

*NOTE: We assume that pred ~ Normal(mu, sigma).
*/
{
    if (x.size() == 0 || mu.size() == 0 || sigma.size() == 0) {
        LOG(LogLevel::ERROR, "Invalid inputs for normal density");
    }
    float sum = 0;
    float PI_C = 3.141592653f;

    for (int i = 0; i < x.size(); i++) {
        sum += -0.5 * log(2 * PI_C * pow(sigma[i], 2)) -
               0.5 * pow((x[i] - mu[i]) / sigma[i], 2);
    }

    return sum / x.size();
}

float compute_average_error_rate(std::vector<int> &error_rate, int curr_idx,
                                 int n_past_data)
/*Compute running error rate.

  Args:
    error_rate: Vector of error rate
    curr_idx: Index of the current error rate
    n_past_data: Number of past data from the current index
*/
{
    int end_idx = curr_idx - n_past_data;
    if (end_idx < 0) {
        end_idx = 0;
        n_past_data = curr_idx;
    }

    float tmp = 0;
    for (int i = 0; i < n_past_data; i++) {
        tmp += error_rate[end_idx + i];
    }

    float avg_error = tmp / n_past_data;

    return avg_error;
}

/////////////////////////////////
// TEST UNI
////////////////////////////////
void test_class_to_obs() {
    int n_classes = 10;
    HRCSoftmax hs = class_to_obs(n_classes);

    std::cout << "Observation = "
              << "\n";
    print_matrix(hs.obs, hs.n_obs, n_classes);
    std::cout << "Index = "
              << "\n";
    print_matrix(hs.idx, hs.n_obs, n_classes);
}

void test_obs_to_class() {
    // Get obs
    int n_classes = 10, B = 2;
    std::vector<int> labels = {2, 3};
    HRCSoftmax hs = class_to_obs(n_classes);

    // Get prob
    std::vector<float> mz = {1, 1, 0, -1, 0, 0, 0, 1,  0, 0, 0,
                             1, 1, 0, -1, 0, 0, 0, -1, 0, 0, 0};

    std::vector<float> Sz(hs.len * 2, 0.02f);
    std::vector<int> er;
    std::vector<float> P;
    std::tie(er, P) = get_error(mz, Sz, labels, n_classes, B);

    std::cout << "Prob = "
              << "\n";
    print_matrix(P, n_classes, B);

    std::cout << "Error"
              << "\n";
    for (int j = 0; j < B; j++) {
        std::cout << er[j] << "\n";
    }
}
