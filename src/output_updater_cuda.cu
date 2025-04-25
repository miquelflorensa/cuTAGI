#include <curand_kernel.h>

#include "../include/activation_cuda.cuh"
#include "../include/output_updater_cuda.cuh"

__global__ void update_delta_z_using_indices_cuda(
    float const *mu_a, float const *var_a, float const *jcb, float const *obs,
    float const *var_obs, int const *selected_idx, int n_obs, int n_enc,
    int size, float *delta_mu, float *delta_var)
/* Update output layer based on selected indices.
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zero_pad = 0.0f;
    float tmp = 0.0f;
    int idx;
    if (col < size) {
        // minus 1 because the encoder index starts at 1
        idx = selected_idx[col] + (col / n_enc) * n_obs - 1;
        tmp = jcb[idx] / (var_a[idx] + var_obs[col]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mu[idx] = zero_pad;
            delta_var[idx] = zero_pad;
        } else {
            delta_mu[idx] = tmp * (obs[col] - mu_a[idx]);
            delta_var[idx] = -tmp * jcb[idx];
        }
    }
}

__global__ void update_delta_z_cuda(float const *mu_a, float const *var_a,
                                    float const *jcb, float const *obs,
                                    float const *var_obs, int size,
                                    float *delta_mu, float *delta_var) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zero_pad = 0;
    float tmp = 0;
    if (col < size) {
        tmp = jcb[col] / (var_a[col] + var_obs[col]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mu[col] = zero_pad;
            delta_var[col] = zero_pad;
        } else {
            delta_mu[col] = tmp * (obs[col] - mu_a[col]);
            delta_var[col] = -tmp * jcb[col];
        }
    }
}

__global__ void update_delta_z_cuda_heteros(float const *mu_a,
    float const *var_a,
    float const *jcb, float const *obs,
    int size, float *delta_mu,
    float *delta_var) {
    /*
    Compute delta hidden states for output layer with learned heteroscedastic
    noise. This function receives a vector of observations and the twice
    output hidden states. Using AGVI, we can infere the posterior for
    observation noise v and use it to update the hidden states Z_out.

    Terminology:
    - V: Gaussian random variable describing the error variance sigma^2. N(0,
    sqrt(V))
    - V2: Square of the error (V^2)
    - V2_bar: Gaussian random variable describing the expected value of V2
    (mu_V2)
    - V2_bar_tilde: Gaussian random variable describing V2 after passing through
    an exponential activation function to restrict values to the positive domain

    */
    const float zero_pad = 0.0f;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Output layer will have twice the size of the common one because one is
    // representing the mean and the other the variance
    int obs_col = col * 2;

    if (col < size) {
        // mean of the Gaussian distribution for the output
        float var_a_col = var_a[obs_col];
        float mu_a_col = mu_a[obs_col];
        float jcb_col = jcb[obs_col];

        // V2_bar_tilde
        float mu_v2_bar_tilde = mu_a[obs_col + 1];
        float var_v2_bar_tilde = var_a[obs_col + 1];
        float cov_v2_bar_tilde = jcb[obs_col + 1];

        // Compute the prior predictive PDF for v2
        float mu_v2 = mu_v2_bar_tilde;
        float var_v2 =
        3.0f * var_v2_bar_tilde + 2.0f * mu_v2_bar_tilde * mu_v2_bar_tilde;
        float cov_y_v = mu_v2;

        // Variance of the output
        float var_sum = var_a_col + mu_v2;

        // Compute updating quantities for the mean of the output
        float tmp1 = jcb_col / var_sum;
        float tmp2 = jcb_col / (var_a_col + 0.01f);

        if (std::isinf(tmp1) || std::isnan(tmp1) || std::isinf(tmp2) || std::isnan(tmp2)) {
            delta_mu[obs_col] = zero_pad;
            delta_var[obs_col] = zero_pad;
        } else {
            float obs_diff = obs[col] - mu_a_col;
            delta_mu[obs_col] = tmp2 * obs_diff;
            delta_var[obs_col] = -tmp1 * jcb_col;
        }

        // Compute the posterior mean and variance for V
        float mu_v_post = cov_y_v / var_sum * (obs[col] - mu_a_col);
        float var_v_post = mu_v2 - cov_y_v / var_sum * cov_y_v;

        // Compute the posterior mean and variance for V2
        float mu_v2_post = mu_v_post * mu_v_post + var_v_post;
        float var_v2_post = 2.0f * var_v_post * var_v_post +
        4.0f * var_v_post * mu_v_post * mu_v_post;

        // Compute the posterior mean and variance for V2_bar_tilde
        float tmp_ratio = var_v2_bar_tilde / var_v2;
        float mu_v2_bar_tilde_post =
        mu_v2_bar_tilde + tmp_ratio * (mu_v2_post - mu_v2);
        float var_v2_bar_tilde_post =
        var_v2_bar_tilde + tmp_ratio * tmp_ratio * (var_v2_post - var_v2);

        // Compute update for V2_bar
        float jv = cov_v2_bar_tilde / var_v2_bar_tilde;
        delta_mu[obs_col + 1] = jv * (mu_v2_bar_tilde_post - mu_v2_bar_tilde);
        delta_var[obs_col + 1] =
        jv * jv * (var_v2_bar_tilde_post - var_v2_bar_tilde);
    }
}


__global__ void update_delta_z_cuda_heteros_softmax(float const *mu_a,
    float const *var_a,
    float const *jcb, float const *obs,
    int size, float *delta_mu,
    float *delta_var, float *mu_zv, float *var_zv, float *var_z) {
    /*
    Compute delta hidden states for output layer with learned heteroscedastic
    noise. This function receives a vector of observations and the twice
    output hidden states. Using AGVI, we can infere the posterior for
    observation noise v and use it to update the hidden states Z_out.

    Terminology:
    - V: Gaussian random variable describing the error variance sigma^2. N(0,
    sqrt(V))
    - V2: Square of the error (V^2)
    - V2_bar: Gaussian random variable describing the expected value of V2
    (mu_V2)
    - V2_bar_tilde: Gaussian random variable describing V2 after passing through
    an exponential activation function to restrict values to the positive domain
    - ZV: Z_out + V; N(mu_z, var_v)
    */
    const float zero_pad = 0.0f;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Output layer will have twice the size of the common one because one is
    // representing the mean and the other the variance. A contains twice the
    // number of observations, the even positions corresponds to Z_out and the
    // odd positions corresponds to V2_bar_tilde
    int obs_col = col * 2;

    if (col < size) {
        // Z_out
        float var_a_col = var_a[obs_col];
        float mu_a_col = mu_a[obs_col];
        float jcb_col = jcb[obs_col];

        // V2_bar_tilde
        float mu_v2_bar_tilde = mu_a[obs_col + 1];
        float var_v2_bar_tilde = var_a[obs_col + 1];
        float cov_v2_bar_tilde = jcb[obs_col + 1];

        // Compute the prior predictive PDF for v2
        float mu_v2 = mu_v2_bar_tilde;
        float var_v2 =
        3.0f * var_v2_bar_tilde + 2.0f * mu_v2_bar_tilde * mu_v2_bar_tilde;
        float cov_y_v = mu_v2;

        float mu_v = 0.0f;
        float var_v = mu_v2;

        // Variance of the output
        float var_sum = var_a_col;

        // Compute updating quantities for the mean of the output
        float tmp1 = jcb_col / (var_a_col + 0.0001f);
        float tmp2 = jcb_col / (var_a_col + 0.0001f);


        float mu_zv_post = mu_zv[obs_col];
        float var_zv_post = var_zv[obs_col];

        if (std::isinf(tmp1) || std::isnan(tmp1) || std::isinf(tmp2) || std::isnan(tmp2)) {
            delta_mu[obs_col] = zero_pad;
            delta_var[obs_col] = zero_pad;
        } else {
            float obs_diff = obs[col] - mu_a_col;
            // Compute the posterior mean and variance for ZV
            mu_zv_post += tmp2 * obs_diff;
            var_zv_post -= tmp1 * jcb_col;
        }

        // Compute deltas for Z
        float Jz_mu = 1.0f / (0.001f * var_zv[obs_col]);
        float Jz = 1.0f / (var_zv[obs_col]);
        delta_mu[obs_col] = Jz_mu * (mu_zv_post - mu_zv[obs_col]);
        delta_var[obs_col] = Jz * Jz * (var_zv_post - var_zv[obs_col]);

        // Smooth back V given ZV
        float Jv = var_v / (var_zv[obs_col]);
        float Jv_mu = var_v / 0.01f;
        float mu_v_post = mu_v + Jv * (mu_zv_post - mu_zv[obs_col]);
        float var_v_post = var_v + Jv * Jv * (var_zv_post - var_zv[obs_col]);

        // Compute the posterior mean and variance for V2
        float mu_v2_post = mu_v_post * mu_v_post + var_v_post;
        float var_v2_post = 2.0f * var_v_post * var_v_post +
        4.0f * var_v_post * mu_v_post * mu_v_post;

        // Compute the posterior mean and variance for V2_bar_tilde
        float tmp_ratio = var_v2_bar_tilde / var_v2;
        float mu_v2_bar_tilde_post =
        mu_v2_bar_tilde + tmp_ratio * (mu_v2_post - mu_v2);
        float var_v2_bar_tilde_post =
        var_v2_bar_tilde + tmp_ratio * tmp_ratio * (var_v2_post - var_v2);

        // Compute update for V2_bar
        float jv2 = cov_v2_bar_tilde / var_v2_bar_tilde;
        delta_mu[obs_col + 1] = jv2 * (mu_v2_bar_tilde_post - mu_v2_bar_tilde);
        delta_var[obs_col + 1] =
        jv2 * jv2 * (var_v2_bar_tilde_post - var_v2_bar_tilde);
    }
}

__global__ void update_delta_z_cuda_remax(float const *mu_a, float const *var_a,
                                          float const *jcb, float const *obs,
                                          float const *var_obs, int size,
                                          float *delta_mu, float *delta_var,
                                          float const *d_var_a_copy) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zero_pad = 0;
    float tmp = 0;
    float jcb_tmp = jcb[col];
    if (col < size) {
        tmp = jcb[col] / var_a[col];

        if (isinf(tmp) || isnan(tmp)) {
            delta_mu[col] = zero_pad;
            delta_var[col] = zero_pad;
        } else {
            delta_mu[col] = tmp * (obs[col] - mu_a[col]);
            delta_var[col] = -tmp * jcb_tmp;
        }
    }
}

__global__ void update_delta_z_cuda_fullcov(
    float const *mu_a, float const *var_a, float const *jcb, float const *obs,
    float const *var_obs, int size, float *delta_mu, float *delta_var,
    float const *mu_zv, float const *var_zv, float const *jcb_zv, float const *mu_z,
    float const *var_z, float *mu_e, float *var_e, float *full_jcb, int no) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int i = idx / no;
    int j = idx % no;

    float mZV_post = mu_zv[idx];
    float s2ZV_post = var_zv[idx];

    float tmp = jcb[idx] / (var_a[idx] + var_obs[idx]);
    // float tmp = jcb[idx] / var_a[idx];

    if (!isnan(tmp) && !isinf(tmp)) {
        // delta_mu[idx] = tmp * (obs[idx] - mu_a[idx]);
        // delta_var[idx] = - tmp * jcb[idx];

        mZV_post += tmp * (obs[idx] - mu_a[idx]);
        s2ZV_post -= tmp * jcb[idx];

        // // Update Z from ZV
        float Jz = jcb_zv[idx] / var_zv[idx];
        delta_mu[idx] = Jz * (mZV_post - mu_zv[idx]);
        delta_var[idx] = Jz * Jz * (s2ZV_post - var_zv[idx]);
    } else {
        delta_mu[idx] = 0.0f;
        delta_var[idx] = 0.0f;
    }

    // float delta_mu_tmp = 0.0f;
    // float delta_var_tmp = 0.0f;

    // float J = 0.0f;
    // float sV = 0.01f;
    // float sV2 = sV * sV;

    // float y = fmaxf(fminf(obs[idx], 0.991f), 0.0001f);
    // y = logf(y);

    // // printf("jcb[%d]: %f\n", idx, jcb[idx]);
    // // printf("var_a[%d]: %f\n", idx, var_a[idx]);
    // // printf("mu_a[%d]: %f\n", idx, mu_a[idx]);
    // printf("y: %f\n", y);
    // // printf("mu_z[%d]: %f\n", idx, mu_z[idx]);
    // // printf("var_z[%d]: %f\n", idx, var_z[idx]);
    // J = fmin(1.0f, jcb[idx] / var_a[idx]);
    // if (!isnan(J) && !isinf(J)) {
    //     delta_mu_tmp = J * (y - mu_a[idx]);
    //     delta_var_tmp = -J * jcb[idx];
    // }
    // delta_mu[idx] = delta_mu_tmp;
    // delta_var[idx] = delta_var_tmp;

    // float mZ_post = mu_z[idx] + J * (y - mu_a[idx]);
    // float s2Z_post = var_z[idx] - J * jcb[idx];
    // printf("mZ_post: %f\n", mZ_post);
    // printf("s2Z_post: %f\n", s2Z_post);

    // // float mZpost = mu_zv[idx] + J * (mu_e[idx] - mu_z[idx]);
    // // float s2Zpost = var_zv[idx] + J * J * (var_e[idx] - var_z[idx]);

    // // // Update Z from ZV
    // // float Jz = 1.0f / (var_zv[idx]);
    // // delta_mu[idx] = Jz * delta_mu_tmp;
    // // delta_var[idx] = Jz * Jz * delta_var_tmp;
}

__global__ void update_delta_z_cuda_fullcov_4step(
    float const *mu_a, float const *var_a, float const *cov_e_a,
    float const *obs, float const *var_obs, int size, float *delta_mu,
    float *delta_var, float const *mu_zv, float const *var_zv,
    float const *mu_z, float const *var_z, float *mu_e, float *var_e,
    float *full_jcb, int no) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= size) return;

    int i = col / no;
    int j = col % no;

    float mA_post = 0.0f;
    float s2A_post = 0.0f;

    float tmp = 0.0f;
    float sV = 0.05f;
    float sV2 = sV * sV;

    // Step 1: Update A from Y
    tmp = var_a[col] / (var_a[col] + sV2);

    if (!isnan(tmp) && !isinf(tmp)) {
        mA_post = mu_a[col] + tmp * (obs[col] - mu_a[col]);
        s2A_post = var_a[col] - tmp * var_a[col];
    }

    // Step 2: Update exp(ZV) = E from A
    // float J = full_jcb[i * no * no + j * no + j] / var_a[col];
    float J = cov_e_a[col] / var_a[col];
    float mE_post =
        mu_e[col] + J * (mA_post - mu_a[col]);  // mE + J * delta_mu_tmp;
    float s2E_post = var_e[col] + J * J * (s2A_post - var_a[col]);
    // float mE_post = 0.0f;
    // float s2E_post = 0.0f;
    // for (int k = 0; k < no; k++) {
    //     int idx_k = i * no + k;
    //     float J = full_jcb[i * no * no + j * no + k] / var_a[idx_k];
    //     mE_post += J * (mA_post - mu_a[idx_k]);
    //     s2E_post += J * J * (s2A_post - var_a[idx_k]);
    // }

    // Step 3: Transform E to ZV
    // float s2ZV_post = fmaxf(logf(1.0f + s2E_post / (mE_post * mE_post)),
    // 1e-6f); float mZV_post = fmaxf(logf(mE_post), -6.0f) - 0.5f * s2ZV_post;

    float cov_zv_e = mu_e[col] * var_zv[col];
    float Je = cov_zv_e / var_e[col];
    float mZV_post = mu_zv[col] + Je * (mE_post - mu_e[col]);
    float s2ZV_post = var_zv[col] + Je * Je * (s2E_post - var_e[col]);

    // Step 4: Update Z from ZV
    float Jz = 1.0f / (var_zv[col]);
    delta_mu[col] = Jz * (mZV_post - mu_zv[col]);
    delta_var[col] = Jz * Jz * (s2ZV_post - var_zv[col]);
}

OutputUpdaterCuda::OutputUpdaterCuda() {}

void OutputUpdaterCuda::set_num_cuda_threads(unsigned int num_threads) {
    this->num_cuda_threads = num_threads;
}

void OutputUpdaterCuda::update_output_delta_z(BaseHiddenStates &output_states,
                                              BaseObservation &obs,
                                              BaseDeltaStates &delta_states)
/*
 */
{
    // Cast to cuda object
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    ObservationCuda *cu_obs = dynamic_cast<ObservationCuda *>(&obs);
    DeltaStateCuda *cu_delta_states =
        dynamic_cast<DeltaStateCuda *>(&delta_states);

    if (cu_obs->d_mu_obs == nullptr) {
        cu_obs->allocate_memory();
    }

    cu_obs->to_device();

    // Reset delta to zero
    cu_delta_states->reset_zeros();

    // Kernel
    int num_states = cu_obs->size;
    int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    update_delta_z_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb, cu_obs->d_mu_obs, cu_obs->d_var_obs,
        num_states, cu_delta_states->d_delta_mu, cu_delta_states->d_delta_var);
}

void OutputUpdaterCuda::update_selected_output_delta_z(
    BaseHiddenStates &output_states, BaseObservation &obs,
    BaseDeltaStates &delta_states)
/*
 */
{
    // Cast to cuda object
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    ObservationCuda *cu_obs = dynamic_cast<ObservationCuda *>(&obs);
    DeltaStateCuda *cu_delta_states =
        dynamic_cast<DeltaStateCuda *>(&delta_states);

    if (cu_obs->d_mu_obs == nullptr) {
        cu_obs->allocate_memory();
    }

    cu_obs->to_device();

    // Reset delta to zero
    cu_delta_states->reset_zeros();

    // Kernel
    int num_states = cu_obs->idx_size;
    int num_enc = cu_obs->idx_size / cu_obs->block_size;
    int num_outputs = cu_output_states->actual_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    update_delta_z_using_indices_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb, cu_obs->d_mu_obs, cu_obs->d_var_obs,
        cu_obs->d_selected_idx, num_outputs, num_enc, num_states,
        cu_delta_states->d_delta_mu, cu_delta_states->d_delta_var);
}

void OutputUpdaterCuda::update_output_delta_z_heteros(
    BaseHiddenStates &output_states, BaseObservation &obs,
    BaseDeltaStates &delta_states)
/*
 */
{
    // Cast to cuda object
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    ObservationCuda *cu_obs = dynamic_cast<ObservationCuda *>(&obs);
    DeltaStateCuda *cu_delta_states =
        dynamic_cast<DeltaStateCuda *>(&delta_states);

    if (cu_obs->d_mu_obs == nullptr) {
        cu_obs->allocate_memory();
    }

    cu_obs->to_device();

    // Reset delta to zero
    cu_delta_states->reset_zeros();

    // Kernel
    int num_states = cu_obs->size;
    int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    update_delta_z_cuda_heteros<<<blocks, this->num_cuda_threads>>>(
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb, cu_obs->d_mu_obs, num_states,
        cu_delta_states->d_delta_mu, cu_delta_states->d_delta_var);
}

__global__ void exp_cuda(float *mu_a, float *var_a, float *jcb, int size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < size) {
        float muZ = mu_a[col];
        float varZ = var_a[col];
        float tmp = expf(muZ + 0.5f * varZ);
        mu_a[col] = tmp;
        var_a[col] = expf(2.0f * muZ + varZ) * (expf(varZ) - 1.0f);
        jcb[col] = tmp;
        // if (std::isnan(mu_a[col]) || std::isinf(mu_a[col]) || std::isnan(var_a[col]) || std::isinf(var_a[col])) {
        //     mu_a[col] = 1e-12f;
        //     var_a[col] = 1e-6f;
        // }

        // Exp activation function for TAGI-V
        // if (col % 2 == 0) {
        // jcb[col] = tmp * varZ;
        // }
        // else {
        //     jcb[col] = tmp;
        // }

    }
}

__device__ float normpdf_cuda_2(float x) {
    constexpr float INV_SQRT_2PI = 0.3989422804014327f;
    return INV_SQRT_2PI * expf(-0.5f * x * x);
}

__device__ float normcdf_cuda_2(float x)
/*
Normal cumulative distribution function
 */
{
    return 0.5f * erfcf(-x * 0.7071067811865475f);
}

// Assumes your normcdf_cuda(x) is already defined elsewhere

__global__ void mixture_celu_mean_var_cuda(const float *mu_z, const float *s2_z,
                                           int num_states, float *mu_a,
                                           float *var_a, float *cov_za)
{
    float scale = 1.0f;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr float EPSILON = 1e-6f;

    if (col < num_states) {
        if (col % 2 != 0) {
            return;
        }
        float mz = mu_z[col];
        float var_z = s2_z[col];
        float sz = powf(var_z, 0.5f);

        // CLOSED FORM
        // float z = mz / sz;
        // float z_alpha = z + sz / alpha;
        // float z_2alpha = z + 2.0f * sz / alpha;

        // float phi_z = normpdf_cuda_2(z);
        // float phi_z_alpha = normpdf_cuda_2(z_alpha);
        // float phi_z_2alpha = normpdf_cuda_2(z_2alpha);

        // float Phi_z = normcdf_cuda_2(z);
        // float Phi_z_alpha = normcdf_cuda_2(z_alpha);
        // float Phi_z_2alpha = normcdf_cuda_2(z_2alpha);

        // // Mean (ma_LA)
        // float mean = mz
        // + sz * phi_z
        // - 0.5f * (alpha + mz) * (2.0f - 2.0f * Phi_z)
        // + (alpha * phi_z * (2.0f - 2.0f * Phi_z_alpha)) / (2.0f * phi_z_alpha);

        // // Variance (Sa_LA)
        // float var = mz * mz
        // + sz * phi_z * mz
        // + var_z
        // + 0.5f * (
        // -2.0f * phi_z * (2.0f - 2.0f * Phi_z_alpha) * alpha * alpha / phi_z_alpha
        // + phi_z * (2.0f - 2.0f * Phi_z_2alpha) * alpha * alpha / phi_z_2alpha
        // + (alpha * alpha - mz * mz - var_z) * (2.0f - 2.0f * Phi_z)
        // )
        // - mean * mean;

        // // Covariance (covza_LA)
        // float cov = var_z * (
        // Phi_z + (phi_z * (1.0f - Phi_z_alpha)) / phi_z_alpha
        // );

        // printf("mu_celu[%d]: %f\n", col, mean + alpha);
        // printf("var_celu[%d]: %f\n", col, var);
        // printf("cov_celu[%d]: %f\n", col, cov);

        // mu_a[col] = mean + alpha;
        // var_a[col] = var;
        // cov_za[col] = cov / var_z;


        // Local Linearization
        float delta = 1.0f / scale;
        float z = mz + delta;
        z *= scale;

        float a = 0.0f;
        float diff = 0.0f;

        if (z >= 1.0f) {
            a = z;
            diff = 1.0f;
        }
        else {
            a = expf(z - 1.0f);
            diff = expf(z - 1.0f);
        }

        a /= scale;
        diff /= scale;

        // printf("mu_celu[%d]: %f\n", col, a);
        // printf("var_celu[%d]: %f\n", col, var_z * diff * diff);
        mu_a[col] = a;
        var_a[col] = diff * diff * var_z;
        cov_za[col] = diff * var_z;
    }
}

// EXP * C function for ConsMax Function
// Could work in a "homoscedastic" setup (one constant per ConsMax), seems to
// not work in a "heteroscedastic setp" (depending on the input)
__global__ void expc_cuda(float *mu_a, float *var_a, float *jcb, int size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < size && col % 12 == 0) {
        float muC1 = mu_a[col + 10];
        float varC1 = var_a[col + 10];

        float muC2 = mu_a[col + 11];
        float varC2 = var_a[col + 11];

        for (int i = 0; i < 10; i++) {
            float muE = mu_a[col + i];
            float varE = var_a[col + i];

            float muC1C2 = muC1 * muC2;
            float varC1C2 =
                varC1 * varC2 + muC1 * muC1 * varC2 + muC2 * muC2 * varC1;

            mu_a[col + i] = muE * muC1C2;

            var_a[col + i] =
                varE * varC1C2 + muE * muE * varC1C2 + muC1C2 * muC1C2 * varE;
            jcb[col + i] = varE * muC1C2;
        }
    }
}

// Plain analitical softmax
// __global__ void compute_softmax_outputs(float *mu_z, float *var_z,
//                                         float *d_mu_a, float *d_var_a,
//                                         float *d_jcb, float
//                                         *sum_mu_global, float
//                                         *sum_var_global, float *mu_e,
//                                         float *var_e, int no, int B) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < B * no) {
//         int i = idx / no;
//         int j = idx % no;

//         float muZ = mu_z[idx];
//         float varZ = var_z[idx];  // Clamp input variance

//         float muE = mu_e[idx];
//         float varE = var_e[idx];

//         float sum_muE = sum_mu_global[i];
//         float sum_varE = sum_var_global[i];

//         float varlnE_sum = logf(1.0f + sum_varE / (sum_muE * sum_muE));
//         float mulnE_sum = logf(sum_muE) - 0.5f * varlnE_sum;

//         float covlnE_sumlnE = logf(1.0f + varE / muE / sum_muE);

//         float mulnE = muZ - mulnE_sum;
//         float varlnE = varZ + varlnE_sum - 2.0f * covlnE_sumlnE;

//         float cov_cElnM = varZ - covlnE_sumlnE;

//         float muA = expf(mulnE + 0.5f * varlnE);
//         float varA = muA * muA * (expf(varlnE) - 1.0f);

//         float covZA = muA * cov_cElnM;

//         d_mu_a[idx] = muA;
//         d_var_a[idx] = varA;
//         d_jcb[idx] = muA * cov_cElnM;

//         // d_mu_a[idx] = mulnE;
//         // d_var_a[idx] = varlnE;
//         // d_jcb[idx] = cov_cElnM;
//     }
// }

// Local Linearized softmax
__global__ void compute_softmax_outputs(float *mu_z, float *var_z,
                                        float *mu_zv, float *var_zv,
                                        float *d_mu_a, float *d_var_a,
                                        float *d_jcb, float *sum_mu_global,
                                        float *sum_var_global, float *mu_e,
                                        float *var_e, float *cov_e,
                                        int no, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * no) {
        if (idx % 2 == 0) {
            int i = idx / no;
            int j = idx % no;

            float varZ = var_zv[idx];
            float muZ = mu_zv[idx];

            float muE = mu_e[idx];
            float varE = var_e[idx];

            float sum_muE = sum_mu_global[i];
            float sum_varE = sum_var_global[i];

            // E_sum^{-1} = 1/E_sum
            float mEsum_inv = 1.0f / sum_muE;
            float s2Esum_inv = 1.0f / (sum_muE * sum_muE * sum_muE * sum_muE) *
                            sum_varE;
            float covE_sumE_sum_inv =
                -1.0f / (sum_muE * sum_muE) * sum_varE;
            // float covEE_sum_inv = -1.0f / (sum_muE * sum_muE) * sum_varE;
            float covEE_sum_inv = 0.0f;  // Test -> not the right theoretical solution

            // float covZE_sum_inv = -1.0f / (sum_muE * sum_muE) * cov_e[idx];
            float covZE_sum_inv = 0.0f;  // Test -> not the right theoretical solution

            // A_i = normal
            float mA = muE * mEsum_inv + covEE_sum_inv;
            float s2A = varE * s2Esum_inv + covEE_sum_inv * covEE_sum_inv +
                        2.0f * covEE_sum_inv * muE * mEsum_inv +
                        s2Esum_inv * muE * muE + varE * mEsum_inv * mEsum_inv;
            // float sA = sqrtf(s2A);

            float cov_ZA = cov_e[idx] * mEsum_inv + covZE_sum_inv * muE;

            d_mu_a[idx] = mA;
            d_var_a[idx] = s2A;
            d_jcb[idx] = cov_ZA;
        }
    }
}

// Remax function with full covariance calculation
__global__ void compute_remax(float *mu_m, float *var_m, int no, int B,
                              float *mu_a, float *var_a, float *jcb,
                              float *sum_mu_global, float *sum_var_global,
                              float *mu_z, float *var_z, float *full_jcb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < B * no) {
        int i = idx / no;  // Batch index
        int j = idx % no;  // Output index
        float mu_m_in = mu_m[idx];

        // float mu_square = fmaxf(mu_m[idx] * mu_m[idx], 1e-6f);
        // float mu_log_square = fmaxf(sum_mu_global[i] * sum_mu_global[i],
        // 1e-6f);
        float mu_square = mu_m[idx] * mu_m[idx];
        float mu_log_square = sum_mu_global[i] * sum_mu_global[i];

        float var_log = logf(1.0f + (var_m[idx] / mu_square));
        float mu_log = logf(mu_m[idx]) - 0.5f * var_log;
        float cov_M_ln_M = var_log * mu_m[idx];
        float cov_M_M = var_m[idx];
        float cov_Z_M = jcb[idx];

        float var_logsum = logf(1.0f + (sum_var_global[i] / mu_log_square));
        float mu_logsum = logf(sum_mu_global[i]) - 0.5f * var_logsum;

        float cov_log_logsum =
            logf(1.0f + cov_M_M / mu_m[idx] / sum_mu_global[i]);

        float cov_a_hat_ln_M = var_log - cov_log_logsum;
        float cov_a_hat_M = cov_a_hat_ln_M * mu_m[idx];

        float tmp_mu = mu_log - mu_logsum;
        float tmp_var = var_log + var_logsum - 2 * cov_log_logsum;

        float tmp_mu_a = expf(tmp_mu + 0.5f * tmp_var);
        float var = var_m[idx];
        mu_a[idx] = tmp_mu_a;
        // var_a[idx] = tmp_mu_a * tmp_mu_a * (expf(tmp_var) - 1.0f);
        var_a[idx] = expf(2.0f * tmp_mu + tmp_var) * (expf(tmp_var) - 1.0f);
        // jcb[idx] *= tmp_mu_a * cov_a_hat_M / var;
        float cov_a_m = (expf(cov_a_hat_ln_M) - 1.0f) * mu_m_in * tmp_mu_a;

        for (int k = 0; k < no; k++) {
            int idx_j = i * no + k;
            // printf("mu_z[%d]: %f\n", idx_j, mu_z[idx_j]);
            // printf("sum_mu_global[%d]: %f\n", i, sum_mu_global[i]);
            // printf("tmp_mu_a: %f\n", tmp_mu_a);
            float cov_M_sum_Z = jcb[idx_j];
            // printf("cov_M_sum_Z: %f\n", cov_M_sum_Z);

            float cov_ln_M_sum_ln_Z =
                logf(1.0f + cov_M_sum_Z / sum_mu_global[i] / mu_z[idx_j]);
            if (isnan(cov_ln_M_sum_ln_Z) || isinf(cov_ln_M_sum_ln_Z)) {
                cov_ln_M_sum_ln_Z = -10.0f;
            }
            float cov_a_hat_ln_Z = -cov_ln_M_sum_ln_Z;
            // printf("cov_a_hat_ln_Z: %f\n", cov_a_hat_ln_Z);
            full_jcb[i * no * no + j * no + k] =
                (expf(cov_a_hat_ln_Z) - 1.0f) * mu_z[idx_j] * tmp_mu_a;
            if (j == k) {
                full_jcb[i * no * no + j * no + k] = cov_a_m / var * cov_Z_M;
            }
            // printf("full_jcb[%d]: %f\n", i * no * no + j * no + k,
            //        full_jcb[i * no * no + j * no + k]);
        }
    }
}

// Monte Carlo method for computing remax
__global__ void compute_remax_MC(float *mu_a, float *var_a, float *jcb,
                                 float *full_jcb, const float *mu_zv,
                                 const float *var_zv, int no, int B,
                                 int N_samples, unsigned int seed, float *mu_e,
                                 float *var_e) {
    extern __shared__ float shared[];
    float *M_samples = shared;           // [no]
    float *exp_samples = &shared[no];    // [no]
    float *sum_a = &shared[2 * no];      // [no]
    float *sum_E = &shared[3 * no];      // [no]
    float *sum_a_var = &shared[4 * no];  // [no]
    float *sum_aM = &shared[5 * no];     // [no*no]

    const int batch_idx = blockIdx.x;
    const int j = threadIdx.x;
    const int idx = batch_idx * no + j;

    curandState state;
    curand_init(seed + idx, 0, 0,
                &state);  // Initialize RNG once per thread

    // Initialize accumulators
    sum_a[j] = 0.0f;
    sum_E[j] = 0.0f;
    sum_a_var[j] = 0.0f;
    for (int k = 0; k < no; k++) {
        sum_aM[j * no + k] = 0.0f;
    }
    __syncthreads();

    // First pass: Compute means of A and Z
    float sum_exp = 0.0f;  // Use local memory instead of shared
    for (int s = 0; s < N_samples; s++) {
        // Sample from Gaussian
        M_samples[j] = curand_normal(&state) * sqrtf(var_zv[idx]) + mu_zv[idx];
        exp_samples[j] = expf(M_samples[j]);

        // Reduce sum_exp across all threads
        sum_exp = 0.0f;
        for (int k = 0; k < no; k++) {
            sum_exp += exp_samples[k];
        }

        __syncthreads();  // Ensure sum_exp is computed before next step

        // Compute softmax probability
        const float a = exp_samples[j] / sum_exp;

        // Accumulate means
        sum_a[j] += a;
        sum_E[j] += M_samples[j];  // Now summing Z instead of E
    }

    // Compute means
    const float a_bar = sum_a[j] / N_samples;
    const float Z_bar_j = sum_E[j] / N_samples;  // Mean of Z
    mu_a[idx] = a_bar;

    // Second pass: Compute Var(A[j]) and Cov(Z[j], A[k])
    for (int s = 0; s < N_samples; s++) {
        // Sample from Gaussian again
        M_samples[j] = curand_normal(&state) * sqrtf(var_zv[idx]) + mu_zv[idx];
        exp_samples[j] = expf(M_samples[j]);

        __syncthreads();  // Ensure samples are computed before next step

        // Reduce sum_exp across all threads
        sum_exp = 0.0f;
        for (int k = 0; k < no; k++) {
            sum_exp += exp_samples[k];
        }

        __syncthreads();  // Ensure sum_exp is computed before next step

        // Compute softmax probability
        const float a = exp_samples[j] / sum_exp;
        const float a_dev = a - a_bar;
        const float Z_dev_j = M_samples[j] - Z_bar_j;  // Deviation of Z

        // Accumulate variance of A[j]
        sum_a_var[j] += a_dev * a_dev;

        // Compute Cov(Z[j], A[k])
        for (int k = 0; k < no; k++) {
            const float a_k = exp_samples[k] / sum_exp;
            sum_aM[j * no + k] += Z_dev_j * (a_k - (sum_a[k] / N_samples));
        }
    }

    // Store results
    var_a[idx] = sum_a_var[j] / (N_samples - 1);
    jcb[idx] = sum_aM[j * no + j] / (N_samples - 1);  // Cov(Z[j], A[j])

    for (int k = 0; k < no; k++) {
        const float cov = sum_aM[j * no + k] / (N_samples - 1);
        full_jcb[batch_idx * no * no + j * no + k] = cov;  // Cov(Z[j], A[k])
    }
}

// Compute max variance for each mini-batch for an idea I had
__global__ void compute_max_var(const float *var_z, float *max_var, int no,
                                int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / no;
    int j = idx % no;
    if (idx < B * no && j == 0) {
        for (int k = 0; k < no; k++) {
            int idx_k = i * no + k;
            if (var_z[idx_k] > max_var[i]) {
                max_var[i] = var_z[idx_k];
            }
        }
    }
}

// Add variance of observations (sigma_v) to variance of Z. We should add sigma_v
// before the softmax
__global__ void add_var_obs_to_var_a(float *d_mu_a, float *d_var_a, float *d_jcb,
                                     const float *d_var_obs,
                                     int no, int B, float *max_var) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * no) {
        // d_var_a[idx] += d_var_obs[idx];

        // int i = idx / no;

        // float a = 1.0f;
        // if (max_var[i] > 1.0f) {
        //     a = powf(1.0f / max_var[i], 0.5f);
        // }
        // float varZ = d_var_a[idx];
        // d_mu_a[idx] = a * d_mu_a[idx];
        // d_var_a[idx] = a * a * d_var_a[idx];
        // d_jcb[idx] = a;
        if (idx % 2 == 0) {
            // printf("d_var_a[%d]: %f\n", idx, d_var_a[idx]);
            // printf("d_mu_a[%d + 1]: %f\n", idx, d_mu_a[idx + 1]);
            d_var_a[idx] += d_mu_a[idx+1];
            // d_var_a[idx] += 0.001f;
            // d_mu_a[idx + 1] = 0.01f;
            // d_var_a[idx + 1] = 0.01f;
        }
    }
}

void OutputUpdaterCuda::update_output_delta_z_remax(
    BaseHiddenStates &output_states, BaseObservation &obs,
    BaseDeltaStates &delta_states, int no, int B) {
    // Cast to cuda object
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    ObservationCuda *cu_obs = dynamic_cast<ObservationCuda *>(&obs);
    DeltaStateCuda *cu_delta_states =
        dynamic_cast<DeltaStateCuda *>(&delta_states);

    if (cu_obs->d_mu_obs == nullptr) {
        cu_obs->allocate_memory();
    }

    cu_obs->to_device();

    // Reset delta to zero
    cu_delta_states->reset_zeros();

    // B: Batch size
    // no: Number of outputs per mini-batch

    // In case of TAGI-V
    no = no * 2;

    int num_states = no * B;
    int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;


    float *mu_z, *var_z, *max_var;
    cudaMalloc(&max_var, B * sizeof(float));
    cudaMalloc(&mu_z, no * B * sizeof(float));
    cudaMalloc(&var_z, no * B * sizeof(float));
    cudaMemcpy(mu_z, cu_output_states->d_mu_a, no * B * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(var_z, cu_output_states->d_var_a, no * B * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemset(max_var, 0, B * sizeof(float));

    // Compute max variance for each mini-batch
    // compute_max_var<<<blocks, this->num_cuda_threads>>>(
    //     var_z, max_var, no, B);
    // cudaDeviceSynchronize();

    add_var_obs_to_var_a<<<blocks, this->num_cuda_threads>>>(
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb ,cu_obs->d_var_obs, no, B, max_var);
    cudaDeviceSynchronize();

    // Save ZV: Z_output + sigma_v
    float *mu_zv, *var_zv, *full_jcb, *jcb_zv;
    cudaMalloc(&mu_zv, no * B * sizeof(float));
    cudaMalloc(&var_zv, no * B * sizeof(float));
    cudaMalloc(&full_jcb, no * no * B * sizeof(float));
    cudaMalloc(&jcb_zv, no * B * sizeof(float));
    cudaMemcpy(mu_zv, cu_output_states->d_mu_a, no * B * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(var_zv, cu_output_states->d_var_a, no * B * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(jcb_zv, cu_output_states->d_jcb, no * B * sizeof(float),
               cudaMemcpyDeviceToDevice);

    // Exp
    // exp_cuda<<<blocks, this->num_cuda_threads>>>(
    //     cu_output_states->d_mu_a, cu_output_states->d_var_a,
    //     cu_output_states->d_jcb, num_states);

    // CELU
    mixture_celu_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        mu_zv, var_zv, no * B, cu_output_states->d_mu_a,
        cu_output_states->d_var_a, cu_output_states->d_jcb);

    cudaDeviceSynchronize();

    // ExpC
    // expc_cuda<<<blocks, this->num_cuda_threads>>>(
    //     cu_output_states->d_mu_a, cu_output_states->d_var_a,
    //     cu_output_states->d_jcb, num_states);

    // mReLU
    // mixture_relu_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
    //     mu_zv, var_zv, no * B, cu_output_states->d_mu_a,
    //     cu_output_states->d_jcb, cu_output_states->d_var_a);

    // cudaDeviceSynchronize();

    // Allocate memory for batch-wise sums and E: exp(ZV)
    float *sum_mu_global, *sum_var_global;
    float *mu_e, *var_e, *cov_e;
    cudaMalloc(&sum_mu_global, B * sizeof(float));
    cudaMalloc(&sum_var_global, B * sizeof(float));
    cudaMemset(sum_mu_global, 0, B * sizeof(float));
    cudaMemset(sum_var_global, 0, B * sizeof(float));
    cudaMalloc(&mu_e, no * B * sizeof(float));
    cudaMalloc(&var_e, no * B * sizeof(float));
    cudaMalloc(&cov_e, no * B * sizeof(float));
    cudaMemcpy(mu_e, cu_output_states->d_mu_a, no * B * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(var_e, cu_output_states->d_var_a, no * B * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(cov_e, cu_output_states->d_jcb, no * B * sizeof(float),
                cudaMemcpyDeviceToDevice);

    // First phase of Remax: Just to get the sum of each mini-batch (denominator)
    // I know the name of the function is not correct... I'll correct it
    remax_forward_cuda<<<blocks, this->num_cuda_threads>>>(
        mu_e, var_e, no, B, cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb, sum_mu_global, sum_var_global);
    cudaDeviceSynchronize();

    // Second phase of Remax: numerator / denominator
    // compute_remax<<<blocks, this->num_cuda_threads>>>(
    //     cu_output_states->d_mu_a, cu_output_states->d_var_a, no, B,
    //     cu_output_states->d_mu_a, cu_output_states->d_var_a,
    //     cu_output_states->d_jcb, sum_mu_global, sum_var_global, mu_zv,
    //     var_zv, full_jcb);

    // MC sampling
    // int N_samples = 10000;
    // const size_t shared_size = (2 * no + 1 + no * no + no) *
    // sizeof(float);
    // compute_remax_MC<<<B, no, shared_size>>>(
    //     cu_output_states->d_mu_a, cu_output_states->d_var_a,
    //     cu_output_states->d_jcb, full_jcb, mu_zv, var_zv, no, B,
    // N_samples,
    //     time(0), mu_e, var_e);

    // cudaDeviceSynchronize();

    // Second phase of Softmax
    compute_softmax_outputs<<<blocks, this->num_cuda_threads>>>(
        mu_z, var_z, mu_zv, var_zv, cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb, sum_mu_global, sum_var_global, mu_e, var_e, cov_e,
        no, B);
    cudaDeviceSynchronize();

    // Free memory for batch-wise sums
    cudaFree(sum_mu_global);
    cudaFree(sum_var_global);

    // Update delta
    num_states = cu_obs->size;
    blocks = (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    // update_delta_z_cuda_remax<<<blocks, this->num_cuda_threads>>>(
    //     cu_output_states->d_mu_a, cu_output_states->d_var_a,
    //     cu_output_states->d_jcb, cu_obs->d_mu_obs, cu_obs->d_var_obs,
    //     num_states, cu_delta_states->d_delta_mu,
    //     cu_delta_states->d_delta_var, var_zv);

    // update_delta_z_cuda_fullcov<<<blocks, this->num_cuda_threads>>>(
    //     cu_output_states->d_mu_a, cu_output_states->d_var_a,
    //     cu_output_states->d_jcb, cu_obs->d_mu_obs, cu_obs->d_var_obs,
    //     num_states, cu_delta_states->d_delta_mu, cu_delta_states->d_delta_var,
    //     mu_zv, var_zv, jcb_zv, mu_z, var_z, mu_e, var_e, full_jcb, no);

    update_delta_z_cuda_heteros_softmax<<<blocks, this->num_cuda_threads>>>(
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb, cu_obs->d_mu_obs, num_states,
        cu_delta_states->d_delta_mu, cu_delta_states->d_delta_var, mu_zv, var_zv, var_z);

    // update_delta_z_cuda<<<blocks, this->num_cuda_threads>>>(
    //     cu_output_states->d_mu_a, cu_output_states->d_var_a,
    //     cu_output_states->d_jcb, cu_obs->d_mu_obs, cu_obs->d_var_obs,
    //     num_states, cu_delta_states->d_delta_mu, cu_delta_states->d_delta_var);

    cudaDeviceSynchronize();

    // Free memory
    cudaFree(max_var);
    cudaFree(mu_zv);
    cudaFree(var_zv);
    cudaFree(full_jcb);
    cudaFree(jcb_zv);
    cudaFree(var_z);
    cudaFree(mu_z);
    cudaFree(mu_e);
    cudaFree(var_e);
    cudaFree(cov_e);
}