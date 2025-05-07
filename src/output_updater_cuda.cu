#include "../include/custom_logger.h"
#include "../include/output_updater_cuda.cuh"
#include "../include/activation_cuda.cuh"

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

        float delta_mu_zv = 0;
        float delta_var_zv = 0;

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

__global__ void update_delta_z_cuda_2(float const *mu_a, float const *var_a,
    float const *jcb, float const *obs,
    float const *var_obs, int size,
    float *delta_mu, float *delta_var, float *var_z, float *var_zv) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zero_pad = 0;
    float tmp = 0;

    if (col < size) {

        float delta_mu_zv = 0;
        float delta_var_zv = 0;

        tmp = jcb[col] / (var_a[col]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mu_zv = zero_pad;
            delta_var_zv = zero_pad;
        } else {
            delta_mu_zv = tmp * (obs[col] - mu_a[col]);
            delta_var_zv = -tmp * jcb[col];
        }

        // Compute deltas for Z
        float Jz = 1.0f / (var_zv[col]);
        float Jz_mu = 1.0f / (var_z[col]);

        // TODO: Use Jz_mu for both
        delta_mu[col] = Jz_mu * delta_mu_zv;
        delta_var[col] = Jz * Jz * delta_var_zv;
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
        float tmp = jcb_col / var_sum;
        if (std::isinf(tmp) || std::isnan(tmp)) {
            delta_mu[obs_col] = zero_pad;
            delta_var[obs_col] = zero_pad;
        } else {
            float obs_diff = obs[col] - mu_a_col;
            delta_mu[obs_col] = tmp * obs_diff;
            delta_var[obs_col] = -tmp * jcb_col;
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

__global__ void update_delta_z_cuda_heteros_2(float const *mu_a,
                                              float const *var_a,
                                              float const *jcb, float const *obs,
                                              int size, float *delta_mu,
                                              float *delta_var, float *var_z,
                                                float *var_zv) {
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
        float cov_col = jcb[obs_col];

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

        float delta_mu_zv = 0.0f;
        float delta_var_zv = 0.0f;

        // Compute updating quantities for the mean of the output
        float tmp = cov_col / var_sum;
        if (std::isinf(tmp) || std::isnan(tmp)) {
            delta_mu_zv = zero_pad;
            delta_var_zv = zero_pad;
        } else {
            float obs_diff = obs[col] - mu_a_col;
            delta_mu_zv = tmp * obs_diff;
            delta_var_zv = -tmp * cov_col;
        }

        // Compute deltas for Z
        float Jz_mu = 1.0f / (var_z[obs_col]); // Force ratio to be 1
        float Jz = 1.0f / (var_zv[obs_col]);
        // TODO: Use Jz_mu for both
        delta_mu[obs_col] = Jz_mu * delta_mu_zv;
        delta_var[obs_col] = Jz * Jz * delta_var_zv;

        // Smooth back V given ZV
        float Jv = var_v / (var_zv[obs_col]);
        float Jv_mu = var_v / (var_z[obs_col]);
        float mu_v_post = mu_v + Jv * delta_mu_zv;
        float var_v_post = var_v + Jv * Jv * delta_var_zv;

        // Compute the posterior mean and variance for V2
        float mu_v2_post = mu_v_post * mu_v_post + var_v_post;
        float var_v2_post = 2.0f * var_v_post * var_v_post +
        4.0f * var_v_post * mu_v_post * mu_v_post;

        // Compute the posterior mean and variance for V2_bar_tilde
        float Jv2_bar_tilde = var_v2_bar_tilde / var_v2;
        float mu_v2_bar_tilde_post =
        mu_v2_bar_tilde + Jv2_bar_tilde * (mu_v2_post - mu_v2);
        float var_v2_bar_tilde_post =
        var_v2_bar_tilde + Jv2_bar_tilde * Jv2_bar_tilde * (var_v2_post - var_v2);

        // Compute update for V2_bar
        float Jv2_bar = cov_v2_bar_tilde / var_v2_bar_tilde;
        delta_mu[obs_col + 1] = Jv2_bar * (mu_v2_bar_tilde_post - mu_v2_bar_tilde);
        delta_var[obs_col + 1] =
        Jv2_bar * Jv2_bar * (var_v2_bar_tilde_post - var_v2_bar_tilde);
    }
}

OutputUpdaterCuda::OutputUpdaterCuda(int device_idx) {
    this->device_idx = device_idx;
    cudaSetDevice(device_idx);
}

void OutputUpdaterCuda::set_num_cuda_threads(unsigned int num_threads) {
    this->num_cuda_threads = num_threads;
}

__global__ void add_var_obs_to_var_a(float *d_mu_a, float *d_var_a, float *d_jcb,
    const float *d_var_obs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx % 2 == 0) {
        d_var_a[idx] += d_mu_a[idx + 1];
        d_mu_a[idx] *= 10.0f;
        d_var_a[idx] *= 100.0f;
        d_jcb[idx] *= 10.0f;
    }
    // d_var_a[idx] += d_var_obs[idx];
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

    float *var_z;
    cudaMalloc(&var_z, sizeof(float) * num_states);
    cudaMemcpy(var_z, cu_output_states->d_var_a, sizeof(float) * num_states,
               cudaMemcpyDeviceToDevice);


    add_var_obs_to_var_a<<<blocks, this->num_cuda_threads>>>(
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb ,cu_obs->d_var_obs);
    cudaDeviceSynchronize();

    float *var_zv;
    cudaMalloc(&var_zv, sizeof(float) * num_states);
    cudaMemcpy(var_zv, cu_output_states->d_var_a, sizeof(float) * num_states,
               cudaMemcpyDeviceToDevice);

    blocks = 8;

    // Softmax
    softmax_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->actual_size, cu_output_states->block_size,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);
    cudaDeviceSynchronize();

    blocks =
    (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    // update_delta_z_cuda<<<blocks, this->num_cuda_threads>>>(
    //     cu_output_states->d_mu_a, cu_output_states->d_var_a,
    //     cu_output_states->d_jcb, cu_obs->d_mu_obs, cu_obs->d_var_obs,
    //     num_states, cu_delta_states->d_delta_mu, cu_delta_states->d_delta_var);

    update_delta_z_cuda_2<<<blocks, this->num_cuda_threads>>>(
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb, cu_obs->d_mu_obs, cu_obs->d_var_obs,
        num_states, cu_delta_states->d_delta_mu, cu_delta_states->d_delta_var, var_z, var_zv);

    cudaFree(var_z);
    cudaFree(var_zv);
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

    // check if device index match
    if (cu_obs->device_idx != cu_output_states->device_idx &&
        cu_obs->device_idx != cu_delta_states->device_idx) {
        std::string message =
            "Device index mismatch: " + std::to_string(cu_obs->device_idx) +
            " vs " + std::to_string(cu_output_states->device_idx) + " vs " +
            std::to_string(cu_delta_states->device_idx);
        LOG(LogLevel::ERROR, message);
    }

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

    // check if device index match
    if (cu_obs->device_idx != cu_output_states->device_idx &&
        cu_obs->device_idx != cu_delta_states->device_idx) {
        std::string message =
            "Device index mismatch: " + std::to_string(cu_obs->device_idx) +
            " vs " + std::to_string(cu_output_states->device_idx) + " vs " +
            std::to_string(cu_delta_states->device_idx);
        LOG(LogLevel::ERROR, message);
    }

    // Reset delta to zero
    cu_delta_states->reset_zeros();

    // Kernel
    int num_states = cu_obs->size * 2;
    int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;
    // printf("num_states: %d\n", num_states);

    float *var_z;
    cudaMalloc(&var_z, sizeof(float) * num_states);
    cudaMemcpy(var_z, cu_output_states->d_var_a, sizeof(float) * num_states,
               cudaMemcpyDeviceToDevice);


    add_var_obs_to_var_a<<<blocks, this->num_cuda_threads>>>(
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb ,cu_obs->d_var_obs);
    cudaDeviceSynchronize();

    float *var_zv;
    cudaMalloc(&var_zv, sizeof(float) * num_states);
    cudaMemcpy(var_zv, cu_output_states->d_var_a, sizeof(float) * num_states,
               cudaMemcpyDeviceToDevice);

    // printf("cu_output_states->actual_size %d\n", cu_output_states->actual_size);
    // printf("num_states: %d\n", num_states);
    blocks = 8;

    // Softmax
    softmax_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->actual_size, cu_output_states->block_size,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);
    cudaDeviceSynchronize();


    num_states = cu_obs->size;
    blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    // update_delta_z_cuda_heteros<<<blocks, this->num_cuda_threads>>>(
    //     cu_output_states->d_mu_a, cu_output_states->d_var_a,
    //     cu_output_states->d_jcb, cu_obs->d_mu_obs, num_states,
    //     cu_delta_states->d_delta_mu, cu_delta_states->d_delta_var);

    update_delta_z_cuda_heteros_2<<<blocks, this->num_cuda_threads>>>(
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb, cu_obs->d_mu_obs, num_states,
        cu_delta_states->d_delta_mu, cu_delta_states->d_delta_var, var_z, var_zv);

    cudaFree(var_z);
    cudaFree(var_zv);
}
