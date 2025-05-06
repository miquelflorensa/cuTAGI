#include "../include/activation.h"
#include "../include/activation_cuda.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>  // For CUDART_PI_F

////////////////////////////////////////////////////////////////////////////////
// Softmax Kernel
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// ReLU
////////////////////////////////////////////////////////////////////////////////
ReLUCuda::ReLUCuda() {}
ReLUCuda::~ReLUCuda() {}

std::string ReLUCuda::get_layer_info() const
/*
 */
{
    return "Relu()";
}

std::string ReLUCuda::get_layer_name() const
/*
 */
{
    return "ReLUCuda";
}

LayerType ReLUCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void ReLUCuda::forward(BaseHiddenStates &input_states,
                       BaseHiddenStates &output_states,
                       BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    // Assign output dimensions
    cu_output_states->height = cu_input_states->height;
    cu_output_states->depth = cu_input_states->depth;
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;

    constexpr unsigned int THREADS = 256;
    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks = (num_states + THREADS - 1) / THREADS;

    relu_mean_var_cuda<<<blocks, THREADS>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }
}

std::unique_ptr<BaseLayer> ReLUCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<ReLU>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Sigmoid
////////////////////////////////////////////////////////////////////////////////
SigmoidCuda::SigmoidCuda() {}
SigmoidCuda::~SigmoidCuda() {}

std::string SigmoidCuda::get_layer_info() const
/*
 */
{
    return "Sigmoid()";
}

std::string SigmoidCuda::get_layer_name() const
/*
 */
{
    return "SigmoidCuda";
}

LayerType SigmoidCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void SigmoidCuda::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    sigmoid_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> SigmoidCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<Sigmoid>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Tanh
////////////////////////////////////////////////////////////////////////////////
TanhCuda::TanhCuda() {}
TanhCuda::~TanhCuda() {}

std::string TanhCuda::get_layer_info() const
/*
 */
{
    return "Tanh()";
}

std::string TanhCuda::get_layer_name() const
/*
 */
{
    return "TanhCuda";
}

LayerType TanhCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void TanhCuda::forward(BaseHiddenStates &input_states,
                       BaseHiddenStates &output_states,
                       BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    tanh_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> TanhCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<Tanh>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Mixture Relu
////////////////////////////////////////////////////////////////////////////////
MixtureReLUCuda::MixtureReLUCuda() {}
MixtureReLUCuda ::~MixtureReLUCuda() {}

std::string MixtureReLUCuda::get_layer_info() const
/*
 */
{
    return "MixtureReLU()";
}

std::string MixtureReLUCuda::get_layer_name() const
/*
 */
{
    return "MixtureReLUCuda";
}

LayerType MixtureReLUCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureReLUCuda::forward(BaseHiddenStates &input_states,
                              BaseHiddenStates &output_states,
                              BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    mixture_relu_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    // cu_output_states->to_device();

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> MixtureReLUCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<MixtureReLU>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Mixture Sigmoid
////////////////////////////////////////////////////////////////////////////////
MixtureSigmoidCuda::MixtureSigmoidCuda() {}
MixtureSigmoidCuda ::~MixtureSigmoidCuda() {}

std::string MixtureSigmoidCuda::get_layer_info() const
/*
 */
{
    return "MixtureSigmoid()";
}

std::string MixtureSigmoidCuda::get_layer_name() const
/*
 */
{
    return "MixtureSigmoidCuda";
}

LayerType MixtureSigmoidCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureSigmoidCuda::forward(BaseHiddenStates &input_states,
                                 BaseHiddenStates &output_states,
                                 BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    mixture_sigmoid_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> MixtureSigmoidCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<MixtureSigmoid>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Mixture Tanh
////////////////////////////////////////////////////////////////////////////////
MixtureTanhCuda::MixtureTanhCuda() {}
MixtureTanhCuda ::~MixtureTanhCuda() {}

std::string MixtureTanhCuda::get_layer_info() const
/*
 */
{
    return "MixtureTanh()";
}

std::string MixtureTanhCuda::get_layer_name() const
/*
 */
{
    return "MixtureTanhCuda";
}

LayerType MixtureTanhCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureTanhCuda::forward(BaseHiddenStates &input_states,
                              BaseHiddenStates &output_states,
                              BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    mixture_tanh_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> MixtureTanhCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<MixtureTanh>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Softplus
////////////////////////////////////////////////////////////////////////////////
SoftplusCuda::SoftplusCuda() {}
SoftplusCuda::~SoftplusCuda() {}

std::string SoftplusCuda::get_layer_info() const
/*
 */
{
    return "Softplus()";
}

std::string SoftplusCuda::get_layer_name() const
/*
 */
{
    return "SoftplusCuda";
}

LayerType SoftplusCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void SoftplusCuda::forward(BaseHiddenStates &input_states,
                           BaseHiddenStates &output_states,
                           BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    softplus_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> SoftplusCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<Softplus>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// LeakyRelu
////////////////////////////////////////////////////////////////////////////////
LeakyReLUCuda::LeakyReLUCuda() {}
LeakyReLUCuda::~LeakyReLUCuda() {}

std::string LeakyReLUCuda::get_layer_info() const
/*
 */
{
    return "leakyRelu()";
}

std::string LeakyReLUCuda::get_layer_name() const
/*
 */
{
    return "leakyReluCuda";
}

LayerType LeakyReLUCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void LeakyReLUCuda::forward(BaseHiddenStates &input_states,
                            BaseHiddenStates &output_states,
                            BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    leakyrelu_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, this->alpha,
        num_states, cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> LeakyReLUCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<LeakyReLU>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Softmax
////////////////////////////////////////////////////////////////////////////////
SoftmaxCuda::SoftmaxCuda() {}
SoftmaxCuda::~SoftmaxCuda() {}

std::string SoftmaxCuda::get_layer_info() const
/*
 */
{
    return "Softmax()";
}

std::string SoftmaxCuda::get_layer_name() const
/*
 */
{
    return "SoftmaxCuda";
}

LayerType SoftmaxCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void SoftmaxCuda::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    unsigned int blocks =
        (input_states.block_size + this->num_cuda_threads - 1) /
        this->num_cuda_threads;

    printf("SoftmaxCuda::forward: blocks %d\n", blocks);

    softmax_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a,
        cu_input_states->actual_size, cu_input_states->block_size,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> SoftmaxCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<Softmax>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// EvenExp
////////////////////////////////////////////////////////////////////////////////
EvenExpCuda::EvenExpCuda() {}
EvenExpCuda::~EvenExpCuda() {}

std::string EvenExpCuda::get_layer_info() const
/*
 */
{
    return "EvenExp()";
}

std::string EvenExpCuda::get_layer_name() const
/*
 */
{
    return "EvenExpCuda";
}

LayerType EvenExpCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void EvenExpCuda::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    // Assign output dimensions
    cu_output_states->height = cu_input_states->height;
    cu_output_states->depth = cu_input_states->depth;
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;

    even_exp_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a,
        cu_input_states->d_jcb, num_states, cu_output_states->d_mu_a,
        cu_output_states->d_var_a, cu_output_states->d_jcb);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> EvenExpCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<EvenExp>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
// CUDA kernels
////////////////////////////////////////////////////////////////////////////////

__global__ void relu_mean_var_cuda(float const *mu_z, float const *var_z,
                                   int num_states, float *mu_a, float *jcb,
                                   float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < num_states) {
        float tmp = fmaxf(mu_z[col], 0.0f);
        mu_a[col] = tmp;

        bool is_zero = (tmp == 0.0f);
        jcb[col] = is_zero ? 0.0f : 1.0f;
        var_a[col] = is_zero ? 0.0f : var_z[col];
    }
}

__global__ void relu_mean_var_cuda_vectorized(float const *mu_z,
                                              float const *var_z,
                                              int num_states, float *mu_a,
                                              float *jcb, float *var_a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;

    if (vec_idx < num_states) {
        float4 mu_z_vec, var_z_vec, mu_a_vec, jcb_vec, var_a_vec;

        // Load 4 float values into float4 vectors
        mu_z_vec.x = mu_z[vec_idx];
        mu_z_vec.y = vec_idx + 1 < num_states ? mu_z[vec_idx + 1] : 0.0f;
        mu_z_vec.z = vec_idx + 2 < num_states ? mu_z[vec_idx + 2] : 0.0f;
        mu_z_vec.w = vec_idx + 3 < num_states ? mu_z[vec_idx + 3] : 0.0f;

        var_z_vec.x = var_z[vec_idx];
        var_z_vec.y = vec_idx + 1 < num_states ? var_z[vec_idx + 1] : 0.0f;
        var_z_vec.z = vec_idx + 2 < num_states ? var_z[vec_idx + 2] : 0.0f;
        var_z_vec.w = vec_idx + 3 < num_states ? var_z[vec_idx + 3] : 0.0f;

        // Process the data
        mu_a_vec.x = fmaxf(mu_z_vec.x, 0.0f);
        mu_a_vec.y = fmaxf(mu_z_vec.y, 0.0f);
        mu_a_vec.z = fmaxf(mu_z_vec.z, 0.0f);
        mu_a_vec.w = fmaxf(mu_z_vec.w, 0.0f);

        jcb_vec.x = (mu_a_vec.x == 0.0f) ? 0.0f : 1.0f;
        jcb_vec.y = (mu_a_vec.y == 0.0f) ? 0.0f : 1.0f;
        jcb_vec.z = (mu_a_vec.z == 0.0f) ? 0.0f : 1.0f;
        jcb_vec.w = (mu_a_vec.w == 0.0f) ? 0.0f : 1.0f;

        var_a_vec.x = (mu_a_vec.x == 0.0f) ? 0.0f : var_z_vec.x;
        var_a_vec.y = (mu_a_vec.y == 0.0f) ? 0.0f : var_z_vec.y;
        var_a_vec.z = (mu_a_vec.z == 0.0f) ? 0.0f : var_z_vec.z;
        var_a_vec.w = (mu_a_vec.w == 0.0f) ? 0.0f : var_z_vec.w;

        // Store the results back as individual floats
        mu_a[vec_idx] = mu_a_vec.x;
        jcb[vec_idx] = jcb_vec.x;
        var_a[vec_idx] = var_a_vec.x;

        if (vec_idx + 1 < num_states) {
            mu_a[vec_idx + 1] = mu_a_vec.y;
            jcb[vec_idx + 1] = jcb_vec.y;
            var_a[vec_idx + 1] = var_a_vec.y;
        }

        if (vec_idx + 2 < num_states) {
            mu_a[vec_idx + 2] = mu_a_vec.z;
            jcb[vec_idx + 2] = jcb_vec.z;
            var_a[vec_idx + 2] = var_a_vec.z;
        }

        if (vec_idx + 3 < num_states) {
            mu_a[vec_idx + 3] = mu_a_vec.w;
            jcb[vec_idx + 3] = jcb_vec.w;
            var_a[vec_idx + 3] = var_a_vec.w;
        }
    }
}

__global__ void sigmoid_mean_var_cuda(float const *mu_z, float const *var_z,
                                      int num_states, float *mu_a, float *jcb,
                                      float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0.0f;

    if (col < num_states) {
        tmp = 1.0f / (1.0f + expf(-mu_z[col]));
        mu_a[col] = tmp;
        jcb[col] = tmp * (1.0f - tmp);
        var_a[col] = tmp * (1.0f - tmp) * var_z[col] * tmp * (1.0f - tmp);
    }
}

__global__ void tanh_mean_var_cuda(float const *mu_z, float const *var_z,
                                   int num_states, float *mu_a, float *jcb,
                                   float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0.0f;
    if (col < num_states) {
        tmp = tanhf(mu_z[col]);
        float tmp_2 = tmp * tmp;
        mu_a[col] = tmp;
        jcb[col] = (1.0f - tmp_2);
        var_a[col] = (1.0f - tmp_2) * var_z[col] * (1.0f - tmp_2);
    }
}

__device__ float normcdf_cuda(float x)
/*
Normal cumulative distribution function
 */
{
    return 0.5f * erfcf(-x * 0.7071067811865475f);
}

__global__ void mixture_relu_mean_var_cuda(float const *mu_z,
                                           float const *var_z, int num_states,
                                           float *mu_a, float *jcb,
                                           float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr float SQRT_2PI = 2.5066282746310002f;
    if (col < num_states) {
        // Reused components for moments calculations
        float tmp_mu_z = mu_z[col];
        float std_z = powf(var_z[col], 0.5);
        float alpha = tmp_mu_z / std_z;
        float pdf_alpha = (1.0f / SQRT_2PI) * expf(-0.5f * alpha * alpha);
        float cdf_alpha = normcdf_cuda(alpha);

        // Moments calculations (L. Alric, 2024)
        float tmp_mu_a = mu_z[col] * cdf_alpha + std_z * pdf_alpha;
        mu_a[col] = tmp_mu_a;
        var_a[col] = -tmp_mu_a * tmp_mu_a + 2 * tmp_mu_a * tmp_mu_z -
                     tmp_mu_z * std_z * pdf_alpha +
                     (var_z[col] - tmp_mu_z * tmp_mu_z) * cdf_alpha;
        jcb[col] = cdf_alpha;
    }
}

__global__ void mixture_sigmoid_mean_var_cuda(float const *mu_z,
                                              float const *var_z,
                                              int num_states, float *mu_a,
                                              float *jcb, float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float std_z, alpha_l, alpha_u, pdf_l, pdf_u, cdf_l, cdf_u;
    constexpr float SQRT_2PI = 2.5066282746310002f;

    if (col < num_states) {
        // cdf and pdf for truncated normal distribution
        std_z = powf(var_z[col], 0.5);
        alpha_l = (1.0f + mu_z[col]) / std_z;  // Lower truncation
        alpha_u = (1.0f - mu_z[col]) / std_z;  // Upper truncation
        cdf_l = normcdf_cuda(alpha_l);
        cdf_u = normcdf_cuda(alpha_u);
        pdf_l = (1.0f / SQRT_2PI) * expf(-0.5f * alpha_l * alpha_l);
        pdf_u = (1.0f / SQRT_2PI) * expf(-0.5f * alpha_u * alpha_u);

        // Moments calculations (L. Alric, 2024)
        float tmp_mu_z = mu_z[col];
        float tmp_mu_z_2 = tmp_mu_z * tmp_mu_z;
        float tmp_mu_a = (tmp_mu_z + 1) * cdf_l + (tmp_mu_z - 1) * cdf_u +
                         std_z * (pdf_l - pdf_u) - tmp_mu_z;

        mu_a[col] = tmp_mu_a;
        var_a[col] =
            max(0.000001f,
                (cdf_l * (var_z[col] - tmp_mu_z_2 - 2 * tmp_mu_z - 1) +
                 cdf_u * (var_z[col] - tmp_mu_z_2 + 2 * tmp_mu_z - 1) +
                 std_z * (pdf_u * (tmp_mu_z - 1) - pdf_l * (tmp_mu_z + 1)) -
                 tmp_mu_a * tmp_mu_a + 2 * mu_a[col] * tmp_mu_z +
                 tmp_mu_z * tmp_mu_z - var_z[col] + 2) /
                    4.0f);
        mu_a[col] = tmp_mu_a / 2.0f + 0.5f;
        jcb[col] = (cdf_u + cdf_l - 1) / 2.0f;
    }
}

__global__ void mixture_tanh_mean_var_cuda(float const *mu_z,
                                           float const *var_z, int num_states,
                                           float *mu_a, float *jcb,
                                           float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float std_z, alpha_l, alpha_u, pdf_l, pdf_u, cdf_l, cdf_u;
    constexpr float SQRT_2PI = 2.5066282746310002f;

    if (col < num_states) {
        // cdf and pdf for truncated normal distribution
        float tmp_mu_z = mu_z[col];
        std_z = powf(var_z[col], 0.5);
        alpha_l = (1.0f + tmp_mu_z) / std_z;  // Lower truncation
        alpha_u = (1.0f - tmp_mu_z) / std_z;  // Upper truncation
        cdf_l = normcdf_cuda(alpha_l);
        cdf_u = normcdf_cuda(alpha_u);
        pdf_l = (1.0f / SQRT_2PI) * expf(-0.5f * alpha_l * alpha_l);
        pdf_u = (1.0f / SQRT_2PI) * expf(-0.5f * alpha_u * alpha_u);

        // Moments calculations (L. Alric, 2024)
        float tmp_mu_a = (tmp_mu_z + 1) * cdf_l + (tmp_mu_z - 1) * cdf_u +
                         std_z * (pdf_l - pdf_u) - tmp_mu_z;

        mu_a[col] = tmp_mu_a;
        var_a[col] = max(
            0.000001f,
            cdf_l * (var_z[col] - tmp_mu_z * tmp_mu_z - 2 * tmp_mu_z - 1) +
                cdf_u * (var_z[col] - tmp_mu_z * tmp_mu_z + 2 * tmp_mu_z - 1) +
                std_z * (pdf_u * (tmp_mu_z - 1) - pdf_l * (tmp_mu_z + 1)) -
                tmp_mu_a + 2 * tmp_mu_a * tmp_mu_z + tmp_mu_z - var_z[col] + 2);

        jcb[col] = cdf_u + cdf_l - 1;
    }
}

__global__ void softplus_mean_var_cuda(float const *mu_z, float const *var_z,
                                       int num_states, float *mu_a, float *jcb,
                                       float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0;
    if (col < num_states) {
        mu_a[col] = logf(1 + expf(mu_z[col]));
        tmp = 1 / (1 + expf(-mu_z[col]));
        jcb[col] = tmp;
        var_a[col] = tmp * var_z[col] * tmp;
    }
}

__global__ void leakyrelu_mean_var_cuda(float const *mu_z, float const *var_z,
                                        float alpha, int num_states,
                                        float *mu_a, float *jcb, float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zero_pad = 0.0f;
    float one_pad = 1.0f;
    float tmp = 0.0f;
    if (col < num_states) {
        tmp = max(mu_z[col], zero_pad);
        if (tmp == 0) {
            mu_a[col] = alpha * mu_z[col];
            jcb[col] = alpha;
            var_a[col] = alpha * var_z[col] * alpha;

        } else {
            mu_a[col] = tmp;
            jcb[col] = one_pad;
            var_a[col] = var_z[col];
        }
    }
}

// __global__ void softmax_mean_var_cuda(float const *mu_z, float *var_z,
//                                       size_t output_size, int batch_size,
//                                       float *mu_a, float *jcb, float *var_a)
// /*
//  */
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= batch_size) return;
//     constexpr float SQRT_2PI = 2.5066282746310002f;

//     // Create a temporary variable to store vector of muM, varM, and cov_ZM
//     float muM[10];
//     float varM[10];
//     float cov_ZM[10];
//     float cov_M_M_sum[10];

//     for (int j = 0; j < output_size; j++) {
//         // mReLU
//         float muZ = mu_z[j + i * output_size];
//         float varZ = var_z[j + i * output_size];
//         float stdZ = sqrtf(varZ);
//         float alpha = muZ / stdZ;
//         float cdfn = fmaxf(1e-20, normcdf_cuda(alpha));
//         float pdfn = fmaxf(1e-20, (1.0f / SQRT_2PI) * expf(-0.5f * alpha * alpha));
//         muM[j] = fmaxf(1e-20, stdZ * pdfn + muZ * cdfn);
//         varM[j] = fmaxf(1e-6, -muM[j] * muM[j] + 2 * muM[j] * muZ - muZ * stdZ * pdfn +
//                     (varZ - muZ * muZ) * cdfn);
//         cov_ZM[j] = cdfn * varZ;
//         cov_M_M_sum[j] = varM[j];
//     }

//     // \tilde{M} = sum(M_i)
//     float muM_sum = 0.0f;
//     float varM_sum = 0.0f;
//     for (int j = 0; j < output_size; j++) {
//         muM_sum += muM[j];
//         varM_sum += varM[j];
//     }

//     // lnM = log(M_i)
//     float mulnM[10];
//     float varlnM[10];
//     float cov_M_lnM[10];
//     float cov_lnM_lnM_sum[10];
//     for (int j = 0; j < output_size; j++) {
//         varlnM[j] = logf(1.0f + fminf(10.0f, varM[j] / (muM[j] * muM[j])));
//         mulnM[j] = logf(muM[j]) - 0.5f * varlnM[j];
//         cov_M_lnM[j] = varlnM[j] * muM[j];
//         cov_lnM_lnM_sum[j] = logf(1.0f + cov_M_M_sum[j] / muM[j] / muM_sum);
//     }

//     // ln\tilde{M} log(\tilde{M}_i)
//     float varlnM_sum = logf(1.0f + fminf(10.0f, varM_sum / (muM_sum * muM_sum)));
//     float mulnM_sum = logf(muM_sum) - 0.5f * varlnM_sum;

//     // 1/\tilde{M} -> 1-ln\tilde{M}
//     float varlnM_sum_inv = 1.0f / varlnM_sum;
//     float mulnM_sum_inv = 1.0f - mulnM_sum;

//     float muM_sum_inv = expf(mulnM_sum_inv + 0.5f * varlnM_sum_inv);
//     float varM_sum_inv = muM_sum_inv * muM_sum_inv * (expf(varlnM_sum_inv) - 1.0f);

//     float cov_M_M_sum_inv[10];
//     for (int j = 0; j < output_size; j++) {
//         cov_M_M_sum_inv[j] = (expf(cov_lnM_lnM_sum[j]) - 1.0f) *
//                              muM_sum * muM_sum_inv;
//     }

//     // \check{A}_i = lnM_i - ln\tilde{M}
//     float mulnA[10];
//     float varlnA[10];
//     for (int j = 0; j < output_size; j++) {
//         mulnA[j] = mulnM[j] - mulnM_sum;
//         varlnA[j] = varlnM[j] + varlnM_sum -
//                     2 * cov_lnM_lnM_sum[j];
//     }

//     // A_i = normal
//     float muA[10];
//     float varA[10];
//     float cov_ZA[10];
//     float cov_cAlnM[10];
//     float cov_AM[10];
//     for (int j = 0; j < output_size; j++) {
//         muA[j] = expf(mulnA[j] + 0.5f * varlnA[j]);
//         cov_cAlnM[j] = varlnM[j] - cov_lnM_lnM_sum[j];    
//     }
//     float muA_sum = 0.0f;
//     for (int j = 0; j < output_size; j++) {
//         muA_sum += muA[j];
//     }
//     for (int j = 0; j < output_size; j++) {
//         muA[j] = muA[j] / muA_sum;
//         varA[j] = muA[j] * muA[j] * (expf(varlnA[j]) - 1.0f);
//         cov_AM[j] = (expf(cov_cAlnM[j]) - 1.0f) * muA[j] * muM[j];
        
        
//         cov_ZA[j] = fminf(cov_AM[j] * cov_ZM[j] / varM[j], sqrtf(varA[j]) * sqrtf(var_z[j]));
//         // cov_ZA[j] = fminf(cov_AM[j] / cov_ZM[j] * var_z[j], sqrtf(varA[j]) * sqrtf(var_z[j]));

//         mu_a[j + i * output_size] = muA[j];
//         var_a[j + i * output_size] = varA[j];
//         jcb[j + i * output_size] = cov_ZA[j];
//     }

// }

// constexpr double INV_SQRT2PI = 0.3989422804014327;  // 1/√(2π)
// constexpr double INV_SQRT2   = 0.7071067811865476;  // 1/√2
// constexpr double EPS_VAR     = 1e-8;                // min variance
// constexpr double TINY        = 1e-300;              // min PDF/CDF
// constexpr double MIN_VAR    = 1e-8;                  // floor on var
// constexpr double NEG_THRESH = -8.0;                  // deep-tail threshold

// __global__ void softmax_mean_var_cuda(
//     float const *mu_z,
//     float *var_z,
//     size_t output_size,   // now 20
//     int batch_size,
//     float *mu_a,
//     float *jcb,
//     float *var_a
// ) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= batch_size) return;

//     constexpr float SQRT_2PI = 2.5066282746310002f;

//     // We only care about the even positions, so:
//     const int K = output_size / 2;  // expect K=10 when output_size=20

//     // Temporary buffers, size K instead of output_size
//     float muM[10], varM[10], cov_ZM[10], cov_M_M_sum[10];

//     // 1) compute muM, varM, cov_ZM, cov_M_M_sum for even indices
//     for (int k = 0; k < K; ++k) {
//         int j = 2 * k;
        
//         float muZ = mu_z[j + i * output_size];
//         float varZ = var_z[j + i * output_size];
//         float stdZ = sqrtf(varZ);
//         float alpha = muZ / stdZ;
//         float cdfn  = fmaxf(1e-20f, normcdf_cuda(alpha));
//         float pdfn  = fmaxf(1e-20f,
//                             (1.0f / SQRT_2PI) * expf(-0.5f * alpha * alpha));

//         muM[k] = fmaxf(1e-20f, stdZ * pdfn + muZ * cdfn);
//         varM[k] = fmaxf(1e-6f,
//                         -muM[k]*muM[k]
//                         + 2*muM[k]*muZ
//                         - muZ*stdZ*pdfn
//                         + (varZ - muZ*muZ)*cdfn);
//         cov_ZM[k]       = cdfn * varZ;
//         cov_M_M_sum[k]  = varM[k];
//     }

//     // 2) sum over the K entries
//     float muM_sum = 0.0f, varM_sum = 0.0f;
//     for (int k = 0; k < K; ++k) {
//         muM_sum  += muM[k];
//         varM_sum += varM[k];
//     }

//     // 3) ln-transform statistics
//     float mulnM[10], varlnM[10], cov_M_lnM[10], cov_lnM_lnM_sum[10];
//     for (int k = 0; k < K; ++k) {
//         varlnM[k]          = logf(1.0f + fminf(10.0f, varM[k]/(muM[k]*muM[k])));
//         mulnM[k]           = logf(muM[k]) - 0.5f * varlnM[k];
//         cov_M_lnM[k]       = varlnM[k] * muM[k];
//         cov_lnM_lnM_sum[k] = logf(1.0f + cov_M_M_sum[k] / muM[k] / muM_sum);
//     }

//     float varlnM_sum = logf(1.0f + fminf(10.0f, varM_sum/(muM_sum*muM_sum)));
//     float mulnM_sum  = logf(muM_sum) - 0.5f * varlnM_sum;

//     // 4) prepare inverse transforms
//     float varlnM_sum_inv = 1.0f / varlnM_sum;
//     float mulnM_sum_inv  = 1.0f - mulnM_sum;
//     float muM_sum_inv    = expf(mulnM_sum_inv + 0.5f * varlnM_sum_inv);
//     float varM_sum_inv   = muM_sum_inv*muM_sum_inv * (expf(varlnM_sum_inv) - 1.0f);

//     float cov_M_M_sum_inv[10];
//     for (int k = 0; k < K; ++k) {
//         cov_M_M_sum_inv[k] = (expf(cov_lnM_lnM_sum[k]) - 1.0f)
//                             * muM_sum * muM_sum_inv;
//     }

//     // 5) compute “check A” (log-domain)
//     float mulnA[10], varlnA[10];
//     for (int k = 0; k < K; ++k) {
//         mulnA[k]  = mulnM[k] - mulnM_sum;
//         varlnA[k] = varlnM[k] + varlnM_sum - 2*cov_lnM_lnM_sum[k];
//     }

//     // 6) back to original domain: A ~ LogNormal
//     float muA[10], varA_loc[10], cov_cAlnM[10], cov_AM[10], cov_ZA[10];
//     for (int k = 0; k < K; ++k) {
//         muA[k]        = expf(mulnA[k] + 0.5f*varlnA[k]);
//         cov_cAlnM[k]  = varlnM[k] - cov_lnM_lnM_sum[k];
//     }
//     float muA_sum = 0.0f;
//     for (int k = 0; k < K; ++k) {
//         muA_sum += muA[k];
//     }

//     for (int k = 0; k < K; ++k) {
//         int j = 2 * k;
//         muA[k]    /= muA_sum;
//         varA_loc[k] = muA[k]*muA[k] * (expf(varlnA[k]) - 1.0f);
//         cov_AM[k]   = (expf(cov_cAlnM[k]) - 1.0f) * muA[k] * muM[k];
//         cov_ZA[k]   = fminf(
//                           cov_AM[k] * cov_ZM[k] / varM[k],
//                           sqrtf(varA_loc[k]) * sqrtf(var_z[j + i*output_size])
//                       );
//         // write back *only* the even slots
//         mu_a[j + i*output_size]  = muA[k];
//         var_a[j + i*output_size] = varA_loc[k];
//         jcb[j + i*output_size]   = cov_ZA[k];
//     }
// }

// __global__ void softmax_mean_var_cuda(
//     float const *mu_z,
//     float       *var_z,
//     size_t        output_size,   // e.g. 20
//     int           batch_size,
//     float       *mu_a,
//     float       *jcb,
//     float       *var_a
// ) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= batch_size) return;

//     // we only care about even slots
//     const int K = output_size / 2;  // e.g. 10 when output_size=20

//     // double-precision temporaries
//     double muM_d[10], varM_d[10], cov_ZM_d[10], cov_M_M_sum_d[10];

//     // constants in double
//     constexpr double SQRT_2PI = 2.5066282746310002;
//     constexpr double INV_SQRT2 = 0.7071067811865475;

//     // 1) compute muM, varM, cov_ZM, cov_M_M_sum for even indices
//     for (int k = 0; k < K; ++k) {
//         int j = 2 * k;

//         // load & promote
//         double muZ  = (double)mu_z[j + i*output_size];
//         double varZ = (double)var_z[j + i*output_size];
//         double stdZ = sqrt(varZ);

//         // printf(
//         //     "col=%d  mu_z=%.17g  var_z=%.17g\n",
//         //     j+i*output_size,
//         //     (double)mu_z[j+i*output_size],
//         //     (double)var_z[j+i*output_size]
//         // );

//         // standardized variable
//         double alpha = muZ / stdZ;

//         // PDF and CDF in double
//         double pdfn = (1.0 / SQRT_2PI) * exp(-0.5 * alpha * alpha);
//         double cdfn = 0.5 * erfc(-alpha * INV_SQRT2);

//         // compute moments
//         double E1 = stdZ * pdfn + muZ * cdfn;                             // E[a]
//         double E2 = (muZ*muZ + varZ) * cdfn + muZ * stdZ * pdfn;          // E[a^2]
//         double V  = E2 - E1 * E1;                                          // Var[a]
//         double C  = varZ * cdfn;                                           // Cov(Z,a)

//         // store into double buffers
//         muM_d[k]         = E1;
//         varM_d[k]        = V;
//         cov_ZM_d[k]      = C;
//         cov_M_M_sum_d[k] = V;
//     }

//     // 2) sum muM and varM
//     double muM_sum_d = 0.0, varM_sum_d = 0.0;
//     for (int k = 0; k < K; ++k) {
//         muM_sum_d  += muM_d[k];
//         varM_sum_d += varM_d[k];
//     }

//     // 3) ln-transform statistics
//     double mulnM_d[10], varlnM_d[10], cov_M_lnM_d[10], cov_lnM_lnM_sum_d[10];
//     for (int k = 0; k < K; ++k) {
//         double ratio = varM_d[k] / (muM_d[k] * muM_d[k]);
//         if (ratio > 10.0) ratio = 10.0;
//         varlnM_d[k] = log(1.0 + ratio);
//         mulnM_d[k]  = log(muM_d[k]) - 0.5 * varlnM_d[k];
//         cov_M_lnM_d[k]       = varlnM_d[k] * muM_d[k];
//         cov_lnM_lnM_sum_d[k] = log(1.0 + cov_M_M_sum_d[k] / muM_d[k] / muM_sum_d);
//     }
//     double ratio_sum = varM_sum_d / (muM_sum_d * muM_sum_d);
//     if (ratio_sum > 10.0) ratio_sum = 10.0;
//     double varlnM_sum_d = log(1.0 + ratio_sum);
//     double mulnM_sum_d  = log(muM_sum_d) - 0.5 * varlnM_sum_d;

//     // 4) prepare inverse transforms
//     double varlnM_sum_inv_d = 1.0 / varlnM_sum_d;
//     double mulnM_sum_inv_d  = 1.0 - mulnM_sum_d;
//     double muM_sum_inv_d    = exp(mulnM_sum_inv_d + 0.5 * varlnM_sum_inv_d);
//     double varM_sum_inv_d   = muM_sum_inv_d * muM_sum_inv_d * (exp(varlnM_sum_inv_d) - 1.0);

//     double cov_M_M_sum_inv_d[10];
//     for (int k = 0; k < K; ++k) {
//         cov_M_M_sum_inv_d[k] =
//             (exp(cov_lnM_lnM_sum_d[k]) - 1.0) * muM_sum_d * muM_sum_inv_d;
//     }

//     // 5) compute “check A” in log-domain
//     double mulnA_d[10], varlnA_d[10];
//     for (int k = 0; k < K; ++k) {
//         mulnA_d[k]  = mulnM_d[k] - mulnM_sum_d;
//         varlnA_d[k] = varlnM_d[k] + varlnM_sum_d - 2.0 * cov_lnM_lnM_sum_d[k];
//     }

//     // 6) back to original domain: A ~ LogNormal
//     double muA_d[10], varA_loc_d[10], cov_cAlnM_d[10], cov_AM_d[10], cov_ZA_d[10];
//     double muA_sum_d = 0.0;
//     for (int k = 0; k < K; ++k) {
//         muA_d[k]       = exp(mulnA_d[k] + 0.5 * varlnA_d[k]);
//         cov_cAlnM_d[k] = varlnM_d[k] - cov_lnM_lnM_sum_d[k];
//         muA_sum_d    += muA_d[k];
//     }

//     // 7) normalize and write back to floats
//     for (int k = 0; k < K; ++k) {
//         int j = 2 * k;
//         double mAk = muA_d[k] / muA_sum_d;
//         double vAk = mAk * mAk * (exp(varlnA_d[k]) - 1.0);
//         double cAM = (exp(cov_cAlnM_d[k]) - 1.0) * mAk * muM_d[k];
//         double cZA = cAM * cov_ZM_d[k] / varM_d[k];
//         // clamp to valid covariance range
//         double maxC = sqrt(vAk) * sqrt((double)var_z[j + i*output_size]);
//         if (cZA > maxC) cZA = maxC;

//         mu_a[j + i*output_size]  = (float)mAk;
//         var_a[j + i*output_size] = (float)vAk;
//         jcb[j + i*output_size]   = (float)cZA;

//         // printf(
//         //     "col=%d  mu_a=%.17g  var_a=%.17g  jcb=%.17g\n",
//         //     j+i*output_size,
//         //     (float)mAk,
//         //     (float)vAk,
//         //     (float)cZA
//         // );
//     }
// }


// __global__ void softmax_mean_var_cuda(
//     float const *mu_z,
//     float       *var_z,
//     size_t        output_size,   // e.g. 20
//     int           batch_size,
//     float       *mu_a,
//     float       *jcb,
//     float       *var_a
// ) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= batch_size) return;

//     const int K = output_size / 2;

//     // buffers
//     double muM_d[10], varM_d[10], cov_ZM_d[10], cov_M_M_sum_d[10];
//     double mulnM_d[10], varlnM_d[10], cov_M_lnM_d[10], cov_lnM_lnM_sum_d[10];
//     double cov_M_M_sum_inv_d[10];
//     double mulnA_d[10], varlnA_d[10];
//     double muA_d[10], cov_cAlnM_d[10];
//     double out_mu_d[10], out_var_d[10], out_jcb_d[10];

//     double muM_sum_d = 0.0, varM_sum_d = 0.0, muA_sum_d = 0.0;
//     double varlnM_sum_d, mulnM_sum_d;
//     double varlnM_sum_inv_d, mulnM_sum_inv_d;
//     double muM_sum_inv_d, varM_sum_inv_d;

//     constexpr double SQRT_2PI = 2.5066282746310002;
//     constexpr double INV_SQRT2 = 0.7071067811865475;

//     // 1) compute muM, varM, cov_ZM, cov_M_M_sum
//     for (int k = 0; k < K; ++k) {
//         int j = 2*k;
//         double muZ  = mu_z[j + i*output_size];
//         double varZ = var_z[j + i*output_size];
//         double stdZ = sqrt(varZ);
//         double alpha = muZ / stdZ;
//         double pdfn  = fmax((1.0 / SQRT_2PI) * exp(-0.5 * alpha*alpha), 1e-20);
//         double cdfn  = fmax(0.5 * erfc(-alpha * INV_SQRT2), 1e-20);

//         double E1 = fmax(stdZ*pdfn + muZ*cdfn, 1e-20);
//         double E2 = (muZ*muZ + varZ)*cdfn + muZ*stdZ*pdfn;
//         double V  = fmax(E2 - E1*E1, 1e-6);
//         double C  = varZ * cdfn;

//         muM_d[k]         = E1;
//         varM_d[k]        = V;
//         cov_ZM_d[k]      = C;
//         cov_M_M_sum_d[k] = V;
//         muM_sum_d  += E1;
//         varM_sum_d += V;
//     }

//     // 2) ??? (original code skips directly to ln-transform)

//     // 3) ln-transform
//     for (int k = 0; k < K; ++k) {
//         double ratio = varM_d[k]/(muM_d[k]*muM_d[k]);
//         ratio = fmin(ratio,10.0);
//         varlnM_d[k] = log(1.0+ratio);
//         mulnM_d[k]  = log(muM_d[k]) - 0.5*varlnM_d[k];
//         cov_M_lnM_d[k]       = varlnM_d[k]*muM_d[k];
//         cov_lnM_lnM_sum_d[k] = log(1.0 + cov_M_M_sum_d[k]/muM_d[k]/muM_sum_d);
//     }
//     {
//         double rs = varM_sum_d/(muM_sum_d*muM_sum_d);
//         rs = fmin(rs,10.0);
//         varlnM_sum_d = log(1.0+rs);
//         mulnM_sum_d  = log(muM_sum_d) - 0.5*varlnM_sum_d;
//     }

//     // 4) inverse transforms
//     varlnM_sum_inv_d = 1.0/varlnM_sum_d;
//     mulnM_sum_inv_d  = 1.0 - mulnM_sum_d;
//     muM_sum_inv_d    = exp(mulnM_sum_inv_d + 0.5*varlnM_sum_inv_d);
//     varM_sum_inv_d   = muM_sum_inv_d*muM_sum_inv_d*(exp(varlnM_sum_inv_d)-1.0);
//     for (int k = 0; k < K; ++k) {
//         cov_M_M_sum_inv_d[k] =
//             (exp(cov_lnM_lnM_sum_d[k]) - 1.0)
//             * muM_sum_d * muM_sum_inv_d;
//     }

//     // 5) compute A in log-domain
//     for (int k = 0; k < K; ++k) {
//         mulnA_d[k]  = mulnM_d[k] - mulnM_sum_d;
//         // compute log-variance and clamp to >= 0
//         varlnA_d[k] = varlnM_d[k] + varlnM_sum_d
//                       - 2.0*cov_lnM_lnM_sum_d[k];
//         varlnA_d[k] = fmax(varlnA_d[k], 1e-20f);
//     }

//     // 6) back to original domain
//     for (int k = 0; k < K; ++k) {
//         muA_d[k]       = exp(mulnA_d[k] + 0.5*varlnA_d[k]);
//         cov_cAlnM_d[k] = varlnM_d[k] - cov_lnM_lnM_sum_d[k];
//         muA_sum_d    += muA_d[k];
//     }

//     // 7) compute final locals
//     bool nan_flag = false;
//     for (int k = 0; k < K; ++k) {
//         double mAk = muA_d[k]/muA_sum_d;
//         // guaranteed exp(varlnA_d[k])>=1 => vAk>=0, clamp again for safety
//         double vAk = mAk*mAk*(exp(varlnA_d[k]) - 1.0);
//         vAk = fmax(vAk, 0.0);
//         double cAM = (exp(cov_cAlnM_d[k]) - 1.0)*mAk*muM_d[k];
//         double cZA = cAM * cov_ZM_d[k] / varM_d[k];
//         double maxC = sqrt(vAk)*sqrt(double(var_z[2*k + i*output_size]));
//         if (cZA > maxC) cZA = maxC;

//         out_mu_d[k]  = mAk;
//         out_var_d[k] = vAk;
//         // clamp jacobian if negative
//         out_jcb_d[k] = fmax(cZA, 1e-20);

//         if (isnan(mAk) || isnan(vAk) || isnan(cZA) || vAk < 0.0) {
//             nan_flag = true;
//         }
//     }

//     // 8) dump everything *before* writing to mu_a/var_a/jcb
//     if (nan_flag) {
//         printf("=== THREAD %d: NaN in outputs, dumping all ===\n", i);

//         // Step 1
//         for (int k = 0; k < K; ++k) {
//             int j = 2*k;
//             double muZ = mu_z[j + i*output_size];
//             double varZ = var_z[j + i*output_size];
//             double stdZ = sqrt(varZ);
//             double alpha = muZ/stdZ;
//             double pdfn  = fmaxf((1.0/SQRT_2PI)*exp(-0.5*alpha*alpha), 1e-20);
//             double cdfn  = fmaxf(0.5*erfc(-alpha*INV_SQRT2), 1e-20);
//             printf(
//               "k=%d muZ=%g varZ=%g stdZ=%g alpha=%g pdfn=%g cdfn=%g "
//               "muM=%g varM=%g covZM=%g\n",
//               k, muZ, varZ, stdZ, alpha, pdfn, cdfn,
//               muM_d[k], varM_d[k], cov_ZM_d[k]
//             );
//         }

//         // Steps 2–7
//         printf("muM_sum=%g varM_sum=%g muA_sum=%g\n",
//                muM_sum_d, varM_sum_d, muA_sum_d);
//         for (int k = 0; k < K; ++k) {
//             printf(
//               "k=%d varlnM=%g mulnM=%g covLnLnSum=%g varlnA=%g mulnA=%g "
//               "out_mu=%g out_var=%g out_jcb=%g\n",
//               k,
//               varlnM_d[k], mulnM_d[k], cov_lnM_lnM_sum_d[k],
//               varlnA_d[k], mulnA_d[k],
//               out_mu_d[k], out_var_d[k], out_jcb_d[k]
//             );
//         }
//         printf("=== end dump ===\n");
//     }

//     // 9) now safe to write back
//     for (int k = 0; k < K; ++k) {
//         int j = 2*k;
//         mu_a[j + i*output_size]  = float(out_mu_d[k]);
//         var_a[j + i*output_size] = float(out_var_d[k]);
//         jcb[j + i*output_size]   = float(out_jcb_d[k]);
//     }
// }

__global__ void softmax_mean_var_cuda(
    const float *mu_z,
    float       *var_z,
    size_t        output_size,   // expect even (e.g. 20)
    int           batch_size,
    float       *mu_a,
    float       *jcb,
    float       *var_a
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    // Stability constants in double precision
    constexpr double EPS            = 1e-8;
    constexpr double MIN_CDF        = 1e-20;
    constexpr double MAX_RATIO      = 1e3;
    constexpr double MAX_LOG        = 10.0;
    constexpr double MAX_INV_VARLN  = 50.0;
    constexpr double SQRT_2PI       = 2.5066282746310002;

    const int K = output_size / 2;

    // Intermediate buffers in double
    double muM[10], varM[10], cov_ZM[10], cov_M_M_sum[10];

    // 1) Truncated-normal moments per component
    for (int k = 0; k < K; ++k) {
        int j = 2*k;
        int idx = j + i*output_size;
        double muZ  = (double)mu_z[idx];
        double varZ = fmax((double)var_z[idx], EPS);
        double stdZ = sqrt(varZ);
        double alpha = muZ / stdZ;

        double cdfn = fmax((double)normcdf_cuda((float)alpha), MIN_CDF);
        double pdfn = fmax((1.0/SQRT_2PI) * exp(-0.5*alpha*alpha), MIN_CDF);

        double muMk = stdZ * pdfn + muZ * cdfn;
        muMk = fmax(muMk, EPS);

        double E2    = (muZ*muZ + varZ) * cdfn + muZ * stdZ * pdfn;
        double varMk = fmax(E2 - muMk*muMk, EPS);

        muM[k]          = muMk;
        varM[k]         = varMk;
        cov_ZM[k]       = fmax(cdfn * varZ, 0.0);
        cov_M_M_sum[k]  = varMk;
    }

    // 2) Summations
    double muM_sum = 0.0, varM_sum = 0.0;
    for (int k = 0; k < K; ++k) {
        muM_sum  += muM[k];
        varM_sum += varM[k];
    }
    muM_sum  = fmax(muM_sum, EPS);
    varM_sum = fmax(varM_sum, EPS);

    // 3) Log-domain transforms
    double mulnM[10], varlnM[10], cov_lnM_lnM_sum[10];
    for (int k = 0; k < K; ++k) {
        double ratio = varM[k] / (muM[k]*muM[k]);
        ratio = fmin(fmax(ratio, MIN_CDF), MAX_RATIO);
        varlnM[k] = fmin(fmax(log(1.0 + ratio), MIN_CDF), MAX_LOG);

        double mln = log(muM[k]) - 0.5 * varlnM[k];
        mulnM[k]   = fmin(fmax(mln, -MAX_LOG), MAX_LOG);

        double corr = cov_M_M_sum[k] / (muM[k] * muM_sum);
        corr = fmin(fmax(corr, MIN_CDF), MAX_RATIO);
        cov_lnM_lnM_sum[k] = fmax(log(1.0 + corr), MIN_CDF);
    }
    
    double sum_ratio   = varM_sum / (muM_sum * muM_sum);
    sum_ratio   = fmin(fmax(sum_ratio, MIN_CDF), MAX_RATIO);
    double varlnM_sum = fmin(fmax(log(1.0 + sum_ratio), MIN_CDF), MAX_LOG);
    double mulnM_sum  = fmin(fmax(log(muM_sum) - 0.5 * varlnM_sum,
                                  -MAX_LOG), MAX_LOG);

    // 4) Inverse transforms with overflow protection
    double inv_varln = 1.0 / varlnM_sum;
    inv_varln = fmin(inv_varln, MAX_INV_VARLN);
    double inv_mln   = 1.0 - mulnM_sum;
    double arg       = fmin(fmax(inv_mln + 0.5 * inv_varln, -MAX_LOG), MAX_LOG);
    double muM_inv   = exp(arg);
    double var_exp   = fmax(exp(inv_varln) - 1.0, MIN_CDF);
    double varM_inv  = muM_inv * muM_inv * var_exp;

    double cov_inv[10];
    for (int k = 0; k < K; ++k) {
        double ce = fmax(exp(cov_lnM_lnM_sum[k]) - 1.0, 0.0);
        cov_inv[k] = ce * muM_sum * muM_inv;
    }

    // 5) Compute A in log-domain
    double mulnA[10], varlnA[10];
    for (int k = 0; k < K; ++k) {
        double ma = mulnM[k] - mulnM_sum;
        mulnA[k]   = fmin(fmax(ma, -MAX_LOG), MAX_LOG);

        double va = varlnM[k] + varlnM_sum - 2.0 * cov_lnM_lnM_sum[k];
        varlnA[k]  = fmin(fmax(va, MIN_CDF), MAX_LOG);
    }

    // 6) Back to original domain & write outputs
    double muA[10], muA_sum = 0.0;
    for (int k = 0; k < K; ++k) {
        double e    = fmin(fmax(mulnA[k] + 0.5 * varlnA[k], -MAX_LOG), MAX_LOG);
        muA[k]      = fmax(exp(e), MIN_CDF);
        muA_sum    += muA[k];
    }
    muA_sum = fmax(muA_sum, EPS);

    for (int k = 0; k < K; ++k) {
        int j   = 2*k;
        int idx = j + i*output_size;
        double norm_muA = muA[k] / muA_sum;
        norm_muA        = fmin(fmax(norm_muA, 0.0), 1.0);

        double ve     = fmax(exp(varlnA[k]) - 1.0, 0.0);
        double varA_l = norm_muA * norm_muA * ve;
        varA_l        = fmin(fmax(varA_l, 0.0), norm_muA*(1.0 - norm_muA));

        double ce     = fmax(exp(varlnM[k] - cov_lnM_lnM_sum[k]) - 1.0, 0.0);
        double covAM  = ce * norm_muA * muM[k];
        double varMk  = fmax(varM[k], EPS);

        double covZA  = covAM * cov_ZM[k] / varMk;
        double maxC   = sqrt(varA_l) * sqrt(fmax((double)var_z[idx], EPS));
        covZA         = fmin(fmax(covZA, 0.0), maxC);

        mu_a[idx]     = (float)norm_muA;
        var_a[idx]    = (float)varA_l;
        jcb[idx]      = (float)covZA;
    }
}


__global__ void even_exp_mean_var_cuda(float const *mu_z, float const *var_z,
                                       float const *jcb_z, int num_states,
                                       float *mu_a, float *var_a, float *jcb_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < num_states) {
        if (col % 2 == 0) {
            mu_a[col] = mu_z[col];
            var_a[col] = var_z[col];
            jcb_a[col] = jcb_z[col];
        } else {
            // EXPONENTIAL

            // float tmp_mu = expf(mu_z[col] + 0.5f * var_z[col]);
            // float tmp_var = expf(2.0f * mu_z[col] + var_z[col]) * (expf(var_z[col]) - 1.0f);
            // mu_a[col] = tmp_mu;
            // var_a[col] = tmp_var;
            // jcb_a[col] = tmp_mu;

            // // constants to keep exp() in range of a 32-bit float
            // constexpr double LOGF_MAX = 88.0;    // exp(88) ≃ 1.6e38 < FLT_MAX
            // constexpr double VAR_CLAMP = 80.0;   // leave headroom: exp(var) won't overflow

            // // …

            // // 1) Load & promote
            // double mu_d  = (double)mu_z[col];
            // double var_d = (double)var_z[col];

            // // 2) Clamp variance into [0, VAR_CLAMP]
            // var_d = fmax(var_d, 0.0);
            // var_d = fmin(var_d, VAR_CLAMP);

            // // 3) Compute tmp_mu = exp(mu + 0.5*var) safely
            // double log_mu = mu_d + 0.5 * var_d;
            // log_mu = fmin(log_mu, LOGF_MAX);
            // double tmp_mu_d = exp(log_mu);

            // // 4) Compute tmp_var = exp(2*mu + var) * (exp(var) - 1) safely
            // double log_v = 2.0 * mu_d + var_d;
            // log_v = fmin(log_v, LOGF_MAX);
            // double term1 = exp(log_v);

            // // use expm1 for better accuracy when var_d ≪ 1
            // double term2 = expm1(var_d);

            // double tmp_var_d = term1 * term2;

            // // 5) Cast back to float and store
            // mu_a[col]  = (float) tmp_mu_d;
            // var_a[col] = (float) tmp_var_d;
            // jcb_a[col] = (float) tmp_mu_d;

            // ###########################################//
            // mReLU
            // constexpr float SQRT_2PI = 2.5066282746310002f;
            // float beta = 1.0f;
            // // Reused components for moments calculations
            // float tmp_mu_z = mu_z[col];
            // float std_z = sqrtf(var_z[col]);
            // float alpha = tmp_mu_z / std_z;
            // float pdf_alpha = fmaxf((1.0f / SQRT_2PI) * expf(-0.5f * alpha * alpha), 1e-20f);
            // float cdf_alpha = fmaxf(normcdf_cuda(alpha), 1e-20f);
    
            // // Moments calculations (L. Alric, 2024)
            // float tmp_mu_a = fmaxf(beta * (mu_z[col] * cdf_alpha + std_z * pdf_alpha), 1e-20f);
            // float tmp_var_a = fmaxf(
            //     -tmp_mu_a * tmp_mu_a + beta * beta * (2 * tmp_mu_a * tmp_mu_z -
            //     tmp_mu_z * std_z * pdf_alpha +
            //     (var_z[col] - tmp_mu_z * tmp_mu_z) * cdf_alpha), 1e-6f);

            // if (std::isnan(tmp_var_a) || std::isinf(tmp_var_a) ||
            //     std::isnan(tmp_mu_a) || std::isinf(tmp_mu_a)) {
            //     printf("col=%d  mu_z=%.17g  var_z=%.17g\n",
            //            col,
            //            (double)mu_z[col],
            //            (double)var_z[col]);
            //     printf("tmp_mu_a=%.17g  tmp_var_a=%.17g jcb_a=%.17g\n",
            //            (double)tmp_mu_a,
            //            (double)tmp_var_a,
            //            (double)cdf_alpha);
            // }

            // mu_a[col] = tmp_mu_a;
            // var_a[col] = tmp_var_a;
            // jcb_a[col] = beta * cdf_alpha;

            // constexpr double SQRT_2PI_D = 2.5066282746310002;
            // constexpr double EPS        = 1e-8;
            // constexpr double MIN_PDF    = 1e-20;

            // double beta = 1.0;

            // // Load & clip inputs
            // double muZ  = (double)mu_z[col];
            // double varZ = fmax((double)var_z[col], EPS);
            // double stdZ = sqrt(varZ);

            // // Standardize & compute PDF/CDF with floors
            // double alpha     = muZ / stdZ;
            // double pdf_alpha = fmax((1.0/SQRT_2PI_D) * exp(-0.5*alpha*alpha), MIN_PDF);
            // double cdf_alpha = fmax((double)normcdf_cuda((float)alpha), MIN_PDF);

            // // Compute mean: muA = β·(μZ·Φ + σZ·φ)   then floor
            // double muA_d = beta * (muZ * cdf_alpha + stdZ * pdf_alpha);
            // muA_d = fmax(muA_d, MIN_PDF);

            // // Compute variance: varA = –μA² + β²·(2μAμZ – μZσZφ + (σZ² – μZ²)Φ)   then floor
            // double term = -muA_d*muA_d
            //             + beta*beta * (
            //                 2.0*muA_d*muZ
            //                 - muZ*stdZ*pdf_alpha
            //                 + (varZ - muZ*muZ)*cdf_alpha
            //             );
            // double varA_d = fmax(term, EPS);

            // // Jacobian: jcb = β·Φ, clamp to [0,1]
            // double jcb_d = beta * cdf_alpha;
            // jcb_d = fmin(jcb_d, 1.0);

            // // Optional debug print
            // if (!isfinite(muA_d) || !isfinite(varA_d)) {
            //     printf("col=%d  mu_z=%.7g  var_z=%.7g -> muA=%.7g varA=%.7g\n",
            //         col, muZ, varZ, muA_d, varA_d);
            // }

            // // Store back in float arrays
            // mu_a[col]  = (float)muA_d;
            // var_a[col] = (float)varA_d;
            // jcb_a[col] = (float)jcb_d;


            //###########################################//
            // CELU CLOSED FORM
            // float alpha = 2.0f;
            // float mz = mu_z[col];
            // float var_z_tmp = var_z[col];
            // float sz = sqrtf(var_z_tmp);
            // float z = mz / sz;
            // float z_alpha = z + sz / alpha;
            // float z_2alpha = z + 2.0f * sz / alpha;

            // float phi_z = fmaxf((1.0f / SQRT_2PI) * expf(-0.5f * z * z), 1e-20f);
            // float phi_z_alpha = fmaxf((1.0f / SQRT_2PI) * expf(-0.5f * z_alpha * z_alpha), 1e-20f);
            // float phi_z_2alpha = fmaxf((1.0f / SQRT_2PI) * expf(-0.5f * z_2alpha * z_2alpha), 1e-20f);

            // float Phi_z = fmaxf(normcdf_cuda(z), 1e-20f);
            // float Phi_z_alpha = fmaxf(normcdf_cuda(z_alpha), 1e-20f);
            // float Phi_z_2alpha = fmaxf(normcdf_cuda(z_2alpha), 1e-20f);

            // // Mean (ma_LA)
            // float mean = mz
            // + sz * phi_z
            // - 0.5f * (alpha + mz) * (2.0f - 2.0f * Phi_z)
            // + (alpha * phi_z * (2.0f - 2.0f * Phi_z_alpha)) / (2.0f * phi_z_alpha);

            // // Variance (Sa_LA)
            // float var = mz * mz
            // + sz * phi_z * mz
            // + var_z_tmp
            // + 0.5f * (
            // -2.0f * phi_z * (2.0f - 2.0f * Phi_z_alpha) * alpha * alpha / phi_z_alpha
            // + phi_z * (2.0f - 2.0f * Phi_z_2alpha) * alpha * alpha / phi_z_2alpha
            // + (alpha * alpha - mz * mz - var_z_tmp) * (2.0f - 2.0f * Phi_z)
            // )
            // - mean * mean;

            // // Covariance (covza_LA)
            // float cov = var_z_tmp * (
            // Phi_z + (phi_z * (1.0f - Phi_z_alpha)) / phi_z_alpha
            // );

            // mu_a[col] = fmaxf(mean + alpha, 1e-20f);
            // var_a[col] = fmaxf(var, 1e-6f);
            // jcb_a[col] = cov / var_z_tmp;


            // Configuration constants
            constexpr double EPS        = 1e-6;
            constexpr double MIN_PHI    = 1e-20;
            constexpr double MIN_PHI_ARG= 1e-20;
            constexpr double SQRT_2PI   = 2.5066282746310002;
            constexpr double INV_SQRT2  = 0.7071067811865475;
            constexpr double ALPHA      = 0.5;

            // 1) Clip input variance, compute standardized z
            double mz    = (double)mu_z[col];
            double varz  = fmax((double)var_z[col], EPS);
            double sz    = sqrt(varz);
            double z     = mz / sz;

            // 2) Shifted points for alpha and 2*alpha
            double z_a   = z + sz/ALPHA;
            double z_2a  = z + 2.0*sz/ALPHA;

            // 3) Gaussian PDF and CDF with lower bounds
            auto safe_pdf = [&](double x){
            double p = (1.0/SQRT_2PI) * exp(-0.5*x*x);
            return fmax(p, MIN_PHI);
            };
            auto safe_cdf = [&](double x){
            double c = 0.5 * erfc(-x * INV_SQRT2);
            return fmax(c, MIN_PHI);
            };

            double  φ_z      = safe_pdf(z);
            double  φ_za     = safe_pdf(z_a);
            double  φ_z2a    = safe_pdf(z_2a);
            double  Φ_z      = safe_cdf(z);
            double  Φ_za     = safe_cdf(z_a);
            double  Φ_z2a    = safe_cdf(z_2a);

            // 4) Compute mean in double, then clip
            double mean_d = mz
                + sz * φ_z
                - 0.5*(ALPHA + mz)*(2.0 - 2.0*Φ_z)
                + (ALPHA*φ_z * (2.0 - 2.0*Φ_za)) / (2.0*φ_za);
            // shift by +alpha as in your original, then clip >0
            mean_d = fmax(mean_d + ALPHA, MIN_PHI);

            // 5) Compute variance in double, ensure >= EPS
            double var_d = mz*mz
                + sz*φ_z * mz
                + varz
                + 0.5 * (
                    -2.0*φ_z*(2.0 - 2.0*Φ_za)*ALPHA*ALPHA/φ_za
                    +   φ_z*(2.0 - 2.0*Φ_z2a)*ALPHA*ALPHA/φ_z2a
                    + (ALPHA*ALPHA - mz*mz - varz)*(2.0 - 2.0*Φ_z)
                )
                - mean_d*mean_d;
            var_d = fmax(var_d, EPS);

            // 6) Compute covariance, then jacobian = cov/varz
            double cov_d = varz * (
                Φ_z
                + (φ_z*(1.0 - Φ_za)) / φ_za
            );
            // jcb = cov_d / var_z, clip to [0,1] for safety
            double jcb_d = cov_d / varz;
            jcb_d = fmin(fmax(jcb_d, 0.0), 1.0);

            // 7) Store results back to float
            mu_a[col] = static_cast<float>(mean_d);
            var_a[col]  = static_cast<float>(var_d);
            jcb_a[col]  = static_cast<float>(jcb_d);
        }
    }
}
