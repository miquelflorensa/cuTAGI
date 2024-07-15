///////////////////////////////////////////////////////////////////////////////
// File:         output_updater_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 27, 2023
// Updated:      May 21, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "base_output_updater.h"
#include "data_struct_cuda.cuh"

class OutputUpdaterCuda : public BaseOutputUpdater {
   public:
    unsigned int num_cuda_threads = 16;

    OutputUpdaterCuda();
    ~OutputUpdaterCuda() = default;

    void set_num_cuda_threads(unsigned int num_threads);

    void update_output_delta_z(BaseHiddenStates &output_states,
                               BaseObservation &obs,
                               BaseDeltaStates &delta_states) override;

    void update_selected_output_delta_z(BaseHiddenStates &output_states,
                                        BaseObservation &obs,
                                        BaseDeltaStates &delta_states) override;

    std::string get_name() const override { return "OutputUpdaterCuda"; };
};

class NoiseOutputUpdaterCuda : public BaseOutputUpdater {
   public:
    unsigned int num_cuda_threads = 16;

    NoiseOutputUpdaterCuda();
    ~NoiseOutputUpdaterCuda() = default;

    void set_num_cuda_threads(unsigned int num_threads);

    void update_output_delta_z_noise(BaseHiddenStates &output_states,
                                     BaseObservation &obs,
                                     BaseDeltaStates &delta_states) override;

    std::string get_name() const override { return "NoiseOutputUpdaterCuda"; };
};