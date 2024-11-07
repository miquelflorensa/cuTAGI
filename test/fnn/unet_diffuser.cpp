////////////////////////////////////////////////////////////////////////////////
// File:         unet_diffuser.cpp
// Description:  ...
// Authors:      Miquel Florensa Montilla & Luong-Ha Nguyen & James-A. Goulet
// Created:      October 08, 2024
// Updated:      October 08, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "unet_diffuser.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../../include/activation.h"
#include "../../include/base_output_updater.h"
#include "../../include/conv2d_layer.h"
#include "../../include/convtranspose2d_layer.h"
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/layer_block.h"
#include "../../include/linear_layer.h"
#include "../../include/module.h"
#include "../../include/norm_layer.h"
#include "../../include/pooling_layer.h"
#include "../../include/resnet_block.h"
#include "../include/base_layer_cuda.cuh"
#include "../include/linear_layer_cuda.cuh"
#include "../include/resnet_block_cuda.cuh"
#include "./stb_image_write.h"

const int DIFFUSION_STEPS = 1000;
const int IMG_SIZE = 24;
const int N_CHANNELS = 64;
const int BATCH_SIZE = 256;
const int NEPOCHS = 50;
const float SIGMA_V = 0.1;

const float gain = 0.15f;

LayerBlock create_layer_block(int in_channels, int out_channels, int size_img) {
    return LayerBlock(
        ReLU(), LayerNorm(std::vector<int>({in_channels, size_img, size_img})),
        Conv2d(in_channels, out_channels, 3, false, 1, 1, 1), ReLU(),
        LayerNorm(std::vector<int>({out_channels, size_img, size_img})),
        Conv2d(out_channels, out_channels, 3, false, 1, 1, 1));
}

class UNetDiffusion : public Module {
   private:
    std::vector<float> baralphas;
    std::vector<float> alphas;
    std::vector<float> betas;
    std::normal_distribution<float> normal_dist;
    float mean = 0.0f;
    float std_dev = 1.0f;
    std::mt19937 rng;

   public:
    UNetDiffusion()
        : Module(
              // First Conv layer
              Conv2d(1, N_CHANNELS, 3, false, 1, 1, 1, IMG_SIZE, IMG_SIZE),

              // Level 0 - 2 ResNet blocks
              ResNetBlock(create_layer_block(N_CHANNELS, N_CHANNELS, IMG_SIZE)),
              ResNetBlock(create_layer_block(N_CHANNELS, N_CHANNELS, IMG_SIZE)),

              // Level 1
              ResNetBlock(LayerBlock(
                  // Downsample
                  ReLU(),
                  LayerNorm(std::vector<int>({N_CHANNELS, IMG_SIZE, IMG_SIZE})),
                  Conv2d(N_CHANNELS, N_CHANNELS * 2, 4, false, 2, 1, 1),

                  // Level 1 - 2 ResNet blocks
                  ResNetBlock(create_layer_block(N_CHANNELS * 2, N_CHANNELS * 2,
                                                 IMG_SIZE / 2)),
                  ResNetBlock(create_layer_block(N_CHANNELS * 2, N_CHANNELS * 2,
                                                 IMG_SIZE / 2)),

                  // Level 2
                  ResNetBlock(LayerBlock(
                      // Downsample
                      ReLU(),
                      LayerNorm(std::vector<int>(
                          {N_CHANNELS * 2, IMG_SIZE / 2, IMG_SIZE / 2})),
                      Conv2d(N_CHANNELS * 2, N_CHANNELS * 4, 4, false, 2, 1, 1),

                      // Level 2 - 2 ResNet blocks
                      ResNetBlock(create_layer_block(
                          N_CHANNELS * 4, N_CHANNELS * 4, IMG_SIZE / 4)),
                      ResNetBlock(create_layer_block(
                          N_CHANNELS * 4, N_CHANNELS * 4, IMG_SIZE / 4)),

                      // Level 3
                      //   ResNetBlock(LayerBlock(
                      //       // Downsample
                      //       ReLU(),
                      //       LayerNorm(std::vector<int>(
                      //           {N_CHANNELS * 4, IMG_SIZE / 4, IMG_SIZE /
                      //           4})),
                      //       Conv2d(N_CHANNELS * 4, N_CHANNELS * 8, 4, false,
                      //       2, 1,
                      //              1),

                      //       // Level 3 - 2 ResNet blocks
                      //       ResNetBlock(create_layer_block(
                      //           N_CHANNELS * 8, N_CHANNELS * 8, IMG_SIZE /
                      //           8)),
                      //       ResNetBlock(create_layer_block(
                      //           N_CHANNELS * 8, N_CHANNELS * 8, IMG_SIZE /
                      //           8)),

                      //       // Middle Block
                      //       ResNetBlock(create_layer_block(
                      //           N_CHANNELS * 8, N_CHANNELS * 8, IMG_SIZE /
                      //           8)),
                      //       ResNetBlock(create_layer_block(
                      //           N_CHANNELS * 8, N_CHANNELS * 8, IMG_SIZE /
                      //           8)),

                      //       // Level 3 - 2 ResNet blocks
                      //       ResNetBlock(create_layer_block(
                      //           N_CHANNELS * 8, N_CHANNELS * 8, IMG_SIZE /
                      //           8)),
                      //       ResNetBlock(create_layer_block(
                      //           N_CHANNELS * 8, N_CHANNELS * 8, IMG_SIZE /
                      //           8)),

                      //       // Upsample
                      //       ReLU(),
                      //       LayerNorm(std::vector<int>(
                      //           {N_CHANNELS * 8, IMG_SIZE / 8, IMG_SIZE /
                      //           8})),
                      //       ConvTranspose2d(N_CHANNELS * 8, N_CHANNELS * 4,
                      //       4,
                      //                       false, 2, 1, 1))),

                      // Middle Block
                      ResNetBlock(create_layer_block(
                          N_CHANNELS * 4, N_CHANNELS * 4, IMG_SIZE / 4)),
                      ResNetBlock(create_layer_block(
                          N_CHANNELS * 4, N_CHANNELS * 4, IMG_SIZE / 4)),

                      // Level 2 - 2 ResNet blocks
                      ResNetBlock(create_layer_block(
                          N_CHANNELS * 4, N_CHANNELS * 4, IMG_SIZE / 4)),
                      ResNetBlock(create_layer_block(
                          N_CHANNELS * 4, N_CHANNELS * 4, IMG_SIZE / 4)),

                      // Upsample
                      ReLU(),
                      LayerNorm(std::vector<int>(
                          {N_CHANNELS * 4, IMG_SIZE / 4, IMG_SIZE / 4})),
                      ConvTranspose2d(N_CHANNELS * 4, N_CHANNELS * 2, 4, false,
                                      2, 1, 1))),

                  // Level 1 - 2 ResNet blocks
                  ResNetBlock(create_layer_block(N_CHANNELS * 2, N_CHANNELS * 2,
                                                 IMG_SIZE / 2)),
                  ResNetBlock(create_layer_block(N_CHANNELS * 2, N_CHANNELS * 2,
                                                 IMG_SIZE / 2)),

                  // Upsample
                  ReLU(),
                  LayerNorm(std::vector<int>(
                      {N_CHANNELS * 2, IMG_SIZE / 2, IMG_SIZE / 2})),
                  ConvTranspose2d(N_CHANNELS * 2, N_CHANNELS, 4, true, 2, 1,
                                  1))),

              // Level 0 - 2 ResNet blocks
              //   ResNetBlock(create_layer_block(N_CHANNELS, N_CHANNELS,
              //   IMG_SIZE)),
              ResNetBlock(create_layer_block(N_CHANNELS, N_CHANNELS, IMG_SIZE)),

              // Last Conv layer
              ReLU(),
              LayerNorm(std::vector<int>({N_CHANNELS, IMG_SIZE, IMG_SIZE})),
              Conv2d(N_CHANNELS, 1, 3, true, 1, 1, 1)),

          normal_dist(0.0f, 1.0f),
          rng(std::random_device{}()) {}

    void setup_diffusion() {
        float s = 0.008;
        std::vector<float> timesteps(DIFFUSION_STEPS);
        std::vector<float> schedule(DIFFUSION_STEPS);

        for (int i = 0; i < DIFFUSION_STEPS; ++i) {
            timesteps[i] = static_cast<float>(i);
            schedule[i] =
                std::pow(std::cos((timesteps[i] / DIFFUSION_STEPS + s) /
                                  (1.0f + s) * M_PI / 2.0f),
                         2);
        }

        baralphas.resize(DIFFUSION_STEPS);
        for (int i = 0; i < DIFFUSION_STEPS; ++i) {
            baralphas[i] = schedule[i] / schedule[0];
        }

        betas.resize(DIFFUSION_STEPS);
        alphas.resize(DIFFUSION_STEPS);

        for (int t = 0; t < DIFFUSION_STEPS; ++t) {
            float baralpha_prev = (t == 0) ? baralphas[0] : baralphas[t - 1];
            betas[t] = 1.0f - (baralphas[t] / baralpha_prev);
            alphas[t] = 1.0f - betas[t];
        }
    }

    static std::vector<std::vector<float>> load_mnist_images(
        const std::string& file_path) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + file_path);
        }

        int magic_number, num_images, num_rows, num_cols;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
        file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
        file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));

        magic_number = __builtin_bswap32(magic_number);
        num_images = __builtin_bswap32(num_images);
        num_rows = __builtin_bswap32(num_rows);
        num_cols = __builtin_bswap32(num_cols);

        int num_pixels = num_rows * num_cols;
        std::vector<std::vector<float>> images(num_images,
                                               std::vector<float>(num_pixels));

        for (int i = 0; i < num_images; ++i) {
            for (int j = 0; j < num_pixels; ++j) {
                unsigned char pixel;
                file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
                images[i][j] = static_cast<float>(pixel);
            }
        }

        return images;
    }

    // Function to save image in PNG format
    static void savePNG(const std::string& filename,
                        const std::vector<std::vector<float>>& images,
                        int img_width, int img_height, int num_images) {
        // Define the total width and height of the combined image
        int combined_width = img_width * num_images;
        int combined_height = img_height;

        // Create a buffer for the combined image
        std::vector<unsigned char> combined_image(combined_width *
                                                  combined_height);

        // Loop over each image and copy it into the correct position in the
        // combined buffer
        for (int n = 0; n < num_images; ++n) {
            for (int y = 0; y < img_height; ++y) {
                for (int x = 0; x < img_width; ++x) {
                    // Calculate the index for the combined image buffer
                    int combined_idx = y * combined_width + (n * img_width + x);

                    // Convert float (0-1) to 8-bit grayscale (0-255)
                    combined_image[combined_idx] = static_cast<unsigned char>(
                        images[n][y * img_width + x] * 255.0f);
                }
            }
        }

        // Save the combined image as PNG
        stbi_write_png(filename.c_str(), combined_width, combined_height, 1,
                       combined_image.data(), combined_width);
    }

    // Function to perform bilinear interpolation on a single image
    static std::vector<float> resize_image(const std::vector<float>& image,
                                           int oldWidth, int oldHeight,
                                           int newWidth, int newHeight) {
        std::vector<float> resizedImage(newWidth * newHeight);

        float xRatio = static_cast<float>(oldWidth - 1) / (newWidth - 1);
        float yRatio = static_cast<float>(oldHeight - 1) / (newHeight - 1);

        for (int y = 0; y < newHeight; ++y) {
            for (int x = 0; x < newWidth; ++x) {
                float gx = x * xRatio;
                float gy = y * yRatio;

                int gxi = static_cast<int>(gx);
                int gyi = static_cast<int>(gy);

                float dx = gx - gxi;
                float dy = gy - gyi;

                int topLeft = gyi * oldWidth + gxi;
                int topRight = gyi * oldWidth + std::min(gxi + 1, oldWidth - 1);
                int bottomLeft =
                    std::min(gyi + 1, oldHeight - 1) * oldWidth + gxi;
                int bottomRight = std::min(gyi + 1, oldHeight - 1) * oldWidth +
                                  std::min(gxi + 1, oldWidth - 1);

                float value = (1 - dx) * (1 - dy) * image[topLeft] +
                              dx * (1 - dy) * image[topRight] +
                              (1 - dx) * dy * image[bottomLeft] +
                              dx * dy * image[bottomRight];

                resizedImage[y * newWidth + x] = value;
            }
        }

        return resizedImage;
    }

    // Function to resize a batch of images
    static std::vector<std::vector<float>> resize_images(
        const std::vector<std::vector<float>>& images, int old_width,
        int old_height, int new_width, int new_height) {
        std::vector<std::vector<float>> resized_images;

        for (const auto& image : images) {
            resized_images.push_back(resize_image(image, old_width, old_height,
                                                  new_width, new_height));
        }

        return resized_images;
    }

    void forward(const std::vector<float>& mu_x,
                 const std::vector<float>& var_x = std::vector<float>(),
                 const std::vector<float>& timesteps = std::vector<float>()) {
        /*
         */
        // Batch size
        int batch_size = mu_x.size() / this->layers.front()->get_input_size();

        // Lazy initialization
        if (this->z_buffer_block_size == 0) {
            this->z_buffer_block_size = batch_size;
            this->z_buffer_size = batch_size * this->z_buffer_size;

            init_output_state_buffer();
            if (this->training) {
                init_delta_state_buffer();
            }
        }

        // Reallocate the buffer if batch size changes
        if (batch_size != this->z_buffer_block_size) {
            this->z_buffer_size =
                batch_size * (this->z_buffer_size / this->z_buffer_block_size);
            this->z_buffer_block_size = batch_size;

            this->input_z_buffer->set_size(this->z_buffer_size, batch_size);
            if (this->training) {
                this->input_delta_z_buffer->set_size(this->z_buffer_size,
                                                     batch_size);
                this->output_delta_z_buffer->set_size(this->z_buffer_size,
                                                      batch_size);
            }
        }

        // Merge input data to the input buffer
        this->input_z_buffer->set_input_x(mu_x, var_x, batch_size);

        // Forward pass for all layers
        for (auto& layer : this->layers) {
            auto* current_layer = layer.get();

            if (current_layer->get_layer_type() == LayerType::ResNetBlockCuda) {
                // int num_blocks = this->input_z_buffer->size / batch_size;

                ResNetBlockCuda* resnet_block =
                    dynamic_cast<ResNetBlockCuda*>(current_layer);

                resnet_block->forward_cuda(*this->input_z_buffer,
                                           *this->output_z_buffer,
                                           *this->temp_states, timesteps);
            } else {
                current_layer->forward(*this->input_z_buffer,
                                       *this->output_z_buffer,
                                       *this->temp_states);
            }

#ifdef USE_CUDA
            HiddenStateCuda* cu_input_states =
                dynamic_cast<HiddenStateCuda*>(this->input_z_buffer.get());
            cu_input_states->to_host();

            // Operations on hidden states
            for (int i = 0; i < this->input_z_buffer->size; i++) {
                if (cu_input_states->var_a[i] < 0) {
                    std::cout << "Main Negative variance on layer: "
                              << current_layer->get_layer_name() << std::endl;
                    abort();
                }
            }
            cu_input_states->chunks_to_device(this->input_z_buffer->size);
#endif

            // Swap the pointer holding class
            std::swap(this->input_z_buffer, this->output_z_buffer);
        }

        // Output buffer is considered as the final output of network
        std::swap(this->output_z_buffer, this->input_z_buffer);
    }

    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
    noise(const std::vector<std::vector<float>>& Xbatch,
          const std::vector<int>& t) {
        // Xbatch: vector of vectors, each inner vector has 784 elements
        // (IMG_SIZE*IMG_SIZE) t: vector of timesteps, same length as
        // Xbatch's outer dimension

        std::vector<std::vector<float>> eps(
            Xbatch.size(), std::vector<float>(Xbatch[0].size()));
        std::vector<std::vector<float>> noised(
            Xbatch.size(), std::vector<float>(Xbatch[0].size()));

        for (size_t i = 0; i < Xbatch.size(); ++i) {
            float sqrt_baralpha_t = std::sqrt(baralphas[t[i]]);
            float sqrt_1_minus_baralpha_t = std::sqrt(1.0 - baralphas[t[i]]);

            for (size_t j = 0; j < Xbatch[i].size(); ++j) {
                eps[i][j] = normal_dist(rng);
                // eps[i][j] = 1.3f;
                noised[i][j] = sqrt_baralpha_t * Xbatch[i][j] +
                               sqrt_1_minus_baralpha_t * eps[i][j];
            }
        }

        return {noised, eps};
    }

    std::vector<std::vector<float>> sample_ddpm(int nsamples, int nfeatures) {
        std::normal_distribution<float> dist(0.0, 1.0);
        std::vector<std::vector<float>> x(nsamples,
                                          std::vector<float>(nfeatures));
        std::vector<std::vector<std::vector<float>>> xt;

        for (auto& row : x) {
            for (auto& val : row) {
                val = dist(rng);
            }
        }
        xt.push_back(x);

        std::vector<std::vector<float>> mz;
        std::vector<std::vector<float>> Sv;
        mz.reserve(nsamples);
        Sv.reserve(nsamples);

        for (int t = DIFFUSION_STEPS - 1; t > 0; --t) {
            // Clear mz and Sv
            mz.clear();
            Sv.clear();

            std::vector<float> mu_a_output;
            std::vector<float> var_a_output;

            for (int i = 0; i < nsamples; i += BATCH_SIZE) {
                int batch_size = std::min(BATCH_SIZE, nsamples - i);
                std::vector<std::vector<float>> xbatch(
                    batch_size, std::vector<float>(nfeatures));

                std::vector<float> timestep(
                    batch_size, static_cast<float>(t) /
                                    static_cast<float>(DIFFUSION_STEPS));

                for (int j = 0; j < batch_size; ++j) {
                    xbatch[j] = x[i + j];
                }

                // Flatten xbatch
                std::vector<float> xbatch_flat;
                xbatch_flat.reserve(batch_size * nfeatures);
                for (const auto& row : xbatch) {
                    xbatch_flat.insert(xbatch_flat.end(), row.begin(),
                                       row.end());
                }

                // Forward pass
                forward(xbatch_flat, std::vector<float>(), timestep);

                if (device == "cuda") {
                    output_to_host();
                }

                // Retrive the output
                for (int j = 0; j < batch_size * nfeatures; ++j) {
                    mu_a_output.push_back(output_z_buffer->mu_a[j]);
                    var_a_output.push_back(output_z_buffer->var_a[j]);
                }
            }

            for (int i = 0; i < nsamples; ++i) {
                std::vector<float> mz_i(nfeatures, 0.0f);
                std::vector<float> Sv_i(nfeatures, 0.0f);

                for (int j = 0; j < nfeatures; ++j) {
                    mz_i[j] = mu_a_output[i * nfeatures + j];
                    Sv_i[j] = var_a_output[i * nfeatures + j];
                }

                mz.push_back(mz_i);
                Sv.push_back(Sv_i);
            }

            for (int i = 0; i < nsamples; ++i) {
                for (int j = 0; j < nfeatures; ++j) {
                    x[i][j] =
                        1 / std::sqrt(alphas[t]) *
                        (x[i][j] - (1 - alphas[t]) /
                                       std::sqrt(1 - baralphas[t]) * mz[i][j]);
                    if (t > 1) {
                        float variance = betas[t];
                        float std = std::sqrt(variance);
                        x[i][j] += std * dist(rng);
                    }
                }
            }
            xt.push_back(x);
        }

        return x;
    }

    void train(const std::vector<std::vector<float>>& X) {
        OutputUpdater output_updater(device);

        std::vector<float> var_y(BATCH_SIZE * IMG_SIZE * IMG_SIZE,
                                 SIGMA_V * SIGMA_V);

        std::vector<float> mu_a_output(BATCH_SIZE * IMG_SIZE * IMG_SIZE, 0.0f);
        std::vector<float> var_a_output(BATCH_SIZE * IMG_SIZE * IMG_SIZE, 0.0f);
        int iter = 0;

        for (int epoch = 0; epoch < NEPOCHS; ++epoch) {
            float mse = 0.0;
            for (int i = 0; i < X.size(); i += BATCH_SIZE) {
                std::vector<std::vector<float>> Xbatch(
                    std::min(BATCH_SIZE, static_cast<int>(X.size() - i)));
                for (size_t j = 0; j < Xbatch.size(); ++j) {
                    Xbatch[j] = X[i + j];
                }

                std::uniform_int_distribution<int> dist(0, DIFFUSION_STEPS - 1);
                std::vector<int> timesteps(Xbatch.size());
                for (auto& t : timesteps) {
                    t = dist(rng);
                }

                auto aux = noise(Xbatch, timesteps);

                std::vector<std::vector<float>> noised = aux.first;
                std::vector<std::vector<float>> eps = aux.second;

                // Timestep embedding
                std::vector<float> timesteps_embedding(Xbatch.size());

                for (int i = 0; i < timesteps.size(); ++i) {
                    timesteps_embedding[i] =
                        static_cast<float>(timesteps[i]) / DIFFUSION_STEPS;
                }

                // Flatten noised
                std::vector<float> noised_flat;
                std::vector<float> eps_flat;
                for (const auto& row : noised) {
                    noised_flat.insert(noised_flat.end(), row.begin(),
                                       row.end());
                }
                for (const auto& row : eps) {
                    eps_flat.insert(eps_flat.end(), row.begin(), row.end());
                }

                // // Forward pass
                forward(noised_flat, std::vector<float>(), timesteps_embedding);

                if (device == "cuda") {
                    output_to_host();
                }

                // Retrive the output
                for (int j = 0; j < BATCH_SIZE * IMG_SIZE * IMG_SIZE; ++j) {
                    mu_a_output[j] = output_z_buffer->mu_a[j];
                    var_a_output[j] = output_z_buffer->var_a[j];
                }

                // Output layer
                output_updater.update(*output_z_buffer, eps_flat, var_y,
                                      *input_delta_z_buffer);

                // Backward pass
                backward();
                step();

                // Compute the mean squared error from the output and
                // eps_flat
                for (int j = 0; j < BATCH_SIZE * IMG_SIZE * IMG_SIZE; ++j) {
                    mse += std::pow(mu_a_output[j] - eps_flat[j], 2);
                }

                mse /= BATCH_SIZE * IMG_SIZE * IMG_SIZE;

                std::cout << "Iteration " << i << ": MSE = " << mse
                          << std::endl;
                // Mean variance of the output
                float mean_var = 0.0;
                for (int j = 0; j < BATCH_SIZE * IMG_SIZE * IMG_SIZE; ++j) {
                    mean_var += var_a_output[j];
                    // std::cout << var_a_output[j] << " ";
                }
                mean_var /= BATCH_SIZE * IMG_SIZE * IMG_SIZE;
                std::cout << "Mean variance of the output: " << mean_var
                          << std::endl;

                // if (i % 10000 == 0 && i > 0) {
                //     generate_samples(i);
                // }
            }

            std::cout << "Epoch " << epoch << " completed" << std::endl;
            generate_samples(epoch);
            // Save Model
            std::string model_file =
                "models_unet/model_" + std::to_string(epoch) + ".bin";

            save(model_file);
        }
    }

    void generate_samples(int epoch) {
        auto Xgen = sample_ddpm(10, IMG_SIZE * IMG_SIZE);

        // Unormalize the generated images
        unnormalize(Xgen);

        std::string output_file = "test/fnn/images_unet/output_image_" +
                                  std::to_string(epoch) + ".png";
        // Save the image as a PGM file
        UNetDiffusion::savePNG(output_file, Xgen, IMG_SIZE, IMG_SIZE, 10);

        std::cout << "Generated samples for epoch " << epoch << std::endl;
    }

    void normalize(std::vector<std::vector<float>>& images) {
        // Compute the mean and standard deviation of the images
        float sum = 0.0;
        float sum_sq = 0.0;
        for (const auto& image : images) {
            for (float pixel : image) {
                sum += pixel;
                sum_sq += pixel * pixel;
            }
        }
        mean = sum / (images.size() * images[0].size());
        std_dev = std::sqrt(sum_sq / (images.size() * images[0].size()) -
                            mean * mean);

        // Normalize the images
        for (auto& image : images) {
            for (float& pixel : image) {
                pixel = (pixel - mean) / std_dev;
            }
        }
    }

    void unnormalize(std::vector<std::vector<float>>& images) {
        for (auto& image : images) {
            auto [min_iter, max_iter] =
                std::minmax_element(image.begin(), image.end());
            float min_val = *min_iter;
            float max_val = *max_iter;

            // Print the first image
            for (int i = 0; i < IMG_SIZE; ++i) {
                for (int j = 0; j < IMG_SIZE; ++j) {
                    image[i * IMG_SIZE + j] =
                        (image[i * IMG_SIZE + j] - min_val) /
                        (max_val - min_val);
                }
            }
        }
    }
};

int unet_diffuser() {
    std::string x_file =
        "/home/mf/Documents/TAGI-V/cuTAGI/data/mnist/"
        "train-images-idx3-ubyte";

    UNetDiffusion diffusion;
    diffusion.setup_diffusion();

    std::vector<std::vector<float>> images =
        UNetDiffusion::load_mnist_images(x_file);

    images = UNetDiffusion::resize_images(images, 28, 28, IMG_SIZE, IMG_SIZE);

    // Shuffle the images
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(images.begin(), images.end(), g);

    // Normalize the images
    diffusion.normalize(images);

    // diffusion.unnormalize(images);
    // Save first 10 images as PNG
    UNetDiffusion::savePNG("test/fnn/images_unet/input_image.png", images,
                           IMG_SIZE, IMG_SIZE, 10);

    diffusion.to_device("cuda");
    // diffusion.load("./test/fnn/models_unet/model_10.bin");

    diffusion.train(images);

    // diffusion.generate_samples(10);

    return 0;
}