///////////////////////////////////////////////////////////////////////////////
// File:         fnn_diffuser.cpp
// Description:  ...
// Authors:      Miquel Florensa Montilla & Luong-Ha Nguyen & James-A. Goulet
// Created:      October 08, 2024
// Updated:      October 08, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "fnn_diffuser.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <algorithm>
#include <fstream>
#include <iostream>

#include "../../include/activation.h"
#include "../../include/base_output_updater.h"
#include "../../include/conv2d_layer.h"
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/linear_layer.h"
#include "../../include/module.h"
#include "../../include/norm_layer.h"
#include "../../include/pooling_layer.h"
#include "../include/base_layer_cuda.cuh"
#include "../include/linear_layer_cuda.cuh"
#include "./stb_image_write.h"

const int DIFFUSION_STEPS = 500;
const int IMG_SIZE = 28;
const int N_CHANNELS = 6000;
const int BATCH_SIZE = 8;
const int NEPOCHS = 50;
const float SIGMA_V = 0.6;

class MNISTDiffusion : public Module {
   private:
    std::vector<float> baralphas;
    std::vector<float> alphas;
    std::vector<float> betas;
    std::normal_distribution<float> normal_dist;
    float mean = 0.0f;
    float std_dev = 1.0f;
    std::mt19937 rng;

   public:
    MNISTDiffusion()
        : Module(Linear(IMG_SIZE * IMG_SIZE, N_CHANNELS), ReLU(),
                 Linear(N_CHANNELS, IMG_SIZE * IMG_SIZE)),
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

    void forward(const std::vector<float>& mu_x,
                 const std::vector<float>& var_x = std::vector<float>())
    /*
     */
    {
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
            current_layer->forward(*this->input_z_buffer,
                                   *this->output_z_buffer, *this->temp_states);

            // Swap the pointer holding class
            std::swap(this->input_z_buffer, this->output_z_buffer);
        }

        // Output buffer is considered as the final output of network
        std::swap(this->output_z_buffer, this->input_z_buffer);
    }

    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
    noise(const std::vector<std::vector<float>>& Xbatch,
          const std::vector<int>& t) {
        // Xbatch: vector of vectors, each inner vector has 784 elements (28*28)
        // t: vector of timesteps, same length as Xbatch's outer dimension

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
                std::vector<std::vector<float>> timestep(
                    batch_size, std::vector<float>(4, static_cast<float>(t) /
                                                          DIFFUSION_STEPS));

                for (int j = 0; j < batch_size; ++j) {
                    xbatch[j] = x[i + j];
                }

                // Flatten xbatch
                std::vector<float> xbatch_flat;
                for (const auto& row : xbatch) {
                    xbatch_flat.insert(xbatch_flat.end(), row.begin(),
                                       row.end());
                }

                // // Forward pass
                forward(xbatch_flat);

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
        to_device("cuda");
        OutputUpdater output_updater(device);

        std::vector<float> var_y(BATCH_SIZE * IMG_SIZE * IMG_SIZE,
                                 SIGMA_V * SIGMA_V);

        std::vector<float> mu_a_output(BATCH_SIZE * IMG_SIZE * IMG_SIZE, 0.0f);

        for (int epoch = 0; epoch < NEPOCHS; ++epoch) {
            float mse = 0.0;
            for (size_t i = 0; i < X.size(); i += BATCH_SIZE) {
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
                forward(noised_flat);

                if (device == "cuda") {
                    output_to_host();
                }

                // Retrive the output
                for (int j = 0; j < BATCH_SIZE * IMG_SIZE * IMG_SIZE; ++j) {
                    mu_a_output[j] = output_z_buffer->mu_a[j];
                }

                // Output layer
                output_updater.update(*output_z_buffer, eps_flat, var_y,
                                      *input_delta_z_buffer);

                // Backward pass
                backward();
                step();

                // Compute the mean squared error from the output and eps_flat
                for (int j = 0; j < BATCH_SIZE * IMG_SIZE * IMG_SIZE; ++j) {
                    mse += std::pow(mu_a_output[j] - eps_flat[j], 2);
                }

                mse /= BATCH_SIZE * IMG_SIZE * IMG_SIZE;

                std::cout << "Iteration " << i << ": MSE = " << mse
                          << std::endl;
            }

            std::cout << "Epoch " << epoch << " completed" << std::endl;
            generate_samples(epoch);
        }
    }

    void generate_samples(int epoch) {
        auto Xgen = sample_ddpm(10, IMG_SIZE * IMG_SIZE);

        // std::mt19937 rng2;
        // std::uniform_int_distribution<int> dist(0, DIFFUSION_STEPS - 1);
        // std::vector<int> timesteps(images10.size());
        // for (auto& t : timesteps) {
        //     t = 300;
        // }

        // auto aux = diffusion.noise(images10, timesteps);

        // std::vector<std::vector<float>> noised = aux.first;

        // Unormalize the generated images
        unnormalize(Xgen);

        std::string output_file =
            "./test/fnn/output_image_" + std::to_string(epoch) + ".png";
        // Save the image as a PGM file
        MNISTDiffusion::savePNG(output_file, Xgen, 28, 28, 10);

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
            for (int i = 0; i < 28; ++i) {
                for (int j = 0; j < 28; ++j) {
                    image[i * 28 + j] =
                        (image[i * 28 + j] - min_val) / (max_val - min_val);
                }
            }
        }
    }
};

int fnn_diffuser() {
    std::string x_file =
        "/home/mf/Documents/TAGI-V/cuTAGI/data/mnist/"
        "train-images-idx3-ubyte";

    MNISTDiffusion diffusion;
    diffusion.setup_diffusion();

    std::vector<std::vector<float>> images =
        MNISTDiffusion::load_mnist_images(x_file);

    // Shuffle the images
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(images.begin(), images.end(), g);

    // Normalize the images
    diffusion.normalize(images);

    // // Take 10 first images
    // std::vector<std::vector<float>> images10(images.begin(),
    //                                          images.begin() + 10);

    // // Noise images10
    // std::mt19937 rng;
    // std::uniform_int_distribution<int> dist(0, DIFFUSION_STEPS - 1);
    // std::vector<int> timesteps(images10.size());
    // for (auto& t : timesteps) {
    //     t = 30;
    // }
    // auto aux = diffusion.noise(images10, timesteps);

    // std::vector<std::vector<float>> noised = aux.first;

    // // Unormalize the images
    // diffusion.unnormalize(noised);

    // // Save the images as PNG files
    // MNISTDiffusion::savePNG("./test/fnn/noised_image.png", noised, 28, 28,
    // 10);

    diffusion.train(images);

    return 0;
}
