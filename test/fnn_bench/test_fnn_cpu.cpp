///////////////////////////////////////////////////////////////////////////////
// File:         test_fnn_cpu.cpp
// Description:  CPU version for testing the FNN
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 15, 2023
// Updated:      January 31, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "test_fnn_cpu.h"

// Specific constant for the network
const std::vector<int> LAYERS = {1, 1, 1, 1};
const std::vector<int> NODES = {13, 10, 15, 1};
const std::vector<int> ACTIVATIONS = {0, 0, 0, 0};
const int BATCH_SIZE = 5;
const int EPOCHS = 50;
const bool NORMALIZE = true;

bool compare_csv_files(const std::string &file1, const std::string &file2) {
    std::ifstream f1(file1);
    std::ifstream f2(file2);

    if (!f1.is_open() || !f2.is_open()) {
        std::cout << "Error opening one of the files." << std::endl;
        return false;
    }

    std::string line1, line2;
    int lineNumber = 1;

    while (std::getline(f1, line1) && std::getline(f2, line2)) {
        if (line1 != line2) {
            std::cout << "Files differ at line " << lineNumber << std::endl;
            return false;
        }
        lineNumber++;
    }

    if (std::getline(f1, line1) || std::getline(f2, line2)) {
        std::cout << "Files have different number of lines." << std::endl;
        return false;
    }

    f1.close();
    f2.close();

    return true;
}

/**
 * @brief Perform a linear regression on the train data.
 *
 * @param[out] net the network to train with specified architecture and
 * parameters
 * @param[in] db the database to train the network on
 */
void regression_train(TagiNetworkCPU &net, auto &db) {
    // Number of data points
    int n_iter = db.num_data / net.prop.batch_size;
    std::vector<int> data_idx = create_range(db.num_data);

    // Initialize the data's variables
    std::vector<float> x_batch(net.prop.batch_size * net.prop.n_x, 0);
    std::vector<float> Sx_batch(net.prop.batch_size * net.prop.n_x,
                                pow(net.prop.sigma_x, 2));
    std::vector<float> Sx_f_batch;
    std::vector<float> y_batch(net.prop.batch_size * net.prop.n_y, 0);
    std::vector<float> V_batch(net.prop.batch_size * net.prop.n_y,
                               pow(net.prop.sigma_v, 2));
    std::vector<int> batch_idx(net.prop.batch_size);
    std::vector<int> idx_ud_batch(net.prop.nye * net.prop.batch_size, 0);

    for (int e = 0; e < EPOCHS; e++) {
        if (e > 0) {
            // Decay observation noise
            decay_obs_noise(net.prop.sigma_v, net.prop.decay_factor_sigma_v,
                            net.prop.sigma_v_min);
        }

        std::vector<float> V_batch(net.prop.batch_size * net.prop.n_y,
                                   pow(net.prop.sigma_v, 2));

        for (int i = 0; i < n_iter; i++) {
            // Load data
            get_batch_idx(data_idx, i * net.prop.batch_size,
                          net.prop.batch_size, batch_idx);
            get_batch_data(db.x, batch_idx, net.prop.n_x, x_batch);
            get_batch_data(db.y, batch_idx, net.prop.n_y, y_batch);

            // Feed forward
            net.feed_forward(x_batch, Sx_batch, Sx_f_batch);

            // Feed backward for hidden states
            net.state_feed_backward(y_batch, V_batch, idx_ud_batch);

            // Feed backward for parameters
            net.param_feed_backward();
        }
    }
}

/**
 * @brief Perform a linear regression on the test data.
 *
 * @param[out] net the network to test with specified architecture and
 * parameters
 * @param[in] db the database to test the network on
 */
void regression_test(TagiNetworkCPU &net, auto &db) {
    std::cout << "Testing...\n";

    // Number of data points
    int n_iter = db.num_data / net.prop.batch_size;
    int derivative_layer = 0;

    std::vector<int> data_idx = create_range(db.num_data);

    // Initialize the data's variables
    std::vector<float> x_batch(net.prop.batch_size * net.prop.n_x, 0);
    std::vector<float> Sx_batch(net.prop.batch_size * net.prop.n_x,
                                pow(net.prop.sigma_x, 2));
    std::vector<float> Sx_f_batch;
    std::vector<float> y_batch(net.prop.batch_size * net.prop.n_y, 0);
    std::vector<float> V_batch(net.prop.batch_size * net.prop.n_y,
                               pow(net.prop.sigma_v, 2));
    std::vector<int> batch_idx(net.prop.batch_size);
    std::vector<int> idx_ud_batch(net.prop.nye * net.prop.batch_size, 0);

    // Output results
    std::vector<float> ma_batch_out(net.prop.batch_size * net.prop.n_y, 0);
    std::vector<float> Sa_batch_out(net.prop.batch_size * net.prop.n_y, 0);
    std::vector<float> ma_out(db.num_data * net.prop.n_y, 0);
    std::vector<float> Sa_out(db.num_data * net.prop.n_y, 0);

    // Derivative results for the input layers
    std::vector<float> mdy_batch_in, Sdy_batch_in, mdy_in, Sdy_in;
    if (net.prop.collect_derivative) {
        mdy_batch_in.resize(net.prop.batch_size * net.prop.n_x, 0);
        Sdy_batch_in.resize(net.prop.batch_size * net.prop.n_x, 0);
        mdy_in.resize(db.num_data * net.prop.n_x, 0);
        Sdy_in.resize(db.num_data * net.prop.n_x, 0);
    }

    int mt_idx = 0;

    // Prediction
    for (int i = 0; i < n_iter; i++) {
        mt_idx = i * net.prop.batch_size * net.prop.n_y;

        // Load data
        get_batch_idx(data_idx, i * net.prop.batch_size, net.prop.batch_size,
                      batch_idx);
        get_batch_data(db.x, batch_idx, net.prop.n_x, x_batch);
        get_batch_data(db.y, batch_idx, net.prop.n_y, y_batch);

        // Feed forward
        net.feed_forward(x_batch, Sx_batch, Sx_f_batch);

        // Derivatives
        if (net.prop.collect_derivative) {
            compute_network_derivatives_cpu(net.prop, net.theta, net.state,
                                            derivative_layer);
            get_input_derv_states(net.state.derv_state.md_layer,
                                  net.state.derv_state.Sd_layer, mdy_batch_in,
                                  Sdy_batch_in);
            update_vector(mdy_in, mdy_batch_in, mt_idx, net.prop.n_x);
            update_vector(Sdy_in, Sdy_batch_in, mt_idx, net.prop.n_x);
        }

        // Get hidden states for output layers
        output_hidden_states(net.state, net.prop, ma_batch_out, Sa_batch_out);

        // Update the final hidden state vector for last layer
        update_vector(ma_out, ma_batch_out, mt_idx, net.prop.n_y);
        update_vector(Sa_out, Sa_batch_out, mt_idx, net.prop.n_y);
    }
    // Denormalize data
    std::vector<float> sy_norm(db.y.size(), 0);
    std::vector<float> my(sy_norm.size(), 0);
    std::vector<float> sy(sy_norm.size(), 0);
    std::vector<float> y_test(sy_norm.size(), 0);

    // Compute log-likelihood
    for (int k = 0; k < db.y.size(); k++) {
        sy_norm[k] = pow(Sa_out[k] + pow(net.prop.sigma_v, 2), 0.5);
    }
    denormalize_mean(ma_out, db.mu_y, db.sigma_y, net.prop.n_y, my);
    denormalize_mean(db.y, db.mu_y, db.sigma_y, net.prop.n_y, y_test);
    denormalize_std(sy_norm, db.mu_y, db.sigma_y, net.prop.n_y, sy);

    // Compute metrics
    auto mse = mean_squared_error(my, y_test);
    auto log_lik = avg_univar_log_lik(my, y_test, sy);
}

/**
 * @brief Train the data
 *
 * @param problem contains a string of the problem name
 * @param net contains the network
 * @param data_path contains the path to the data
 */
auto train_data(std::string problem, TagiNetworkCPU &net,
                std::string data_path) {
    // Directory of the data
    std::string x_dir, y_dir;

    // Num of data in each set
    int num_train_data;

    if (problem == "Boston_housing") {
        x_dir = data_path + "/Boston_housing/x_train.csv";
        y_dir = data_path + "/Boston_housing/y_train.csv";
        num_train_data = 455;
    } else {
        num_train_data = 0;
        x_dir = data_path + "/1D/x_train.csv";
        y_dir = data_path + "/1D/y_train.csv";
    }

    std::vector<std::string> x_path;
    std::vector<std::string> y_path;
    x_path.push_back(x_dir);
    y_path.push_back(y_dir);

    // Train data

    // Initialize the mu and sigma
    std::vector<float> mu_x, sigma_x, mu_y, sigma_y;
    return get_dataloader(x_path, y_path, mu_x, sigma_x, mu_y, sigma_y,
                          num_train_data, net.prop.n_x, net.prop.n_y,
                          NORMALIZE);
}

/**
 * @brief Test the data
 *
 * @param problem contains a string of the problem name
 * @param net contains the network
 * @param data_path contains the path to the data
 * @param train_db contains the training data
 */
auto test_data(std::string problem, TagiNetworkCPU &net, std::string data_path,
               auto &train_db) {
    // Directory of the data
    std::string x_dir, y_dir;

    // Num of data in each set
    int num_test_data;

    if (problem == "Boston_housing") {
        x_dir = data_path + "/Boston_housing/x_test.csv";
        y_dir = data_path + "/Boston_housing/y_test.csv";
        num_test_data = 51;
    } else {
        num_test_data = 0;
        x_dir = data_path + "/1D/x_test.csv";
        y_dir = data_path + "/1D/y_test.csv";
    }

    std::vector<std::string> x_path;
    std::vector<std::string> y_path;
    x_path.push_back(x_dir);
    y_path.push_back(y_dir);

    // Test data
    return get_dataloader(x_path, y_path, train_db.mu_x, train_db.sigma_x,
                          train_db.mu_y, train_db.sigma_y, num_test_data,
                          net.prop.n_x, net.prop.n_y, NORMALIZE);
}

/**
 * @brief Test the FNN network
 *
 */
void test_fnn_cpu() {
    // Create TAGI network
    Network net;

    net.layers = LAYERS;
    net.nodes = NODES;
    net.activations = ACTIVATIONS;
    net.batch_size = BATCH_SIZE;

    TagiNetworkCPU tagi_net(net);

    // Put it in the main test file
    SavePath path;
    path.curr_path = get_current_dir();
    std::string data_path = path.curr_path + "/test/data";
    std::string init_param_path =
        path.curr_path +
        "/test/fnn_bench/data/"
        "2023_01_26_init_param_fnn_bench_Boston_housing.csv";
    std::string opt_param_path =
        path.curr_path +
        "/test/fnn_bench/data/"
        "2023_01_26_opt_param_fnn_bench_Boston_housing.csv";
    std::string opt_param_path_2 =
        path.curr_path +
        "/test/fnn_bench/data/"
        "2023_01_26_opt_2_param_fnn_bench_Boston_housing.csv";
    std::string forward_states_path =
        path.curr_path +
        "/test/fnn_bench/data/"
        "2023_01_26_forward_hidden_states_fnn_bench_Boston_housing.csv";
    std::string backward_states_path =
        path.curr_path +
        "/test/fnn_bench/data/"
        "2023_01_26_backward_hidden_states_fnn_bench_Boston_housing.csv";

    // Train data
    auto train_db = train_data("Boston_housing", tagi_net, data_path);
    // Test data
    auto test_db = test_data("Boston_housing", tagi_net, data_path, train_db);

    // Read the initial parameters (see tes_utils.cpp for more details)
    read_params(init_param_path, tagi_net.theta);

    // Train the network
    regression_train(tagi_net, train_db);

    // Test the network
    regression_test(tagi_net, test_db);

    bool save_new_values = false;

    if (save_new_values) {
        // Write the parameters and hidden states
        write_params(opt_param_path, tagi_net.theta);

        write_forward_hidden_states(forward_states_path, tagi_net.state);

        write_backward_hidden_states(backward_states_path, tagi_net,
                                     net.layers.size() - 2);
    } else {
        // Read optimal parameters for the initial ones
        // Param optimal_theta;
        // read_params(opt_param_path, optimal_theta);

        write_params(opt_param_path_2, tagi_net.theta);

        // Compare optimal values with the ones we got
        if (compare_csv_files(opt_param_path, opt_param_path_2)) {
            std::cout << "\033[1;32mTEST FOR FNN HAS PASSED\033[0m\n"
                      << std::endl;
        } else {
            std::cout << "\033[1;31mTEST FOR FNN HAS FAILED\033[0m\n"
                      << std::endl;
        }

        if (remove(opt_param_path_2.c_str()) != 0)
            std::cout << "Error deleting " << opt_param_path_2 << std::endl;
    }

    std::cout << "\033[1;33mTEST FOR FNN HAS FINISHED\033[0m\n" << std::endl;
}