///////////////////////////////////////////////////////////////////////////////
// File:         test_dataloader.cpp
// Description:  Header of data loader file for testing
// Authors:      Florensa, Miquel & Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      February 20, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "../include/common.h"
#include "../include/cost.h"
#include "../include/data_transfer_cpu.h"
#include "../include/dataloader.h"
#include "../include/derivative_calcul_cpu.h"
#include "../include/feed_forward_cpu.h"
#include "../include/global_param_update_cpu.h"
#include "../include/indices.h"
#include "../include/net_init.h"
#include "../include/net_prop.h"
#include "../include/param_feed_backward_cpu.h"
#include "../include/state_feed_backward_cpu.h"
#include "../include/struct_var.h"
#include "../include/tagi_network_cpu.h"
#include "../include/task_cpu.h"
#include "../include/user_input.h"
#include "../include/utils.h"
#include "test_utils.h"

bool compare_csv_files(const std::string &file1, const std::string &file2);

Dataloader train_data(std::string problem, TagiNetworkCPU &net,
                      std::string data_path, bool normalize);

Dataloader test_data(std::string problem, TagiNetworkCPU &net,
                     std::string data_path, Dataloader &train_db,
                     bool normalize);

ImageData image_train_data(std::string data_path, std::vector<float> mu,
                           std::vector<float> sigma, int w, int h, int d,
                           HrSoftmax &hrs);

ImageData image_test_data(std::string data_path, std::vector<float> mu,
                          std::vector<float> sigma, int w, int h, int d,
                          HrSoftmax &hrs);

Dataloader test_time_series_datloader(Network &net, std::string mode,
                                      int num_features, std::string data_path,
                                      std::vector<int> output_col,
                                      bool data_norm);