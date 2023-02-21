///////////////////////////////////////////////////////////////////////////////
// File:         test_cpu.cpp
// Description:  Main script to test the CPU implementation of cuTAGI
// Authors:      Florensa, Miquel & Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      February 20, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "test_cpu.h"

std::vector<std::string> read_dates() {
    std::ifstream file("test/data/last_dates.csv");
    std::string line;
    std::getline(file, line);
    std::getline(file, line);
    std::istringstream iss(line);
    std::string value;
    std::vector<std::string> dates;

    while (std::getline(iss, value, ',')) {
        dates.push_back(value);
    }
    return dates;
}

void write_dates(std::vector<std::string> dates, int column, std::string date) {
    std::ofstream file("test/data/last_dates.csv");
    file << "fnn,fnn_hetero,fnn_full_cov,fnn_derivates,cnn,cnn_batch_norm,"
            "autoencoder,lstm,cnn_resnet"
         << std::endl;

    for (int i = 0; i < dates.size(); i++) {
        if (i == column) {
            file << date;
        } else {
            file << dates[i];
        }
        if (i != dates.size() - 1) {
            file << ",";
        }
    }
    file << std::endl;
}

void test_cpu(std::vector<std::string>& user_input_options) {
    std::string reinizialize_test_outputs = "";
    std::string test_architecture = "";
    std::string date = "";

    if (user_input_options.size() == 1 &&
        (user_input_options[0] == "-h" || user_input_options[0] == "--help")) {
        std::cout << "Perform Tests:                    build/main test"
                  << std::endl;
        std::cout << "Run one specific test:            build/main test "
                     "[architecture-name]"
                  << std::endl;
        std::cout
            << "Reinizialize all test outputs:    build/main test -reset all"
            << std::endl;
        std::cout << "Reinizialize one specific outupt: build/main test -reset "
                     "[architecture-name]"
                  << std::endl;
        std::cout << "Available architectures: fnn, fnn_hetero, "
                     "fnn_full_cov, "
                     "fnn_derivates, cnn, cnn_batch_norm, autoencoder, lstm, "
                     "cnn_resnet, lstm, cnn_resnet"
                  << std::endl;
        return;
    } else if (user_input_options.size() > 0) {
        if (user_input_options[0] == "-reset") {
            if (user_input_options.size() == 1) {
                reinizialize_test_outputs = "all";
            } else {
                reinizialize_test_outputs = user_input_options[1];
            }
        } else {
            test_architecture = user_input_options[0];
        }
        std::time_t t = std::time(0);  // get time now
        std::tm* now = std::localtime(&t);
        std::string year = std::to_string(now->tm_year + 1900);
        std::string month = std::to_string(now->tm_mon + 1);
        if (month.size() == 1) month = "0" + month;
        std::string day = std::to_string(now->tm_mday);
        if (day.size() == 1) day = "0" + day;

        date = year + "_" + month + "_" + day;

    } else if (user_input_options.size() == 0) {
        test_architecture = "all";
    } else if (user_input_options.size() > 1) {
        std::cout << "Too many arguments" << std::endl;
        return;
    }

    // Read last test dates
    std::vector<std::string> test_dates = read_dates();

    ////////////////////////////
    //      PERFORM TESTS     //
    ////////////////////////////

    if (test_architecture.size() > 0) {
        if (test_architecture == "all" || test_architecture == "fnn") {
            // Perform test on CPU for the FNN architectures
            std::cout << "Performing FNN tests" << std::endl;

            if (test_fnn_cpu(false, test_dates[0], "fnn", "1D") &&
                test_fnn_cpu(false, test_dates[0], "fnn", "Boston_housing")) {
                std::cout << "[ " << floor((100 / 9) * 1) << "%] "
                          << "FNN tests passed" << std::endl;
            } else {
                std::cout << "[ " << floor((100 / 9) * 1) << "%] "
                          << "FNN tests failed" << std::endl;
            }
        }
    }

    ///////////////////////////////
    // REINIZIALIZE TEST OUTPUTS //
    ///////////////////////////////

    if (reinizialize_test_outputs.size() > 0) {
        if (reinizialize_test_outputs == "all" ||
            reinizialize_test_outputs == "fnn") {
            // Reinizialize test outputs for the FNN architectures
            std::cout << "Reinizializing FNN test outputs" << std::endl;

            test_fnn_cpu(true, date, "fnn", "1D");
            test_fnn_cpu(true, date, "fnn", "Boston_housing");

            // Update de last date of the test
            write_dates(test_dates, 0, date);
        }
    }
}
