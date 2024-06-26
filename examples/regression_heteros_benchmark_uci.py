import fire
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time

import pytagi.metric as metric
from examples.data_loader import RegressionDataLoader
from examples.time_series_forecasting import PredictionViz
from pytagi import Normalizer
from pytagi.nn import Linear, NoiseOutputUpdater, ReLU, Sequential, AGVI

from sklearn.model_selection import train_test_split


def predict(batch_size, train_dtl, test_dtl, net, cuda):
    test_batch_iter = test_dtl.create_data_loader(batch_size, shuffle=True)
    mu_preds = []
    var_preds = []
    y_test = []
    x_test = []

    for x, y in test_batch_iter:
        # Predicion
        m_pred, v_pred = net(x)

        if cuda:
            aux = np.exp(m_pred[1::2] + 0.5 * v_pred[1::2])
            var_preds.extend(aux + v_pred[::2])
        else:
            var_preds.extend(v_pred[::2] + m_pred[1::2])

        mu_preds.extend(m_pred[::2])

        x_test.extend(x)
        y_test.extend(y)

    mu_preds = np.array(mu_preds)
    std_preds = np.array(var_preds) ** 0.5
    y_test = np.array(y_test)
    x_test = np.array(x_test)

    mu_preds = Normalizer.unstandardize(mu_preds, train_dtl.y_mean, train_dtl.y_std)
    std_preds = Normalizer.unstandardize_std(std_preds, train_dtl.y_std)

    #x_test = Normalizer.unstandardize(x_test, train_dtl.x_mean, train_dtl.x_std)
    y_test = Normalizer.unstandardize(y_test, train_dtl.y_mean, train_dtl.y_std)

    # Compute log-likelihood
    mse = metric.mse(mu_preds, y_test)
    log_lik = metric.log_likelihood(
        prediction=mu_preds, observation=y_test, std=std_preds
    )

    return mse**0.5, log_lik



def run_benchmark(data_name: str, num_epochs, batch_size, n_splits):
    """Run benchmark for each UCI dataset"""
    rmse_list = []
    log_lik_list = []
    times_list = []

    num_nodes = 50
    if data_name == "Protein":
        num_nodes = 100
        n_splits = 5
    if data_name == "Yatch":
        batch_size = 5


    for split in range(n_splits):
        # Check if x_train, y_train, x_test, y_test files are available
        if os.path.exists("./data/UCI/" + data_name + "/x_train.csv"):
            # Remove the files
            os.remove("./data/UCI/" + data_name + "/x_train.csv")
            os.remove("./data/UCI/" + data_name + "/y_train.csv")
            os.remove("./data/UCI/" + data_name + "/x_test.csv")
            os.remove("./data/UCI/" + data_name + "/y_test.csv")
            os.remove("./data/UCI/" + data_name + "/x_val.csv")
            os.remove("./data/UCI/" + data_name + "/y_val.csv")


            data_file = "./data/UCI/" + data_name +  "/data/data.txt"

            # Read and split the data
            data = np.loadtxt(data_file)

            # Split the data into training and testing
            #np.random.shuffle(data)
            # Read index split
            train_index = np.loadtxt("./data/UCI/" + data_name + "/data/index_train_" + str(split) + ".txt")
            test_index = np.loadtxt("./data/UCI/" + data_name + "/data/index_test_" + str(split) + ".txt")

            train_data = data[train_index.astype(int)]
            test_data = data[test_index.astype(int)]

            train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

            # Save the training and testing data in x and y files
            np.savetxt("./data/UCI/" + data_name + "/x_train.csv", train_data[:, :-1], delimiter=",")
            np.savetxt("./data/UCI/" + data_name + "/y_train.csv", train_data[:, -1], delimiter=",")
            np.savetxt("./data/UCI/" + data_name + "/x_test.csv", test_data[:, :-1], delimiter=",")
            np.savetxt("./data/UCI/" + data_name + "/y_test.csv", test_data[:, -1], delimiter=",")
            np.savetxt("./data/UCI/" + data_name + "/x_val.csv", val_data[:, :-1], delimiter=",")
            np.savetxt("./data/UCI/" + data_name + "/y_val.csv", val_data[:, -1], delimiter=",")

        # Dataset
        x_train_file = "./data/UCI/" + data_name + "/x_train.csv"
        y_train_file = "./data/UCI/" + data_name + "/y_train.csv"
        x_test_file = "./data/UCI/" + data_name + "/x_test.csv"
        y_test_file = "./data/UCI/" + data_name + "/y_test.csv"
        x_val_file = "./data/UCI/" + data_name + "/x_val.csv"
        y_val_file = "./data/UCI/" + data_name + "/y_val.csv"


        train_dtl = RegressionDataLoader(x_file=x_train_file, y_file=y_train_file)
        test_dtl = RegressionDataLoader(
            x_file=x_test_file,
            y_file=y_test_file,
            x_mean=train_dtl.x_mean,
            x_std=train_dtl.x_std,
            y_mean=train_dtl.y_mean,
            y_std=train_dtl.y_std,
        )
        val_dtl = RegressionDataLoader(
            x_file=x_val_file,
            y_file=y_val_file,
            x_mean=train_dtl.x_mean,
            x_std=train_dtl.x_std,
            y_mean=train_dtl.y_mean,
            y_std=train_dtl.y_std,
        )

        # Read num_coulmns in x_train
        x_train_data = pd.read_csv(x_train_file, skiprows=1, delimiter=",", header=None)
        num_inputs = x_train_data.shape[1]

        cuda = False

        if cuda:
            net = Sequential(
                Linear(num_inputs, num_nodes),
                ReLU(),
                Linear(num_nodes, num_nodes),
                ReLU(),
                Linear(num_nodes, 2),
            )
            net.to_device("cuda")
        else:
            # Network
            net = Sequential(
                Linear(num_inputs, num_nodes),
                ReLU(),
                Linear(num_nodes, num_nodes),
                ReLU(),
                Linear(num_nodes, 2),
                AGVI(),
            )
            #net.set_threads(8)


        out_updater = NoiseOutputUpdater(net.device)
        delta = 0.01
        patience = 5
        best_log_lik = -np.inf
        best_rmse = np.inf

        start_time = time.time()

        # -------------------------------------------------------------------------#
        # Training
        #pbar = tqdm(range(num_epochs), desc="Training Progress")
        for epoch in range(num_epochs):
            batch_iter = train_dtl.create_data_loader(batch_size)

            for x, y in batch_iter:
                # Feed forward
                m_pred, _ = net(x)

                # Update output layer
                out_updater.update(
                    output_states=net.output_z_buffer,
                    mu_obs=y,
                    delta_states=net.input_delta_z_buffer,
                )

                # Feed backward
                net.backward()
                net.step()

                # Training metric
                pred = Normalizer.unstandardize(m_pred, train_dtl.y_mean, train_dtl.y_std)
                obs = Normalizer.unstandardize(y, train_dtl.y_mean, train_dtl.y_std)

                # Even positions correspond to Z_out
                pred = pred[::2]


            rmse, log_lik = predict(batch_size, train_dtl, val_dtl, net, cuda)

            # Early stopping
            if (log_lik - best_log_lik) > delta:
                best_log_lik = log_lik
                counter = 0
            elif (log_lik - best_log_lik) < delta:
                counter += 1
                if counter == patience:
                    print(f"Early stopping at epoch {epoch}")
                    break




        times_list.append(time.time() - start_time)

        # -------------------------------------------------------------------------#
        # Testing
        rmse, log_lik = predict(batch_size, train_dtl, test_dtl, net, cuda)

        rmse_list.append(rmse)
        log_lik_list.append(log_lik)

        # print("#############")
        print(f"RMSE           : {rmse: 0.3f}")
        print(f"Log-likelihood: {log_lik: 0.3f}")

    # Compute +- 1 std
    print(f"RMSE           : {np.mean(rmse_list): 0.3f} +- {np.std(rmse_list): 0.3f}")
    print(f"Log-likelihood: {np.mean(log_lik_list): 0.3f} +- {np.std(log_lik_list): 0.3f}")
    print(f"Time           : {np.mean(times_list): 0.3f} +- {np.std(times_list): 0.3f}")
    print("#############")



def main():
    """Run benchmark for regression tasks on UCI dataset"""

    data_names = ["Boston_housing","Concrete", "Energy", "Yacht", "Wine", "Kin8nm","Naval", "Power-plant","Protein"]
    #data_names = ["Protein"]

    for data_name in data_names:
        print(f"Running benchmark for {data_name}")
        run_benchmark(data_name, num_epochs=100, batch_size=10, n_splits=20)

if __name__ == "__main__":
    fire.Fire(main)
