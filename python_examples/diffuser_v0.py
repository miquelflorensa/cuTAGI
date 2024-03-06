###############################################################################
# File:         diffuser.py
# Description:  Diffusion process using pytagi_v0
# Authors:      Miquel Florensa & Luong-Ha Nguyen & James-A. Goulet
# Created:      February 13, 2024
# Updated:      February 16, 2024
# Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com
#               & james.goulet@polymtl.ca
# License:      This code is released under the MIT License.
###############################################################################
from typing import Union, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import pytagi.metric as metric
from pytagi import NetProp, TagiNetwork
from pytagi import Normalizer as normalizer
from pytagi import Utils, exponential_scheduler


class RegressionMLP(NetProp):
    """Multi-layer perceptron for regression task"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = [1, 1, 1, 1, 1, 1]
        self.nodes = [3, 64, 64, 64, 64, 2]
        self.activations = [0, 4, 4, 4, 4, 0]
        self.batch_size = 2500
        self.sigma_v = 1.0
        self.sigma_v_min: float = 0.1
        self.decay_factor_sigma_v: float = 0.95
        self.cap_factor: float = 1
        self.noise_gain = 1.0
        self.device = "cuda"

class Diffuser:
    def __init__(
        self,
        num_epochs: int,
        batch_size: int,
        X_data: np.ndarray,
        diffusion_steps: int,
        sampling_dim: [int, int],
        alphas: np.ndarray,
        betas: np.ndarray,
    ) -> None:
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        # self.X_data = normalizer(X_data)
        self.X_data = self.normalize(X_data)
        self.diffusion_steps = diffusion_steps
        self.sampling_dim = sampling_dim
        self.alphas = alphas
        self.betas = betas

        self.network = TagiNetwork(RegressionMLP())
        self.nodes_output = 2
        self.sigma_v_ini = 0

        self.s = 0.008
        self.timesteps = np.arange(0, self.diffusion_steps)
        self.schedule = np.cos((self.timesteps / self.diffusion_steps + self.s) / (1 + self.s) * np.pi / 2)**2

        self.baralphas = self.schedule / self.schedule[0]

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize the data"""
        X = (X - X.mean()) / X.std()
        return X

    def noise(self, Xbatch, t):
        eps = np.random.randn(*Xbatch.shape)
        baralphas_t_reshaped = self.baralphas[t][:, np.newaxis]  # Reshape to have one column
        noised = (baralphas_t_reshaped ** 0.5) * Xbatch + ((1 - baralphas_t_reshaped) ** 0.5) * eps
        return noised, eps

    def train(self) -> None:
        """Train the network using TAGI"""

        num_iter = int(self.X_data.shape[0] / self.batch_size)
        pbar = tqdm(range(self.num_epochs))

        # Inputs
        Sx_batch, Sx_f_batch = self.init_inputs(
            self.batch_size,
            self.nodes_output
            )

        # Outputs
        V_batch, ud_idx_batch = self.init_outputs(
            self.batch_size,
            self.sigma_v_ini,
            self.nodes_output
        )

        var_error = []

        for epoch in pbar:
            sum_likelihood = 0
            mse = 0
            mse_accumulated = []
            sigma_v = exponential_scheduler(
                curr_v=3,
                min_v=0.1,
                decaying_factor=0.99,
                curr_iter=epoch,
            )
            V_batch = V_batch * 0.0 + sigma_v**2

            for i in range(num_iter):
                timestep = np.random.randint(0, self.diffusion_steps, self.batch_size)
                x_batch = self.X_data[i * self.batch_size : (i + 1) * self.batch_size]
                noised, eps_batch = self.noise(x_batch, timestep)

                # Add timestep to the noised data
                #Divide timestep by 40 to normalize it
                noised = np.concatenate((noised, timestep.reshape(-1, 1)/40), axis=1)

                # Feed forward
                self.network.feed_forward(noised.flatten(), Sx_batch, Sx_f_batch)

                # Update hidden states
                self.network.state_feed_backward(eps_batch.flatten(), V_batch, ud_idx_batch)

                # Update parameters
                self.network.param_feed_backward()


                ma_pred, Sa_pred = self.network.get_network_predictions()
                #print("ma_pred", len(ma_pred))
                #print("Sa_pred", len(Sa_pred))
                #print("eps_batch", len(eps_batch.flatten()))

                # Compute MSE
                pred = ma_pred
                obs = eps_batch
                #print("pred", pred)
                #print("obs", obs)
                #print("Sa_pred", Sa_pred)
                #mse = metric.mse(pred, obs.flatten())
                #pbar.set_description(f"MSE: {mse:.4f}")
                #LOGlikelihood

                #sum_likelihood += metric.log_likelihood(pred, obs.flatten(), Sa_pred)
                mse_accumulated.append(metric.mse(pred, obs.flatten()))
                mse += metric.mse(pred, obs.flatten())

            mse = mse / num_iter
            pbar.set_description(f"MSE: {mse:.4f}")
            #pbar.set_description(f"Loglikelihood: {sum_likelihood:.4f}")

            var_error.append(np.var(mse_accumulated))

        return var_error


    def sample(self) -> None:
        """Sampling with TAGI"""

        # Generate noisy data
        x = np.random.randn(self.sampling_dim[0], self.sampling_dim[1])

        # Denoising history
        xt = [x]
        vart = [np.ones_like(x)]

        num_iter = int(x.shape[0] / self.batch_size)

        Sx_batch, Sx_f_batch = self.init_inputs(
            self.batch_size,
            self.nodes_output
        )

        # From T to t_0
        for t in range(self.diffusion_steps-1, 0, -1):
            x_temp = []
            var_temp = []

            associated_variance = np.zeros_like(num_iter * self.batch_size)

            for i in range(num_iter):

                x_batch = x[i * self.batch_size : (i + 1) * self.batch_size]

                self.network.feed_forward(np.concatenate((x_batch, np.full((len(x_batch), 1), fill_value=t/40)), axis=1), Sx_batch, Sx_f_batch)
                predicted_noise, predicted_variance = self.network.get_network_predictions()


                # Update Variance associated with the X_t-1
                # \Sigma_{t-1}^X = 1 / \alpha_t * \Sigma_t^X + \frac{(1-\alpha_t)^2}{\alpha_t-\alpha_t^2} * \Sigma_t^\epsilon
                associated_variance = 1 / self.alphas[t] * associated_variance + (1 - self.alphas[t]) ** 2 / (self.alphas[t] - self.alphas[t] ** 2) * predicted_variance


                # Denoise
                #epsilon = z(O)(x,t) + v, v: V ~ N(0, predicted_variance)
                epsilon = predicted_noise + np.random.normal(0, self.sigma_v_ini**2)
                aux = 1 / (self.alphas[t] ** 0.5) * (x_batch.flatten() - (1 - self.alphas[t]) / ((1-self.baralphas[t]) ** 0.5) * epsilon)

                if t > 1:
                    variance = self.betas[t]
                    std = variance ** 0.5
                    aux += std * np.random.randn(*predicted_noise.shape)

                x_temp = np.concatenate((x_temp, aux))
                var_temp = np.concatenate((var_temp, associated_variance))

            #print("x_temp", x_temp.shape)
            #print("x", x.shape)
            # Transform x_temp to x shape
            x_temp = x_temp.reshape(x.shape)
            var_temp = var_temp.reshape(x.shape)

            xt += [x_temp]
            vart += [var_temp]
            x = x_temp
            variance = var_temp

        return x, xt, variance, vart


    def init_inputs(self, batch_size: int, nodes_output) -> Tuple[np.ndarray, np.ndarray]:
        """Initnitalize the covariance matrix for inputs"""
        Sx_batch = np.zeros((batch_size, nodes_output), dtype=np.float32)

        Sx_f_batch = np.array([], dtype=np.float32)

        return Sx_batch, Sx_f_batch

    def init_outputs(self, batch_size: int, sigma_v, nodes_output) -> Tuple[np.ndarray, np.ndarray]:
        """Initnitalize the covariance matrix for outputs"""
        # Outputs
        V_batch = (
            np.zeros((batch_size, nodes_output), dtype=np.float32)
            + sigma_v**2
        )
        ud_idx_batch = np.zeros((batch_size, 1), dtype=np.int32)

        return V_batch, ud_idx_batch