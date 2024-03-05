###############################################################################
# File:         diffuser.py
# Description:  Diffusion process using pytagi
# Authors:      Miquel Florensa & Luong-Ha Nguyen & James-A. Goulet
# Created:      February 13, 2024
# Updated:      February 13, 2024
# Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com
#               & james.goulet@polymtl.ca
# License:      This code is released under the MIT License.
###############################################################################
import numpy as np
from tqdm import tqdm

import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from pytagi import Utils, exponential_scheduler
from sequential import Sequential
from activation import ReLU
from linear import Linear
from output_updater import OutputUpdater

class Diffuser:
    def __init__(
        self,
        num_epochs: int,
        batch_size: int,
        X_data: np.ndarray,
        diffusion_steps: int,
        alphas: np.ndarray,
        betas: np.ndarray,
    ) -> None:
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        # self.X_data = normalizer(X_data)
        self.X_data = X_data/2
        self.diffusion_steps = diffusion_steps
        self.alphas = alphas
        self.betas = betas

        self.network = Sequential(
            Linear(3, 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
            Linear(64, 2),
        )

        # Set noising variances betas as in Nichol and Dariwal paper (https://arxiv.org/pdf/2102.09672.pdf)
        self.s = 0.008
        self.timesteps = np.arange(0, self.diffusion_steps)
        self.schedule = np.cos((self.timesteps / self.diffusion_steps + self.s) / (1 + self.s) * np.pi / 2)**2

        self.baralphas = self.schedule / self.schedule[0]

        # self.network.set_threads(8)
        #self.network.to_device("cuda")


    # Function to add noise to the data
    def noise(self, Xbatch, t):
        eps = np.random.randn(*Xbatch.shape)
        baralphas_t_reshaped = self.baralphas[t][:, np.newaxis]  # Reshape to have one column
        noised = (baralphas_t_reshaped ** 0.5) * Xbatch + ((1 - baralphas_t_reshaped) ** 0.5) * eps
        return noised, eps

    def train(self) -> None:
        """Train the network using TAGI"""

        # Updater for output layer (i.e., equivalent to loss function)
        output_updater = OutputUpdater(self.network.device)

        num_iter = int(self.X_data.shape[0] / self.batch_size)
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:
            mse = 0
            for i in range(num_iter):
                timestep = np.random.randint(0, self.diffusion_steps, self.batch_size)
                x_batch = self.X_data[i * self.batch_size : (i + 1) * self.batch_size]
                noised, eps_batch = self.noise(x_batch, timestep)

                # Add timestep to the noised data
                #Divide timestep by 40 to normalize it
                noised = np.concatenate((noised, timestep.reshape(-1, 1)/40), axis=1)

                # Feed Forward
                self.network(noised.flatten())


                sigma_v = exponential_scheduler(
                    curr_v=0.5,
                    min_v=0.1,
                    decaying_factor=0.95,
                    curr_iter=epoch,
                )


                # Update output layer
                output_updater.update(
                    output_states=self.network.output_z_buffer,
                    mu_obs=eps_batch.flatten(),
                    var_obs=np.ones_like(eps_batch.flatten())*sigma_v**2,
                    delta_states=self.network.input_delta_z_buffer,
                )

                # Update hidden states
                self.network.backward()
                self.network.step()

                ma_pred, Sa_pred = self.network.get_outputs()

                # Compute MSE
                pred = ma_pred
                obs = eps_batch

                mse += metric.mse(pred, obs.flatten())

            mse = mse / num_iter
            pbar.set_description(f"MSE: {mse:.4f}")

    def sample(self) -> None:
        """Sampling with TAGI"""

        # Generate noisy data
        x = np.random.randn(100000, 2)

        # Denoising history
        xt = [x]

        num_iter = int(self.X_data.shape[0] / self.batch_size)

        # From T to t_0
        for t in range(self.diffusion_steps-1, 0, -1):
            x_temp = []
            var_temp = []


            for i in range(num_iter):

                x_batch = x[i * self.batch_size : (i + 1) * self.batch_size]

                # Forward pass with input noise and timestep
                self.network.forward(np.concatenate((x_batch, np.full((len(x_batch), 1), fill_value=t/40)), axis=1))

                # Retrieve the predicted noise
                predicted_noise, predicted_variance = self.network.get_outputs()

                # Denoise
                aux = 1 / (self.alphas[t] ** 0.5) * (x_batch.flatten() - (1 - self.alphas[t]) / ((1-self.baralphas[t]) ** 0.5) * predicted_noise)

                if t > 1:
                    variance = self.betas[t]
                    std = variance ** 0.5
                    aux += std * np.random.randn(*predicted_noise.shape)

                x_temp = np.concatenate((x_temp, aux))
                var_temp = np.concatenate((var_temp, predicted_variance))

            x_temp = x_temp.reshape(x.shape)

            xt += [x_temp]
            x = x_temp
            variance = var_temp

        return x, xt, variance




