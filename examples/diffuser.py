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
from pytagi import exponential_scheduler
from pytagi.nn import Linear, OutputUpdater, ReLU, Sequential

class Diffuser:
    def __init__(
        self,
        num_epochs: int,
        batch_size: int,
        X_data: np.ndarray,
        diffusion_steps: int,
    ) -> None:
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        # self.X_data = normalizer(X_data)
        self.X_data = X_data
        self.diffusion_steps = diffusion_steps

        self.network = Sequential(
            Linear(3, 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
            Linear(64, 2),
        )

        # self.network.set_threads(8)
        self.network.to_device("cuda")

        # Set noising variances betas as in Nichol and Dariwal paper (https://arxiv.org/pdf/2102.09672.pdf)
        self.s = 0.008
        self.timesteps = np.arange(0, self.diffusion_steps)
        self.schedule = np.cos((self.timesteps / self.diffusion_steps + self.s) / (1 + self.s) * np.pi / 2)**2

        self.baralphas = self.schedule / self.schedule[0]

        self.betas = 1 - self.baralphas / np.concatenate([self.baralphas[0:1], self.baralphas[0:-1]])
        self.alphas = 1 - self.betas


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
                # Divide timestep by 40 to normalize it
                noised = np.concatenate((noised, timestep.reshape(-1, 1)/40), axis=1)

                noised = np.array(noised, dtype=np.float32)

                # Feed Forward
                m_pred, _ = self.network(noised.flatten())


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

                # Compute MSE
                pred = m_pred
                obs = eps_batch

                mse += metric.mse(pred, obs.flatten())

            mse = mse / num_iter
            pbar.set_description(f"MSE: {mse:.4f}")

    def sample(self) -> None:
        """Sampling with TAGI"""

        # Num points of the data
        num_points = 500

        # Num samples for Monte Carlo Sampling
        num_samples = 100

        # Generate noisy data
        mx = np.random.randn(num_points, 2)

        # Coordiantes history
        X_hist = [mx]

        # Variance associated with X
        Sx = np.zeros_like(mx)


        # From T to t_0
        pbar = tqdm(range(self.diffusion_steps-1, 0, -1))
        for t in pbar:
            m_temp = [] # Temporal variable to store the new mean
            S_temp = [] # Temporal variable to store the new variance


            if t < self.diffusion_steps-1:
                # Generate samples from the previous timestep
                accumulated_samples_x = []
                accumulated_samples_y = []
                for i in range(mx.shape[0]):
                    s_x = np.random.normal(mx[i][0], Sx[i][0]**0.5, num_samples)
                    s_y = np.random.normal(mx[i][1], Sx[i][1]**0.5, num_samples)
                    accumulated_samples_x.append(s_x)
                    accumulated_samples_y.append(s_y)

                # Combine x and y samples
                accumulated_samples_x = np.array(accumulated_samples_x).flatten()
                accumulated_samples_y = np.array(accumulated_samples_y).flatten()
                new_samples = np.concatenate((accumulated_samples_x[:, np.newaxis], accumulated_samples_y[:, np.newaxis]), axis=1)

                # Concatenate the new samples with the previous data
                mx = np.concatenate((mx, new_samples), axis=0)

            # Iterations over the data
            num_iter = int(mx.shape[0] / self.batch_size)

            for i in range(num_iter):
                # Create batch
                x_batch = mx[i * self.batch_size : (i + 1) * self.batch_size]

                # Input for TAGI containing x and y coordinates and the timestep
                x = np.concatenate((x_batch, np.full((len(x_batch), 1), fill_value=t/self.diffusion_steps)), axis=1, dtype=np.float32)

                # Forward pass (m_pred: mean epsilon, v_pred: variance epsilon)
                m_pred, v_pred = self.network(x.flatten())


                m_temp = np.concatenate((m_temp, m_pred))
                S_temp = np.concatenate((S_temp, v_pred))


            # Reshape the mean and variance into columns for each coordinate
            m_temp = m_temp.reshape(mx.shape)
            S_temp = S_temp.reshape(mx.shape)

            # Update varinance associated with X
            #print(np.sum(1 / self.alphas[t] * Sx))
            term_1 = np.sum(1 / self.alphas[t] * Sx)
            Sx = np.array(1 / self.alphas[t] * Sx)
            Sz = []
            cov_X_Z = []


            if t < self.diffusion_steps-1:
                # Compute mean and variance in groups of num_samples of m_temp
                for i in range(num_points, m_temp.shape[0], num_samples):
                    var_x = np.var(m_temp[i:i+num_samples, 0])
                    var_y = np.var(m_temp[i:i+num_samples, 1])
                    Sz += [[var_x, var_y]]

                # Compute covariance between every x from 0 to num_points
                # and every group of num_samples of m_temp from num_points to the end
                for i in range(num_points):
                    covariance_x = np.cov(mx[(i+1)*num_samples:(i+2)*num_samples, 0], m_temp[(i+1)*num_samples:(i+2)*num_samples, 0])
                    covariance_y = np.cov(mx[(i+1)*num_samples:(i+2)*num_samples, 1], m_temp[(i+1)*num_samples:(i+2)*num_samples, 1])
                    cov_X_Z += [[covariance_x[0][1], covariance_y[0][1]]]

                m_temp = m_temp[:num_points]
                Sz =  np.array(Sz)
                cov_X_Z = np.array(cov_X_Z)

            else:
                Sz = np.zeros_like(Sx)
                cov_X_Z = np.zeros_like(Sx)

            term_2 = np.sum((1 - self.alphas[t]) ** 2 / (self.alphas[t] * (1 - self.baralphas[t])) * (S_temp[:num_points] + Sz))
            Sx += np.array((1 - self.alphas[t]) ** 2 / (self.alphas[t] * (1 - self.baralphas[t])) * (S_temp[:num_points] + Sz))
            term_3 = np.sum(-2 * (1 - self.alphas[t]) / (self.alphas[t] * (1 - self.baralphas[t])**0.5) * cov_X_Z)
            Sx += np.array(-2 * (1 - self.alphas[t]) / (self.alphas[t] * (1 - self.baralphas[t])**0.5) * cov_X_Z)

            print(-2 * (1 - self.alphas[t]) / (self.alphas[t] * (1 - self.baralphas[t])**0.5))

            if t > self.diffusion_steps-1:
                Sx = S_temp[:num_points]

            # Clip variance to avoid negative values to 10^-6
            #Sx = np.clip(Sx, 1e-6, None)

            print(f"Term 1: {term_1}")
            print(f"Term 2: {term_2}")
            print(f"Term 3: {term_3}")
            print(f"Sum Sx: {np.sum(Sx)}")


            mx = 1 / (self.alphas[t] ** 0.5) * (mx[:num_points] - (1 - self.alphas[t]) / ((1-self.baralphas[t]) ** 0.5) * m_temp)

            if t > 1:
                # Sigma t
                std = self.betas[t] ** 0.5
                mx += std * np.random.randn(*m_temp.shape)

            X_hist += [mx]

        return mx, X_hist, Sx

def main():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import make_swiss_roll

    x, _ = make_swiss_roll(n_samples=100000, noise=0.5)
    # Make two-dimensional to easen visualization
    x = x[:, [0, 2]]

    x = (x - x.mean()) / x.std()

    X_true = np.array(x, dtype=np.float32)

    diffuser = Diffuser(
        num_epochs=100,
        #batch_size=2048,
        batch_size=500,
        X_data=X_true,
        diffusion_steps=40,
    )

    #diffuser.train()
    #diffuser.network.save("test_model/test.bin")

    diffuser.network.load("test_model/test.bin")


    X, X_hist, Sx = diffuser.sample()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(X_true[:, 0], X_true[:, 1], c="b", s=1)
    ax[0].set_title("Original data")
    ax[1].scatter(X[:, 0], X[:, 1], c="r", s=1)
    ax[1].set_title("Diffused data")

    plt.savefig("diffusion_swiss_roll.png")

    # Plot variance with colors inn the points
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sc = ax.scatter(X[:, 0], X[:, 1], c=Sx[:, 0], s=1)
    plt.colorbar(sc)
    plt.title("Variance of the points")
    plt.savefig("variance_swiss_roll.png")

if __name__ == "__main__":
    main()

    # X = np.random.randn(1000)
    # # Y = np.random.randn(1)
    # Z = np.random.randn(1000)

    # varX = np.var(X)
    # varZ = np.var(Z)

    # cov = np.cov(X, Z)
    # corr = np.corrcoef(X, Z)
    # # var = np.var(X)

    # # print(X)
    # # print(Y)
    # print(cov)
    # print(corr[0][1] * varX**0.5 * varZ**0.5)
    # # print(var)