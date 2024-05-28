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
from pytagi.nn import Linear, NoiseOutputUpdater, ReLU, Sequential

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
            Linear(64, 2*2),
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
        output_updater = NoiseOutputUpdater(self.network.device)

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
                m_pred = m_pred[::2]

                # Update output layer
                output_updater.update(
                    output_states=self.network.output_z_buffer,
                    mu_obs=eps_batch.flatten(),
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

        term1 = []
        term2_a = []
        term2_b = []
        term2_c = []
        term3 = []

        # Num points of the data
        num_points = 1000

        # Num samples for Monte Carlo Sampling
        num_samples = 10000

        # Generate noisy data
        mx = np.random.randn(num_points, 2) # 500 x 2
        mx = np.array(mx, dtype=np.float32)

        # Coordiantes history
        mx_hist = [mx]

        # Variance associated with X
        Sx = np.zeros_like(mx) # 500 x 2

        mx_samples = np.zeros((num_points * num_samples, 2)) # (500 * 100) x 2


        # From T to t_0
        pbar = tqdm(range(self.diffusion_steps-1, 0, -1))
        for t in pbar:
            mz = [] # Mean Z output
            Sz = [] # Variance Z output
            Sv = [] # Variance given by V
            mz_samples = [] # Mean Z output for MC Sampling
            Sz_samples = [] # Empirical varaince obtained with MC Sampling
            cov_X_Z_samples = [] # Covariance between X_samples and Z_samples


            # Iterations over the data
            num_iter = int(mx.shape[0] / self.batch_size)

            for i in range(num_iter):
                # Create mini-batch
                x_batch = mx[i * self.batch_size : (i + 1) * self.batch_size]

                # Input for TAGI containing x and y coordinates and the timestep
                x = np.concatenate((x_batch, np.full((len(x_batch), 1), fill_value=t/self.diffusion_steps)), axis=1, dtype=np.float32)
                # x = 500 x 3

                # Forward pass (m_pred: mean epsilon, v_pred: variance epsilon)
                m_pred, v_pred = self.network(x.flatten())
                # m_pred = 1000

                mv = np.exp(m_pred[1::2] + 0.5 * v_pred[1::2])
                m_pred = m_pred[::2]
                v_pred = v_pred[::2]

                mz.extend(m_pred)
                Sz.extend(v_pred)
                Sv.extend(mv)

            # MC Sampling
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
                mx_samples = np.concatenate((accumulated_samples_x[:, np.newaxis], accumulated_samples_y[:, np.newaxis]), axis=1)

                num_iter_samples = int(num_samples * num_points / self.batch_size)
                # Pass mx_samples through the network
                for i in range(num_iter_samples):
                    x_batch = mx_samples[i * self.batch_size : (i + 1) * self.batch_size]

                    x = np.concatenate((x_batch, np.full((len(x_batch), 1), fill_value=t/self.diffusion_steps)), axis=1, dtype=np.float32)

                    m_pred, _ = self.network(x.flatten())
                    m_pred = m_pred[::2]
                    #m_pred = x_batch.flatten() * 2

                    mz_samples.extend(m_pred)

                mz_samples = np.array(mz_samples)
                mz_samples = mz_samples.reshape(mx_samples.shape)

            # Reshape the mean and variance into columns for each coordinate
            mz = np.array(mz)
            Sz = np.array(Sz)
            Sv = np.array(Sv)
            mz = mz.reshape(mx.shape)
            Sz = Sz.reshape(mx.shape)
            Sv = Sv.reshape(mx.shape)


            # Update varinance associated with X
            term_1 = np.mean(1 / self.alphas[t] * Sx)
            Sx = np.array(1 / self.alphas[t] * Sx)

            if t < self.diffusion_steps-1:
                # For every group of num_samples
                for i in range(num_points):
                    variance = []
                    # For every coordinate
                    for j in range(mz_samples.shape[1]):
                        variance += [np.var(mz_samples[i*num_samples:(i+1)*num_samples, j])]

                    Sz_samples += [variance]

                # Compute covariance between every x from 0 to num_points
                # and every group of num_samples of m_temp from num_points to the end
                for i in range(num_points):
                    covariance = []
                    # For every coordinate
                    for j in range(mz_samples.shape[1]):
                        covariance += [np.cov(mx_samples[i*num_samples:(i+1)*num_samples, j], mz_samples[i*num_samples:(i+1)*num_samples, j])[0][1]]

                    cov_X_Z_samples += [covariance]
            else:
                Sz_samples = np.zeros_like(mx)
                cov_X_Z_samples = np.zeros_like(mx)

            cov_X_Z_samples = np.array(cov_X_Z_samples, dtype=np.float32)
            Sz_samples = np.array(Sz_samples, dtype=np.float32)

            factor_term_2 = (1 - self.alphas[t]) ** 2 / (self.alphas[t] * (1 - self.baralphas[t]))
            term_2_a = np.mean(factor_term_2 * Sz)
            Sx += np.array(factor_term_2 * Sz)
            term_2_b = np.mean(factor_term_2 * Sz_samples)
            Sx += np.array(factor_term_2 * Sz_samples)
            term_2_c = np.mean(factor_term_2 * Sv)
            Sx += np.array(factor_term_2 * Sv)


            factor_term_3 = -2 * (1 - self.alphas[t]) / (self.alphas[t] * (1 - self.baralphas[t])**0.5)
            term_3 = np.mean(factor_term_3 * cov_X_Z_samples)
            Sx += np.array(factor_term_3 * cov_X_Z_samples)

            if t > self.diffusion_steps-1:
                Sx = Sz

            # Clip variance to avoid negative values to 10^-6
            #Sx = np.clip(Sx, 1e-6, None)

            print(f"Term 1: {term_1}")
            print(f"Term 2 a: {term_2_a}")
            print(f"Term 2 b: {term_2_b}")
            print(f"Term 2 c: {term_2_c}")
            print(f"Term 3: {term_3}")
            print(f"Sum Sx: {np.mean(Sx)}")

            term1 += [term_1]
            term2_a += [term_2_a]
            term2_b += [term_2_b]
            term2_c += [term_2_c]
            term3 += [term_3]

            mx = 1 / (self.alphas[t] ** 0.5) * (mx - (1 - self.alphas[t]) / ((1-self.baralphas[t]) ** 0.5) * mz)

            if t > 1:
                # Sigma t
                #std = self.betas[t] ** 0.5
                std = ((1 - self.baralphas[t-1]) / (1 - self.baralphas[t]) * self.betas[t]) ** 0.5
                mx += std * np.random.randn(*mx.shape)

            mx_hist += [mz]

        terms = {
            "term1": term1,
            "term2_a": term2_a,
            "term2_b": term2_b,
            "term2_c": term2_c,
            "term3": term3,
        }

        return mx, mx_hist, Sx, terms

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
    #diffuser.network.save("test_model_tagi_v_heteros/test.bin")

    diffuser.network.load("test_model_tagi_v_heteros/test.bin")


    X, X_hist, Sx, terms = diffuser.sample()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(X_true[:, 0], X_true[:, 1], c="b", s=1)
    ax[0].set_title("Original data")
    ax[1].scatter(X[:, 0], X[:, 1], c="r", s=1)
    ax[1].set_title("Diffused data")

    plt.savefig("diffusion_swiss_roll.png")

    # Plot variance with colors inn the points
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sc = ax.scatter(X[:, 0], X[:, 1], c=Sx[:, 0]**0.5, s=1)
    plt.colorbar(sc)
    plt.title("Std x of the points")
    plt.savefig("std_swiss_roll.png")

    # Plot variance with colors inn the points
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sc = ax.scatter(X[:, 0], X[:, 1], c=Sx[:, 1]**0.5, s=1)
    plt.colorbar(sc)
    plt.title("Std y of the points_2")
    plt.savefig("std_swiss_roll_2.png")

    # Plot terms history
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(terms["term1"], label="Term 1")
    ax.plot(terms["term2_a"], label="Term 2 a")
    ax.plot(terms["term2_b"], label="Term 2 b")
    ax.plot(terms["term2_c"], label="Term 2 c")
    ax.plot(terms["term3"], label="Term 3")
    plt.legend()
    plt.savefig("terms_swiss_roll.png")

if __name__ == "__main__":
    main()