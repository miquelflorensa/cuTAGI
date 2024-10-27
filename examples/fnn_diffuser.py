import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from pytagi.nn import Linear, OutputUpdater, ReLU, Sequential
import pytagi.metric as metric

# Constants
DIFFUSION_STEPS = 50
IMG_SIZE = 28
N_CHANNELS = 6000
BATCH_SIZE = 8
NEPOCHS = 50
SIGMA_V = 0.6

class MNISTDiffusion:
    def __init__(self):
        self.setup_diffusion()
        self.setup_model()

    def setup_diffusion(self):
        s = 0.008
        timesteps = np.arange(0, DIFFUSION_STEPS, dtype=np.float32)
        schedule = np.cos((timesteps / DIFFUSION_STEPS + s) / (1 + s) * np.pi / 2) ** 2
        self.baralphas = schedule / schedule[0]
        betas = 1 - self.baralphas / np.concatenate(([self.baralphas[0]], self.baralphas[:-1]))
        self.alphas = 1 - betas
        self.betas = betas

    def setup_model(self):
        self.net_unet = Sequential(
            Linear(IMG_SIZE * IMG_SIZE + 4, N_CHANNELS),
            ReLU(),
            Linear(N_CHANNELS, IMG_SIZE * IMG_SIZE),
        )
        self.net_unet.to_device("cuda")
        self.output_updater = OutputUpdater(self.net_unet.device)

    @staticmethod
    def load_mnist_images(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, 'rb') as f:
            magic_number = int.from_bytes(f.read(4), byteorder='big')
            num_images = int.from_bytes(f.read(4), byteorder='big')
            num_rows = int.from_bytes(f.read(4), byteorder='big')
            num_cols = int.from_bytes(f.read(4), byteorder='big')
            num_pixels = num_rows * num_cols
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape((num_images, num_pixels))

        return images

    def noise(self, Xbatch, t):
        # Xbatch: 8 x 784
        # t: 8 x 1
        # eps = np.random.randn(*Xbatch.shape)
        eps = np.full(Xbatch.shape, 1.3)

        noised = (np.sqrt(self.baralphas[t]) * Xbatch) + (np.sqrt(1 - self.baralphas[t]) * eps)
        return noised, eps

    def sample_ddpm(self, nsamples, nfeatures):
        x = np.random.randn(nsamples, nfeatures)
        xt = [x]
        for t in range(DIFFUSION_STEPS - 1, 0, -1):
            mz = []
            Sv = []
            for i in range(0, len(x), BATCH_SIZE):
                xbatch = x[i:i+BATCH_SIZE]
                timestep = np.full((len(xbatch), 4), t / DIFFUSION_STEPS, dtype=np.float32)

                net_input = np.concatenate([xbatch, timestep], axis=1, dtype=np.float32)
                m_pred, v_pred = self.net_unet(net_input)
                mz.extend(m_pred)
                Sv.extend(v_pred)

            mz = np.array(mz).reshape(nsamples, nfeatures)
            Sv = np.array(Sv).reshape(nsamples, nfeatures)

            x = 1 / (self.alphas[t] ** 0.5) * (x - (1 - self.alphas[t]) / ((1-self.baralphas[t]) ** 0.5) * mz)
            if t > 1:
                variance = self.betas[t]
                std = variance ** 0.5
                x += std * np.random.randn(nsamples, nfeatures)
            xt.append(x)

        return x, xt, Sv

    def train(self, X):
        var_y = np.full((BATCH_SIZE * IMG_SIZE * IMG_SIZE,), SIGMA_V**2, dtype=np.float32)

        for epoch in range(NEPOCHS):
            mse = 0.0
            for i in range(0, len(X), BATCH_SIZE):
                Xbatch = X[i:i+BATCH_SIZE] # 8 x 784
                timesteps = np.random.randint(0, DIFFUSION_STEPS, size=[len(Xbatch), 1]) # 8 x 1
                noised, eps = self.noise(Xbatch, timesteps) # 8 x 784, 8 x 784

                timesteps = timesteps / DIFFUSION_STEPS # 8 x 1
                net_input = np.concatenate([noised, timesteps.repeat(4, axis=1)], axis=1, dtype=np.float32) # 8 x 788

                net_input = net_input.flatten() # 6304
                unet_output, var_pred = self.net_unet(net_input)

                self.output_updater.update(
                    output_states=self.net_unet.output_z_buffer,
                    mu_obs=eps.flatten(),
                    var_obs=var_y,
                    delta_states=self.net_unet.input_delta_z_buffer,
                )

                self.net_unet.backward()
                self.net_unet.step()

                mse = metric.mse(unet_output, eps.flatten())
                print(f"Iteration {i} loss = {mse}")

            print(f"Epoch {epoch} loss = {mse}")
            self.generate_samples(epoch)

    def generate_samples(self, epoch):
        Xgen, _, Sv = self.sample_ddpm(10, IMG_SIZE * IMG_SIZE)
        Sv = Sv**0.5

        fig, ax = plt.subplots(1, 10, figsize=(20,3))
        for i in range(10):
            ax[i].imshow(Xgen[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
            ax[i].set_title(f"Sample {i}")
            ax[i].axis("off")

        plt.savefig(f"examples/mnist_images/tagi_unet_epoch_{epoch}.png")
        plt.close(fig)

def main():
    x_file = "/home/mf/Documents/TAGI-V/cuTAGI/data/mnist/train-images-idx3-ubyte"

    diffusion = MNISTDiffusion()

    images = diffusion.load_mnist_images(x_file)
    x = images
    # x = np.random.permutation(images)

    x = (x - x.mean()) / x.std()
    X = np.array(x, dtype=np.float32)


    # timesteps = np.random.randint(0, DIFFUSION_STEPS, size=[10, 1])
    # timesteps vector of 10 elements of 400
    timesteps = np.full((10, 1), 30)
    noised, eps = diffusion.noise(X[:10], timesteps)

    print(noised[0])

    # Noise 10 images
    for i in range(10):
        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        vmin = min(X[i].min(), noised[i].min())
        vmax = max(X[i].max(), noised[i].max())

        ax[0].imshow(X[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray", vmin=vmin, vmax=vmax)
        ax[0].set_title("Original")
        ax[0].axis("off")

        ax[1].imshow(noised[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray", vmin=vmin, vmax=vmax)
        ax[1].set_title("Noised")
        ax[1].axis("off")

        plt.savefig(f"examples/mnist_images/tagi_unet_noised_{i}.png")
        plt.close(fig)

    # diffusion.train(X)

if __name__ == "__main__":
    main()