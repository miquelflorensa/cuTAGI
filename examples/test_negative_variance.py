import fire
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from pytagi import metric
from pytagi.nn import (
    Linear,
    OutputUpdater,
    ReLU,
    Sequential,
)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


TAGI_FNN = Sequential(
    Linear(784, 6000),
    ReLU(),
    Linear(6000, 784),
)

img_size = 28

diffusion_steps = 200  # Number of steps in the diffusion process
# Set noising variances betas as in Nichol and Dariwal paper (https://arxiv.org/pdf/2102.09672.pdf)
s = 0.008

timesteps = np.arange(0, diffusion_steps, dtype=np.float32)
schedule = np.cos((timesteps / diffusion_steps + s) / (1 + s) * np.pi / 2)**2
baralphas = schedule / schedule[0]
betas = 1 - baralphas / np.concatenate([baralphas[0:1], baralphas[0:-1]])
alphas = 1 - betas

def noise(Xbatch, t):
    eps = np.random.randn(*Xbatch.shape)
    baralphas_t_reshaped = baralphas[t][:, np.newaxis]  # Reshape to have one column
    noised = (baralphas_t_reshaped ** 0.5) * Xbatch + ((1 - baralphas_t_reshaped) ** 0.5) * eps
    return noised, eps


DATA_FOLDER = "./data/mnist"

def custom_collate_fn(batch):
    # batch is a list of tuples (image, label)
    batch_images, batch_labels = zip(*batch)

    # Convert to a single tensor and then to numpy
    batch_images = torch.stack(batch_images)
    batch_labels = torch.tensor(batch_labels)

    # Flatten images and labels to 1D
    batch_images = batch_images.numpy().reshape(len(batch_images), -1).flatten()
    batch_labels = batch_labels.numpy().flatten()

    return batch_images, batch_labels


def sample_ddpm(net_fnn, nsamples, nfeatures, batch_size):
    """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
    # x = torch.randn(size=(nsamples, nfeatures))
    x = np.random.randn(nsamples, nfeatures)

    xt = [x]
    for t in range(diffusion_steps-1, 0, -1):
        mz = []
        Sv = []
        for i in range(0, len(x), batch_size):
            xbatch = x[i:i+batch_size]

            # net_input = xbatch.view(-1, 1, img_size, img_size)
            net_input = xbatch

            for j in range(net_input.shape[0]):
                net_input[j,0] = t

            timesteps = torch.full([nsamples, 1], t)
            timesteps = timesteps.view(-1)


            net_input = net_input.cpu().numpy()
            timesteps_emb = timesteps_emb.cpu().numpy()

            # net_input = np.concatenate((net_input, timesteps_emb), axis=1)

            m_pred, v_pred = net_fnn(net_input)


            mz.extend(m_pred)
            Sv.extend(v_pred)

        mz = np.array(mz)
        mz = mz.reshape(nsamples, nfeatures)
        mz = torch.tensor(mz)

        Sv = np.array(Sv)
        Sv = Sv.reshape(nsamples, nfeatures)

        # See DDPM paper between equations 11 and 12
        x = 1 / (alphas[t] ** 0.5) * (x - (1 - alphas[t]) / ((1-baralphas[t]) ** 0.5) * mz)
        if t > 1:
            # See DDPM paper section 3.2.
            # Choosing the variance through beta_t is optimal for x_0 a normal distribution
            variance = betas[t]
            std = variance ** (0.5)
            x += std * torch.randn(size=(nsamples, nfeatures))

        xt += [x]

    return x, xt, Sv


def tagi_trainer(
    batch_size: int, num_epochs: int, device: str = "cuda", sigma_v: float = 0.0
):
    # Data loading and preprocessing
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root=DATA_FOLDER, train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root=DATA_FOLDER, train=False, transform=transform, download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=custom_collate_fn,
    )


    net = TAGI_FNN
    net.to_device(device)
    # net.set_threads(16)

    out_updater = OutputUpdater(net.device)
    print("len(x) = ")
    sigma_v = 0.1
    var_y = np.full((batch_size * img_size*img_size,), sigma_v**2, dtype=np.float32)

    # Training
    error_rates = []
    pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in pbar:
        mse = 0
        net.train()
        for x, labels in train_loader:
            timesteps = np.random.randint(0, diffusion_steps, batch_size)
            noised, eps = noise(x, timesteps)
            # print(noised.shape)
            net_input = np.concatenate((noised, timesteps.reshape(-1, 1)/40), axis=1)
            net_input = np.array(net_input, dtype=np.float32)

            # print(net_input.shape)

            # Feedforward and backward pass
            m_pred, v_pred = net(net_input.flatten())

            out_updater.update(
                output_states=net.output_z_buffer,
                mu_obs=eps,
                var_obs=var_y,
                delta_states=net.input_delta_z_buffer,
            )

            # Update parameters
            net.backward()
            net.step()


            mse += metric.mse(m_pred, eps)
            print(f"Iteration loss = {mse}")


        Xgen, Xgen_hist, Sv = sample_ddpm(net, 9, img_size*img_size, batch_size)
        Xgen = Xgen.cpu()
        Sv = Sv**0.5

        fig, ax = plt.subplots(1, 9, figsize=(15,2))
        for i in range(9):
            ax[i].imshow(Xgen[i].reshape(img_size,img_size), cmap="gray")
            ax[i].set_title(f"Sample {i}")
            ax[i].axis("off")

        plt.savefig("examples/mnist_images_fnn/tagi_fnn_epoch_{}.png".format(epoch))


def main(
    framework: str = "tagi",
    batch_size: int = 256,
    epochs: int = 20,
    device: str = "cuda",
):
    if framework == "tagi":
        tagi_trainer(batch_size=batch_size, num_epochs=epochs, device=device)
    else:
        raise RuntimeError(f"Invalid Framework: {framework}")


if __name__ == "__main__":
    fire.Fire(main)
