import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_mnist_images(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Open the file in binary mode
    with open(file_path, 'rb') as f:
        # Read the magic number and dimensions (metadata)
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_images = int.from_bytes(f.read(4), byteorder='big')
        num_rows = int.from_bytes(f.read(4), byteorder='big')
        num_cols = int.from_bytes(f.read(4), byteorder='big')

        # Calculate the total number of pixels per image
        num_pixels = num_rows * num_cols

        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8)

        # Reshape the data to (num_images, num_pixels)
        images = images.reshape((num_images, num_pixels))

    return images

def load_mnist_labels(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Open the file in binary mode
    with open(file_path, 'rb') as f:
        # Read the magic number and number of labels (metadata)
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_labels = int.from_bytes(f.read(4), byteorder='big')

        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels

def filter_images_by_label(images, labels, target_label):
    # Get indices of images with the target label
    target_indices = np.where(labels == target_label)[0]

    # Filter the images and labels
    filtered_images = images[target_indices, :]
    filtered_labels = labels[target_indices]

    return filtered_images, filtered_labels

# Load Mnist dataset
#x_file="/data/mnist/train-images-idx3-ubyte"
x_file = "/home/mf/Documents/TAGI-V/cuTAGI/data/mnist/train-images-idx3-ubyte"
y_file = "/home/mf/Documents/TAGI-V/cuTAGI/data/mnist/train-labels-idx1-ubyte"

images = load_mnist_images(x_file)
labels = load_mnist_labels(y_file)

# target_label = 8
# x, _ = filter_images_by_label(images, labels, target_label)

#x = images shuffle
x = np.random.permutation(images)
# x = images

x = (x - x.mean()) / x.std()

print(x.shape)

import torch

X = torch.tensor(x, dtype=torch.float32)

diffusion_steps = 500  # Number of steps in the diffusion process

# Set noising variances betas as in Nichol and Dariwal paper (https://arxiv.org/pdf/2102.09672.pdf)
s = 0.008
timesteps = torch.tensor(range(0, diffusion_steps), dtype=torch.float32)
schedule = torch.cos((timesteps / diffusion_steps + s) / (1 + s) * torch.pi / 2)**2

baralphas = schedule / schedule[0]
betas = 1 - baralphas / torch.concatenate([baralphas[0:1], baralphas[0:-1]])
alphas = 1 - betas

# Check the cumulative alphas follow the distribution recommended in the paper
# sns.lineplot(baralphas)
# plt.xlabel("Diffusion step")
# plt.ylabel(r"$\bar{\alpha}$")
# plt.show()
# plt.savefig("examples/mnist_images/baralphas.png")

def noise(Xbatch, t):
    eps = torch.randn(size=Xbatch.shape)
    noised = (baralphas[t] ** 0.5).repeat(1, Xbatch.shape[1]) * Xbatch + ((1 - baralphas[t]) ** 0.5).repeat(1, Xbatch.shape[1]) * eps
    return noised, eps

from pytagi import HRCSoftmaxMetric, Utils, exponential_scheduler
import pytagi.metric as metric
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    LayerNorm,
    Conv2d,
    ConvTranspose2d,
    Linear,
    OutputUpdater,
    ReLU,
    ResNetBlock,
    Sequential,
    LayerBlock
)

n_channels = 64
img_size = 14


def ResBlock(in_channels: int, out_channels: int):
    return LayerBlock(
        # BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels, out_channels, 3, bias=False, padding=1),
        # BatchNorm2d(out_channels),
        ReLU(),
        Conv2d(out_channels, out_channels, 3, bias=False, padding=1),
    )

def DownBlock(in_channels: int, out_channels: int):
    return ResNetBlock(ResBlock(in_channels, out_channels))

def UpBlock(in_channels: int, out_channels: int):
    return LayerBlock(
        # BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels, out_channels, 3, bias=False, padding=1),
        # BatchNorm2d(out_channels),
        ReLU(),
        ConvTranspose2d(out_channels, out_channels, 3, bias=True, padding=1),
    )

def DownSample(n_channels: int):
    return Conv2d(n_channels, n_channels, 4, stride=2, padding=1)

def UpSample(n_channels: int):
    return ConvTranspose2d(n_channels, n_channels, 3, stride=2, padding=1)

TAGI_UNET = Sequential(

    Conv2d(1, n_channels, 3, bias=False, padding=1, stride=1, in_width=img_size, in_height=img_size),

    # DownBlock(n_channels, n_channels),

    # ReLU(),

    ResNetBlock(
        LayerBlock(
            ReLU(),
            LayerNorm((64, img_size, img_size)),
            Conv2d(64, 64, 3, bias=False, padding=1),
            ReLU(),
            LayerNorm((64, img_size, img_size)),
            Conv2d(64, 64, 3, bias=False, padding=1),
        )
    ),
    ReLU(),
    Conv2d(64, 128, 4, bias=False, padding=1),
    # LayerNorm((128, img_size, img_size)),
    # # ReLU(),
    # ResNetBlock(
    #     LayerBlock(
            # ResNetBlock(
            #     LayerBlock(
            #         ReLU(),
            #         LayerNorm((128, img_size, img_size)),
            #         Conv2d(128, 128, 3, bias=False, padding=1),
            #         ReLU(),
            #         LayerNorm((128, img_size, img_size)),
            #         Conv2d(128, 128, 3, bias=False, padding=1),
            #     )
            # ),
            ResNetBlock(
                LayerBlock(
                    ReLU(),
                    LayerNorm((128, img_size, img_size)),
                    Conv2d(128, 128, 3, bias=False, padding=1),
                    ReLU(),
                    LayerNorm((128, img_size, img_size)),
                    Conv2d(128, 128, 3, bias=False, padding=1),
                )
            ),
            ReLU(),
            Conv2d(128, 256, 4, bias=False, padding=1),
            ResNetBlock(
                LayerBlock(
                    ReLU(),
                    LayerNorm((256, img_size, img_size)),
                    Conv2d(256, 256, 3, bias=False, padding=1),
                    ReLU(),
                    LayerNorm((256, img_size, img_size)),
                    Conv2d(256, 256, 3, bias=False, padding=1),
                )
            ),
            ReLU(),
            ConvTranspose2d(256, 128, 4, bias=True, padding=1),
            ResNetBlock(
                LayerBlock(
                    ReLU(),
                    LayerNorm((128, img_size, img_size)),
                    Conv2d(128, 128, 3, bias=False, padding=1),
                    ReLU(),
                    LayerNorm((128, img_size, img_size)),
                    Conv2d(128, 128, 3, bias=False, padding=1),
                )
            ),
            # ResNetBlock(
            #     LayerBlock(
            #         # LayerNorm((128, img_size, img_size)),
            #         ReLU(),
            #         Conv2d(128, 128, 3, bias=False, padding=1),
            #         # LayerNorm((128, img_size, img_size)),
            #         ReLU(),
            #         Conv2d(128, 128, 3, bias=False, padding=1),
            #     )
            # ),
    #     )
    # ),

    ReLU(),
    ConvTranspose2d(128, 64, 4, bias=True, padding=1),


    # LayerNorm((64, img_size, img_size)),
    # ReLU(),
    ResNetBlock(
        LayerBlock(
            ReLU(),
            LayerNorm((64, img_size, img_size)),
            Conv2d(64, 64, 3, bias=False, padding=1),
            ReLU(),
            LayerNorm((64, img_size, img_size)),
            Conv2d(64, 64, 3, bias=False, padding=1),
        )
    ),
    ReLU(),
    Conv2d(64, 1, 3, bias=True, padding=1),
)

import math

def sinusoidal_positional_encoding(t):
    # t has dimension [batch_size]
    half_dim = n_channels // 8
    emb = math.log(10_000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=1)

    return emb


def sample_ddpm(net_unet, nsamples, nfeatures):
    """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
    x = torch.randn(size=(nsamples, nfeatures))
    xt = [x]
    for t in range(diffusion_steps-1, 0, -1):
        mz = []
        Sv = []
        for i in range(0, len(x), batch_size):
            xbatch = x[i:i+batch_size]

            net_input = xbatch.view(-1, 1, img_size, img_size)

            timesteps = torch.full([nsamples, 1], t)
            timesteps = timesteps.view(-1)
            timesteps_emb = timesteps / diffusion_steps

            net_input = net_input.cpu().numpy()
            timesteps_emb = timesteps_emb.cpu().numpy()


            for j in range(net_input.shape[0]):
                net_input[j, 0, 0, 0] = timesteps_emb[j]
                # net_input[j, 0, 0, 0] = 0

            # m_pred, v_pred = net_unet(net_input + time_emb_output)
            m_pred, v_pred = net_unet(net_input)

            # v_pred = np.exp(m_pred[1::2] + 0.5 * v_pred[1::2])
            # m_pred = m_pred[::2]


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

import torch
from torch import nn

# net_unet = TAGI_UNET
# net_unet.to_device("cuda")

# output_updater = OutputUpdater(net_unet.device)

nepochs = 50
batch_size = 1


# Resize the data of 28*28 to img_size*img_size
X = torch.tensor(x, dtype=torch.float32)
X = nn.functional.interpolate(X.view(-1, 1, 28, 28), size=(img_size, img_size)).view(-1, img_size*img_size)


# sigma_v = 0.0
# var_y = np.full((batch_size * img_size*img_size,), sigma_v**2, dtype=np.float32)

# # net_unet.load("mnist_models/tagi_unet_epoch_{}.bin".format(0))

# for epoch in range(nepochs):
#     mse = 0.0

#     var_y = np.full((batch_size * img_size*img_size,), sigma_v**2, dtype=np.float32)

#     for i in range(0, len(X), batch_size):
#         Xbatch = X[i:i+batch_size]
#         timesteps = torch.randint(0, diffusion_steps, size=[len(Xbatch), 1])
#         noised, eps = noise(Xbatch, timesteps)

#         noised = noised.view(-1, 1, img_size, img_size)
#         eps = eps.view(-1, 1, img_size, img_size)

#         timesteps_emb = timesteps / diffusion_steps

#         net_input = noised.cpu().numpy() # [batch_size, 1, img_size, img_size]
#         eps = eps.cpu().numpy()
#         timesteps_emb = timesteps_emb.cpu().numpy() # [batch_size, 1]
#         # print(timesteps_emb)

#         for j in range(net_input.shape[0]):
#             net_input[j,0,0,0] = timesteps_emb[j,0]
#             # net_input[j, 0, 0, 0] = 0

#         # print(net_input.shape)

#         unet_output, var_pred = net_unet(net_input)
#         # print("unet_output", unet_output)
#         # print("var_pred", var_pred)

#         # unet_output = unet_output[::2]

#         # print(unet_output.shape)


#         # Update the output layer
#         output_updater.update(
#             output_states=net_unet.output_z_buffer,
#             mu_obs=eps.flatten(),
#             var_obs=var_y,
#             delta_states=net_unet.input_delta_z_buffer,
#         )

#         net_unet.backward()
#         net_unet.step()


#         # Compute the loss
#         mse = metric.mse(unet_output, eps.flatten())

#         print(f"Iteration {i} loss = {mse}")


#     # mse /= len(X) / batch_size
#     print(f"Epoch {epoch} loss = {mse}")

#     # # Save the models
#     net_unet.save("mnist_models/tagi_unet_epoch_{}.bin".format(epoch))

#     Xgen, Xgen_hist, Sv = sample_ddpm(net_unet, 9, img_size*img_size)
#     Xgen = Xgen.cpu()
#     Sv = Sv**0.5

fig, ax = plt.subplots(1, 9, figsize=(20,4))
for i in range(9):
    ax[i].imshow(X[i].reshape(img_size,img_size), cmap="gray")
    # ax[1,i].imshow(Xgen[i+9].reshape(img_size,img_size), cmap="gray")
    # ax[1,i].imshow(Sv[i].reshape(img_size,img_size), cmap="inferno")
    ax[i].set_title(f"Sample {i}")
    ax[i].axis("off")
    ax[i].axis("off")

    # Set colorbar
    # cbar = fig.colorbar(ax[1,i].imshow(Sv[i].reshape(img_size,img_size), cmap="inferno"), ax=ax[1,i], orientation="horizontal")
    # cbar.set_label("Standard deviation")
# plt.show()

# Save the generated images
plt.savefig("examples/mnist_images/tagi_unet_epoch_{}.png".format(0))
