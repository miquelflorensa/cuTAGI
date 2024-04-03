###############################################################################
# File:         diffusion_runner.py
# Description:  Diffusion runner using pytagi_v0
# Authors:      Miquel Florensa & Luong-Ha Nguyen & James-A. Goulet
# Created:      February 13, 2024
# Updated:      March 4, 2024
# Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com
#               & james.goulet@polymtl.ca
# License:      This code is released under the MIT License.
###############################################################################

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from sklearn.datasets import make_swiss_roll
from sklearn.datasets import make_moons
from datetime import datetime


from pytagi import NetProp


x, _ = make_swiss_roll(n_samples=100000, noise=0.5)
# Make two-dimensional to easen visualization
x = x[:, [0, 2]]
x = (x - x.mean()) / x.std()

#make_moons, labels = make_moons(n_samples=100000, noise=0.01)
#moons = make_moons * [5., 10.]
#x = make_moons


diffusion_steps = 40  # Number of steps in the diffusion process

# Set noising variances betas as in Nichol and Dariwal paper (https://arxiv.org/pdf/2102.09672.pdf)
s = 0.008
timesteps = np.arange(0, diffusion_steps)
schedule = np.cos((timesteps / diffusion_steps + s) / (1 + s) * np.pi / 2)**2

baralphas = schedule / schedule[0]
betas = 1 - baralphas / np.concatenate([baralphas[0:1], baralphas[0:-1]])
alphas = 1 - betas

X = np.array(x, dtype=np.float32)

#from .diffuser_v0 import Diffuser
from .diffuser_heteros_v0 import Diffuser

diffuser = Diffuser(
    num_epochs=1,
    #batch_size=2048,
    batch_size=10,
    X_data=X,
    diffusion_steps=diffusion_steps,
    sampling_dim=(10000, 2),
    alphas=alphas,
    betas=betas,
)

mse_epochs, fid_epochs = diffuser.train()

# Make saving path with date and time
now = datetime.now()
date_time = now.strftime("%m-%d-%Y_%H-%M-%S")

# Create new directory diffusion_results
import os
try:
    os.mkdir("diffusion_results/")
except FileExistsError:
    pass

# Create new directory for saving
import os
try:
    os.mkdir("diffusion_results/" + date_time)
except FileExistsError:
    pass


# Plot error variance during epochs
plt.plot(mse_epochs)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Mean Square Error during epochs training")
plt.savefig("diffusion_results/" + date_time + "/error_mse.png")


# Plot FID during epochs
plt.figure()
plt.plot(fid_epochs)
plt.xlabel("Epoch")
plt.ylabel("FID")
plt.title("FID during epochs training")
plt.savefig("diffusion_results/" + date_time + "/fid_epochs.png")

x, xt, var, var_temp = diffuser.sample()

act1 = x
act2 = X[:x.shape[0], :]


x2 = x
xt2 = xt

original_data = X[:10000, :]
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(original_data[:, 0], original_data[:, 1], c="b", s=1)
ax[0].set_title("Original data")
ax[1].scatter(x2[:, 0], x2[:, 1], c="r", s=1)
ax[1].set_title("Diffused data")
plt.savefig("diffusion_results/" + date_time + "/diffusion_swiss_roll.png")


var2 = var.reshape(x.shape)


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plotting original data
sc = ax[0].scatter(x2[:, 0], x2[:, 1], c=var2[:,0], cmap='viridis', s=1, vmin=0.0, vmax=0.01)
ax[0].set_title("Horizontal variance")

# Plotting diffused data with colors based on variance
sc = ax[1].scatter(x2[:, 0], x2[:, 1], c=var2[:,1], cmap='viridis', s=1, vmin=0.0, vmax=0.01)
ax[1].set_title("Vertical variance")

# Plot vertical colorbar without distorting the plot
cbar = plt.colorbar(sc, ax=ax.ravel().tolist(), orientation='vertical', pad=0.01)
cbar.set_label('Variance')

plt.savefig("diffusion_results/" + date_time + "/diffusion_swiss_roll_variance.png")

import matplotlib.animation as animation

def draw_frame(i):
    plt.clf()
    Xvis = xt2[i]
    fig = plt.scatter(Xvis[:, 0], Xvis[:, 1], marker="1", animated=True, c=var_temp[i][:,0], cmap='viridis', s=1, vmin=0.0, vmax=0.01)
    plt.xlim([-2.2, 2.2])
    plt.ylim([-2.2, 2.2])
    return fig,


fig = plt.figure()
anim = animation.FuncAnimation(fig, draw_frame, frames=40, interval=1, blit=True)
anim.save("diffusion_results/" + date_time + "/diffusion_swiss_roll_variance.gif", writer="imagemagick")

import matplotlib.animation as animation

# Select 3 random points
num_points = 3
random_indices = np.random.choice(len(xt2[0]), num_points, replace=False)
random_points = xt2[0][random_indices]  # assuming all frames have the same number of points

# Initialize trajectory list for each point
trajectories = [[] for _ in range(num_points)]

def draw_frame(i):
    plt.clf()
    Xvis = xt2[i]
    sc = plt.scatter(Xvis[:, 0], Xvis[:, 1], marker="1", c=var_temp[i][:,0], cmap='viridis', s=1, vmin=0.0, vmax=0.01, alpha=0.1)
    plt.xlim([-2.2, 2.2])
    plt.ylim([-2.2, 2.2])

    # Update and plot trajectories
    for j, point_index in enumerate(random_indices):
        if j >= num_points:
            break
        trajectories[j].append(Xvis[point_index])
        trajectory = np.array(trajectories[j])
        color = plt.cm.viridis(j / num_points)  # Assign color based on index
        plt.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.4)

        # Plot arrows indicating direction of movement
        if len(trajectory) > 1:
            plt.arrow(trajectory[-2, 0], trajectory[-2, 1],
                      trajectory[-1, 0] - trajectory[-2, 0], trajectory[-1, 1] - trajectory[-2, 1],
                      color=color, alpha=0.8, width=0.005, head_width=0.05, head_length=0.1)

    return sc,


fig = plt.figure()
anim = animation.FuncAnimation(fig, draw_frame, frames=40, interval=1, blit=True)
anim.save("diffusion_results/" + date_time + "/diffusion_swiss_roll_variance_trajectories.gif", writer="imagemagick")