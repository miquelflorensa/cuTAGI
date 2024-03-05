import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from sklearn.datasets import make_swiss_roll

x, _ = make_swiss_roll(n_samples=100000, noise=0.5)
# Make two-dimensional to easen visualization
x = x[:, [0, 2]]

x = (x - x.mean()) / x.std()


diffusion_steps = 40  # Number of steps in the diffusion process

# Set noising variances betas as in Nichol and Dariwal paper (https://arxiv.org/pdf/2102.09672.pdf)
s = 0.008
timesteps = np.arange(0, diffusion_steps)
schedule = np.cos((timesteps / diffusion_steps + s) / (1 + s) * np.pi / 2)**2

baralphas = schedule / schedule[0]
betas = 1 - baralphas / np.concatenate([baralphas[0:1], baralphas[0:-1]])
alphas = 1 - betas




def noise(Xbatch, t):
    eps = np.random.randn(*Xbatch.shape)
    noised = (baralphas[t] ** 0.5).repeat(Xbatch.shape[1], axis=1) * Xbatch + ((1 - baralphas[t]) ** 0.5).repeat(Xbatch.shape[1], axis=1) * eps
    return noised, eps

X = np.array(x, dtype=np.float32)

from diffuser_v1 import Diffuser

diffuser = Diffuser(
    num_epochs=10,
    #batch_size=2048,
    batch_size=20,
    X_data=X,
    diffusion_steps=diffusion_steps,
    alphas=alphas,
    betas=betas,
)

diffuser.train()

x, xt, var = diffuser.sample()

x2 = x
xt2 = xt

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(X[:, 0], X[:, 1], c="b", s=1)
ax[0].set_title("Original data")
ax[1].scatter(x2[:, 0], x2[:, 1], c="r", s=1)
ax[1].set_title("Diffused data")

plt.savefig("diffusion_swiss_roll.png")
var2 = var.reshape(x.shape)


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plotting original data
ax[0].scatter(X[:, 0], X[:, 1], c="b", s=1)
ax[0].set_title("Original data")

# Plotting diffused data with colors based on variance
sc = ax[1].scatter(x2[:, 0], x2[:, 1], c=var2[:,0], cmap='viridis', s=1, vmin=0, vmax=0.01)
ax[1].set_title("Diffused data")
plt.colorbar(sc, ax=ax[1], label='Variance')

plt.savefig("diffusion_swiss_roll_variance.png")
