###############################################################################
# File:         visualizer.py
# Description:  Visualization tool for images data
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      May 10, 2022
# Updated:      June 12, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
###############################################################################
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.typing as npt

plt.rcParams.update({
    'font.size': 18,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


class ImageViz:
    """ Visualization of image data
    Attributes:
        task_name: Name of the task such as autoencoder
        data_name: Name of dataset such as mnist or cifar10
        mu: Mean of each input, e.g., for 3 channels; mu: 0.5, 0.5, 0.5 
        sigma: Standard deviation of each input
        img_size: Size of image input e.g. mnist size = [28, 28, 1]
    """

    def __init__(self, task_name: str, data_name: str, mu: npt.NDArray,
                 sigma: npt.NDArray, img_size: npt.NDArray) -> None:
        self.task_name = task_name
        self.data_name = data_name
        self.mu = mu
        self.sigma = sigma
        self.img_size = img_size

    def load_generated_images(self) -> npt.NDArray:
        """ Load the images generated using TAGI-neural netwrok"""

        # Get file name
        path_dir = f'./saved_results/'
        file_name = f'{path_dir}/generated_images_test.csv'

        # Load image data from *.csv file
        df = pd.read_csv(file_name, skiprows=0, delimiter=",", header=None)
        imgs = df[0].values

        # Reshape data for plot
        num_imgs = int(len(imgs) / np.prod(self.img_size))
        imgs = np.reshape(
            imgs,
            (num_imgs, self.img_size[0], self.img_size[1], self.img_size[2]))
        mu = np.reshape(self.mu, (self.img_size[0], 1, 1))
        sigma = np.reshape(self.sigma, (self.img_size[0], 1, 1))

        imgs = (imgs * sigma + mu) * 255.0
        imgs = imgs.transpose(0, 2, 3, 1)

        return imgs

    def plot_images(self, n_row: int, n_col: int) -> None:
        """Plot and save figure
        Args:
            n_row: Number of rows for exported image
            n_col: Number of colums for exported image
        """

        # Load images
        imgs = self.load_generated_images()
        (num, _, _, _) = imgs.shape

        # Plot images
        path_dir = './saved_results'
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        fig_path = f'{path_dir}/{self.data_name}_{self.task_name}.png'

        _, axes = plt.subplots(n_row, n_col, figsize=((10, 10)))
        for i in range(num):
            ax = axes[i // n_col, i % n_col]
            ax.imshow(imgs[i], cmap='gray')
            ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches='tight')


class PredictionViz:
    """Visualization of prediction
    Attributes:
        task_name: Name of the task such as autoencoder
        data_name: Name of dataset such as Boston housing or toy example
        figsize: Size of figure
        fontsize: Font size for letter in the figure
        lw: linewidth
        ms: Marker size
        ndiv_x: Number of divisions for x-direction
        ndiv_y: Number of division for y-direciton
    """

    def __init__(self,
                 task_name: str,
                 data_name: str,
                 figsize: tuple = (12, 12),
                 fontsize: int = 28,
                 lw: int = 3,
                 ms: int = 10,
                 ndiv_x: int = 4,
                 ndiv_y: int = 4) -> None:
        self.task_name = task_name
        self.data_name = data_name
        self.figsize = figsize
        self.fontsize = fontsize
        self.lw = lw
        self.ms = ms
        self.ndiv_x = ndiv_x
        self.ndiv_y = ndiv_y

    def load_dataset(self, file_path: str, header: bool = False) -> npt.NDArray:
        """Load dataset (*.csv)
        Args:
            file_path: File path to the data file
            header: Ignore hearder ?

        """

        # Load image data from *.csv file
        if header:
            df = pd.read_csv(file_path, skiprows=1, delimiter=",", header=None)
        else:
            df = pd.read_csv(file_path, skiprows=0, delimiter=",", header=None)

        return df[0].values

    def plot_predictions(self,
                         x_train: npt.NDArray,
                         y_train: npt.NDArray,
                         x_test: npt.NDArray,
                         y_test: npt.NDArray,
                         y_pred: npt.NDArray,
                         sy_pred: npt.NDArray,
                         sy_test: npt.NDArray | None = None,
                         label: str = 'diag',
                         title: str | None = None,
                         eq: str | None = None) -> None:
        """Compare prediciton distribution with theorical distribution

            x_train: Input train data
            y_train: Output train data
            x_test: Input test data
            y_test: Output test data
            y_pred: Prediciton of network
            sy_pred: Standard deviation of the prediction
            sy_test: Output test's theorical standard deviation
            label: Name of file
            title: Figure title
            eq: Math equation for data

        """

        # Get max and min values
        max_y = max(y_test)
        min_y = min(y_test)
        max_x = max(x_test) + 1
        min_x = min(x_test) - 1

        # Plot figure
        plt.figure(figsize=self.figsize)
        ax = plt.axes()
        ax.set_title(title, fontsize=1.1 * self.fontsize, fontweight='bold')
        ax.text(-5.5, 100, eq, color='k', fontsize=self.fontsize)
        ax.plot(x_test, y_pred, 'r', lw=self.lw, label=r"$\mathbb{E}[Y]$")
        ax.plot(x_test, y_test, 'k', lw=self.lw, label=r"$y_{true}$")

        ax.fill_between(x_test,
                        y_pred - 1 * sy_pred,
                        y_pred + 1 * sy_pred,
                        facecolor='red',
                        alpha=0.3,
                        label=r"$\mathbb{E}[Y]\pm\sigma$")
        if sy_test is not None:
            ax.fill_between(x_test,
                            y_test - 1 * sy_test,
                            y_test + 1 * sy_test,
                            facecolor='blue',
                            alpha=0.3,
                            label=r"$y_{true}\pm\sigma$")
        if x_train is not None:
            ax.plot(x_train,
                    y_train,
                    'k',
                    marker='o',
                    mfc='none',
                    lw=self.lw,
                    ms=self.ms,
                    linestyle='',
                    label=r'$y_{train}$')

        ax.set_xlabel(r'$x$', fontsize=self.fontsize)
        ax.set_ylabel(r'$y$', fontsize=self.fontsize)
        x_ticks = np.linspace(min_x, max_x, self.ndiv_x)
        y_ticks = np.linspace(min_y, max_y, self.ndiv_y)
        ax.set_yticks(y_ticks)
        ax.set_xticks(x_ticks)
        ax.tick_params(axis='both',
                       which='both',
                       direction='inout',
                       labelsize=self.fontsize)
        ax.legend(loc='lower center',
                  edgecolor='black',
                  fontsize=1 * self.fontsize,
                  ncol=2,
                  framealpha=0.3)
        ax.set_ylim([min_y, max_y])
        ax.set_xlim([min_x, max_x])

        # Save figure
        saving_path = f'saved_results/pred_{label}_{self.data_name}.png'
        plt.savefig(saving_path, bbox_inches='tight')
        plt.close()


def autoencoder():
    # User input data
    task_name = 'autoencoder'
    data_name = 'mnist'
    mu = np.array([0.1309])
    sigma = np.array([1])
    img_size = np.array([1, 28, 28])
    n_row = 10
    n_col = 10

    viz = ImageViz(task_name=task_name,
                   data_name=data_name,
                   mu=mu,
                   sigma=sigma,
                   img_size=img_size)
    viz.plot_images(n_row=n_row, n_col=n_col)


def regression():
    # Equation
    eq = r"$Y = x^{3} + V, ~V\sim\mathcal{N}(0, 3^{2})$"

    # User input data
    task_name = 'regression'
    data_name = 'toy_example'
    x_train_path = './data/toy_example/x_train_1D.csv'
    y_train_path = './data/toy_example/y_train_1D.csv'
    x_test_path = './data/toy_example/x_test_1D.csv'
    y_test_path = './data/toy_example/y_test_1D.csv'
    y_pred_path = './saved_results/y_prediction.csv'
    sy_pred_path = './saved_results/sy_prediction.csv'

    viz = PredictionViz(task_name=task_name, data_name=data_name)

    # Load data
    x_train = viz.load_dataset(file_path=x_train_path, header=True)
    y_train = viz.load_dataset(file_path=y_train_path, header=True)
    x_test = viz.load_dataset(file_path=x_test_path, header=True)
    y_test = viz.load_dataset(file_path=y_test_path, header=True)
    y_pred = viz.load_dataset(file_path=y_pred_path)
    sy_pred = viz.load_dataset(file_path=sy_pred_path)

    # Plot
    viz.plot_predictions(x_train=x_train,
                         y_train=y_train,
                         x_test=x_test,
                         y_test=y_test,
                         y_pred=y_pred,
                         sy_pred=sy_pred,
                         label='diag',
                         title=r"\textbf{Diagonal covariance}",
                         eq=eq)


def input_uncertainty_prop():
    """ The analytical formulation for output is defined following
        y = x^3 + Normal(0, \sigma_v), where x ~ Normal(mu_x, \sigma_x).

        Var[y] = (|dy/dy|)^2 * (\sigma_x)^2 + (\sigma_v) ^2.

    """
    # Equation
    eq = r"$Y = X^{3} + V,~X \sim \mathcal{N}(\mu_{X}, 1),~V\sim\mathcal{N}(0, 3^{2})$"

    # Standard deviation. NOTE: the following values must correspond to
    # the data file path specified by user
    sigma_v = 3
    sigma_x = 1

    # User input data
    task_name = 'regression'
    data_name = 'toy_example'
    x_train_path = './data/toy_example/x_train_1D_full_cov.csv'
    y_train_path = './data/toy_example/y_train_1D_full_cov.csv'
    x_test_path = './data/toy_example/x_test_1D_full_cov.csv'
    y_test_path = './data/toy_example/y_test_1D_full_cov.csv'
    y_pred_path = './saved_results/y_prediction.csv'
    sy_pred_path = './saved_results/sy_prediction.csv'

    viz = PredictionViz(task_name=task_name, data_name=data_name)

    # Load data
    x_train = viz.load_dataset(file_path=x_train_path, header=True)
    y_train = viz.load_dataset(file_path=y_train_path, header=True)
    x_test = viz.load_dataset(file_path=x_test_path, header=True)
    y_test = viz.load_dataset(file_path=y_test_path, header=True)
    y_pred = viz.load_dataset(file_path=y_pred_path)
    sy_pred = viz.load_dataset(file_path=sy_pred_path)

    # Compute theorical standard deviation for the output
    sy_test = ((3 * (x_test**2))**2 * (sigma_x**2) + sigma_v**2)**0.5

    # Plot
    viz.plot_predictions(x_train=None,
                         y_train=y_train,
                         x_test=x_test,
                         y_test=y_test,
                         y_pred=y_pred,
                         sy_pred=sy_pred,
                         sy_test=sy_test,
                         label='full_cov',
                         title=r"\textbf{Full covariance",
                         eq=eq)


if __name__ == '__main__':
    #regression()
    autoencoder()
    #input_uncertainty_prop()
