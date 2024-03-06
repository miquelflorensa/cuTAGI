# 004 Heteroscedasticity & Variance associated & batch_size = 1

Heteroscedastic implementation: [https://github.com/miquelflorensa/cuTAGI/blob/diffusion-tagi/python_examples/diffuser_heteros_v0.py](https://github.com/miquelflorensa/cuTAGI/blob/diffusion-tagi/python_examples/diffuser_heteros_v0.py)

We use batch size = 1 and noise_gain = 1. With bigger batch_sizes it explodes. Deactivate weight update of 2 additional output layers for 1st epoch needs to be tried.

![Screenshot 2024-03-05 at 1.40.08 PM.png](002%20Homoscedasticity%20&%20Variance%20associated%2021122c163d8d4352851d82786dfa9cc0/Screenshot_2024-03-05_at_1.40.08_PM.png)

Variance Associated with $X_{t-1}$ is obtained from:

![Screenshot 2024-03-05 at 1.34.36 PM.png](002%20Homoscedasticity%20&%20Variance%20associated%2021122c163d8d4352851d82786dfa9cc0/Screenshot_2024-03-05_at_1.34.36_PM.png)

![diffusion_swiss_roll.png](004%20Heteroscedasticity%20&%20Variance%20associated%20&%20bat%208aba2c6606c34573989746c9dde274aa/diffusion_swiss_roll.png)

![diffusion_swiss_roll_variance.png](004%20Heteroscedasticity%20&%20Variance%20associated%20&%20bat%208aba2c6606c34573989746c9dde274aa/diffusion_swiss_roll_variance.png)

![diffusion.gif](004%20Heteroscedasticity%20&%20Variance%20associated%20&%20bat%208aba2c6606c34573989746c9dde274aa/diffusion.gif)