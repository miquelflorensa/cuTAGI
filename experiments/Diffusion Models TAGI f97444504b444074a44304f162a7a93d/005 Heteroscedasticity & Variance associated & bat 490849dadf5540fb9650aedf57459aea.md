# 005 Heteroscedasticity & Variance associated & batch_size = 5

Heteroscedastic implementation: [https://github.com/miquelflorensa/cuTAGI/blob/diffusion-tagi/python_examples/diffuser_heteros_v0.py](https://github.com/miquelflorensa/cuTAGI/blob/diffusion-tagi/python_examples/diffuser_heteros_v0.py)

We use batch size = 5 and noise_gain = 1. With bigger batch_sizes it explodes. Deactivate weight update of 2 additional output layers for 1st epoch needs to be tried.

![Screenshot 2024-03-05 at 1.40.08 PM.png](002%20Homoscedasticity%20&%20Variance%20associated%2021122c163d8d4352851d82786dfa9cc0/Screenshot_2024-03-05_at_1.40.08_PM.png)

Variance Associated with $X_{t-1}$ is obtained from:

![Screenshot 2024-03-05 at 1.34.36 PM.png](002%20Homoscedasticity%20&%20Variance%20associated%2021122c163d8d4352851d82786dfa9cc0/Screenshot_2024-03-05_at_1.34.36_PM.png)

![diffusion_swiss_roll.png](005%20Heteroscedasticity%20&%20Variance%20associated%20&%20bat%20490849dadf5540fb9650aedf57459aea/diffusion_swiss_roll.png)

![diffusion_swiss_roll_variance.png](005%20Heteroscedasticity%20&%20Variance%20associated%20&%20bat%20490849dadf5540fb9650aedf57459aea/diffusion_swiss_roll_variance.png)

![diffusion_with_trajectories.gif](005%20Heteroscedasticity%20&%20Variance%20associated%20&%20bat%20490849dadf5540fb9650aedf57459aea/diffusion_with_trajectories.gif)

![error_variance.png](005%20Heteroscedasticity%20&%20Variance%20associated%20&%20bat%20490849dadf5540fb9650aedf57459aea/error_variance.png)