# 003 Heteroscedasticity & Variance associated & batch_size = 20

Heteroscedastic implementation: [https://github.com/miquelflorensa/cuTAGI/blob/diffusion-tagi/python_examples/diffuser_heteros_v0.py](https://github.com/miquelflorensa/cuTAGI/blob/diffusion-tagi/python_examples/diffuser_heteros_v0.py)

We use batch small batch size and noise_gain = 1. With bigger batch_sizes it explodes. Deactivate weight update of 2 additional output layers for 1st epoch needs to be tried.

![Screenshot 2024-03-05 at 1.40.08 PM.png](002%20Homoscedasticity%20&%20Variance%20associated%2021122c163d8d4352851d82786dfa9cc0/Screenshot_2024-03-05_at_1.40.08_PM.png)

Variance Associated with $X_{t-1}$ is obtained from:

![Screenshot 2024-03-05 at 1.34.36 PM.png](002%20Homoscedasticity%20&%20Variance%20associated%2021122c163d8d4352851d82786dfa9cc0/Screenshot_2024-03-05_at_1.34.36_PM.png)

![diffusion_swiss_roll.png](003%20Heteroscedasticity%20&%20Variance%20associated%20&%20bat%20ea0cb08c28f043daa60c5eea3c0155b0/diffusion_swiss_roll.png)

![diffusion_swiss_roll_variance.png](003%20Heteroscedasticity%20&%20Variance%20associated%20&%20bat%20ea0cb08c28f043daa60c5eea3c0155b0/diffusion_swiss_roll_variance.png)

![diffusion.gif](003%20Heteroscedasticity%20&%20Variance%20associated%20&%20bat%20ea0cb08c28f043daa60c5eea3c0155b0/diffusion.gif)