# Diffusion Models TAGI

To-do list:

- [x]  pytagi_v0 implementation
- [x]  Try GPU and bath_size
- [ ]  Explore with empirical scheduler (James tried and it seems that there is no visible improvement with small networks)
- [ ]  Metrics: GMM (Gaussian mixture model)
- [ ]  Motion and direction of some points during sampling
- [ ]  Check starting points sampling

![Screenshot 2024-02-22 at 11.14.48 AM.png](Diffusion%20Models%20TAGI%20f97444504b444074a44304f162a7a93d/Screenshot_2024-02-22_at_11.14.48_AM.png)

## Experiment

[001 Basic Implementation, homosc, no associated variance](Diffusion%20Models%20TAGI%20f97444504b444074a44304f162a7a93d/001%20Basic%20Implementation,%20homosc,%20no%20associated%20va%20d014d507ebbc48e6a5fa0449802488bd.md)

[002 Homoscedasticity & Variance associated](Diffusion%20Models%20TAGI%20f97444504b444074a44304f162a7a93d/002%20Homoscedasticity%20&%20Variance%20associated%2021122c163d8d4352851d82786dfa9cc0.md)

[003 Heteroscedasticity & Variance associated & batch_size = 20](Diffusion%20Models%20TAGI%20f97444504b444074a44304f162a7a93d/003%20Heteroscedasticity%20&%20Variance%20associated%20&%20bat%20ea0cb08c28f043daa60c5eea3c0155b0.md)

[004 Heteroscedasticity & Variance associated & batch_size = 1](Diffusion%20Models%20TAGI%20f97444504b444074a44304f162a7a93d/004%20Heteroscedasticity%20&%20Variance%20associated%20&%20bat%208aba2c6606c34573989746c9dde274aa.md)

[005 Heteroscedasticity & Variance associated & batch_size = 5](Diffusion%20Models%20TAGI%20f97444504b444074a44304f162a7a93d/005%20Heteroscedasticity%20&%20Variance%20associated%20&%20bat%20490849dadf5540fb9650aedf57459aea.md)