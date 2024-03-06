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

[001 Basic Implementation, homosc, no associated variance](https://www.notion.so/001-Basic-Implementation-homosc-no-associated-variance-d014d507ebbc48e6a5fa0449802488bd?pvs=21)

[002 Homoscedasticity & Variance associated](https://www.notion.so/002-Homoscedasticity-Variance-associated-21122c163d8d4352851d82786dfa9cc0?pvs=21)

[003 Heteroscedasticity & Variance associated & batch_size = 20](https://www.notion.so/003-Heteroscedasticity-Variance-associated-batch_size-20-ea0cb08c28f043daa60c5eea3c0155b0?pvs=21)

[004 Heteroscedasticity & Variance associated & batch_size = 1](https://www.notion.so/004-Heteroscedasticity-Variance-associated-batch_size-1-8aba2c6606c34573989746c9dde274aa?pvs=21)

[005 Heteroscedasticity & Variance associated & batch_size = 5](https://www.notion.so/005-Heteroscedasticity-Variance-associated-batch_size-5-490849dadf5540fb9650aedf57459aea?pvs=21)