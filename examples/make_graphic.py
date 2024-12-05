import matplotlib.pyplot as plt

# Data
initial_var_mean = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

# Gain = 0.05
training_errors = [25.93, 19.35, 20.14, 21.64, 22.40, 23.82, 24.72, 25.07, 26.75, 26.03, 27.32, 27.18, 27.93, 27.66]
test_errors = [27.88, 22.01, 24.21, 24.43, 25.26, 26.00, 28.45, 27.52, 28.29, 30.17, 30.62, 30.26, 30.46, 29.79]

# Gain = 0.1
training_errors = [25.68, 29.02, 15.62, 16.46, 16.65, 17.19, 17.65, 17.78, 18.18, 18.48, 18.15, 20.16, 18.80, 18.22]
test_errors = [25.43, 28.96, 18.08, 19.29, 20.41, 20.18, 20.87, 21.26, 21.43, 21.19, 21.58, 23.13, 21.75, 21.49]

#  Gain = 0.15
training_errors = [21.35, 24.24, 25.59, 18.30, 13.36, 14.81, 15.67, 15.58, 15.11, 15.49, 15.23, 15.44, 16.02, 16.47]
test_errors = [22.82, 25.65, 25.37, 19.57, 17.67, 17.91, 17.82, 19.70, 18.73, 19.19, 21.05, 19.87, 19.56, 18.93]

# Gain = 0.2
training_errors = [17.56, 29.39, 25.45, 23.31, 23.80, 18.77, 15.18, 14.57, 13.93, 14.65, 14.60, 13.66, 14.26, 15.17]
test_errors = [19.56, 27.28, 25.36, 23.84, 24.25, 20.42, 17.42, 17.25, 17.03, 17.82, 17.67, 17.46, 18.86, 18.61]

# Gain = 0.4
training_errors = [32.02, 21.53, 21.07, 17.86, 22.21, 31.43, 23.12, 20.16, 20.00, 18.21, 21.44, 19.27, 17.64, 16.98]
test_errors = [31.66, 24.07, 22.41, 20.93, 24.20, 31.79, 22.60, 21.80, 22.03, 20.51, 21.19, 20.62, 19.88, 20.09]

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(initial_var_mean, training_errors, marker='o', label='Training Error')
# plt.plot(initial_var_mean, test_errors, marker='s', label='Test Error')

# # Adding labels and title
# plt.xlabel('Initial Variance Mean')
# plt.ylabel('Error (%)')
# plt.title('Training and Test Errors vs Initial Variance Mean')
# plt.legend()
# plt.grid()

# # Save the plot
# plt.savefig('training_and_test_errors_vs_initial_variance_mean.png')

# Plot 3D graph
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Data
initial_var_mean = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
gain = np.array([0.05, 0.1, 0.15, 0.2, 0.4])

training_errors = np.array([
    [25.93, 19.35, 20.14, 21.64, 22.40, 23.82, 24.72, 25.07, 26.75, 26.03, 27.32, 27.18, 27.93, 27.66],
    [25.68, 29.02, 15.62, 16.46, 16.65, 17.19, 17.65, 17.78, 18.18, 18.48, 18.15, 20.16, 18.80, 18.22],
    [21.35, 24.24, 25.59, 18.30, 13.36, 14.81, 15.67, 15.58, 15.11, 15.49, 15.23, 15.44, 16.02, 16.47],
    [17.56, 29.39, 25.45, 23.31, 23.80, 18.77, 15.18, 14.57, 13.93, 14.65, 14.60, 13.66, 14.26, 15.17],
    [32.02, 21.53, 21.07, 17.86, 22.21, 31.43, 23.12, 20.16, 20.00, 18.21, 21.44, 19.27, 17.64, 16.98]
])

test_errors = np.array([
    [27.88, 22.01, 24.21, 24.43, 25.26, 26.00, 28.45, 27.52, 28.29, 30.17, 30.62, 30.26, 30.46, 29.79],
    [25.43, 28.96, 18.08, 19.29, 20.41, 20.18, 20.87, 21.26, 21.43, 21.19, 21.58, 23.13, 21.75, 21.49],
    [22.82, 25.65, 25.37, 19.57, 17.67, 17.91, 17.82, 19.70, 18.73, 19.19, 21.05, 19.87, 19.56, 18.93],
    [19.56, 27.28, 25.36, 23.84, 24.25, 20.42, 17.42, 17.25, 17.03, 17.82, 17.67, 17.46, 18.86, 18.61],
    [31.66, 24.07, 22.41, 20.93, 24.20, 31.79, 22.60, 21.80, 22.03, 20.51, 21.19, 20.62, 19.88, 20.09]
])

# Create a meshgrid for the initial variance mean and gain
X, Y = np.meshgrid(initial_var_mean, gain)

# Flatten the meshgrid to match the shape of the error arrays
X_flat = X.flatten()
Y_flat = Y.flatten()

# Flatten the training and test errors
training_errors_flat = training_errors.flatten()
test_errors_flat = test_errors.flatten()

# Create the 3D bar chart
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Width of the bars
width = 0.005
depth = 0.02

# Plot training errors
ax.bar3d(X_flat - width/2, Y_flat - depth/2, np.zeros_like(training_errors_flat), width, depth, training_errors_flat, shade=True, color='b', alpha=0.6, label='Training Error')

# Plot test errors
ax.bar3d(X_flat + width/2, Y_flat + depth/2, np.zeros_like(test_errors_flat), width, depth, test_errors_flat, shade=True, color='r', alpha=0.6, label='Test Error')

# Adding labels and title
ax.set_xlabel('Initial Variance Mean')
ax.set_ylabel('Gain')
ax.set_zlabel('Error (%)')
ax.set_title('Training and Test Errors vs Initial Variance Mean and Gain')

# Save the plot
plt.savefig('training_and_test_errors_vs_initial_variance_mean_and_gain_bar.png')