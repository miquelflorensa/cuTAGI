import numpy as np

data_file = "./data/UCI/Concrete/data/data.txt"

# Read and split the data
data = np.loadtxt(data_file)

# Split the data into training and testing
np.random.shuffle(data)
train_data = data[:int(0.5 * len(data))]
test_data = data[int(0.5 * len(data)):]

# Save the training and testing data in x and y files
np.savetxt("./data/UCI/Concrete/data/x_train.csv", train_data[:, :-1], delimiter=",")
np.savetxt("./data/UCI/Concrete/data/y_train.csv", train_data[:, -1], delimiter=",")
np.savetxt("./data/UCI/Concrete/data/x_test.csv", test_data[:, :-1], delimiter=",")
np.savetxt("./data/UCI/Concrete/data/y_test.csv", test_data[:, -1], delimiter=",")

