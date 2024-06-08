
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# Example: Downloading the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, return_X_y=False)
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train.to_numpy()[shuffle_index], y_train.to_numpy()[shuffle_index]

some_digit = X.loc[36001]
some_digit = some_digit.to_numpy()
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
print(y[36001])
plt.show()
