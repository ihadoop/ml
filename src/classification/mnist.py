
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt

# Example: Downloading the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, return_X_y=False)
X, y = mnist["data"], mnist["target"]
some_digit = X.loc[36001]
some_digit = some_digit.to_numpy()
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
