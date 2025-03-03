import tensorflow as tf
from tensorflow.keras import datasets,layers,optimizers

(xs,ys),_ = datasets.mnist.load_data()

print("datasets:",xs.shape,ys.shape)