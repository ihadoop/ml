import tensorflow as tf

tf.constant('Hello, TensorFlow!')
tf.constant(1)

tf.constant(2,dtype=tf.float32)

a = tf.constant([True,False])
a = a.numpy()
a.ndim
tf.is_tensor(a)

tf.convert_to_tensor(a)