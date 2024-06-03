import tensorflow as tf
import numpy as np
import collections
import random
import math

# Parameters
vocabulary_size = 487
embedding_dim = 128
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.

# Sample data preparation (replace with your data)
def read_data(filename):
    with open(filename, 'r') as f:
        return f.read().split()

# Build the dictionary and replace rare words with UNK token.
def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

filename =  "../../data/word.txt"
words = read_data(filename)
data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
del words  # Hint to reduce memory.

# Generate a training batch for the skip-gram model.
def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

# Build and train a skip-gram model.
class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.nce_weights = tf.Variable(tf.random.truncated_normal([vocab_size, embedding_dim], stddev=1.0 / math.sqrt(embedding_dim)))
        self.nce_biases = tf.Variable(tf.zeros([vocab_size]))

    def call(self, inputs, labels):
        embed = self.embedding(inputs)
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weights, biases=self.nce_biases, labels=labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))
        return loss

vocab_size = len(dictionary)
data_index = 0
batch_size = 128
embedding_dim = 128
num_steps = 100001

word2vec = Word2Vec(vocab_size, embedding_dim)
optimizer = tf.optimizers.Adam()

for step in range(num_steps):
    batch_inputs, batch_labels = generate_batch(data, batch_size, num_skips, skip_window)
    with tf.GradientTape() as tape:
        loss = word2vec(batch_inputs, batch_labels)
    gradients = tape.gradient(loss, word2vec.trainable_variables)
    optimizer.apply_gradients(zip(gradients, word2vec.trainable_variables))
    if step % 2000 == 0:
        print(f'Step {step}, Loss: {loss.numpy()}')

# Extract trained embeddings
embeddings = word2vec.embedding.get_weights()[0]

# Save embeddings for later use
np.save('word2vec_embeddings.npy', embeddings)
