from nltk.text import TextCollection
import nltk
from nltk import FreqDist
from sklearn.preprocessing import StandardScaler

sentences = ["This book is very informative, a must-read.",
             "The book was poorly written, I don't recommend it.",
             "It's one of the most interesting books I've read.",
             "The plot of this book is too slow, not enjoyable.",
             "I found this book to be very useful and insightful.",
             "The story in this book is too bland, not engaging.",
             "This book offers a unique perspective, refreshing to read.",
             "The book lacks depth, it was quite boring.",
             "The writing style of this book is beautiful, worth reading.",
             "The structure of this book is confusing, it's tiring to read."]

corpus = TextCollection(sentences)
sentence = "The structure of this book is confusing, it's tiring to read."
tokens = nltk.word_tokenize("".join(sentences))
freq = FreqDist(tokens)
freq_common = freq.most_common(50)
vocabulary = []
for word,times in freq_common:
    vocabulary.append(word)
vector = []
for word in vocabulary:
    vector.append(corpus.tf_idf(word,"story in this book is too bland, not engaging."))
print(vector)