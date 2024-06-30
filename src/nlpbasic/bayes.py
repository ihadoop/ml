from nltk.classify import NaiveBayesClassifier

def preprocess(s):
    s = s.lower()
    return {word: True for word in s.split()}

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
labels = ["pos", "neg", "pos", "neg", "pos", "neg", "pos", "neg", "pos", "neg"]
traning_data = [
    [preprocess(sentences[0]), labels[0]],
    [preprocess(sentences[1]), labels[1]],
    [preprocess(sentences[2]), labels[2]],
    [preprocess(sentences[3]), labels[3]],
    [preprocess(sentences[4]), labels[4]],
    [preprocess(sentences[5]), labels[5]],
    [preprocess(sentences[6]), labels[6]],
    [preprocess(sentences[7]), labels[7]],
    [preprocess(sentences[8]), labels[8]]
]
model = NaiveBayesClassifier.train(traning_data)

print(model.classify(preprocess(sentences[9])))
