from nltk import FreqDist
import nltk


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

tokens = nltk.word_tokenize("".join(sentences))
freq = FreqDist(tokens)
freq_common = freq.most_common(50)

for word, freq in freq_common:
    print(word, freq)
