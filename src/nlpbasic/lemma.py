from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
wordnet_lemmatizer = WordNetLemmatizer()
print(wordnet_lemmatizer.lemmatize("is"))
print(wordnet_lemmatizer.lemmatize("is", pos="v"))

text = nltk.word_tokenize("Process finished with exit code 0")
print(text)

print(nltk.pos_tag(text))


print(stopwords.words("english"))

