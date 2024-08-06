import nltk

nltk.download("punkt")

# NLTK and Spacy Tools
# Tokenization of word, sentence, character and subclass and punctuation
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize


# Word
input_text = "We are enjoying the party"
nltk_word = word_tokenize(input_text)
print(nltk_word)

# Sentence
input_text = "We are enjoying the party. We are having fun."
nltk_sent = sent_tokenize(input_text)
print(nltk_sent)

# Character
input_text = "We are having fun. We are having fun"
nltk_char = list(input_text)
print(nltk_char)

# Subword - convert word to root forms
input_text = "We are having-fun. We are enjoying the party"
nltk_subword = regexp_tokenize(input_text, pattern="\w+")
print(nltk_subword)
