import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('wordnet')

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Word Tokenization
input_text = "We are enjoying the party"
nltk_word = word_tokenize(input_text)
print("Word Tokenization:", nltk_word)

# Stemming
stemmed_words = [stemmer.stem(word) for word in nltk_word]
print("Stemmed Words:", stemmed_words)

# Lemmatization
lemmatized_words = [lemmatizer.lemmatize(word, pos="v") for word in nltk_word]
print("Lemmatized Words:", lemmatized_words)

# Sentence Tokenization
input_text = "We are enjoying the party. We are having fun."
nltk_sent = sent_tokenize(input_text)
print("Sentence Tokenization:", nltk_sent)

# Character Tokenization
input_text = "We are having fun. We are having fun"
nltk_char = list(input_text)
print("Character Tokenization:", nltk_char)

# Subword Tokenization
input_text = "We are having-fun. We are enjoying the party"
nltk_subword = regexp_tokenize(input_text, pattern="\w+")
print("Subword Tokenization:", nltk_subword)
