# Install the required library
# !pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Step 1: Define a function to analyze sentiment using VADER
def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)  # Get sentiment scores for the text
    return score


# Sample text data for sentiment analysis
text_data = [
    "I love the new features in this product! It's amazing.",
    "The service was terrible, and I will never return.",
    "I'm not sure how I feel about the changes. It's a bit confusing.",
    "What a fantastic experience! Highly recommend it to everyone.",
    "This is the worst day ever; I'm so upset.",
]

# Analyze sentiment for each text using VADER
for sentence in text_data:
    vader_result = analyze_sentiment_vader(sentence)
    print(f"Text: {sentence}")
    print(f"VADER Sentiment Score: {vader_result}")
    print("\n" + "=" * 80 + "\n")


# Install the required library
# !pip install textblob

from textblob import TextBlob


# Step 1: Define a function to analyze sentiment using TextBlob
def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
    }


# Analyze sentiment for each text using TextBlob
for sentence in text_data:
    textblob_result = analyze_sentiment_textblob(sentence)
    print(f"Text: {sentence}")
    print(
        f"TextBlob Polarity: {textblob_result['polarity']}, Subjectivity: {textblob_result['subjectivity']}"
    )
    print("\n" + "=" * 80 + "\n")
