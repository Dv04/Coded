import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups

import os  # For file path operations


def load_data():
    # Here, we're using a placeholder dataset. Replace this with your dataset.
    # Ensure that your dataset is in a structure where one column is the text, and another column is the sentiment label.

    # Loading some sample data
    categories = [
        "rec.sport.baseball",
        "sci.crypt",
    ]  # Just as an example, these categories could be 'positive' and 'negative' in your case
    data = fetch_20newsgroups(subset="all", categories=categories)

    # Create a DataFrame from the loaded data
    df = pd.DataFrame({"text": data["data"], "sentiment": data["target"]})
    return df


def preprocess_and_split(df):
    # If you have any specific text preprocessing steps, implement them here
    # For now, we'll just split the dataset into training and testing subsets

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"], test_size=0.2, random_state=0
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # Creating a pipeline that first creates bag of words (after removing stop words), then applies TF-IDF transformer,
    # and then fits a logistic regression model
    text_clf = Pipeline(
        [
            ("vect", CountVectorizer(stop_words="english")),
            ("tfidf", TfidfTransformer()),
            ("clf", LogisticRegression()),
        ]
    )

    # Train the model with the training data
    text_clf.fit(X_train, y_train)
    return text_clf


def evaluate_model(text_clf, X_test, y_test):
    # Predict the test set results
    y_pred = text_clf.predict(X_test)

    # Calculate the accuracy of the model
    print(f"\n\nAccuracy: {accuracy_score(y_test, y_pred)}")


def predict_sentiment(text_clf, new_texts):
    # Ensure the input is a list of strings
    if not isinstance(new_texts, list):
        new_texts = [new_texts]

    # Predict the sentiment of each text
    predicted_sentiments = text_clf.predict(new_texts)

    return predicted_sentiments


def main():
    # Load the data
    df = load_data()

    # Preprocess and split the data
    X_train, X_test, y_train, y_test = preprocess_and_split(df)

    # Train the model
    text_clf = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(text_clf, X_test, y_test)

    while True:  # Allowing multiple predictions without restarting the script
        new_texts = input("\nInput a custom statement (or 'exit' to quit): ")
        if new_texts.lower() == "exit":
            print("Exiting the program.")
            break
        predictions = predict_sentiment(text_clf, new_texts)
        print(
            f"\nText: {new_texts}\nPredicted sentiment: {'Positive' if predictions else 'Negative'}"
        )


if __name__ == "__main__":
    main()
