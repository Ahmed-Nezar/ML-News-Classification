import os
import joblib
import pandas as pd

from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def create_model():
    """
    Creates a machine learning model for news analysis.
    Returns:
        pipeline (Pipeline): A pipeline object that consists of the following steps:
            - vectorizer: CountVectorizer for converting text data into numerical features.
            - tfidf: TfidfTransformer for transforming the count matrix into a normalized tf-idf representation.
            - classifier: RandomForestClassifier for classifying the sentiment of the input text.
    """

    print("-- Creating model --")
    pipeline = Pipeline(
        [
            ("vectorizer", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("classifier", RandomForestClassifier(n_jobs=-1)),
        ]
    )
    print("-- Model created --")

    return pipeline


def train_model(model, X_train, y_train):
    print("-- Training model --")
    model.fit(X_train, y_train)
    print("-- Model trained --")
    return model


def save_model(model):
    os.makedirs("./models", exist_ok=True)
    model_name = f"model_{_get_time_str()}.pkl"
    joblib.dump(model, os.path.join("./models", model_name))
    print("-- Model saved --")


def load_model(path):
    model = joblib.load(path)
    print("-- Model loaded --")
    return model


def predict(model, X_test):
    print("-- Predicting --")
    y_pred = model.predict(X_test)
    print("-- Predicted --")
    return y_pred


def make_classification_report(y_test, y_pred):
    print("-- Classification report --")
    print(classification_report(y_test, y_pred))


def results_df(X_test, y_test, y_pred):
    results = pd.DataFrame({"Test_case": X_test, "actual": y_test, "predicted": y_pred})
    return results


def _get_time_str():
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")
