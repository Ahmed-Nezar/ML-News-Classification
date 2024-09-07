import pandas as pd
import os


def get_train_test(path: str) -> pd.DataFrame:

    train = pd.read_csv(os.path.join(path, "train.csv"))
    test = pd.read_csv(os.path.join(path, "test.csv"))

    train.columns = ["class", "title", "review"]
    test.columns = ["class", "title", "review"]

    X_train = train["review"]
    y_train = train["class"]
    X_test = test["review"]
    y_test = test["class"]

    label_class(y_train, y_test)

    return X_train, X_test, y_train, y_test


def label_class(y_train, y_test):
    y_train = y_train.map({1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"})
    y_test = y_test.map({1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"})

    return y_train, y_test
