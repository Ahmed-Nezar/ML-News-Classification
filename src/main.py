from preprocess import get_train_test, label_class
from model import (
    create_model,
    train_model,
    save_model,
    load_model,
    predict,
    make_classification_report,
)


if __name__ == "__main__":
    path = "ag_news_csv"
    X_train, X_test, y_train, y_test = get_train_test(path)
    y_train, y_test = label_class(y_train, y_test)
    model = create_model()
    model = train_model(model, X_train, y_train)
    save_model(model)
    y_pred = predict(model, X_test)
    make_classification_report(y_test, y_pred)
