# ML-News-Classification

This project implements a machine learning pipeline for classifying news articles into four categories: World, Sports, Business, and Sci/Tech. The model uses Natural Language Processing (NLP) techniques to process the news articles and train a classification model.

## Project Structure

- **src/**
  - `main.py`: The main script that handles data loading, model training, and evaluation.
  - `preprocess.py`: Contains functions for loading and preprocessing the data, including labeling the news categories.
  - `model.py`: Defines the machine learning pipeline, including model creation, training, saving, and evaluation.

- **notebooks/**
  - `News-Analysis-ML-Based.ipynb`: Jupyter notebook containing exploratory data analysis and initial model experiments.

## Dataset

The dataset used in this project is **AG News** provided by Xiang Zhang, which can be accessed from the following link:  
[AG News Dataset](https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M?resourcekey=0-TLwzfR2O-D2aPitmn5o9VQ)

The dataset consists of four categories of news: World, Sports, Business, and Sci/Tech.

## How to Run

1. **Clone the Repository**  
   Clone this repository to your local machine

2. **Download Dataset**  
   Download the dataset from the provided link and place it in a directory named `ag_news_csv`.

3. **Run the Model**  
   Execute the main script to train the model and evaluate its performance:
   ```bash
   python src/main.py
   ```

4. **View Results**  
   After running the script, the trained model will be saved, and a classification report will be displayed, showing the performance of the model on the test set.

## Model Details

The model uses the following steps in the pipeline:
1. **CountVectorizer**: Converts the news text into a bag-of-words representation.
2. **TfidfTransformer**: Transforms the bag-of-words into a TF-IDF matrix, giving more weight to less frequent but important words.
3. **RandomForestClassifier**: A random forest classifier is used to predict the category of news articles.

## Initial Results
Below is the classification report for the initial model trained on the AG News dataset:
```
              precision    recall  f1-score   support

    Business       0.86      0.80      0.83      1900
    Sci/Tech       0.83      0.85      0.84      1900
      Sports       0.90      0.97      0.93      1900
       World       0.90      0.87      0.88      1900

    accuracy                           0.87      7600
   macro avg       0.87      0.87      0.87      7600
weighted avg       0.87      0.87      0.87      7600

```
   

## Future Work

In future updates, we plan to:
- **Fine-tune the model** using different hyperparameters and additional NLP techniques.
- **Experiment with different models** such as Support Vector Machines (SVM), XGBoost, or even deep learning models like BERT to potentially improve classification performance.

## Credits

- The AG News dataset is credited to **Xiang Zhang**.
- The project uses the **scikit-learn** library for machine learning and **pandas** for data handling.

---
