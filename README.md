# 🌸 Iris Flower Classification

A Streamlit web app that predicts the species of an iris flower
(setosa, versicolor or virginica) from its sepal and petal measurements.

**Live app:** _add your Streamlit link here_

## How it works
1. Loads the Iris dataset (150 flowers, 4 measurements each) from scikit-learn.
2. Splits it 80/20 into training and test sets.
3. Trains a Random Forest classifier.
4. Takes your slider inputs and shows the predicted species with probabilities.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech
Python · Streamlit · scikit-learn · pandas
