# Airline Passenger Satisfaction Classification

A machine learning project that predicts whether an airline passenger is satisfied with their flight, based on survey ratings and flight details. Built as an academic data science project.

The notebook takes the raw dataset through cleaning and encoding, compares four classifiers, tunes the best one, and finishes with a small simulation that draws random passengers from the test set and shows the model predicting their satisfaction live.

## Dataset

[Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) from Kaggle. Over 100,000 passenger records with features such as flight distance, seat comfort, inflight service ratings, delays, and the satisfaction label. The dataset is not included in this repo. Download it from Kaggle to run the notebook.

## What the notebook does

1. **Preprocessing:** drops ID columns, handles the missing values in arrival delay, label encodes the categorical features and the target, and checks feature correlation with satisfaction.
2. **Model comparison:** trains and evaluates Logistic Regression, Decision Tree, K-Nearest Neighbors, and Random Forest on a held out test set with full classification reports.
3. **Random Forest tuning:** a parameter search lands on 500 trees with no max depth limit.
4. **Simulation:** samples individual passengers from the test set, prints their details, and shows the tuned model predicting each one. Screenshots are in the sample output images in this repo.

## Results (test set)

| Model | Accuracy |
|---|---|
| Logistic Regression | 87.1% |
| Decision Tree | 94.7% |
| Random Forest (tuned) | 96.3% |

Tuned Random Forest detail: precision 0.973, recall 0.942, F1 score 0.957.

## Run it

```bash
pip install -r requirements.txt
```

Place the Kaggle files at `../Datasets/archive/train.csv` and `../Datasets/archive/test.csv` relative to the notebook, or edit the two `read_csv` paths in the loading cell to wherever you put them. Then run `airline_passenger_satisfaction.ipynb` top to bottom.

## Tech

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter.
