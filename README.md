# Detection of Cyber Attacks in Wireless Sensor Networks using Machine Learning

MCA final-year project (Sree Vidyanikethan Engineering College, JNTUA, 2023).

## What it does
Classifies wireless sensor network records as Normal or one of four attacks (Blackhole, Grayhole, Flooding, TDMA) using three ML models, compares them, and serves the best one through a Flask web app.

- Models: AdaBoost, Gradient Boosting, Linear Discriminant Analysis (LDA)
- Metrics: accuracy, precision, recall, F1-score
- Stack: Python, scikit-learn, pandas, Flask

## Dataset
WSN-DS (Almomani et al., 2016), a public intrusion-detection dataset for wireless sensor networks, downloaded from Kaggle (search "WSN-DS", author Bassam Kasasbeh).
374,661 records, 17 numeric features, label column "Attack type".
After removing duplicates: 365,788 records (Normal 332,040; Grayhole 13,909; Blackhole 10,049; TDMA 6,633; Flooding 3,157).
The CSV is not stored in this repository because of its size. Download it and place it in the data folder.

## Results (test set, 80/20 stratified split)
| Model | Accuracy | Attack Precision | Attack Recall | Attack F1 |
|---|---|---|---|---|
| AdaBoost | 97.55% | 91.12% | 93.35% | 92.22% |
| Gradient Boosting | 99.54% | 98.40% | 98.09% | 98.24% |
| LDA | 95.92% | 84.01% | 96.00% | 89.61% |

Full details are in RESULTS.md. The classes are imbalanced (about 91% Normal), so attack precision, recall and F1 are more informative than accuracy.

## Run
    pip install -r requirements.txt
    python train.py data/WSN-DS.csv
    python app.py

Then open http://127.0.0.1:5000

## Project structure
    train.py              trains and compares the models, writes RESULTS.md and model/
    app.py                Flask web app
    templates/index.html  web page
    model/                results.csv, confusion matrices, best_model.joblib
