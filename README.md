# Detection of Cyber Attacks in Wireless Sensor Networks using Machine Learning

MCA final-year project (Sree Vidyanikethan Engineering College, JNTUA, 2023).

## What it does
Classifies wireless-sensor-network traffic as **normal** or **attack** using three ML models,
compares them, and serves the best one through a small Flask web app.

- Models: AdaBoost, Gradient Boosting, Linear Discriminant Analysis (LDA)
- Metrics: accuracy, precision, recall, F1-score
- Stack: Python, scikit-learn, pandas, Flask

## Dataset
`generate_synthetic_data.py` creates a **synthetic** demo dataset so the code runs out of the box.
To use a real dataset (for example WSN-DS), place the CSV in `data/` with a final column named `label`
and run `python train.py data/your_file.csv`.
Results on the synthetic data are for demonstration only.

## Run
```bash
pip install -r requirements.txt
python generate_synthetic_data.py
python train.py
python app.py        # open http://127.0.0.1:5000
```

## Project structure
```
generate_synthetic_data.py   demo data
train.py                     trains + compares models, saves the best
app.py                       Flask web app
templates/index.html         web page
```
