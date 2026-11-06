"""Flask app: enter sensor-network traffic values and see whether it looks like an attack."""
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)
bundle = joblib.load("model/best_model.joblib")
pipe, FEATURES, CLASSES = bundle["pipeline"], bundle["features"], bundle["classes"]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    values = {}
    if request.method == "POST":
        try:
            values = {f: float(request.form[f]) for f in FEATURES}
            import pandas as pd
            row = pd.DataFrame([values])
            pred = pipe.predict(row)[0]
            prob = pipe.predict_proba(row)[0].max()
            result = f"{CLASSES[pred].upper()} (confidence {prob:.0%})"
        except ValueError:
            result = "Please enter valid numbers in all fields."
    return render_template("index.html", features=FEATURES, result=result,
                           values=values, model_name=bundle["name"])

if __name__ == "__main__":
    app.run(debug=True)
