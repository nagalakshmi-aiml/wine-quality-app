from flask import Flask, render_template, request, jsonify
import joblib, pandas as pd

app = Flask(__name__)
model = joblib.load("model/wine_model.pkl")

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol",
]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        values = [float(request.form[f]) for f in FEATURES]
        df = pd.DataFrame([values], columns=FEATURES)
        prediction = round(float(model.predict(df)[0]), 2)
    return render_template("index.html", features=FEATURES, prediction=prediction)

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    df = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    return jsonify({"quality": round(float(model.predict(df)[0]), 2)})

if __name__ == "__main__":
    app.run(debug=True)