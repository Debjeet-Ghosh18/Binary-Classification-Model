from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained binary classification model
with open("Binary_Classification_Model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = float(request.form["age"])  # user input
    features = np.array([[age]])      # model expects age as feature

    prediction = model.predict(features)[0]

    if prediction == 1:
        result = "✅ Yes, this person bought insurance."
    else:
        result = "❌ No, this person did not buy insurance."

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
