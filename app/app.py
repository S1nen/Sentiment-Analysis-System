from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("models/xgb_model2.pkl", "rb"))
vectorizer = pickle.load(open("models/TfidfVectorizer.pkl", "rb"))
scaler = pickle.load(open("models/scaler_new2.pkl", "rb")) 

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]
    vector = vectorizer.transform([review]).toarray()
    scaled_vector = scaler.transform(vector)
    prediction = model.predict(scaled_vector)[0]

    proba = model.predict_proba(scaled_vector)[0]
    confidence = max(proba)

    if confidence < 0.4:
        label = "Neutral / Unclear"
    else:
        label = "positive" if prediction == 1 else "negative"

    return render_template("index.html", review=review, prediction=label)

if __name__ == "__main__":
    app.run(debug=True)
