
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model_svm.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        komentar = request.form.get("komentar", "")

        if komentar.strip():
            vector = vectorizer.transform([komentar])
            prediction = model.predict(vector)[0]

            label = str(prediction).lower()

            if label in ["1", "positive", "positif"]:
                result = {
                    "label": "Positive",
                    "icon": "✨",
                    "class": "positive",
                    "desc": "Pemain memberikan pengalaman positif terhadap Magic Chess Go Go."
                }
            else:
                result = {
                    "label": "Negative",
                    "icon": "⚔️",
                    "class": "negative",
                    "desc": "Pemain memberikan kritik atau pengalaman negatif terhadap game."
                }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
