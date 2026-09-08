from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model_svm.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        komentar = request.form.get("komentar", "").strip()

        if komentar:
            try:
                vector = vectorizer.transform([komentar])
                prediction = model.predict(vector)[0]

                label = str(prediction).lower()

                positive_labels = ["1", "positive", "positif", "pos"]

                if label in positive_labels:
                    result = {
                        "label": "POSITIVE",
                        "icon": "🏆",
                        "class": "positive",
                        "desc": "Komentar menunjukkan pengalaman pemain yang baik terhadap Magic Chess Go Go."
                    }
                else:
                    result = {
                        "label": "NEGATIVE",
                        "icon": "⚔️",
                        "class": "negative",
                        "desc": "Komentar menunjukkan adanya keluhan atau pengalaman kurang baik dari pemain."
                    }

            except Exception as e:
                result = {
                    "label": "ERROR",
                    "icon": "⚠️",
                    "class": "negative",
                    "desc": str(e)
                }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)