from pathlib import Path
import os

from flask import Flask, render_template, request
import joblib

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # Batasi request agar ringan di deployment gratis

model = joblib.load(BASE_DIR / "model_svm.pkl")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    komentar = ""

    if request.method == "POST":
        komentar = request.form.get("komentar", "").strip()

        if len(komentar) > 1000:
            result = {
                "label": "TERLALU PANJANG",
                "icon": "!",
                "class": "neutral",
                "desc": "Komentar maksimal 1.000 karakter agar analisis tetap cepat dan ringan."
            }
        elif komentar:
            try:
                vector = vectorizer.transform([komentar])
                prediction = model.predict(vector)[0]
                label = str(prediction).strip().lower()

                positive_labels = {"positive", "positif", "pos"}
                neutral_labels = {"neutral", "netral", "neu"}
                negative_labels = {"negative", "negatif", "neg"}

                if label in positive_labels:
                    result = {
                        "label": "POSITIF",
                        "icon": "+",
                        "class": "positive",
                        "desc": "Komentar menunjukkan pengalaman atau penilaian yang positif terhadap Magic Chess GoGo."
                    }
                elif label in neutral_labels:
                    result = {
                        "label": "NETRAL",
                        "icon": "=",
                        "class": "neutral",
                        "desc": "Komentar cenderung bersifat netral, informatif, atau tidak menunjukkan sentimen positif maupun negatif yang kuat."
                    }
                elif label in negative_labels:
                    result = {
                        "label": "NEGATIF",
                        "icon": "−",
                        "class": "negative",
                        "desc": "Komentar menunjukkan keluhan atau pengalaman yang kurang baik terhadap Magic Chess GoGo."
                    }
                else:
                    result = {
                        "label": label.upper(),
                        "icon": "?",
                        "class": "neutral",
                        "desc": "Model menghasilkan label yang belum dipetakan pada tampilan aplikasi."
                    }

            except Exception:
                app.logger.exception("Gagal melakukan prediksi sentimen")
                result = {
                    "label": "ERROR",
                    "icon": "!",
                    "class": "negative",
                    "desc": "Terjadi kesalahan saat melakukan analisis. Silakan coba lagi."
                }

    return render_template("index.html", result=result, komentar=komentar)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
