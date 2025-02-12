from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
# import requests
import gdown

app = Flask(__name__)
CORS(app)

# Load tokenizer and model
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


MODEL_PATH = "model.keras"
MODEL_URL = (
    "https://drive.google.com/uc?export=download&id=1uqqZiZsmI2wnxG6r9Z4IQq94rT9fGwgH"
)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model using gdown...")

        # Use gdown to download the file
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

        if os.path.exists(MODEL_PATH):
            print("Model downloaded successfully.")
        else:
            print("Error: Model download failed.")

download_model()


print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

MAX_LEN = 30


def predict_next_words(text, top_k=3):
    # Tokenize input text
    sequence = tokenizer.texts_to_sequences([text])[0]

    if len(sequence) == 0:  # If tokenization results in empty list
        return []

    # Keep only the last 30 tokens if input is too long
    sequence = sequence[-MAX_LEN:]

    # Pad sequence to required length (post-padding)
    sequence = pad_sequences([sequence], maxlen=MAX_LEN, padding="post")

    # Predict next word probabilities
    predictions = model.predict(sequence)[0]

    # Get top-k word indices
    top_indices = np.argsort(predictions)[-top_k:][::-1]

    # Convert indices back to words
    word_index = tokenizer.index_word
    top_words = [word_index.get(i, "") for i in top_indices]

    return [word for word in top_words if word]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    predictions = predict_next_words(text)
    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(debug=True)
