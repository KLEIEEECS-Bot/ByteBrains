import joblib
import re
import string
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import pytesseract
import io
import nltk
from nltk.corpus import stopwords
import os

# --- Load the trained model and vectorizer ---
try:
    model = joblib.load('phishing_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
except FileNotFoundError:
    print("Model files not found. Please run phishing_trainer.py first.")
    exit()

# --- Preprocessor Function (same as before) ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for cross-origin requests

# Configure pytesseract to find the Tesseract executable
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception as e:
    print(f"Warning: Could not set Tesseract path. Please check the installation. Error: {e}")

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    input_type = data.get('type')
    input_data = data.get('data')

    input_text = ""
    if input_type == 'text':
        input_text = input_data
    elif input_type == 'image':
        try:
            # Decode the base64 image data
            image_bytes = base64.b64decode(input_data.split(',')[1])
            img = Image.open(io.BytesIO(image_bytes))
            input_text = pytesseract.image_to_string(img)
        except Exception as e:
            return jsonify({'error': f"Error processing image: {e}"}), 400

    if not input_text or len(input_text.split()) < 5:
        return jsonify({'is_phishing': False, 'reason': 'short_text'})

    cleaned_text = clean_text(input_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    
    is_phishing = False
    if prediction[0] == 1.0:
        is_phishing = True
    
    return jsonify({'is_phishing': is_phishing})

if __name__ == '__main__':
    app.run(debug=True)