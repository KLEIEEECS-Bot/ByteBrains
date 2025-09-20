import pandas as pd
import os
import re
import string
import joblib
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from PIL import Image
import pytesseract

# --- Configuration ---
DATA_FOLDER = 'hackathon'
# You must update these column names
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'
MODEL_PATH = 'phishing_model.joblib'
VECTORIZER_PATH = 'tfidf_vectorizer.joblib'

# --- Preprocessor Function ---
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

# --- Training and Prediction Logic ---
def train_model():
    print("Checking for NLTK stopwords...")
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')

    print("--- Starting Phishing Model Training ---")
    print("Step 1: Loading and combining datasets...")
    dataframes = []
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith('.csv'):
            file_path = os.path.join(DATA_FOLDER, filename)
            try:
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                dataframes.append(df)
                print(f"Loaded '{filename}'")
            except Exception as e:
                print(f"Warning: Could not load '{filename}'. Skipping. Error: {e}")
    
    if not dataframes:
        print("An error occurred: No CSV files found. Please ensure they are in the 'hackathon' folder.")
        return

    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Preprocessing
    print("\nStep 2: Preprocessing text data...")
    try:
        df = combined_df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
        df['cleaned_text'] = df[TEXT_COLUMN].apply(clean_text)
    except KeyError as e:
        print(f"An error occurred: {e}. Please check your TEXT_COLUMN and LABEL_COLUMN variables.")
        return
        
    print("Data Preprocessing Complete.")
    
    # Training
    print("\nStep 3: Training and saving the model...")
    X = df['cleaned_text']
    y = df[LABEL_COLUMN]
    
    if X.empty or y.empty:
        print("An error occurred: No data to train on. Check data loading and preprocessing steps.")
        return
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(tfidf_vectorizer, VECTORIZER_PATH)
    print("Model Training Complete. Model and Vectorizer saved.")
    
    # Evaluation
    print("\nStep 4: Evaluating the model on the test set...")
    y_pred = model.predict(X_test_tfidf)
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    print("\n--- Phishing Model Training Complete ---")

def run_interactive_detector():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError:
        print("Error: Model or vectorizer files not found. Please run the training first.")
        return
        
    try:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    except Exception as e:
        print(f"Warning: Could not set Tesseract path. Please check the installation. Error: {e}")
        print("You can still use text input, but image detection will not work.")
        
    print("\n--- Advanced Interactive Phishing Detector ---")
    print("Enter a message or a file path to an image. Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYour input (text or file path): ")
        
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
            
        input_text = ""
        if os.path.isfile(user_input):
            try:
                print("Processing image, this might take a moment...")
                img = Image.open(user_input)
                input_text = pytesseract.image_to_string(img)
                print("Text extracted from image.")
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        else:
            input_text = user_input
            
        if not input_text or len(input_text.split()) < 5:
            print("\n--- Prediction ---")
            print("This message is likely LEGITIMATE. ✅ (Reason: Text is too short for meaningful analysis)")
            continue
            
        cleaned_text = clean_text(input_text)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)
        
        print("\n--- Prediction ---")
        if prediction[0] == '0.0':
            print("This message is likely LEGITIMATE. ✅")
        elif prediction[0] == '1.0':
            print("This message is a potential PHISHING attempt! ⚠️")
            print("\n**Guidance:**")
            print("- Do NOT click any links.")
            print("- Do NOT reply or provide any personal information.")
            print("- If it's from a known company, go to their official website directly to verify.")
        else:
            print("Unable to determine the class.")

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        train_model()
    else:
        print("Model files already exist. Skipping training.")
    
    run_interactive_detector()
