import pandas as pd
import os
import re
import string
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# --- Step 1: Data Loader Function (Self-Contained) ---
def load_and_combine_data(data_folder):
    """Loads all CSV files from a folder and combines them into one DataFrame."""
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder not found at '{data_folder}'")
    dataframes = []
    
    # Iterate over all files in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_folder, filename)
            try:
                # Read the CSV file, skipping bad lines to avoid errors
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                dataframes.append(df)
                print(f"Loaded '{filename}'")
            except Exception as e:
                print(f"Warning: Could not load '{filename}'. Skipping. Error: {e}")
    if not dataframes:
        raise ValueError("No CSV files found in the specified folder.")
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

# --- Step 2: Preprocessor Function (Self-Contained) ---
def clean_text(text):
    """Cleans a single string of text."""
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

def preprocess_data(df, text_column, label_column):
    """Applies the cleaning function to a specified text column in a DataFrame."""
    # Find the actual text and label columns based on a common name list
    text_col_candidates = ['text', 'body', 'message', 'email_content', 'email']
    label_col_candidates = ['label', 'class', 'type']
    
    found_text_col = next((col for col in df.columns if col.lower() in text_col_candidates), None)
    found_label_col = next((col for col in df.columns if col.lower() in label_col_candidates), None)

    if not found_text_col:
        raise KeyError(f"Could not find a suitable text column. Searched for: {text_col_candidates}")
    if not found_label_col:
        raise KeyError(f"Could not find a suitable label column. Searched for: {label_col_candidates}")

    df = df.dropna(subset=[found_text_col, found_label_col]).copy()
    df['cleaned_text'] = df[found_text_col].apply(clean_text)
    df['label'] = df[found_label_col]
    
    return df[['cleaned_text', 'label']]

# --- Step 3: Trainer Function (Self-Contained) ---
def train_and_save_model(df, text_column, label_column, model_path, vectorizer_path):
    """Trains a Naive Bayes classifier and a TF-IDF vectorizer, then saves them."""
    if text_column not in df.columns or label_column not in df.columns:
        raise KeyError(f"Missing required columns. Ensure '{text_column}' and '{label_column}' exist.")
    X = df[text_column]
    y = df[label_column]
    if X.empty or y.empty:
        raise ValueError("No data to train on. Check data loading and preprocessing steps.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    joblib.dump(model, model_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    return model, tfidf_vectorizer, X_test_tfidf, y_test

# --- MAIN EXECUTION SCRIPT ---
if __name__ == '__main__':
    # Download NLTK stopwords. The download function is designed to check
    # if the resource is present and only download if it's not.
    print("Checking for NLTK stopwords...")
    nltk.download('stopwords')

    # --- Configuration ---
    DATA_FOLDER = '.' # Fix: Use '.' for the current directory
    TEXT_COLUMN = 'text' # **UPDATE THIS** to your actual text column name
    LABEL_COLUMN = 'label' # **UPDATE THIS** to your actual label column name
    MODEL_PATH = 'phishing_model.joblib'
    VECTORIZER_PATH = 'tfidf_vectorizer.joblib'

    print("--- Starting Phishing Model Training ---")
    try:
        # Step 1: Load the data
        print("Step 1: Loading and combining datasets...")
        raw_df = load_and_combine_data(DATA_FOLDER)
        print("Raw Data Loaded. Shape:", raw_df.shape)
        print("Columns:", raw_df.columns.tolist())
        
        # Step 2: Preprocess the data
        print("\nStep 2: Preprocessing text data...")
        # We pass a None value for the column names and let the function discover them
        processed_df = preprocess_data(raw_df, text_column=None, label_column=None)
        print("Data Preprocessing Complete.")
        
        # Step 3: Train and save the model
        print("\nStep 3: Training and saving the model...")
        model, vectorizer, X_test_tfidf, y_test = train_and_save_model(
            processed_df,
            text_column='cleaned_text',
            label_column='label',
            model_path=MODEL_PATH,
            vectorizer_path=VECTORIZER_PATH
        )
        print("Model Training Complete. Model and Vectorizer saved.")
        
        # Step 4: Evaluate the model
        print("\nStep 4: Evaluating the model on the test set...")
        y_pred = model.predict(X_test_tfidf)
        print("--- Classification Report ---")
        print(classification_report(y_test, y_pred))
        print("\n--- Phishing Model Training Complete ---")
        
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your data folder path and column names in the configuration section.")
