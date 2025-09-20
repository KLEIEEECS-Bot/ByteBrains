import joblib
import re
import string
from nltk.corpus import stopwords
import nltk

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

# --- Main Testing Logic ---
if __name__ == '__main__':
    # Load the trained model and vectorizer
    try:
        print("Loading trained model and vectorizer...")
        model = joblib.load('phishing_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        print("Model and vectorizer loaded successfully.")
    except FileNotFoundError:
        print("Error: Model or vectorizer files not found. Please run phishing_trainer.py first to create them.")
        exit()

    # --- Sample Emails to Test ---
    # Example 1: Legitimate email
    legit_email = """
    Hello,
    
    This is a quick reminder about our team meeting on Monday at 10 AM. We will be discussing the Q3 results.
    
    Thanks,
    John
    """
    
    # Example 2: Phishing email (notice the urgent tone and unusual link)
    phishing_email = """
    Urgent: Your account has been compromised.
    
    We have detected suspicious activity on your account. To prevent unauthorized access, you must verify your login credentials immediately by clicking the link below:
    
    https://verify.acc0unt-security.com/login?id=456
    
    Failure to do so will result in the suspension of your account.
    """

    test_emails = [
        {"text": legit_email, "type": "Legitimate"},
        {"text": phishing_email, "type": "Phishing"}
    ]

    for i, email_data in enumerate(test_emails):
        print(f"\n--- Testing Sample Email {i + 1} ({email_data['type']}) ---")
        
        # 1. Pre-process the new email text
        print("Preprocessing text...")
        cleaned_text = clean_text(email_data['text'])
        
        # 2. Vectorize the cleaned text
        print("Vectorizing text...")
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # 3. Predict the class
        print("Making a prediction...")
        prediction = model.predict(vectorized_text)
        
        # 4. Print the result
        print("Prediction Result:", prediction)
        if prediction[0] == 0:
            print("Conclusion: Predicted as a Legitimate Email (Ham) ✅")
        elif prediction[0] == 1:
            print("Conclusion: Predicted as a Phishing Attempt (Spam) ⚠️")
        else:
            print("Conclusion: Unable to determine the class.")
            
    print("\nScript finished. All samples tested.")