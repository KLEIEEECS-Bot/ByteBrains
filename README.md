# 62.-ByteBrains-Help Me Say No to Phishing
# 🎣 **62.-ByteBrains-Help Me Say No to Phishing**

An AI-powered assistant designed for an 8-hour hackathon to protect users from phishing attacks. This project provides a simple, interactive web interface where users can paste suspicious messages or upload screenshots. The assistant then uses a machine learning model to quickly analyze the content and provide a verdict, along with actionable guidance.

---

### ✨ **Key Features**

* **Text Analysis:** A machine learning model analyzes the text of a message for linguistic patterns, grammatical errors, and urgency cues common in phishing scams.
* **Image-based Detection:** Uses Optical Character Recognition (OCR) to extract text from image files and screenshots, allowing for the analysis of messages from any source.
* **Intelligent URL Validation:** A logic layer checks if a message's URL belongs to a list of trusted domains, preventing false positives.
* **Interactive Web UI:** A modern, responsive web application provides a seamless user experience and clear, visual feedback.

---

### 🛠️ **Tech Stack**

| Component | Technologies Used |
| :--- | :--- |
| **Frontend** | `HTML`, `Tailwind CSS`, `JavaScript` |
| **Backend** | `Python`, `Flask`, `Flask-CORS` |
| **Machine Learning** | `scikit-learn`, `NLTK`, `joblib` |
| **OCR** | `Tesseract OCR`, `pytesseract`, `Pillow` |

---

### 🚀 **How to Run the Project**

Follow these steps to set up and run the entire application locally.

#### **Prerequisites**

* **Tesseract OCR:** You must have Tesseract installed on your system. If you haven't already, install the executable for Windows and ensure it's added to your system's PATH.

#### **Setup**

1.  **Clone the Repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd [YOUR_REPOSITORY_FOLDER]
    ```

2.  **Download Datasets:** This project requires external datasets to train the model. Download the following datasets from Kaggle and place the `.csv` files in your project directory:
    * **Phishing Email Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
    * **Phishing Website Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning)

3.  **Install Python Dependencies:**
    ```bash
    pip install Flask Flask-Cors scikit-learn pandas nltk pytesseract Pillow
    ```

4.  **Train the Model:** The `improved_phishing_detector.py` file will train your model and save it to disk.
    ```bash
    python improved_phishing_detector.py
    ```

5.  **Run the Backend Server:** Start the Flask server that the website will communicate with.
    ```bash
    python app.py
    ```

6.  **Open the Web App:** With the server running, open the `index.html` file in your web browser.

---

### 🖥️ **Usage**

* **Paste Text:** Copy a message from an email or chat and paste it into the text area.
* **Upload an Image:** Take a screenshot of a suspicious message and upload the image file.
* **Click "Analyze":** The app will send the content to your model for a real-time prediction.
* **Review the Result:** A clear verdict will be displayed, along with guidance on how to respond safely.

---

<br>

***This project was built for the 8-hour hackathon by ByteBrains.***
Review the Result: A clear verdict will be displayed, along with guidance on how to respond safely.

