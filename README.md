# 📩 SMS Spam Detection

A machine learning project that classifies an SMS/email message as **Spam** or **Not Spam** using Natural Language Processing (NLP), TF-IDF vectorization, and a Multinomial Naive Bayes classifier. The trained model is deployed through a simple Streamlit web application.

## 📌 Project Overview

The project follows this workflow:

**Data Cleaning → EDA → Text Preprocessing → TF-IDF Vectorization → Model Building → Evaluation → Streamlit Web App → Deployment**

The notebook uses the SMS spam dataset and prepares the text for machine learning. The final application accepts a message from the user and predicts whether it is spam.

## ✨ Features

- SMS/email spam classification
- Text preprocessing using NLTK
- Tokenization
- Removal of special characters
- Removal of stop words and punctuation
- Porter stemming
- TF-IDF text vectorization
- Multinomial Naive Bayes classification
- Interactive Streamlit interface
- Saved model and vectorizer using Pickle
- Streamlit deployment configuration

## 🛠️ Technologies Used

- **Python**
- **Pandas & NumPy** – data handling
- **NLTK** – natural language preprocessing
- **Scikit-learn** – vectorization and machine learning
- **TF-IDF** – text feature extraction
- **Multinomial Naive Bayes** – final prediction model
- **Streamlit** – web application
- **Pickle** – model/vectorizer serialization

The application imports Streamlit, NLTK, Pickle, and other Python libraries, while the requirements file specifies Streamlit, NLTK, and Scikit-learn. 

## 📂 Project Structure

```text
SMS-Spam-Detection/
│
├── app.py
├── sms-spam-detection.ipynb
├── spam.csv
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── nltk.txt
├── setup.sh
├── Procfile
└── README.md
```

### File Description

| File | Description |
|---|---|
| `app.py` | Streamlit application for predicting spam/not-spam messages |
| `sms-spam-detection.ipynb` | Complete data science and machine learning workflow |
| `spam.csv` | SMS spam dataset |
| `model.pkl` | Saved trained machine learning model |
| `vectorizer.pkl` | Saved TF-IDF vectorizer |
| `requirements.txt` | Python dependencies |
| `nltk.txt` | NLTK resources required by the application |
| `setup.sh` | Streamlit server configuration for deployment |
| `Procfile` | Deployment process configuration |
| `README.md` | Project documentation |

## 🔄 Machine Learning Workflow

### 1. Data Cleaning

The dataset is loaded using Pandas. Unnecessary columns are removed, the original columns are renamed to `target` and `text`, and the target labels are encoded.

Duplicate records are also removed.

### 2. Exploratory Data Analysis

The notebook analyzes:

- Class distribution
- Number of characters
- Number of words
- Number of sentences
- Differences between ham and spam messages
- Common words in spam and ham messages

The dataset is identified as imbalanced during EDA.

### 3. Text Preprocessing

The same preprocessing pipeline is used by the application:

1. Convert text to lowercase
2. Tokenize the text
3. Keep alphanumeric tokens
4. Remove English stop words and punctuation
5. Apply Porter stemming
6. Join the processed tokens back into a string

The Streamlit application implements this preprocessing in `transform_text()`. 

### 4. TF-IDF Vectorization

The processed text is converted into numerical features using:

```python
TfidfVectorizer(max_features=3000)
```

The trained vectorizer is saved as `vectorizer.pkl`.

### 5. Model Training

The notebook evaluates multiple machine learning classifiers, including:

- SVC
- K-Nearest Neighbors
- Multinomial Naive Bayes
- Decision Tree
- Logistic Regression
- Random Forest
- AdaBoost
- Bagging
- Extra Trees
- Gradient Boosting
- XGBoost

The notebook also experiments with voting and stacking classifiers.

The final saved prediction model is the **Multinomial Naive Bayes (`MultinomialNB`)** model.

## 🧠 Prediction Pipeline

When a user enters a message:

```text
Input Message
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Trained Model
     ↓
Prediction
     ↓
Spam / Not Spam
```

The Streamlit application loads `vectorizer.pkl` and `model.pkl`, transforms the entered message, and passes the resulting vector to the model for prediction. 

## 🌐 Streamlit Application

The application provides:

- Title: **Email/SMS Spam Classifier**
- Text box for entering a message
- **Predict** button
- Prediction output as either:
  - `Spam`
  - `Not Spam`

This interface and prediction flow are implemented directly in `app.py`. 

## 🚀 Installation & Running Locally

### 1. Clone/download the project

Place all project files in the same directory.

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux/macOS**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The provided requirements file contains Streamlit, NLTK, and Scikit-learn. 

### 4. Download NLTK resources

The project requires:

```text
stopwords
punkt
```

These resources are also downloaded by the application when it starts. 

### 5. Run the application

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## ☁️ Deployment

The project includes deployment-related files:

- `Procfile`
- `setup.sh`
- `requirements.txt`

The setup script creates Streamlit configuration and uses the deployment-provided `$PORT`, with CORS disabled and headless mode enabled. 

For deployment, make sure these files are included along with:

```text
app.py
model.pkl
vectorizer.pkl
requirements.txt
nltk.txt
Procfile
setup.sh
```

## ⚠️ Important Notes

- `model.pkl` and `vectorizer.pkl` must be available in the same project directory expected by the application.
- The preprocessing used during prediction should remain consistent with the preprocessing used during training.
- The application currently downloads the NLTK `punkt` and `stopwords` resources at startup.
- The README describes the implementation present in the supplied project files; no additional model-performance values are claimed here unless they are explicitly available in the source files.

## 🔮 Future Improvements

- Add prediction probability/confidence
- Improve the user interface
- Add batch CSV prediction
- Add model performance visualizations to the web app
- Add more robust input validation
- Compare additional preprocessing techniques
- Add automated testing
- Improve deployment configuration and dependency pinning

## 👨‍💻 Project Type

**Machine Learning / NLP / Streamlit Deployment**

---

### Author

**Jatin Jangir**

