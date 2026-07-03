## 🌐 Live Demo
https://arilinesentimentnlp-fkslvtcvrcqqtfyrmxmdmv.streamlit.app/

## 📸 Demo
<img width="1907" height="825" alt="Screenshot 2026-07-03 141127" src="https://github.com/user-attachments/assets/13e10996-dc4c-40c0-8d46-5a507b78941f" />
<img width="1907" height="877" alt="Screenshot 2026-07-03 141148" src="https://github.com/user-attachments/assets/2a242bfc-9d73-4ac8-9b63-2955074b9e75" />
<img width="1892" height="875" alt="Screenshot 2026-07-03 141220" src="https://github.com/user-attachments/assets/c6329925-f2bf-4d38-aa6b-8ad93077ba5e" />




# ✈️ Airline Sentiment Analysis using Word2Vec & Artificial Neural Network (ANN)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![NLP](https://img.shields.io/badge/NLP-Word2Vec-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview

This project performs **multiclass sentiment classification** on airline passenger tweets using **Natural Language Processing (NLP)** and **Deep Learning**.

Instead of using pre-trained embeddings like GloVe, this project trains a **custom Word2Vec model** on the dataset, converts each tweet into a dense vector representation, and feeds these vectors into an **Artificial Neural Network (ANN)** for sentiment prediction.

The application is deployed using **Streamlit**, allowing users to classify airline tweets interactively.

---

## 🎯 Problem Statement

Airlines receive thousands of customer tweets every day.

Manually analyzing customer opinions is difficult and time-consuming.

This project automates sentiment classification into:

- 😊 Positive
- 😐 Neutral
- 😠 Negative

allowing organizations to understand customer satisfaction quickly.

---

# 📊 Dataset

**Dataset Name**

Twitter US Airline Sentiment Dataset

Dataset contains airline tweets labeled into three sentiment classes.

### Features

- Tweet Text
- Airline
- Airline Confidence
- Negative Reason
- User Information
- Timestamp
- Sentiment Label

Target Variable

```
airline_sentiment
```

Classes

- Positive
- Neutral
- Negative

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Gensim
- Word2Vec
- Scikit-Learn
- TensorFlow
- Keras
- Optuna
- Streamlit
- Matplotlib

---

# 🚀 Project Workflow

```
Raw Tweets
      │
      ▼
Text Cleaning
      │
      ▼
Tokenization
      │
      ▼
Word2Vec Embedding
      │
      ▼
Sentence Vector Generation
      │
      ▼
Feature Scaling
      │
      ▼
ANN Model
      │
      ▼
Hyperparameter Tuning (Optuna)
      │
      ▼
Prediction
```

---

# 🔍 NLP Preprocessing

The following preprocessing steps were performed:

- Lowercase conversion
- URL removal
- Mention removal
- Hashtag removal
- Punctuation removal
- Number removal
- Tokenization
- Stopword removal
- Word2Vec embedding generation

---

# 🧠 Deep Learning Model

Artificial Neural Network

### Architecture

- Input Layer : 200 Features
- Hidden Layer 1 : 64 Neurons (ReLU)
- Dropout : 0.386
- Hidden Layer 2 : 128 Neurons (ReLU)
- Dropout : 0.405
- Output Layer : 3 Neurons (Softmax)

Optimizer

```
Adam
```

Learning Rate

```
0.0004249817850888504
```

Loss Function

```
Sparse Categorical Crossentropy
```

Evaluation Metric

```
Accuracy
```

---

# ⚙ Hyperparameter Optimization

Hyperparameters were optimized using **Optuna**.

Optimized Parameters

- Number of Hidden Layers
- Number of Neurons
- Dropout Rate
- Optimizer
- Learning Rate
- Batch Size

---

# 📈 Model Performance

Example Metrics

| Metric | Score |
|----------|---------|
| Train Accuracy | 79.6% |
| Test Accuracy | 77.6% |

Additional Evaluation

- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1 Score

---

# 🌐 Streamlit Application

The application allows users to

- Enter airline tweets
- Predict sentiment instantly
- View prediction confidence
- Interactive web interface

---

# 📂 Project Structure

```
Airline_Sentiment_Analysis/
│
├── app.py
├── requirements.txt
├── README.md
├── airline_sentiment_word2vec.keras
├── word2vec.model
├── scaler.pkl
├── label_encoder.pkl
└── Airline_Sentiment.ipynb
```

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Airline_Sentiment_Analysis.git
```

Move into project

```bash
cd Airline_Sentiment_Analysis
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

# 📸 Application Preview

Home Page
<img width="1907" height="825" alt="Screenshot 2026-07-03 141127" src="https://github.com/user-attachments/assets/d845cb47-d64b-41c7-9095-01cfe31aabf9" />


Prediction Page
<img width="1907" height="877" alt="Screenshot 2026-07-03 141148" src="https://github.com/user-attachments/assets/8e838d14-2e76-4c9f-9a57-37ff62ffa205" />

# 🎯 Future Improvements

- Use Transformer Models (BERT)
- Compare Word2Vec vs GloVe
- Deploy on Hugging Face Spaces
- Add Explainable AI (SHAP/LIME)
- Improve Accuracy using Bidirectional LSTM

# 👨‍💻 Author

**K Durga Prasad**

Aspiring Data Scientist | Machine Learning | Deep Learning | NLP

LinkedIn:
https://www.linkedin.com/in/kagithala-durga-prasad-81251a295/


# ⭐ If you found this project useful

Please ⭐ Star this repository and share your feedback!
