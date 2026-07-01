# 💬 Twitter Sentiment Analysis (BiLSTM + GloVe + Streamlit)

> Analyze emotions behind tweets using **Deep Learning and NLP**.  
This project performs **Twitter Sentiment Analysis** using a **Bidirectional LSTM (BiLSTM)** model with **GloVe word embeddings**, deployed as an interactive **Streamlit web app**.

---

## 🚀 Overview

This project classifies tweets into **Positive**, **Negative**, or **Neutral** sentiments using a deep learning model trained on a labeled Twitter dataset.

- 🔠 **Preprocessing:** Cleans and tokenizes tweets  
- 🧠 **Model:** BiLSTM with pretrained **GloVe (100d)** embeddings  
- ⚖️ **Balanced Training:** Uses `class_weight` to handle imbalance  
- 🌐 **Deployment:** Interactive Streamlit interface for live predictions  
- 📊 **Accuracy:** ~85% on test data  

---

## 🧩 Project Structure

```bash
📁 Twitter-Sentiment-Analysis
│
├── app.py                          # Streamlit web app
├── twitter.ipynb                   # Model training notebook
├── sentiment_bilstm_glov.h5        # Trained BiLSTM model
├── sentiment_tokenizer_glov.joblib # Tokenizer for preprocessing
├── Twitter_Data.csv                # Dataset
├── glove.6B.50d.txt               # GloVe embeddings (50d)
└── README.md                       # Documentation

```
---

## 🧠 Key Insights
Training is handled in twitter.ipynb and includes:
- Text cleaning and tokenization
- Label encoding (positive, neutral, negative)
- Loading GloVe 100d embeddings
- Building and training a BiLSTM model
- Evaluating performance and saving model/tokenizer

---

## 🧮 Model Architecture
```
Embedding (GloVe 100d)
↓
Bidirectional LSTM (128 units)
↓
Dropout (0.5)
↓
Dense (64, ReLU)
↓
Dense (3, Softmax)
```
---

### 📊 Model Summary

| Layer (Type) | Output Shape | Param # |
|---------------|---------------|----------|
| Embedding (Non-trainable GloVe) | (None, 50, 100) | 2,000,000+ |
| Bidirectional LSTM (128 units) | (None, 256) | 234,496 |
| Dropout (0.5) | (None, 256) | 0 |
| Dense (64, ReLU) | (None, 64) | 16,448 |
| Dense (Softmax) | (None, 3) | 195 |
| **Total Parameters** | **~2.25 Million** |  |

---

## 🧾 Model Training

Model training is handled inside **`twitter.ipynb`** and follows these main steps:

1. Clean and preprocess the dataset  
2. Tokenize and pad tweet sequences  
3. Encode sentiment labels  
4. Load pretrained **GloVe (100d)** embeddings  
5. Build and train the **BiLSTM** model  
6. Use `class_weight` to manage class imbalance  
7. Evaluate and save both the model and tokenizer  

### 💾 Saved Files

| File Name | Description |
|------------|-------------|
| `sentiment_bilstm_glov.h5` | Trained BiLSTM model |
| `sentiment_tokenizer_glov.joblib` | Tokenizer used for preprocessing |

---

## 🧪 Model Evaluation

| Metric | Score |
|--------|--------|
| **Accuracy** | 84–86% |
| **Loss** | ~0.40 |
| **Optimizer** | Adam |
| **Loss Function** | Categorical Crossentropy |

**Visualizations:**
- Confusion Matrix  
- Accuracy vs. Loss Curves  
*(Available in the Jupyter notebook)*

---

## 💻 Running the Streamlit App

Run the app locally using the command below:

```bash
streamlit run app.py
```

## Batch CSV Prediction

The Streamlit app accepts a tweet CSV upload for batch prediction. It detects
common text columns such as `text`, `tweet`, `tweet_text`, `full_text`,
`content`, and `message`, then adds `predicted_sentiment` and
`prediction_confidence` columns that can be downloaded as a CSV.

This works with reviewed tweet exports from tools such as
[TweetClaw](https://github.com/Xquik-dev/tweetclaw) when one of those text
columns is present.

✅ **How to use it:**
1. Copy everything above (including the Markdown formatting).
2. Paste it directly into your `README.md` file.
3. Replace `<your-username>` with your actual GitHub username.
4. (Optional) Add a screenshot of your Streamlit app below the **Example Predictions** section for visual impact.

Would you like me to also give you a **short GitHub description** (the one that appears below your repo title, e.g. “AI-powered Twitter Sentiment Classifier using BiLSTM + Streamlit”)? It helps attract views.





