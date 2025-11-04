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

📁 Twitter-Sentiment-Analysis
│
├── app.py # Streamlit web app
├── twitter.ipynb # Model training notebook
├── sentiment_bilstm_glov.h5 # Trained BiLSTM model
├── sentiment_tokenizer_glov.joblib # Tokenizer for preprocessing
├── Twitter_Data.csv # Dataset
├── glove.6B.100d.txt # GloVe embeddings (100d)
├── requirements.txt # Python dependencies
└── README.md # Documentation

yaml
Copy code

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Download GloVe Embeddings
Download from GloVe 6B Dataset
Place the file glove.6B.100d.txt in the project folder.

🧠 Model Training
Training is handled in twitter.ipynb and includes:

Text cleaning and tokenization

Label encoding (positive, neutral, negative)

Loading GloVe 100d embeddings

Building and training a BiLSTM model

Evaluating performance and saving model/tokenizer

🧮 Model Architecture
scss
Copy code
Embedding (GloVe 100d)
↓
Bidirectional LSTM (128 units)
↓
Dropout (0.5)
↓
Dense (64, ReLU)
↓
Dense (3, Softmax)
Saved Files

Copy code
sentiment_bilstm_glov.h5
sentiment_tokenizer_glov.joblib
💻 Running the Streamlit App
Run the web app locally:

bash
Copy code
streamlit run app.py
🖥️ Example Predictions
Tweet	Predicted Sentiment
I love this movie!	😀 Positive
This is terrible.	😡 Negative
It’s okay, not great.	😐 Neutral

📊 Model Evaluation
Metric	Score
Accuracy	84–86%
Loss	~0.4
Optimizer	Adam
Loss Function	Categorical Crossentropy

Visualization:

Confusion Matrix

Accuracy vs. Loss curves
(included in the Jupyter notebook)

🚀 Future Improvements
Implement BERT/RoBERTa for higher accuracy

Integrate Twitter API for real-time tweet analysis

Add explainability with SHAP or LIME

Deploy app on Streamlit Cloud or Hugging Face Spaces

👤 Author
Afeef Anversha
Data Analyst | AI & ML Enthusiast
🔗 LinkedIn
🐙 GitHub

🪪 License
This project is licensed under the MIT License.
Feel free to use, modify, and share with attribution.

⭐ Support
If you find this project useful, please consider giving it a ⭐ on GitHub
and connecting with me on LinkedIn!

🔖 Tags
#AI #NLP #DeepLearning #TensorFlow #Streamlit #MachineLearning #Python #DataScience

markdown
Copy code

---

✅ **How to use it:**
1. Copy everything above (including the Markdown formatting).
2. Paste it directly into your `README.md` file.
3. Replace `<your-username>` with your actual GitHub username.
4. (Optional) Add a screenshot of your Streamlit app below the **Example Predictions** section for visual impact.

Would you like me to also give you a **short GitHub description** (the one that appears below your repo title, e.g. “AI-powered Twitter Sentiment Classifier using BiLSTM + Streamlit”)? It helps attract views.






