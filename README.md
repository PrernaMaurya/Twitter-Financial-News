**Twitter-financial-news-(finance-analyst)**

**[Open in Google Colab] (https://colab.research.google.com/drive/1DzvefJx86txYe9NeFouyGSfEwchpoI9S?usp=sharing)**


**🚀 Project Overview**

This project aims to classify real-world financial news tweets into one of 20 distinct categories using advanced natural language processing and deep learning techniques. The dataset comprises over 21,000 annotated tweets sourced from finance-related discussions on Twitter. The ultimate goal is to build a robust classification model that can assist financial analysts and traders in quickly understanding tweet context and category.


**🧠 Problem Statement**

●Objective: Automatically categorize tweets into predefined financial topics such as Earnings, M&A, Central Banks, Legal Regulations, etc.
●Challenge: Highly imbalanced multi-label text classification with noisy real-world Twitter data.


**📂 Dataset**

●Size: 21,107 tweets

●Labels: 20 categories (e.g., Analyst Update, Stock Movement, Fed, IPO, M&A, etc.)

●Format: text, label


**🔧 Tools & Libraries**

●Python, Jupyter Notebook, TensorFlow / Keras

●Scikit-learn, NLTK, Seaborn, Matplotlib

●Hugging Face Transformers

●WordCloud, CountVectorizer, TF-IDF


**🔍 Key Steps & Methodology
📌 Data Preprocessing**

●Removed URLs, mentions, punctuation, and numeric tokens

●Performed lemmatization and stopword removal

●Created clean training and test sets


**📌 EDA (Exploratory Data Analysis)**

●WordCloud of frequent terms

●Sentiment and category distribution

●Highlighted data imbalance and informed class-weighting strategy


**📌 Feature Engineering**

●Applied CountVectorizer and TF-IDF vectorization

●Tokenization and padding for deep learning models

●Vocabulary size of ~25,000+ terms


**📌 Modeling**

Three core models were developed and tested:

🔹 ANN (Artificial Neural Network) – Dense architecture with high dimensional sparse vectors

🔹 CNN (Convolutional Neural Network) – Captures local patterns in token sequences

🔹 Simple RNN – For temporal relationships in token sequences

All models were trained using inverse class weights to handle imbalanced data.


**📌 Evaluation**

●Accuracy, Loss, and Validation metrics tracked across 100 epochs

●Achieved ~83%+ validation accuracy with the ANN and CNN models

●Performance visualized via confusion matrices and training curves


**📈 Notable Insights**

●ANN outperformed RNN on sparse data while being computationally efficient

●CNN achieved a balance of training speed and accuracy on sequential input

●Data imbalance posed a major challenge but was mitigated through weighted losses


**📌 Future Work**

●Experiment with BERT and transformer-based models for higher accuracy

●Use multi-label classification for overlapping topics

●Integrate real-time tweet fetching using Tweepy API

**Conclusion**

This project demonstrates a complete NLP pipeline for classifying financial tweets with significant practical applications in financial analytics, investment tracking, and sentiment monitoring. The CNN and ANN models performed robustly with >80% accuracy in categorizing news into precise financial labels.
