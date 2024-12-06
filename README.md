
Hate Speech Detection

Overview

This project focuses on detecting hate speech in tweets using machine learning models. The goal is to build a model that can classify tweets into two categories: hate speech and non-hate speech. The project uses various machine learning algorithms to process and classify textual data, achieving an overall model accuracy of 90.06%.

Key Features:

-Text Preprocessing: Text data is cleaned using NLP techniques such as stemming, stopword removal, and tokenization.
-Model Selection: Different models are trained, including Random Forest, XGBoost, and Logistic Regression, to identify the most accurate model.
-Performance Metrics: The models are evaluated using confusion matrix and accuracy score to assess their performance.
-Deployment: The best model (XGBoost) is saved for future predictions.

Technologies Used

-Python
-Pandas: For data manipulation and analysis.
-NumPy: For numerical operations.
-scikit-learn: For machine learning algorithms and evaluation.
-XGBoost: For boosting-based classifier.
-NLTK: For natural language processing tasks like stemming and stopword removal.
-joblib: For saving and loading the trained models.
-CountVectorizer: For text vectorization (Bag of Words).

Setup and Installation

Clone the repository:
bash
Copy code
git clone https://github.com/your-username/hate-speech-detection.git

Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
How to Run the Project
Download the dataset (twitter_data.csv) and place it in the project directory.

Run the main.py script to train the model:
bash
Copy code
python main.py

Model Evaluation
-Random Forest Classifier: Evaluated using confusion matrix and accuracy score.
-XGBoost Classifier: Best model with 90.06% accuracy.
-Logistic Regression: Performance also evaluated using the confusion matrix.

Conclusion
This project demonstrates how to preprocess textual data and use machine learning to classify hate speech in social media posts. The XGBoost classifier performs the best, achieving a high accuracy rate of over 90%. This model can be used for automated moderation of user-generated content on social platforms.
