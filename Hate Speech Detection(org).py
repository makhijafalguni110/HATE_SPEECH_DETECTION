#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from wordcloud import WordCloud

#to data preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#NLP tools
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

#train split and fit models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier


#model selection
from sklearn.metrics import confusion_matrix, accuracy_score


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


dataset = pd.read_csv('twitter_data.csv')
dataset.head()


# In[3]:


dataset.info()


# In[4]:


dataset.describe().T


# In[5]:


dt_transformed = dataset[['class', 'tweet']]
y = dt_transformed.iloc[:, 0].values  # Selecting only the 'class' column


# # Encoding the Dependent Variable

# In[6]:


from sklearn.preprocessing import LabelEncoder

# Apply label encoding to convert categorical labels to numerical labels
le = LabelEncoder()
y = le.fit_transform(y)


# In[7]:


print(y)


# In[8]:


y_df = pd.DataFrame(y)
y = np.array(y_df[0])  # Assuming 'y' contains the class labels directly


# In[9]:


# Define the variables
y_hate = "This is a hate-related message"
y_offensive = "This is an offensive message"

# Print the variables
print(y_hate)
print(y_offensive)


# # Cleaning the Text

# In[11]:


dt_trasformed = pd.DataFrame({'tweet': ["This is an example tweet!", "Another example tweet, with some punctuation."]})

# Process the tweets
corpus = []
for i in range(len(dt_trasformed['tweet'])):
    review = re.sub('[^a-zA-Z]', ' ', dt_trasformed['tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Print the resulting corpus
print(corpus)


# In[12]:


cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()


# # Splitting the dataset into the Training set and Test set

# In[18]:


X = corpus  # This should be a list of preprocessed tweets
y_hate = [0, 1, 0, 1]  # This should be the target labels for each tweet in the corpus

# Check lengths
print(f"Length of X: {len(X)}")
print(f"Length of y_hate: {len(y_hate)}")

# Ensure lengths match
if len(X) == len(y_hate):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_hate, test_size=0.30, random_state=0)


# Decision Tree

# In[20]:


corpus = ["example tweet one", "example tweet two", "example tweet three", "example tweet four"]
labels = [0, 1, 0, 1]  # Example labels

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y_hate = labels  # This should be your actual labels corresponding to the tweets

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_hate, test_size=0.30, random_state=0)

# Ensure X_train and y_train are defined
print(f"Length of X_train: {len(X_train)}")
print(f"Length of y_train: {len(y_train)}")

# Train the Decision Tree classifier
classifier_dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier_dt.fit(X_train, y_train)

# Print success message
print("Classifier trained successfully!")


# XG Boost

# In[21]:


pip install xgboost


# In[23]:


from xgboost import XGBClassifier

classifier_xgb = XGBClassifier()
classifier_xgb.fit(X_train, y_train)


# Linear Regression

# In[24]:


from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train, y_train)


# Random Forest

# In[25]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Create a RandomForestClassifier instance
classifier_rf = RandomForestClassifier()

# Fit the classifier to your training data
classifier_rf.fit(X_train, y_train)

# Make predictions on the test data
y_pred_rf = classifier_rf.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)

# Print the confusion matrix
print(cm)


# # Making the Confusion Matrix for each model

# In[26]:


#XGBoost Classifier
y_pred_xgb = classifier_xgb.predict(X_test)
cm = confusion_matrix(y_test, y_pred_xgb)
print(cm)


# In[27]:


#Logistic Regression
y_pred_lr=classifier_lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred_lr)
print(cm)


# In[28]:


#Decision Tree
y_pred_dt = classifier_dt.predict(X_test)
cm = confusion_matrix(y_test, y_pred_dt)
print(cm)


# In[29]:


#Random Florest
y_pred_rf = classifier_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)


# In[30]:


from sklearn.metrics import accuracy_score

# Assuming you have already defined and fitted your decision tree model (classifier_dt)
# Make predictions on the test data
y_pred_dt = classifier_dt.predict(X_test)

# Calculate accuracy scores for each classifier
rf_score = accuracy_score(y_test, y_pred_rf)
xgb_score = accuracy_score(y_test, y_pred_xgb)
lr_score = accuracy_score(y_test, y_pred_lr)
dt_score = accuracy_score(y_test, y_pred_dt)

# Print accuracy scores
print('Random Forest Accuracy: ', rf_score)
print('XGBoost Classifier Accuracy: ', xgb_score)
print('Logistic Regression Accuracy: ', lr_score)
print('Decision Tree Accuracy: ', dt_score)


# In[31]:


# Importing libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# Importing the dataset
dataset = pd.read_csv('twitter_data.csv')

# Cleaning the texts
corpus = []
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(corpus).toarray()
y = dataset['class'].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Training Random Forest Classifier
classifier_rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier_rf.fit(X_train, y_train)

# Training XGBoost Classifier
classifier_xgb = XGBClassifier()
classifier_xgb.fit(X_train, y_train)

# Training Logistic Regression Classifier
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(X_train, y_train)

# Making predictions and computing confusion matrices for each classifier
classifiers = [('Random Forest', classifier_rf), ('XGBoost Classifier', classifier_xgb), ('Logistic Regression', classifier_lr)]

for clf_name, clf in classifiers:
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(clf_name + " Confusion Matrix:")
    print(cm)
    print(clf_name + " Accuracy: " + str(accuracy))

# Saving the best model (XGBoost Classifier)
joblib.dump(classifier_xgb, 'hatespeech.pkl')


# In[32]:


# Calculate accuracies for each classifier
accuracies = []

for clf_name, clf in classifiers:
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Calculate overall accuracy
overall_accuracy = np.mean(accuracies)

print("Overall Model Accuracy:", overall_accuracy)

