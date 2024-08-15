# -*- coding: utf-8 -*-
"""Build a spam filter .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1iXOEAuANBbp4h2cZmnqaGd0uXRthGDET
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the Dataset
df = pd.read_csv('emails.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:\n", df.head())

# Check for any missing values
print("Missing values before cleaning:\n", df.isnull().sum())

# Step 3: Data Preprocessing
# Remove rows with any NaN values in 'text' or 'spam' columns
df.dropna(subset=['text', 'spam'], inplace=True)

df.head()

# Check for NaN values in 'spam' column
print("Missing values in 'spam' column after cleaning:\n", df['spam'].isnull().sum())

df.head()

# Verify unique values in 'spam' column
print("Unique values in 'spam' column after encoding:\n", df['spam'].unique())

df.head()

# Display unique entries and ensure no empty text entries exist
print("Number of empty text entries after cleaning:", (df['text'].str.len() == 0).sum())

df.head()

# Print a few samples of cleaned text to verify content
print("Sample text data after cleaning:\n", df['text'].head())

df.head()

# Step 4: Feature Extraction
# Convert text to numerical data using TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words=None, max_features=3000)

df.head()

# Fit and transform the text data
try:
    X = tfidf.fit_transform(df['text']).toarray()
    y = df['spam']
    print("Shape of TF-IDF features matrix:", X.shape)
except ValueError as e:
    print("Error during TF-IDF vectorization:", e)
    print("Unique values in text after cleaning:", df['text'].unique())

df.head()

# Step 5: Model Training
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_train

# Initialize and train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

"""Hyperparameter Tuning:"""

from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {'alpha': [0.1, 0.5, 1.0]}

# Initialize GridSearchCV
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')

# Fit to training data
grid_search.fit(X_train, y_train)

# Print best hyperparameters and score
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Update the model with best parameters
best_model = grid_search.best_estimator_

"""Feature Importance:"""

import numpy as np

# Get feature names
feature_names = tfidf.get_feature_names_out()

# Get log probabilities for features given each class
log_probabilities = model.feature_log_prob_

# For spam class (1), get top 10 features with highest log probability
spam_class_index = 1  # Index for the spam class
top_indices = np.argsort(log_probabilities[spam_class_index])[-10:]

# Extract top features
top_features = [feature_names[i] for i in top_indices]

print("Top features for 'spam':", top_features)

# You can also do this for the 'ham' class if desired
ham_class_index = 0  # Index for the ham class
top_indices_ham = np.argsort(log_probabilities[ham_class_index])[-10:]
top_features_ham = [feature_names[i] for i in top_indices_ham]

print("Top features for 'ham':", top_features_ham)

"""Cross-Validation:"""

from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(MultinomialNB(), X, y, cv=5, scoring='accuracy')

print("Cross-validation scores:", cv_scores)
print("Mean cross-validation accuracy:", cv_scores.mean())

"""Visualization:"""

import matplotlib.pyplot as plt
import seaborn as sns

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

"""Learning Curve:"""

from sklearn.model_selection import learning_curve

# Calculate learning curve
train_sizes, train_scores, test_scores = learning_curve(MultinomialNB(), X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training accuracy')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Validation accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid()
plt.show()