# aimlmid2026_n_karamnishvili25
Midterm exam in AI and Machine Learning

# assignment 1, Finding the correlation
The purpose of this analysis is to measure the strength and direction of the linear relationship between two numerical variables, x and y, using the Pearson correlation coefficient.

Step 1: Data Preparation

Two datasets were given: x: independent variable y: dependent variable

Step 2: Compute the Means

The arithmetic mean of each dataset is calculated: x_mean = sum(x) / n y_mean = sum(y) / n The means represent the central value of each variable and are used as reference points for further calculations.

Step 3: Compute Deviations from the Mean

For each observation: dx = x[i] - x_mean dy = y[i] - y_mean These deviations show how far each value is from its mean.

Step 4: Covariance Calculation

The covariance measures whether the variables move together: cov += dx * dy

Positive covariance : variables increase together Negative covariance: one increases while the other decreases

In this case, the covariance is negative, indicating an inverse relationship.

Step 5: Standardization

To ensure the result is scale-independent, the covariance is divided by the product of the standard deviations: sigma = math.sqrt(sum_x2 * sum_y2) r = cov / sigma

Step 6: Pearson Correlation Coefficient r = -0.6554845380710971 r ≈ −0.65

The value of r lies between -1 and 1 and indicates a moderate negative linear correlation

As x increases, y tends to decrease

please view correlation visualisation image

# assignment 2, Spam email detection


Spam Email Detection Using Logistic Regression
1. Introduction

The goal of this project is to build a logistic regression model to classify emails as spam or legitimate.
The dataset contains numeric features extracted from emails: number of words, number of links, number of capitalized words, and count of spam-related words. The target variable is is_spam (1 = spam, 0 = legitimate).
The model is trained on 70% of the dataset and validated on the remaining 30%. In addition, the application can parse new email text, extract features, and predict if it is spam.

2. Data Loading and Processing

The dataset is stored in a CSV file (spam.csv) and loaded using Python’s pandas library.

import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("spam.csv")

# Define features and target
X = data[["words", "links", "capital_words", "spam_word_count"]]
y = data["is_spam"]

# Split data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


Explanation:

X contains the numeric features used by the model.

y is the target label.

Data is split to train the model on 70% of emails and evaluate it on 30% of unseen emails.

3. Logistic Regression Model

Logistic Regression is a binary classification model that estimates the probability that an email is spam.
It uses a sigmoid function to output values between 0 and 1, which are then thresholded to produce class predictions.

from sklearn.linear_model import LogisticRegression

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Display coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


Coefficients Explanation:

Positive coefficient → increases likelihood of spam

Negative coefficient → decreases likelihood of spam

Example: A positive coefficient for spam_word_count means emails with more spam words are more likely classified as spam.

4. Model Validation: Confusion Matrix & Accuracy

The model is evaluated on the 30% test set using a confusion matrix and accuracy score.

from sklearn.metrics import confusion_matrix, accuracy_score





# Predict on test data
y_pred = model.predict(X_test)

# Compute evaluation metrics
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy:", acc)


Explanation:

Confusion Matrix: Shows True Positives, True Negatives, False Positives, and False Negatives.

Accuracy: Measures the percentage of correct predictions.

Insight: A high accuracy and a well-diagonalized confusion matrix indicate the model reliably distinguishes spam from legitimate emails.

5. Email Text Checker (New Email Prediction)

The application can parse new emails, extract features, and classify them.

def extract_features(email_text):
    spam_words = ["free", "win", "offer", "money", "urgent"]

    words = email_text.split()
    word_count = len(words)
    link_count = email_text.count("http")
    capital_words = sum(1 for w in words if w.isupper())
    spam_word_count = sum(1 for w in words if w.lower() in spam_words)

    return [[word_count, link_count, capital_words, spam_word_count]]

# Example spam email
email = "FREE money offer just for YOU click now http://spam.com"
features = extract_features(email)
prediction = model.predict(features)
print("Spam" if prediction[0] == 1 else "Not Spam")

6. Manually Composed Spam Email

Email Text:

Subject: URGENT FREE OFFER

YOU HAVE WON FREE MONEY
CLICK THE LINK NOW http://spam.com

LIMITED TIME OFFER

Explanation:

Contains multiple capitalized words, spam keywords, and a link.

The email was composed to trigger high values in features strongly associated with spam.

Model prediction: Spam (1)

7. Manually Composed Legitimate Email

Email Text:

Subject: Meeting Reminder

Hi team,

This is a reminder for our weekly meeting scheduled for Monday at 10 AM.
Please prepare your updates and join via Zoom.

Best regards,
Alice

Explanation:

Contains no spam words, no links, normal capitalization, and moderate word count.

Model prediction: Legitimate (0)

8. Data and Model Visualizations
Visualization 1: Class Distribution of Emails
import matplotlib.pyplot as plt

class_counts = data['is_spam'].value_counts()
labels = ['Legitimate', 'Spam']

plt.figure(figsize=(6,4))
plt.bar(labels, class_counts, color=['green', 'red'])
plt.title("Class Distribution of Emails")
plt.ylabel("Number of Emails")
plt.xlabel("Email Type")
plt.show()


Explanation:

Shows the proportion of spam vs legitimate emails.

Insight: There are more legitimate emails than spam, indicating a slight imbalance that the model must handle.

Visualization 2: Confusion Matrix Heatmap
import seaborn as sns

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legitimate','Spam'], 
            yticklabels=['Legitimate','Spam'])
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


Explanation:

Visual representation of model performance.

Correctly classified emails appear on the diagonal; misclassified emails are off-diagonal.

Insight: Most emails are correctly classified, with only a few false positives or negatives.

9. Conclusion

A logistic regression model was successfully trained to classify emails as spam or legitimate.

Both spam and legitimate emails can be evaluated using the feature extraction function.

Model evaluation using a confusion matrix, accuracy score, and visualizations shows that the model performs reliably on unseen data.

Manually composed emails demonstrate how the model interprets feature values and classifies new inputs.