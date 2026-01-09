import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


data = pd.read_csv("n_karamnishvili25_86295.csv")

X = data[["words", "links", "capital_words", "spam_word_count"]]
y = data["is_spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)



def extract_features(email_text):
    spam_words = ["free", "win", "offer", "money", "urgent"]

    words = email_text.split()
    word_count = len(words)
    link_count = email_text.count("http")
    capital_words = sum(1 for w in words if w.isupper())
    spam_word_count = sum(1 for w in words if w.lower() in spam_words)

    return [[word_count, link_count, capital_words, spam_word_count]]


y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

email = "FREE money offer just for YOU click now http://spam.com"
features = extract_features(email)

prediction = model.predict(features)
print("Spam" if prediction[0] == 1 else "Not Spam")


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy:", acc)


print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


class_counts = data['is_spam'].value_counts()
labels = ['Legitimate', 'Spam']

plt.figure(figsize=(6,4))
plt.bar(labels, class_counts, color=['green', 'red'])
plt.title("Class Distribution of Emails")
plt.ylabel("Number of Emails")
plt.xlabel("Email Type")
plt.show()


plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate','Spam'], yticklabels=['Legitimate','Spam'])
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
