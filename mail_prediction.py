import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Simulated dataset with names and email patterns
data = {
    "Name": ["John Doe", "Jane Smith", "Alice Brown", "Bob Johnson", "Charlie Adams"],
    "Email": ["j.doe@example.com", "j.smith@example.com", "a.brown@example.com", "b.johnson@example.com", "c.adams@example.com"]
}
df = pd.DataFrame(data)

def generate_email(name):
    """Generates a potential email format from a name."""
    name_parts = name.lower().split()
    return f"{name_parts[0][0]}.{name_parts[-1]}@example.com"

df["Predicted_Email"] = df["Name"].apply(generate_email)

def extract_features(emails):
    """Extracts basic features from email addresses."""
    return [re.sub(r'[^a-zA-Z]', '', email.split('@')[0]) for email in emails]

# Preparing data for ML model
X = extract_features(df["Email"])
y = df["Email"]
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualization of email patterns
email_lengths = [len(email) for email in df["Email"]]
plt.figure(figsize=(10, 5))
plt.hist(email_lengths, bins=5, color='blue', alpha=0.7)
plt.xlabel("Email Length")
plt.ylabel("Frequency")
plt.title("Distribution of Email Lengths")
plt.show()

# Scatter plot of email formats
plt.figure(figsize=(10, 5))
plt.scatter(range(len(df)), [len(email) for email in df["Predicted_Email"]], color='red', label='Predicted Email')
plt.scatter(range(len(df)), [len(email) for email in df["Email"]], color='blue', label='Actual Email')
plt.xlabel("Sample Index")
plt.ylabel("Email Length")
plt.legend()
plt.title("Comparison of Predicted vs Actual Email Formats")
plt.show()
