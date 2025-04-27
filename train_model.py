import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib

# Load your data
df = pd.read_csv('data/mlb_nrfi_predictions.csv')

# Confirm the correct features exist
required_features = ['Predicted_NRFI_Probability', 'Away_NRFI_Batting_Rate', 'Home_NRFI_Pitching_Rate']
for feature in required_features:
    if feature not in df.columns:
        raise ValueError(f"Missing feature: {feature}")

# ✅ Create pseudo-labels for training
# You can define NRFI_Hit as 1 if Predicted NRFI Probability >= 50%
df['NRFI_Hit'] = (df['Predicted_NRFI_Probability'] >= 50).astype(int)

# Features and labels
X = df[required_features]
y = df['NRFI_Hit']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n📊 Logistic Regression Model Performance:")
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Save model
joblib.dump(model, "model.pkl")
print("\n✅ Logistic Regression model saved as model.pkl")
