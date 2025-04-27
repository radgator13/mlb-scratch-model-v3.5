import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib

# Load your data
df = pd.read_csv('data/mlb_nrfi_predictions.csv')


# Filter: only games that are completed (no 'Pending' innings)
df = df[
    (df['Away 1th'] != "Pending") &
    (df['Home 1th'] != "Pending")
]

# Also: filter only if YRFI exists
df = df[df['YRFI'].notna()]

# ✅ Build Target directly from YRFI
# Predicting NRFI (No Run First Inning), so flip YRFI:
df['NRFI_Hit'] = (df['YRFI'] == 0).astype(int)

print("\n🧠 Columns available in the dataset:")
print(df.columns.tolist())

# Features to use (make sure they exist)
features = [
    'Predicted_NRFI_Probability',
    'Away_NRFI_Batting_Rate',
    'Home_NRFI_Pitching_Rate'
]

# Confirm features exist
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    raise ValueError(f"Missing features in dataset: {missing_features}")

X = df[features]
y = df['NRFI_Hit']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)

print("\n📊 Model Performance:")
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Save model
joblib.dump(model, "model.pkl")
print("\n✅ Model saved as model.pkl")
