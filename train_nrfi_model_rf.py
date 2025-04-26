import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib

# Load full results
df = pd.read_csv("data/mlb_nrfi_results_full.csv")

# Drop rows with missing actual runs
df = df.dropna(subset=['Actual_1st_Inning_Runs'])

# Build Target
df['NRFI_Hit'] = (df['Actual_1st_Inning_Runs'] == 0).astype(int)

# Features we will use
features = [
    'Predicted_NRFI_Probability',
    'Away_NRFI_Batting_Rate',
    'Home_NRFI_Pitching_Rate'
]

X = df[features]
y = df['NRFI_Hit']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print("\n📊 Random Forest Model Performance:")
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Save model
joblib.dump(model, "model_rf.pkl")
print("\n✅ Random Forest model saved as model_rf.pkl")
