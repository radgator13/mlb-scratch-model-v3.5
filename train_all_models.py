import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# Load real training data
df = pd.read_csv('data/mlb_training_dataset.csv')

features = ['Predicted_NRFI_Probability', 'Away_NRFI_Batting_Rate', 'Home_NRFI_Pitching_Rate']
X = df[features]
y = df['NRFI_Hit']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Placeholder to collect models and results
models = {}
results = {}

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr)
}
models['Logistic'] = lr
joblib.dump(lr, 'model_logistic_real.pkl')

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = {
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf)
}
models['RandomForest'] = rf
joblib.dump(rf, 'model_rf_real.pkl')

# LightGBM
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)
results['LightGBM'] = {
    'Accuracy': accuracy_score(y_test, y_pred_lgbm),
    'Precision': precision_score(y_test, y_pred_lgbm),
    'Recall': recall_score(y_test, y_pred_lgbm)
}
models['LightGBM'] = lgbm
joblib.dump(lgbm, 'model_lgbm_real.pkl')

# Print Results
print("\n📊 Model Comparison (trained on real outcomes):")
for model_name, metrics in results.items():
    print(f"\n🚀 {model_name}:")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.2f}")

print("\n✅ All real models saved: model_logistic_real.pkl, model_rf_real.pkl, model_lgbm_real.pkl")
