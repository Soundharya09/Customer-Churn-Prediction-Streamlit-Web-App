# STEP 1 - TRAIN & SAVE THE MODEL

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATASET
# =========================

train_df = pd.read_csv("customer_churn_dataset-training-master.csv")

print("Dataset Loaded Successfully")
print(train_df.head())

# =========================
# REMOVE UNNECESSARY COLUMNS
# =========================

if 'CustomerID' in train_df.columns:
    train_df.drop('CustomerID', axis=1, inplace=True)

# =========================
# HANDLE MISSING VALUES
# =========================

# Fill numerical columns with median
num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    train_df[col] = train_df[col].fillna(train_df[col].median())

# Fill categorical columns with mode
cat_cols = train_df.select_dtypes(include=['object']).columns

for col in cat_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])

# =========================
# SEPARATE FEATURES & TARGET
# =========================

X = train_df.drop('Churn', axis=1)
y = train_df['Churn']

# =========================
# ONE-HOT ENCODING
# =========================

X = pd.get_dummies(X)

# Save feature column names
feature_columns = X.columns.tolist()

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# TRAIN MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# =========================
# MODEL EVALUATION
# =========================

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# =========================
# SAVE MODEL FILES
# =========================

with open('churn_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('feature_columns.pkl', 'wb') as file:
    pickle.dump(feature_columns, file)

print("\nModel and preprocessing files saved successfully.")