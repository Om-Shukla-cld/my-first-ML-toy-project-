# my-first-ML-toy-project-

# ============================================
# Placement Prediction using Logistic Regression
# Author: Om Shukla
# ============================================

# 📌 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import pickle

# 📌 Step 2: Load Dataset
df = pd.read_csv("placement.csv")

# 📌 Step 3: Basic EDA
print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())
print("\nStatistical Summary:\n", df.describe())
print("\nNull Values:\n", df.isnull().sum())

# Visualizations
sns.pairplot(df, diag_kind="kde")
plt.show()

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# 📌 Step 4: Preprocessing
# Encode categorical columns (if any)
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Features (X) and Target (y)
X = df.drop("placement", axis=1)   # Assuming 'placement' is target column
y = df["placement"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 📌 Step 5: Feature Selection using mlxtend
lr = LogisticRegression(max_iter=1000)

sfs = SFS(lr,
          k_features="best", 
          forward=True, 
          floating=False, 
          scoring='accuracy',
          cv=5)

sfs = sfs.fit(X_train, y_train)

print("Selected features:", sfs.k_feature_names_)

# Update X_train and X_test with selected features
X_train = sfs.transform(X_train)
X_test = sfs.transform(X_test)

# 📌 Step 6: Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 📌 Step 7: Model Evaluation
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 📌 Step 8: Save Model as placement.pkl
with open("placement.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model saved as placement.pkl")
