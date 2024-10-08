import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'TCGA_InfoWithGrade_Normalisasi.csv'
data = pd.read_csv(file_path)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Define features (X) and target (y)
X = data.drop('Grade', axis=1)
y = data['Grade']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize the training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the classifiers
rf_clf = RandomForestClassifier(n_estimators=100, random_state=0)
gaussian = GaussianNB()
clf = DecisionTreeClassifier(random_state=0)

# Train the models
rf_clf.fit(X_train_scaled, y_train)
gaussian.fit(X_train_scaled, y_train)
clf.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rf = rf_clf.predict(X_test_scaled)
y_pred_nb = gaussian.predict(X_test_scaled)
y_pred_dt = clf.predict(X_test_scaled)

# Evaluate the models and store the accuracies
accuracies = {
    'Random Forest': accuracy_score(y_test, y_pred_rf),
    'Naive Bayes': accuracy_score(y_test, y_pred_nb),
    'Decision Tree': accuracy_score(y_test, y_pred_dt)
}

Precision = {
    'Random Forest': precision_score(y_test, y_pred_rf, average='weighted'),
    'Naive Bayes': precision_score(y_test, y_pred_nb, average='weighted'),
    'Decision Tree': precision_score(y_test, y_pred_dt, average='weighted')
}

Recall = {
    'Random Forest': recall_score(y_test, y_pred_rf, average='weighted'),
    'Naive Bayes': recall_score(y_test, y_pred_nb, average='weighted'),
    'Decision Tree': recall_score(y_test, y_pred_dt, average='weighted')
}

F1_Score = {
    'Random Forest': f1_score(y_test, y_pred_rf, average='weighted'),
    'Naive Bayes': f1_score(y_test, y_pred_nb, average='weighted'),
    'Decision Tree': f1_score(y_test, y_pred_dt, average='weighted')
}

# Print classification reports
print("\n========== Random Forest Model Evaluation ==========")
print("Classification Report:\n", classification_report(y_test, y_pred_dt))
print("Accuracy: {:.2f}%".format(accuracies['Random Forest'] * 100))
print("Precision: {:.2f}%".format(Precision['Random Forest'] * 100))
print("Recall: {:.2f}%".format(Recall['Random Forest'] * 100))
print("F1 Score: {:.2f}%".format(F1_Score['Random Forest'] * 100))

print("\n========== Naive Bayes Model Evaluation ==========")
print("Classification Report:\n", classification_report(y_test, y_pred_nb))
print("Accuracy: {:.2f}%".format(accuracies['Naive Bayes'] * 100))
print("Precision: {:.2f}%".format(Precision['Naive Bayes'] * 100))
print("Recall: {:.2f}%".format(Recall['Naive Bayes'] * 100))
print("F1 Score: {:.2f}%".format(F1_Score['Naive Bayes'] * 100))

print("\n========== Decision Tree Model Evaluation ==========")
print("Classification Report:\n", classification_report(y_test, y_pred_dt))
print("Accuracy: {:.2f}%".format(accuracies['Decision Tree'] * 100))
print("Precision: {:.2f}%".format(Precision['Decision Tree'] * 100))
print("Recall: {:.2f}%".format(Recall['Decision Tree'] * 100))
print("F1 Score: {:.2f}%".format(F1_Score['Decision Tree'] * 100))

# Menampilkan akurasi dari masing-masing model
print("\n========== Model Accuracy Comparison ==========")
for model_name, accuracy in accuracies.items():
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")

# Model dengan akurasi tertinggi
best_model_name = max(accuracies, key=accuracies.get)
print(f"\nAlgoritma dengan akurasi tertinggi adalah: {best_model_name}")

# input manual untuk prediksi
print("\n========== Manual Input Prediction ==========")

# Get input for each feature according to the specified mappings
features = []

# Input for each feature
features.append(float(input("Enter Gender (Male = 0, Female = 1): ")))
features.append(float(input("Enter Age at Diagnosis (actual value): ")))
features.append(float(input("Enter Race (white = 0.0, black or african american = 0.333333, asian = 0.666667, american indian or alaska native = 1.0): ")))
features.append(float(input("Enter IDH1 (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter TP53 (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter ATRX (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter PTEN (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter EGFR (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter CIC (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter MUC16 (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter PIK3CA (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter NF1 (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter PIK3R1 (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter FUBP1 (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter RB1 (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter NOTCH1 (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter BCOR (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter CSMD3 (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter SMARCA4 (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter GRIN2A (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter IDH2 (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter FAT4 (NOT_MUTATED = 0, MUTATED = 1): ")))
features.append(float(input("Enter PDGFRA (NOT_MUTATED = 0, MUTATED = 1): ")))

# Melakukan prediksi dengan model terbaik
models = {
    'Random Forest': rf_clf,
    'Naive Bayes': gaussian,
    'Decision Tree': clf
}

# Convert input list to DataFrame
input_data = pd.DataFrame([features], columns=X.columns)

# Normalize the input data using the previously fitted scaler
input_data_scaled = scaler.transform(input_data)

# Predict using the best model
best_model = models[best_model_name]
prediction = best_model.predict(input_data_scaled)

# Interpretasi hasil prediksi
if prediction == 0.0:
    prediction = "LGG"  # Low-Grade Glioma
else:
    prediction = "GBM"  # Glioblastoma Multiforme

print(f"\nPrediksi berdasarkan input manual menggunakan {best_model_name}: {prediction}")

# Visualisasi Confusion Matrix untuk model terbaik
print(f"\nConfusion Matrix for {best_model_name}:")
if best_model_name == 'Random Forest':
    cm = confusion_matrix(y_test, y_pred_rf)
    print(cm)
elif best_model_name == 'Naive Bayes':
    cm = confusion_matrix(y_test, y_pred_nb)
    print(cm)
else:
    cm = confusion_matrix(y_test, y_pred_dt)
    print(cm)

plt.figure(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix for {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
