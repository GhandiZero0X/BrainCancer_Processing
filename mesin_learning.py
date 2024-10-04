import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
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
from sklearn.model_selection import train_test_split
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

best_model_name = max(accuracies, key=accuracies.get)
print(f"\nAlgoritma dengan akurasi tertinggi adalah: {best_model_name}")

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

# Function to predict based on the input data
def predict(input_data):
    models = {
    'Random Forest': rf_clf,
    'Naive Bayes': gaussian,
    'Decision Tree': clf
    }

    # Normalize input data
    input_data_scaled = scaler.transform(input_data)

    best_model = models[best_model_name]
    prediction_rf = best_model.predict(input_data_scaled)

    # Return the prediction result
    return prediction_rf[0]
