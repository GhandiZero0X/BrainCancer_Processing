import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'TCGA_InfoWithGrade_Normalisasi.csv'
data_naiveBayes = pd.read_csv(file_path)

print("========== Pemrosesan Naive Bayes ==========")

print(data_naiveBayes.info())

# Mengecek distribusi frekuensi nilai pada setiap kolom setelah label encoding
for column in data_naiveBayes.columns:
    print("\nFrequency Distribution of", column)
    print(data_naiveBayes[column].value_counts())

target_column = 'Grade' 
X = data_naiveBayes.drop(columns=[target_column])
print("\nFeatures:")
print(X.head())
y = data_naiveBayes[target_column]

# pisahkan dataset menjadi 20% data test dan 80% data training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# tarining model naive bayes dengan GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)

# evaluasi model naive bayes
print("\n========== Naive Bayes Model Evaluation ==========")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Accuracy
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Precision: {:.2f}%".format(precision_score(y_test, y_pred, average='weighted') * 100))
print("Recall: {:.2f}%".format(recall_score(y_test, y_pred, average='weighted') * 100))
print("F1 Score: {:.2f}%".format(f1_score(y_test, y_pred, average='weighted') * 100))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Step 7: Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# print("\n========== Manual Input Prediction ==========")

# # Initialize the MinMaxScaler
# scaler = MinMaxScaler()

# # Get input for each feature according to the specified mappings
# gender = float(input("Enter Gender (Male = 0, Female = 1): "))

# age_at_diagnosis = float(input("Enter Age at Diagnosis (actual value): "))
# age_scaled = scaler.fit_transform([[age_at_diagnosis]])[0][0]
# race = float(input("Enter Race (white = 0.0, black or african american = 0.333333, asian = 0.666667, american indian or alaska native = 1.0): "))
# # Input for mutation status for the genes
# idh1 = float(input("Enter IDH1 (NOT_MUTATED = 0, MUTATED = 1): "))
# tp53 = float(input("Enter TP53 (NOT_MUTATED = 0, MUTATED = 1): "))
# atrx = float(input("Enter ATRX (NOT_MUTATED = 0, MUTATED = 1): "))
# pten = float(input("Enter PTEN (NOT_MUTATED = 0, MUTATED = 1): "))
# egfr = float(input("Enter EGFR (NOT_MUTATED = 0, MUTATED = 1): "))
# cic = float(input("Enter CIC (NOT_MUTATED = 0, MUTATED = 1): "))
# muc16 = float(input("Enter MUC16 (NOT_MUTATED = 0, MUTATED = 1): "))
# pik3ca = float(input("Enter PIK3CA (NOT_MUTATED = 0, MUTATED = 1): "))
# nf1 = float(input("Enter NF1 (NOT_MUTATED = 0, MUTATED = 1): "))
# pik3r1 = float(input("Enter PIK3R1 (NOT_MUTATED = 0, MUTATED = 1): "))
# fubp1 = float(input("Enter FUBP1 (NOT_MUTATED = 0, MUTATED = 1): "))
# rb1 = float(input("Enter RB1 (NOT_MUTATED = 0, MUTATED = 1): "))
# notch1 = float(input("Enter NOTCH1 (NOT_MUTATED = 0, MUTATED = 1): "))
# bcor = float(input("Enter BCOR (NOT_MUTATED = 0, MUTATED = 1): "))
# csmd3 = float(input("Enter CSMD3 (NOT_MUTATED = 0, MUTATED = 1): "))
# smarca4 = float(input("Enter SMARCA4 (NOT_MUTATED = 0, MUTATED = 1): "))
# grin2a = float(input("Enter GRIN2A (NOT_MUTATED = 0, MUTATED = 1): "))
# idh2 = float(input("Enter IDH2 (NOT_MUTATED = 0, MUTATED = 1): "))
# fat4 = float(input("Enter FAT4 (NOT_MUTATED = 0, MUTATED = 1): "))
# pdgf = float(input("Enter PDGFRA (NOT_MUTATED = 0, MUTATED = 1): "))

# # Create manual input DataFrame
# manual_input = pd.DataFrame({
#     'Gender': [gender],  
#     'Age_at_diagnosis': [age_scaled],  # Use the normalized value here
#     'Race': [race],  
#     'IDH1': [idh1],
#     'TP53': [tp53],
#     'ATRX': [atrx],
#     'PTEN': [pten],
#     'EGFR': [egfr],
#     'CIC': [cic],
#     'MUC16': [muc16],
#     'PIK3CA': [pik3ca],
#     'NF1': [nf1],
#     'PIK3R1': [pik3r1],
#     'FUBP1': [fubp1],
#     'RB1': [rb1],
#     'NOTCH1': [notch1],
#     'BCOR': [bcor],
#     'CSMD3': [csmd3],
#     'SMARCA4': [smarca4],
#     'GRIN2A': [grin2a],
#     'IDH2': [idh2],
#     'FAT4': [fat4],
#     'PDGFRA': [pdgf]
# })

# # Predict using the trained model
# manual_prediction = gaussian.predict(manual_input)

# # Condition for the prediction result, where 0 = LGG and 1 = GBM
# if manual_prediction == 0.0:
#     manual_prediction = "LGG"
# else:
#     manual_prediction = "GBM"
# print("Predicted Class for Manual Input:", manual_prediction)