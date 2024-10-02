import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'TCGA_InfoWithGrade_Normalisasi.csv'
data_decisiontree = pd.read_csv(file_path)

# Define features (X) and target (y)
X = data_decisiontree.drop('Grade', axis=1)
y = data_decisiontree['Grade']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize the 'Age_at_diagnosis' feature
X_train['Age_at_diagnosis'] = scaler.fit_transform(X_train[['Age_at_diagnosis']])
X_test['Age_at_diagnosis'] = scaler.transform(X_test[['Age_at_diagnosis']])

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=0)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("\n========== Decision Tree Model Evaluation ==========")
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Precision: {:.2f}%".format(precision_score(y_test, y_pred, average='weighted') * 100))
print("Recall: {:.2f}%".format(recall_score(y_test, y_pred, average='weighted') * 100))
print("F1 Score: {:.2f}%".format(f1_score(y_test, y_pred, average='weighted') * 100))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualize Confusion Matrix
f, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt=".0f", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Confusion Matrix for Decision Tree')
plt.show()

# Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=True, rounded=True)
plt.title('Decision Tree')
plt.show()

# Manual Input Prediction
# print("\n========== Manual Input Prediction ==========")

# # Get input for each feature
# gender = float(input("Enter Gender (Male = 0, Female = 1): "))
# age_at_diagnosis = float(input("Enter Age at Diagnosis (actual value): "))
# age_scaled = scaler.transform([[age_at_diagnosis]])[0][0]  # Normalize the input
# race = float(input("Enter Race (white = 0.0, black or african american = 0.333333, asian = 0.666667, american indian or alaska native = 1.0): "))

# # Input for mutation status of genes
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
#     'Age_at_diagnosis': [age_scaled],  # Use normalized value
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
# manual_prediction = clf.predict(manual_input)[0]

# if manual_prediction == 0.0:
#     manual_prediction = "LGG"
# else:
#     manual_prediction = "GBM"

# print("Predicted Class for Manual Input:", manual_prediction)
