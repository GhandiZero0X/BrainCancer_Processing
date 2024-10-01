import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from decisiontree import DecisionTreeClassifier  # Sesuaikan dengan nama kelas/model di file decisiontree.py
from naivebayes import GaussianNB      # Sesuaikan dengan nama kelas/model di file naivebayes.py
from randomforest import RandomForestClassifier  # Sesuaikan dengan nama kelas/model di file randomforest.py

# Load your dataset (replace 'your_dataset.csv' with the actual dataset file path)
dataset = pd.read_csv('TCGA_InfoWithGrade_Normalisasi.csv')

# Assume the last column is the target label and the rest are features
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inisialisasi model
decision_tree_model = DecisionTreeClassifier()
naive_bayes_model = GaussianNB()
random_forest_model = RandomForestClassifier()

# Melatih dan evaluasi model
decision_tree_model.fit(X_train, y_train)
dt_accuracy = decision_tree_model.score(X_test, y_test)

naive_bayes_model.fit(X_train, y_train)
nb_accuracy = naive_bayes_model.score(X_test, y_test)

random_forest_model.fit(X_train, y_train)
rf_accuracy = random_forest_model.score(X_test, y_test)

# Membandingkan hasil akurasi
print("Akurasi Decision Tree:", dt_accuracy)
print("Akurasi Naive Bayes:", nb_accuracy)
print("Akurasi Random Forest:", rf_accuracy)

# Menentukan model dengan akurasi terbaik
if dt_accuracy > nb_accuracy and dt_accuracy > rf_accuracy:
    best_model = decision_tree_model
    best_model_name = "Decision Tree"
elif nb_accuracy > dt_accuracy and nb_accuracy > rf_accuracy:
    best_model = naive_bayes_model
    best_model_name = "Naive Bayes"
else:
    best_model = random_forest_model
    best_model_name = "Random Forest"

print(f"\nAlgoritma dengan akurasi tertinggi adalah: {best_model_name}")

# Melakukan input manual untuk prediksi
print("\n========== Manual Input Prediction ==========")

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Mendapatkan input untuk setiap fitur sesuai dengan instruksi
gender = float(input("Enter Gender (Male = 0, Female = 1): "))
age_at_diagnosis = float(input("Enter Age at Diagnosis (actual value): "))
age_scaled = scaler.fit_transform([[age_at_diagnosis]])[0][0]
race = float(input("Enter Race (white = 0.0, black or african american = 0.333333, asian = 0.666667, american indian or alaska native = 1.0): "))

# Input untuk status mutasi gen
idh1 = float(input("Enter IDH1 (NOT_MUTATED = 0, MUTATED = 1): "))
tp53 = float(input("Enter TP53 (NOT_MUTATED = 0, MUTATED = 1): "))
atrx = float(input("Enter ATRX (NOT_MUTATED = 0, MUTATED = 1): "))
pten = float(input("Enter PTEN (NOT_MUTATED = 0, MUTATED = 1): "))
egfr = float(input("Enter EGFR (NOT_MUTATED = 0, MUTATED = 1): "))
cic = float(input("Enter CIC (NOT_MUTATED = 0, MUTATED = 1): "))
muc16 = float(input("Enter MUC16 (NOT_MUTATED = 0, MUTATED = 1): "))
pik3ca = float(input("Enter PIK3CA (NOT_MUTATED = 0, MUTATED = 1): "))
nf1 = float(input("Enter NF1 (NOT_MUTATED = 0, MUTATED = 1): "))
pik3r1 = float(input("Enter PIK3R1 (NOT_MUTATED = 0, MUTATED = 1): "))
fubp1 = float(input("Enter FUBP1 (NOT_MUTATED = 0, MUTATED = 1): "))
rb1 = float(input("Enter RB1 (NOT_MUTATED = 0, MUTATED = 1): "))
notch1 = float(input("Enter NOTCH1 (NOT_MUTATED = 0, MUTATED = 1): "))
bcor = float(input("Enter BCOR (NOT_MUTATED = 0, MUTATED = 1): "))
csmd3 = float(input("Enter CSMD3 (NOT_MUTATED = 0, MUTATED = 1): "))
smarca4 = float(input("Enter SMARCA4 (NOT_MUTATED = 0, MUTATED = 1): "))
grin2a = float(input("Enter GRIN2A (NOT_MUTATED = 0, MUTATED = 1): "))
idh2 = float(input("Enter IDH2 (NOT_MUTATED = 0, MUTATED = 1): "))
fat4 = float(input("Enter FAT4 (NOT_MUTATED = 0, MUTATED = 1): "))
pdgf = float(input("Enter PDGFRA (NOT_MUTATED = 0, MUTATED = 1): "))

# Membuat array fitur input
input_data = np.array([[gender, age_scaled, race, idh1, tp53, atrx, pten, egfr, cic, muc16, 
                        pik3ca, nf1, pik3r1, fubp1, rb1, notch1, bcor, csmd3, smarca4, 
                        grin2a, idh2, fat4, pdgf]])

# Melakukan prediksi
prediction = best_model.predict(input_data)

print(f"\nPrediksi berdasarkan input manual menggunakan {best_model_name}: {prediction[0]}")
