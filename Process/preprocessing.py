import pandas as pd
import numpy as np

# Membaca dataset
file_path = 'TCGA_GBM_LGG_Mutations_all.csv'
data = pd.read_csv(file_path)

print("========== Data Awal ==========")
# Menampilkan data awal
print("Data Awal :")
print(data)

# Cek informasi dataset awal
print("\nInfo Dataset awal :")
data.info()

# Mengganti nilai '--' dan 'not reported' dengan NaN
data.replace(['--', 'not reported'], np.nan, inplace=True)

print("\n========== Penanganan Missing Values ==========")
# Deteksi Missing Value
print("\nMissing Values di Setiap Kolom :")
missing = data.isnull().sum()
print(missing)

# Menghapus baris yang memiliki missing values
data_cleaned_missingValue = data.dropna()

# Info dataset setelah penanganan
print("\nInfo Dataset Setelah Penanganan Missing Values :")
print(data_cleaned_missingValue.isnull().sum())

print("\n========== Modifikasi Tipe Data ==========")
# Modifikasi tipe data Gender dan Race
gender_mapping = {'Male': 0, 'Female': 1}
race_mapping = {'white': 0, 'black or african american': 1, 'asian': 2, 'american indian or alaska native': 3}
mutation_mapping = {'MUTATED': 1, 'NOT_MUTATED': 0}

data_cleaned_missingValue.loc[:, 'Gender'] = data_cleaned_missingValue['Gender'].map(gender_mapping)
data_cleaned_missingValue.loc[:, 'Race'] = data_cleaned_missingValue['Race'].map(race_mapping)

# Ekstrak tahun dan hari dari Age_at_diagnosis
# Jika tidak ada hari, kita asumsikan 0 hari
age_diagnosis_extracted = data_cleaned_missingValue['Age_at_diagnosis'].str.extract(r'(\d+)\syears(?:\s(\d+)\sdays)?')

# Mengisi NaN pada kolom hari dengan 0
age_diagnosis_extracted[1].fillna(0, inplace=True)

# Konversi tahun dan hari ke dalam bentuk desimal (tahun + hari/365)
data_cleaned_missingValue.loc[:, 'Age_at_diagnosis'] = age_diagnosis_extracted[0].astype(float) + (age_diagnosis_extracted[1].astype(float) / 365)

# List of mutation columns to be transformed to binary (0 or 1)
mutation_columns = [
    'IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 
    'PIK3CA', 'NF1', 'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 
    'CSMD3', 'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA'
]

# Apply mutation mapping for the listed columns
data_cleaned_missingValue_modifiedtype = data_cleaned_missingValue.copy()  # Copying the cleaned data
data_cleaned_missingValue_modifiedtype[mutation_columns] = data_cleaned_missingValue_modifiedtype[mutation_columns].applymap(lambda x: mutation_mapping.get(x, x))

# Mengecek distribusi frekuensi nilai pada setiap kolom setelah penghapusan missing values
# for column in data_cleaned_missingValue_modifiedtype.columns:
#     print(f"Distribusi frekuensi nilai di kolom '{column}':")
#     print(data_cleaned_missingValue_modifiedtype[column].value_counts())
#     print("\n" + "-"*50 + "\n")

# Menampilkan hasil akhir
print("Dataset yang telah dimodifikasi:")
print(data_cleaned_missingValue_modifiedtype)

print("\nInfo Dataset yang telah dimodifikasi:")
print(data_cleaned_missingValue_modifiedtype.isnull().sum())

# save to csv
data_cleaned_missingValue_modifiedtype.to_csv('TCGA_GBM_LGG_Mutations_cleaned_modifikasi.csv', index=False)

print("\n========== Data Penanganan Outlier ==========")
# Pengecekan dan Penanganan Outlier
# Mencari outlier pada kolom Age_at_diagnosis
Q1 = data_cleaned_missingValue_modifiedtype['Age_at_diagnosis'].quantile(0.25)
Q3 = data_cleaned_missingValue_modifiedtype['Age_at_diagnosis'].quantile(0.75)
IQR = Q3 - Q1

# Identifikasi Outlier
outliers = data_cleaned_missingValue_modifiedtype[(data_cleaned_missingValue_modifiedtype['Age_at_diagnosis'] < (Q1 - 1.5 * IQR)) | (data_cleaned_missingValue_modifiedtype['Age_at_diagnosis'] > (Q3 + 1.5 * IQR))]
jumlah_outliers = len(outliers)
print("Jumlah outlier pada kolom Age_at_diagnosis:", jumlah_outliers)
print("\nOutliers pada kolom Age_at_diagnosis:", outliers)

# Menghapus outliers
data_no_outliers = data_cleaned_missingValue_modifiedtype[(data_cleaned_missingValue_modifiedtype['Age_at_diagnosis'] >= (Q1 - 1.5 * IQR)) & (data_cleaned_missingValue_modifiedtype['Age_at_diagnosis'] <= (Q3 + 1.5 * IQR))]  # Menghapus outliers
print("\nMissing Data setelah menghapus outliers :")
print(data_no_outliers.isnull().sum())

# Cek outlier setelah data bersih
cek_outlier_setelah_bersih = data_no_outliers[(data_no_outliers['Age_at_diagnosis'] < (Q1 - 1.5 * IQR)) | (data_no_outliers['Age_at_diagnosis'] > (Q3 + 1.5 * IQR))]    # Cek outlier setelah data bersih
jumlah_outliers_setelah_bersih = len(cek_outlier_setelah_bersih)
print("Jumlah outlier setelah data bersih:", jumlah_outliers_setelah_bersih)

# Menampilkan data setelah menghapus outliers
print("\nData setelah menghapus outliers pada kolom Age_at_diagnosis:")
print(len(data_cleaned_missingValue_modifiedtype))
print(len(data_no_outliers))

# Menyimpan data ke dalam file CSV
data_no_outliers.to_csv('TCGA_GBM_LGG_Mutations_cleaned_MissingValue_Outlier.csv', index=False)
print("\nDataset yang telah dibersihkan telah disimpan sebagai 'TCGA_GBM_LGG_Mutations_cleaned.csv'")