import pandas as pd
import numpy as np

# Membaca dataset
file_path = 'TCGA_InfoWithGrade.csv'
data = pd.read_csv(file_path)

print("========== Data Awal ==========")
print("Data Awal : \n", data.head())

print("Data Info : \n")
data.info()

for c in list(data):
    print(f"\nValue counts pada kolom '{c}':\n", data[c].value_counts())

print("\n========== Pengecekan Missing Value dan Outlier ==========")
# Pengecekan Missing Value
print("Missing Value Setiap Kolom : \n", data.isnull().sum())

# Penanganan Missing Value dengan Menghapus Baris yang Memiliki Missing Value
data_clean_missingValue = data.dropna()

# Pengcekan Outlier pada Kolom 'Age_at_diagnosis'
Q1 = data_clean_missingValue['Age_at_diagnosis'].quantile(0.25)
Q3 = data_clean_missingValue['Age_at_diagnosis'].quantile(0.75)
IQR = Q3 - Q1

# Batas untuk Outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identifikasi Outlier
outliers = data_clean_missingValue[(data_clean_missingValue['Age_at_diagnosis'] < lower_bound) | 
                                   (data_clean_missingValue['Age_at_diagnosis'] > upper_bound)]
jumlah_outlier = len(outliers)
print(f"\nJumlah outlier pada kolom 'Age_at_diagnosis': {jumlah_outlier}")
print("Outliers pada kolom 'Age_at_diagnosis':\n", outliers)

# Menghapus Outlier
data_cleaned_Outlier = data_clean_missingValue[~((data_clean_missingValue['Age_at_diagnosis'] < lower_bound) | 
                                                (data_clean_missingValue['Age_at_diagnosis'] > upper_bound))]

# Validasi Pengecekan Missing Value dan Outlier Setelah Penghapusan
print("\n========== Informasi Data Setelah Pembersihan ==========")
print("Jumlah Missing Value Setelah Penghapusan:\n", data_cleaned_Outlier.isnull().sum())

# Pengecekan Ulang untuk Outlier
outliers_after_cleaning = data_cleaned_Outlier[(data_cleaned_Outlier['Age_at_diagnosis'] < lower_bound) | 
                                                (data_cleaned_Outlier['Age_at_diagnosis'] > upper_bound)]
print(f"\nJumlah Outlier Setelah Pembersihan: {len(outliers_after_cleaning)}")

# Simpan dataset yang sudah dibersihkan
data_cleaned_Outlier.to_csv('TCGA_InfoWithGrade_MissingValue_Outlier.csv', index=False)
print("\nDataset yang telah dibersihkan telah disimpan sebagai 'TCGA_InfoWithGrade_MissingValue_Outlier.csv'")
