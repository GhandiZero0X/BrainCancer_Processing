import pandas as pd
import numpy as np

# Membaca dataset
file_path = 'TCGA_GBM_LGG_Mutations_all.csv'
data = pd.read_csv(file_path)

# Menampilkan data awal
print("Data Awal :")
print(data)

# Cek informasi dataset awal
print("\nInfo Dataset awal :")
data.info()

# Mengganti nilai '--' dan 'not reported' dengan NaN
data.replace(['--', 'not reported'], np.nan, inplace=True)

# Deteksi Missing Value
print("\nMissing Values di Setiap Kolom :")
missing = data.isnull().sum()
print(missing)

# Menghapus baris yang memiliki missing values
data_cleaned_missingValue = data.dropna()

# Info dataset setelah penanganan
print("\nInfo Dataset Setelah Penanganan Missing Values :")
print(data_cleaned_missingValue.isnull().sum())

# Define mappings for categorical values
gender_mapping = {'Male': 0, 'Female': 1}
race_mapping = {'white': 0, 'black or african american': 1, 'asian': 2, 'american indian or alaska native': 3}
mutation_mapping = {'MUTATED': 1, 'NOT_MUTATED': 0}

# Apply mappings for Gender and Race columns using .loc
data_cleaned_missingValue.loc[:, 'Gender'] = data_cleaned_missingValue['Gender'].map(gender_mapping)
data_cleaned_missingValue.loc[:, 'Race'] = data_cleaned_missingValue['Race'].map(race_mapping)

# Convert Age_at_diagnosis to numeric by extracting the total days
age_diagnosis_extracted = data_cleaned_missingValue['Age_at_diagnosis'].str.extract(r'(\d+)\syears\s(\d+)\sdays')

# Pastikan ada data yang diekstrak sebelum penetapan
if age_diagnosis_extracted.notna().all(axis=1).any():
    data_cleaned_missingValue.loc[:, 'Age_at_diagnosis'] = age_diagnosis_extracted.astype(float).sum(axis=1) * 365
else:
    print("Tidak ada data yang cocok untuk kolom 'Age_at_diagnosis'.")

if age_diagnosis_extracted.isna().any().any():
    # Konversi ke tahun dengan membagi total hari dengan 365
    data_cleaned_missingValue.loc[:, 'Age_at_diagnosis'] = age_diagnosis_extracted[0].astype(float) + (age_diagnosis_extracted[1].astype(float) / 365)
else:
    print("Terdapat data yang tidak dapat diekstrak dengan benar dari kolom 'Age_at_diagnosis'.")

# List of columns to be transformed to binary (0 or 1)
mutation_columns = [
    'IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 
    'PIK3CA', 'NF1', 'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 
    'CSMD3', 'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA'
]

# Apply mutation mapping for the listed columns
data_cleaned_missingValue_modifiedtype = data_cleaned_missingValue.copy()  # Copying the cleaned data
data_cleaned_missingValue_modifiedtype[mutation_columns] = data_cleaned_missingValue_modifiedtype[mutation_columns].applymap(lambda x: mutation_mapping.get(x, x))

# Mencetak dataset yang telah dibersihkan dan dimodifikasi
print("Dataset yang telah dimodifikasi:")
print(data_cleaned_missingValue_modifiedtype)

# Pengecekan dan Penanganan Outlier
# Menghitung IQR untuk kolom Age_at_diagnosis
Q1 = data_cleaned_missingValue_modifiedtype['Age_at_diagnosis'].quantile(0.25)
Q3 = data_cleaned_missingValue_modifiedtype['Age_at_diagnosis'].quantile(0.75)
IQR = Q3 - Q1

# Mendefinisikan batas bawah dan batas atas untuk outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identifikasi Outlier
outliers = data_cleaned_missingValue_modifiedtype[
    (data_cleaned_missingValue_modifiedtype['Age_at_diagnosis'] < lower_bound) | 
    (data_cleaned_missingValue_modifiedtype['Age_at_diagnosis'] > upper_bound)
]

# Menampilkan jumlah dan data outlier
jumlah_outliers = len(outliers)
print("Jumlah outlier:", jumlah_outliers)
print("\nData outliers berdasarkan Age_at_diagnosis:")
print(outliers)

# Menghapus outliers
data_no_outliers = data_cleaned_missingValue_modifiedtype[~((data_cleaned_missingValue_modifiedtype['Age_at_diagnosis'] < lower_bound) | 
                                                            (data_cleaned_missingValue_modifiedtype['Age_at_diagnosis'] > upper_bound))]

print("\nMissing Data setelah menghapus outliers :")
print(data_no_outliers.isnull().sum())

# Mengecek jumlah outlier setelah pembersihan
cek_outlier_setelah_bersih = data_no_outliers[
    (data_no_outliers['Age_at_diagnosis'] < lower_bound) | 
    (data_no_outliers['Age_at_diagnosis'] > upper_bound)
]

jumlah_outliers_setelah_bersih = len(cek_outlier_setelah_bersih)
print("Jumlah outlier setelah data bersih:", jumlah_outliers_setelah_bersih)

# Menyimpan dataset yang telah dibersihkan dan tanpa outlier ke file CSV
output_file_path = 'TCGA_GBM_LGG_Mutations_clean_missingValue_Outlier.csv'
data_no_outliers.to_csv(output_file_path, index=False)

# perbandingan data sebelum dan sesudah Outlier
print("\nPerbandingan data sebelum dan sesudah Outlier:")
print("Jumlah baris data sebelum Outlier:", len(data_cleaned_missingValue_modifiedtype))
print("Jumlah baris data setelah Outlier:", len(data_no_outliers))

print(f"\nDataset yang telah dibersihkan telah disimpan sebagai '{output_file_path}'")