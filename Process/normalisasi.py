import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Membaca dataset setelah penanganan missing value dan outlier
file_path = 'TCGA_GBM_LGG_Mutations_cleaned_MissingValue_Outlier.csv'
data_cleaned = pd.read_csv(file_path)

# Pisahkan kolom numerik dan non-numerik
data_numerik = data_cleaned.select_dtypes(include=['float64', 'int64'])  # Pilih kolom numerik
data_non_numerik = data_cleaned.select_dtypes(exclude=['float64', 'int64'])  # Pilih kolom non-numerik

# Tampilkan nama kolom numerik dan non-numerik
print("\nNama Kolom Numerik:")
print(data_numerik.columns.tolist())

print("\nNama Kolom Non-Numerik:")
print(data_non_numerik.columns.tolist())

# a. Min-Max Scaling
min_max_scaler = MinMaxScaler()
data_min_max = data_numerik.copy()
data_min_max.iloc[:, :] = min_max_scaler.fit_transform(data_numerik)  # Normalisasi hanya kolom numerik

# Gabungkan kembali data non-numerik dan numerik yang telah dinormalisasi
data_min_max_combined = pd.concat([data_non_numerik, data_min_max], axis=1)

# Tampilkan hasil normalisasi
print("\nHasil Min-Max Scaling:")
print(data_min_max_combined.head())

# Simpan hasil normalisasi ke file CSV
output_file_path = 'TCGA_GBM_LGG_Mutations_Min_Max_Normalized.csv'
data_min_max_combined.to_csv(output_file_path, index=False)
print(f"\nDataset yang telah dinormalisasi disimpan ke: {output_file_path}")
