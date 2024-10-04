import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path = 'TCGA_InfoWithGrade_MissingValue_Outlier.csv'
data_normalisasi = pd.read_csv(file_path)

# pilih kolom yang numerik
data_numerik = data_normalisasi.select_dtypes(include=['float64', 'int64']).columns

# kolom numerik
print("\nNama Kolom Numerik:")
print(data_numerik.tolist())

# Min Max scalling
scaler = MinMaxScaler()
data_normalisasi[data_numerik] = scaler.fit_transform(data_normalisasi[data_numerik])

# menampilakn data setelah normalisasi
print(data_normalisasi.head())
print("info data setelah normalisasi", data_normalisasi.info())
# cek distribusi data setelah normalisasi
for c in list(data_normalisasi):
    print(f"\nValue counts pada kolom '{c}':\n", data_normalisasi[c].value_counts())

data_normalisasi.to_csv('TCGA_InfoWithGrade_Normalisasi.csv', index=False)

