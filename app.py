from flask import Flask, render_template, request
from mesin_learning import predict

app = Flask(__name__)

# Route untuk halaman utama (index)
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk halaman form
@app.route('/form')
def form_diagnosa():
    return render_template('form.html')

# Route untuk memproses hasil prediksi
@app.route('/hasil', methods=['POST'])
def result():
    # Ambil data dari form
    data = {
        "Gender": request.form['Gender'],
        "Age": request.form['Age'],
        "Race": request.form['Race'],
        "IDH1": request.form['IDH1'],
        "TP53": request.form['TP53'],
        "ATRX": request.form['ATRX'],
        "PTEN": request.form['PTEN'],
        "EGFR": request.form['EGFR'],
        "CIC": request.form['CIC'],
        "MUC16": request.form['MUC16'],
        "PIK3CA": request.form['PIK3CA'],
        "NF1": request.form['NF1'],
        "PIK3R1": request.form['PIK3R1'],
        "FUBP1": request.form['FUBP1'],
        "RB1": request.form['RB1'],
        "NOTCH1": request.form['NOTCH1'],
        "BCOR": request.form['BCOR'],
        "CSMD3": request.form['CSMD3'],
        "SMARCA4": request.form['SMARCA4'],
        "GRIN2A": request.form['GRIN2A'],
        "IDH2": request.form['IDH2'],
        "FAT4": request.form['FAT4'],
        "PDGFRA": request.form['PDGFRA']
    }

    # Buat list dengan input user untuk prediksi
    input_data = [[data[key] for key in data.keys()]]

    # Prediksi menggunakan fungsi predict dari mesin_learning.py
    prediction = predict(input_data)
    
    if data['Gender'] == '0':
        data['Gender'] = "Male"
    else:
        data['Gender'] = "Female"

    if data['Race'] == '0.0':
        data['Race'] = "White"
    elif data['Race'] == '0.333333':
        data['Race'] = "Black or African American"
    elif data['Race'] == '0.666667':
        data['Race'] = "Asian"
    elif data['Race'] == '1.0':
        data['Race'] = "American Indian or Alaska Native"

    
    for key in ['IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA', 'NF1', 'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA']:
        if data[key] == '0':
            data[key] = "NOT_MUTATED"
        else:
            data[key] = "MUTATED"
    
    if prediction == 0.0:
        prediction = "LGG (Low-Grade Glioma)"  # Low-Grade Glioma
    else:
        prediction = "GBM (Glioblastoma Multiforme)" 

    # Kirim hasil input genomik dan prediksi ke result.html
    return render_template('result.html', data=data, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
