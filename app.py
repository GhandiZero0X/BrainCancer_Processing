from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# Route untuk melayani file statis (CSS, JavaScript, gambar, dll.)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Route untuk halaman utama (index)
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk halaman About
@app.route('/hasil')
def about():
    return render_template('about.html')

# Route untuk halaman Contact
@app.route('/form')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
