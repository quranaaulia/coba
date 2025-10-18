from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
from config import Config
from utils.file_utils import allowed_file, save_uploaded_file
import pandas as pd
from services.preprocessing import get_preprocessing_steps

app = Flask(__name__)
app.config.from_object(Config)

# 🟢 Route: Halaman Upload CSV
@app.route('/', methods=['GET', 'POST'])
def upload():
    # Hapus session saat pertama masuk
    session.clear()
    
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filepath = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
            # Cek file exists
            if os.path.exists(filepath):
                session['uploaded_file'] = filepath
                flash('Upload berhasil! Silakan lanjut ke menu Home.', 'success')
                return redirect(url_for('home'))
            else:
                flash('Gagal menyimpan file!', 'danger')
        else:
            flash('Format file tidak didukung! Harus .csv', 'danger')
    return render_template('upload.html')

# 🟢 Route: Home / Dashboard
@app.route('/home', methods=['GET', 'POST'])
def home():
    filepath = session.get('uploaded_file')
    uploaded = bool(filepath and os.path.exists(filepath))
    
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filepath = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
            if os.path.exists(filepath):
                session['uploaded_file'] = filepath
                flash('Upload berhasil! File diganti.', 'success')
                return redirect(url_for('home'))
            else:
                flash('Gagal menyimpan file!', 'danger')
        else:
            flash('Format file tidak didukung! Harus .csv', 'danger')
    return render_template('home.html', uploaded=uploaded)

# 🟢 Route: Komentar Asli
@app.route('/comments/raw')
def comments_raw():
    filepath = session.get('uploaded_file')
    if not filepath or not os.path.exists(filepath):
        flash('Silakan upload file CSV terlebih dahulu.', 'warning')
        return redirect(url_for('upload'))
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except Exception as e:
        flash(f'Gagal membaca file CSV: {e}', 'danger')
        return redirect(url_for('home'))

    print("Kolom CSV:", df.columns.tolist())

    # Normalisasi nama kolom
    columns_map = {c.strip().lower(): c for c in df.columns}
    if 'text' not in columns_map or 'createtimeiso' not in columns_map:
        flash(f'File CSV harus memiliki kolom \"text\" dan \"createTimeISO\". Kolom ditemukan: {df.columns.tolist()}', 'danger')
        return redirect(url_for('home'))

    text_col = columns_map['text']
    date_col = columns_map['createtimeiso']

    comments = [
        {'text': str(row[text_col]), 'date': str(row[date_col])}
        for _, row in df[[text_col, date_col]].dropna(subset=[text_col]).iterrows()
    ]
    total_comments = len(comments)
    return render_template('comments_raw.html', comments=comments, total_comments=total_comments)

# 🟢 Route: Komentar Setelah Preprocessing
@app.route('/comments/preprocessed')
def comments_preprocessed():
    filepath = session.get('uploaded_file')
    if not filepath or not os.path.exists(filepath):
        flash('Silakan upload file CSV terlebih dahulu.', 'warning')
        return redirect(url_for('upload'))

    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        from services.preprocessing import get_preprocessing_steps

        # Get all preprocessing steps
        preprocessing_results = get_preprocessing_steps(df)

        return render_template(
            'comments_preprocessed.html',
            results=preprocessing_results,
            total_original=len(df),
            total_processed=len(preprocessing_results['hasil_preprocessing'])
        )

    except Exception as e:
        flash(f'Error dalam preprocessing: {str(e)}', 'danger')
        return redirect(url_for('home'))

@app.route('/word2vec')
def word2vec():
    filepath = session.get('uploaded_file')
    if not filepath or not os.path.exists(filepath):
        flash('Silakan upload file CSV terlebih dahulu.', 'warning')
        return redirect(url_for('upload'))
    
    from services.word2vec_service import get_word2vec_analysis
    data = get_word2vec_analysis(filepath)
    return render_template('word2vec.html', data=data)
@app.route('/bertopic_build')
def bertopic_build():
    return render_template('bertopic_build.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)