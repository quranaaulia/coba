# app.py
import os

# Batasi penggunaan thread BLAS/OpenMP agar numpy/scikit-learn tidak mengambil semua core
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS", "1")
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "1")
os.environ["VECLIB_MAXIMUM_THREADS"] = os.environ.get("VECLIB_MAXIMUM_THREADS", "1")
os.environ["NUMEXPR_NUM_THREADS"] = os.environ.get("NUMEXPR_NUM_THREADS", "1")

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import os
from config import Config
from utils.file_utils import allowed_file, save_uploaded_file
import pandas as pd

# import layanan bertopic (fungsi ringan saja at top-level)
from services.bertopic_service import (
    load_bertopic_model,
    start_build_bertopic_async,
    get_bertopic_analysis,
    is_building,
)

app = Flask(__name__)
app.config.from_object(Config)


# -----------------------
# Routes untuk upload / home
# -----------------------
@app.route('/', methods=['GET', 'POST'])
def upload():
    session.clear()
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filepath = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
            if os.path.exists(filepath):
                session['uploaded_file'] = filepath
                flash('Upload berhasil! Silakan lanjut ke menu Home.', 'success')
                return redirect(url_for('home'))
            else:
                flash('Gagal menyimpan file!', 'danger')
        else:
            flash('Format file tidak didukung! Harus .csv', 'danger')
    return render_template('upload.html')


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


# -----------------------
# Routes komentar / preprocessing / word2vec
# -----------------------
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

    # Normalisasi nama kolom
    columns_map = {c.strip().lower(): c for c in df.columns}
    if 'text' not in columns_map or 'createtimeiso' not in columns_map:
        flash(f'File CSV harus memiliki kolom "text" dan "createTimeISO". Kolom ditemukan: {df.columns.tolist()}', 'danger')
        return redirect(url_for('home'))

    text_col = columns_map['text']
    date_col = columns_map['createtimeiso']

    comments = [
        {'text': str(row[text_col]), 'date': str(row[date_col])}
        for _, row in df[[text_col, date_col]].dropna(subset=[text_col]).iterrows()
    ]
    total_comments = len(comments)
    return render_template('comments_raw.html', comments=comments, total_comments=total_comments)


@app.route('/comments/preprocessed')
def comments_preprocessed():
    filepath = session.get('uploaded_file')
    if not filepath or not os.path.exists(filepath):
        flash('Silakan upload file CSV terlebih dahulu.', 'warning')
        return redirect(url_for('upload'))
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        from services.preprocessing import get_preprocessing_steps
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


# -----------------------
# Route: Build BERTopic (non-blocking)
# -----------------------
@app.route('/bertopic_build', methods=['GET', 'POST'])
def bertopic_build():
    filepath = session.get('uploaded_file')
    if not filepath or not os.path.exists(filepath):
        flash('Silakan upload file CSV terlebih dahulu.', 'warning')
        return redirect(url_for('upload'))

    rebuild = request.args.get('rebuild') or request.form.get('rebuild')

    # Jika user klik Rebuild -> hapus model lama
    if rebuild or request.args.get('rebuild') == '1':
        if os.path.exists('models/bertopic_model.pkl'):
            try:
                os.remove('models/bertopic_model.pkl')
                flash('Model lama dihapus. Proses build ulang akan dimulai.', 'info')
            except Exception as e:
                flash(f'Gagal menghapus model lama: {e}', 'danger')

    # Cek apakah sedang building
    if is_building():
        return render_template('bertopic_build.html', data={'status': 'building'})

    # Jika model sudah ada -> tampilkan hasil
    if os.path.exists('models/bertopic_model.pkl'):
        data = get_bertopic_analysis()
        data['status'] = 'done'
        return render_template('bertopic_build.html', data=data)

    # Jika POST → mulai proses background build
    if request.method == 'POST':
        start_build_bertopic_async(filepath, max_samples=200)
        return render_template('bertopic_build.html', data={'status': 'started'})

    # Default GET → tampilkan tombol Start Build
    return render_template('bertopic_build.html', data={'status': 'ready'})


# Endpoint status (dipanggil AJAX polling dari frontend)
@app.route('/bertopic_status')
def bertopic_status():
    if os.path.exists('models/bertopic_model.pkl'):
        return jsonify({'status': 'done'})
    if is_building():
        return jsonify({'status': 'building'})
    return jsonify({'status': 'none'})

# -----------------------
# Route analysis
# -----------------------
@app.route('/analysis')
def analysis():
    return render_template('analysis.html')


# -----------------------
if __name__ == '__main__':
    # Jalankan tanpa debug/reloader agar tidak menggandakan process
    app.run(host='127.0.0.1', port=5000, debug=False)
