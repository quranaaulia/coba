import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download resource NLTK yang diperlukan
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopwords_id = set(stopwords.words('indonesian'))

def get_preprocessing_steps(df):
    """Fungsi untuk melakukan semua tahap preprocessing"""
    
    # 1. Pembersihan Data Awal
    results = {
        'data_awal': {
            'jumlah_data': len(df),
            'jumlah_duplikat': df.duplicated().sum(),
            'data_kosong': df.isnull().sum().to_dict()
        }
    }
    
    # 2. Hapus Duplikat
    df_clean = df.drop_duplicates()
    results['setelah_hapus_duplikat'] = {
        'jumlah_data': len(df_clean)
    }
    
    # 3. Hapus Data Kosong
    df_clean = df_clean.dropna(subset=['text'])
    results['setelah_hapus_kosong'] = {
        'jumlah_data': len(df_clean)
    }
    
    # 4. Preprocessing Text
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Hapus tanda baca
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Hapus karakter khusus
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Tokenisasi sederhana
        tokens = text.split()
        # Hapus stopwords
        tokens = [t for t in tokens if t not in stopwords_id]
        # Stemming
        tokens = [stemmer.stem(t) for t in tokens]
        return ' '.join(tokens)
    
    # Terapkan preprocessing ke semua text
    df_clean['text_clean'] = df_clean['text'].apply(clean_text)
    
    # Simpan hasil
    results['hasil_preprocessing'] = df_clean[['text', 'text_clean']].to_dict('records')
    
    return results
