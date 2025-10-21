import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import plotly.express as px
import joblib

def load_bertopic_model(model_path='models/bertopic_model.pkl'):
    """Load topic model"""
    try:
        if os.path.exists(model_path):
            return joblib.load(model_path)
    except:
        pass
    return None

def build_bertopic_model(filepath=None):
    """Build topic model using LDA from uploaded CSV data"""
    if filepath is None:
        return {
            'error': 'Filepath tidak diberikan. Pastikan file CSV sudah diupload.'
        }

    try:
        # Load and preprocess data
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        from services.preprocessing import get_preprocessing_steps
        preprocessing_results = get_preprocessing_steps(df)
        text_data = [item['text_clean'] for item in preprocessing_results['hasil_preprocessing'] if item['text_clean']]

        if not text_data:
            return {
                'error': 'Tidak ada data teks yang valid setelah preprocessing.'
            }

        # Limit data for performance (take first 200 samples)
        max_samples = 200
        if len(text_data) > max_samples:
            text_data = text_data[:max_samples]

        # Setup Indonesian stopwords
        indonesian_stopwords = stopwords.words('indonesian')

        # Create CountVectorizer for LDA
        count_vectorizer = CountVectorizer(
            max_df=0.9,
            min_df=5,
            stop_words=indonesian_stopwords,
            ngram_range=(1, 2)
        )
        count_matrix = count_vectorizer.fit_transform(text_data)

        # Determine number of topics (between 5-15 based on data size)
        n_topics = min(15, max(5, len(text_data) // 50))

        # Train LDA model with optimized parameters
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10,  # Reduced for speed
            learning_method='online',
            learning_offset=50.0,
            evaluate_every=5
        )

        lda_output = lda_model.fit_transform(count_matrix)

        # Get feature names
        feature_names = count_vectorizer.get_feature_names_out()

        # Extract topics
        topics_summary = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[:-6:-1]  # Top 5 words
            top_words = [feature_names[i] for i in top_words_idx]
            topic_prob = lda_output[:, topic_idx].mean()
            count = int(topic_prob * len(text_data))

            topics_summary.append({
                'topic_id': topic_idx,
                'keywords': top_words,
                'count': count,
                'probability': topic_prob
            })

        # Sort topics by count
        topics_summary.sort(key=lambda x: x['count'], reverse=True)

        # Calculate coherence score
        text_tokenized = [text.split() for text in text_data]
        dictionary = corpora.Dictionary(text_tokenized)
        topic_words = [[word for word in topic['keywords']] for topic in topics_summary]

        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=text_tokenized,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()

        # Save model
        model_data = {
            'lda_model': lda_model,
            'count_vectorizer': count_vectorizer,
            'topics_summary': topics_summary,
            'coherence_score': coherence_score,
            'total_topics': len(topics_summary)
        }
        joblib.dump(model_data, 'models/bertopic_model.pkl')

        # Skip visualizations for performance - will be shown in analysis page
        distribution_html = "<p>Visualisasi dinonaktifkan untuk performa optimal</p>"
        barchart_html = "<p>Visualisasi dinonaktifkan untuk performa optimal</p>"
        hierarchy_html = "<p>Visualisasi hierarchy tidak tersedia untuk model LDA</p>"
        topics_html = "<p>Visualisasi topik 2D tidak tersedia untuk model LDA</p>"

        return {
            'topics_summary': topics_summary,
            'coherence_score': coherence_score,
            'total_topics': len(topics_summary),
            'model_type': 'LDA (Alternatif BERTopic)',
            'visualizations': {
                'barchart': barchart_html,
                'hierarchy': hierarchy_html,
                'topics': topics_html,
                'distribution': distribution_html
            }
        }

    except Exception as e:
        return {
            'error': f'Gagal membangun model topik: {str(e)}'
        }

def get_bertopic_analysis():
    """Get topic analysis data"""
    model_data = load_bertopic_model()
    if model_data is None:
        return {
            'error': 'Model topik tidak ditemukan. Pastikan model sudah dibangun.'
        }

    topic_model = model_data['topic_model']
    topics_summary = model_data['topics_summary']

    # Create visualizations using BERTopic
    try:
        # Distribution chart
        fig_dist = topic_model.visualize_barchart(top_n_topics=10)
        distribution_html = fig_dist.to_html(full_html=False)
    except:
        distribution_html = "<p>Data visualisasi distribusi tidak dapat dibuat</p>"

    try:
        # Hierarchy chart
        fig_hierarchy = topic_model.visualize_hierarchy()
        hierarchy_html = fig_hierarchy.to_html(full_html=False)
    except:
        hierarchy_html = "<p>Data visualisasi hierarchy tidak dapat dibuat</p>"

    try:
        # Topics 2D chart
        fig_topics = topic_model.visualize_topics()
        topics_html = fig_topics.to_html(full_html=False)
    except:
        topics_html = "<p>Data visualisasi topik 2D tidak dapat dibuat</p>"

    # Create word frequency chart for top topics
    top_topics = topics_summary[:5]  # Top 5 topics
    barchart_data = []
    for topic in top_topics:
        for word in topic['keywords']:
            barchart_data.append({
                'topic': f'Topik {topic["topic_id"]}',
                'word': word,
                'weight': 1  # Simplified weight
            })

    if barchart_data:
        barchart_df = pd.DataFrame(barchart_data)
        fig_barchart = px.bar(
            barchart_df,
            x='word',
            y='weight',
            color='topic',
            title='Kata-Kata Utama per Topik',
            barmode='group'
        )
        barchart_html = fig_barchart.to_html(full_html=False)
    else:
        barchart_html = "<p>Data visualisasi tidak dapat dibuat</p>"

    return {
        'topics_summary': topics_summary,
        'coherence_score': model_data.get('coherence_score', 0.0),
        'total_topics': model_data.get('total_topics', 0),
        'model_type': 'BERTopic',
        'visualizations': {
            'barchart': barchart_html,
            'hierarchy': hierarchy_html,
            'topics': topics_html,
            'distribution': distribution_html
        }
    }
