# services/bertopic_service.py
import os
import time
import joblib
import traceback
from multiprocessing import Process
from typing import Optional, Dict, Any

import pandas as pd
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

MODEL_PATH = "models/bertopic_model.pkl"
LOCK_PATH = "models/bertopic_building.lock"
LOG_PATH = "models/bertopic_build.log"


def _log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def load_bertopic_model(model_path: str = MODEL_PATH) -> Optional[Dict[str, Any]]:
    """Load saved model data (joblib)."""
    try:
        if os.path.exists(model_path):
            return joblib.load(model_path)
    except Exception as e:
        _log(f"load_bertopic_model error: {e}\n{traceback.format_exc()}")
    return None


def is_building() -> bool:
    """Return True if background build is running."""
    return os.path.exists(LOCK_PATH)


def start_build_bertopic_async(filepath: str, max_samples: int = 200) -> Dict[str, Any]:
    """Start BERTopic build in background process."""
    if not filepath or not os.path.exists(filepath):
        return {"error": "Filepath tidak ditemukan atau tidak diberikan."}

    if is_building():
        return {"status": "building", "message": "Proses build sedang berjalan."}

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    open(LOCK_PATH, "w").close()

    p = Process(target=_build_worker, args=(filepath, max_samples), daemon=True)
    p.start()
    _log(f"Background build started for {filepath} (PID {p.pid})")
    return {"status": "started", "message": "Proses build dimulai di background."}


def _calculate_coherence(topics_summary, text_data):
    """Helper untuk hitung coherence score secara aman."""
    try:
        text_tokenized = [t.lower().split() for t in text_data if isinstance(t, str) and len(t) > 0]
        dictionary = corpora.Dictionary(text_tokenized)

        topic_words = [t["keywords"][:10] for t in topics_summary if len(t["keywords"]) > 0]
        topic_words = [tw for tw in topic_words if any(isinstance(w, str) and len(w) > 0 for w in tw)]

        if len(topic_words) == 0 or len(dictionary) == 0:
            return 0.0

        coherence_model = CoherenceModel(
            topics=topic_words, texts=text_tokenized, dictionary=dictionary, coherence="c_v"
        )
        coherence_score = round(coherence_model.get_coherence(), 4)
        _log(f"Coherence berhasil dihitung: {coherence_score}")
        return coherence_score
    except Exception as e:
        _log(f"Coherence calculation failed: {e}")
        return 0.0


def _build_worker(filepath: str, max_samples: int = 200):
    """Worker function executed in separate process to build and save model."""
    try:
        _log(f"Worker started for {filepath}")
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            have_bertopic = True
            _log("BERTopic & SentenceTransformer tersedia.")
        except Exception as e:
            have_bertopic = False
            _log(f"BERTopic import failed: {e}; fallback ke LDA.")

        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        import nltk
        from nltk.corpus import stopwords
        try:
            stopwords.words("indonesian")
        except Exception:
            try:
                nltk.download("stopwords")
            except Exception:
                pass

        # Preprocessing
        df = pd.read_csv(filepath, encoding="utf-8-sig")
        from services.preprocessing import get_preprocessing_steps
        preprocessing_results = get_preprocessing_steps(df)
        text_data = [
            item["text_clean"]
            for item in preprocessing_results["hasil_preprocessing"]
            if item.get("text_clean")
        ]

        if not text_data:
            _log("Tidak ada teks valid setelah preprocessing. Aborting.")
            return

        if len(text_data) > max_samples:
            text_data = text_data[:max_samples]
            _log(f"Data disampling ke {max_samples} items untuk performa.")

        # === BERTopic ===
        if have_bertopic:
            try:
                os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                topic_model = BERTopic(
                    embedding_model=embedding_model,
                    low_memory=True,
                    nr_topics="auto",
                    calculate_probabilities=False,
                )
                topics, probs = topic_model.fit_transform(text_data)
                _log("BERTopic training selesai.")

                topics_summary = []
                for t in sorted(set(topics)):
                    if t == -1:
                        continue
                    topic_info = topic_model.get_topic(t)
                    top_words = [w for w, _ in topic_info[:5]]
                    count = int((topics == t).sum())
                    topics_summary.append(
                        {"topic_id": int(t), "keywords": top_words, "count": count}
                    )

                coherence_score = _calculate_coherence(topics_summary, text_data)

                model_data = {
                    "model_type": "BERTopic",
                    "topic_model": topic_model,
                    "topics_summary": topics_summary,
                    "coherence_score": coherence_score,
                    "total_topics": len(topics_summary),
                }

                joblib.dump(model_data, MODEL_PATH)
                _log(f"BERTopic model saved to {MODEL_PATH}")
                return
            except Exception as e:
                _log(f"BERTopic build error: {e}\n{traceback.format_exc()}")

        # === Fallback: LDA ===
        _log("Memulai fallback LDA.")
        try:
            ind_stop = stopwords.words("indonesian")
        except Exception:
            ind_stop = []

        vectorizer = CountVectorizer(
            max_df=0.9, min_df=5, stop_words=ind_stop, ngram_range=(1, 2)
        )
        count_matrix = vectorizer.fit_transform(text_data)

        n_topics = min(15, max(5, len(text_data) // 50))
        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42, max_iter=10, learning_method="online"
        )
        lda_output = lda.fit_transform(count_matrix)
        feature_names = vectorizer.get_feature_names_out()

        topics_summary = []
        for topic_idx, topic in enumerate(lda.components_):
            top_idx = topic.argsort()[:-6:-1]
            top_words = [feature_names[i] for i in top_idx]
            topic_prob = lda_output[:, topic_idx].mean()
            count_est = int(topic_prob * len(text_data))
            topics_summary.append(
                {"topic_id": topic_idx, "keywords": top_words, "count": count_est}
            )

        topics_summary.sort(key=lambda x: x["count"], reverse=True)
        coherence_score = _calculate_coherence(topics_summary, text_data)

        model_data = {
            "model_type": "LDA",
            "lda_model": lda,
            "count_vectorizer": vectorizer,
            "topics_summary": topics_summary,
            "coherence_score": coherence_score,
            "total_topics": len(topics_summary),
        }

        joblib.dump(model_data, MODEL_PATH)
        _log(f"LDA model saved to {MODEL_PATH}")

    except Exception as e:
        _log(f"Unhandled exception in _build_worker: {e}\n{traceback.format_exc()}")
    finally:
        try:
            if os.path.exists(LOCK_PATH):
                os.remove(LOCK_PATH)
                _log("Lock file removed.")
        except Exception as e:
            _log(f"Failed to remove lock file: {e}")


def get_bertopic_analysis() -> Dict[str, Any]:
    """Load saved model and prepare visualization info."""
    model_data = load_bertopic_model()
    if model_data is None:
        return {"error": "Model topik tidak ditemukan. Pastikan model sudah dibangun."}

    model_type = model_data.get("model_type", "unknown")
    topics_summary = model_data.get("topics_summary", [])
    coherence_score = float(model_data.get("coherence_score", 0.0))
    total_topics = int(model_data.get("total_topics", 0))

    visualizations = {
        "distribution": "<p>Visualisasi distribution tidak tersedia</p>",
        "barchart": "<p>Visualisasi barchart tidak tersedia</p>",
        "hierarchy": "<p>Visualisasi hierarchy tidak tersedia</p>",
        "topics": "<p>Visualisasi topik 2D tidak tersedia</p>",
    }

    try:
        if model_type == "BERTopic" and "topic_model" in model_data:
            topic_model = model_data["topic_model"]
            try:
                fig_dist = topic_model.visualize_barchart(top_n_topics=10)
                visualizations["distribution"] = fig_dist.to_html(full_html=False)
            except Exception:
                visualizations["distribution"] = "<p>Distribusi tidak dapat dibuat</p>"

            try:
                fig_h = topic_model.visualize_hierarchy()
                visualizations["hierarchy"] = fig_h.to_html(full_html=False)
            except Exception:
                visualizations["hierarchy"] = "<p>Hierarchy tidak dapat dibuat</p>"

            try:
                fig_t = topic_model.visualize_topics()
                visualizations["topics"] = fig_t.to_html(full_html=False)
            except Exception:
                visualizations["topics"] = "<p>Topik 2D tidak dapat dibuat</p>"
    except Exception as e:
        _log(f"Visualization (BERTopic) failed: {e}")

    # Simple bar chart
    try:
        import plotly.express as px

        top_topics = topics_summary[:5]
        barchart_data = []
        for t in top_topics:
            for w in t["keywords"]:
                barchart_data.append(
                    {"topic": f"Topik {t['topic_id']}", "word": w, "weight": 1}
                )
        if barchart_data:
            df = pd.DataFrame(barchart_data)
            fig = px.bar(
                df,
                x="word",
                y="weight",
                color="topic",
                barmode="group",
                title="Kata Utama per Topik",
            )
            visualizations["barchart"] = fig.to_html(full_html=False)
    except Exception:
        pass

    return {
        "topics_summary": topics_summary,
        "coherence_score": coherence_score,
        "total_topics": total_topics,
        "model_type": model_type,
        "visualizations": visualizations,
    }
