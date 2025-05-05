from sentence_transformers import SentenceTransformer

def load_kobert_model():
    return SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

def encode_books(model, book_texts):
    return model.encode(book_texts, show_progress_bar=True)