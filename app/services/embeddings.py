from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
is_fitted = False

def fit_vectorizer(documents):
    global is_fitted
    vectorizer.fit(documents)
    is_fitted = True

def get_embedding(text: str):
    if not is_fitted:
        raise ValueError("Vectorizer not fitted yet.")

    vector = vectorizer.transform([text])
    return vector.toarray()[0].tolist()
