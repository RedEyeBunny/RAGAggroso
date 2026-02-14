from sklearn.feature_extraction.text import TfidfVectorizer

# Global vectorizer
vectorizer = TfidfVectorizer()

# You must fit once on your corpus
def fit_vectorizer(documents):
    vectorizer.fit(documents)

def get_embedding(text: str):
    vector = vectorizer.transform([text])
    return vector.toarray()[0].tolist()
