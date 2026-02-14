import os
from fastembed import TextEmbedding

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def get_embedding(text: str):
    if not text.strip():
        return []
    embedding = list(model.embed([text]))[0]
    return embedding.tolist()