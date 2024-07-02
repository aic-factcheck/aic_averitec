## reranking of higher number of sentences retrieved
from re import S
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings
from typing import List

def rerank_sentences(claim: str, sentences: List[str], dimensions: int = 512):
    #load model
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions)

    # For retrieval you need to pass this prompt.
    query = f'Represent this sentence for searching relevant passages: {claim}'

    docs = [query] + sentences

    # 2. Encode
    embeddings = model.encode(docs)

    similarities = cos_sim(embeddings[0], embeddings[1:]).flatten()

    #get indices of sorted similarities
    sorted_indices = similarities.argsort(descending=True)

    return similarities, sorted_indices


