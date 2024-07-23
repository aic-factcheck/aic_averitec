from dataclasses import dataclass, field
from typing import Any, List, Dict
from averitec import Datapoint
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


@dataclass
class RetrievalResult:
    documents: List[Document] = field(default_factory=list)
    metadata: Dict[str, Any] = None

    def __iter__(self):
        return iter(self.documents)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]


class Retriever:
    def __call__(self, datapoint: Datapoint, *args, **kwargs) -> RetrievalResult:
        raise NotImplementedError


class SimpleFaissRetriever(Retriever):
    def __init__(self, path: str, embeddings: Embeddings = None, k: int = 10):
        self.path = path
        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
        self.embeddings = embeddings
        self.k = k

    def __call__(self, datapoint: Datapoint, *args, **kwargs) -> RetrievalResult:
        vecstore = FAISS.load_local(
            f"{self.path}/{datapoint.claim_id}",
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        documents = vecstore.similarity_search(datapoint.claim, k=self.k)
        return RetrievalResult(documents=documents)
