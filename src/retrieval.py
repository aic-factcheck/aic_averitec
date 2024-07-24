from dataclasses import dataclass, field
from typing import Any, List, Dict
from averitec import Datapoint
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from utils.chat import SimpleJSONChat


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


class MmrFaissRetriever(Retriever):
    def __init__(
        self, path: str, embeddings: Embeddings = None, k: int = 10, fetch_k: int = 40, lambda_mult=0.7
    ):
        self.path = path
        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
        self.embeddings = embeddings
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult

    def __call__(self, datapoint: Datapoint, *args, **kwargs) -> RetrievalResult:
        vecstore = FAISS.load_local(
            f"{self.path}/{datapoint.claim_id}",
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        documents = vecstore.max_marginal_relevance_search(
            datapoint.claim, k=self.k, fetch_k=self.fetch_k, lambda_mult=self.lambda_mult
        )
        return RetrievalResult(documents=documents)


class SubqueryRetriever(Retriever):
    def __init__(self, retriever: Retriever, k=10, fetch_k=50, subqueries=5, lambda_mult=0.5, model="gpt-4o"):
        self.retriever = retriever
        self.k = k
        self.fetch_k = fetch_k
        self.subqueries = subqueries
        self.lambda_mult = lambda_mult
        self.client = SimpleJSONChat(
            model=model,
            system_prompt=f"""You are a professional researcher who receives a factual claim and its metadata (speaker, date) and your goal is to output a set of pertinent Google/Bing search queries that could be used to find relevant sources for proving or debunking such claim. You may also use the metadata if they can be used to disambiguate claim and facilitate fact-checking. Ideally, each query would focus on one aspect of the claim, independent of others. You may produce up to 5 search queries which should cover all relevant aspects of the claim and lead to the most successful source search, take your time and be thorough.\nPlease, you MUST output only the best search queries in the following JSON format:\n```json\n[\n    "<query 1>",\n    "<query 2>",\n    "<query 3>",\n    "<query 4>",\n    "<query 5>"\n]\n```""",
        )

    def get_subqueries(self, datapoint):
        return self.client(f"{datapoint.claim} ({datapoint.speaker}, {datapoint.claim_date})") + [
            datapoint.claim
        ]

    def __call__(self, datapoint):
        original_claim = datapoint.claim
        queries = self.get_subqueries(datapoint)
        documents = []
        for subquery in queries:
            datapoint.claim = subquery

            for document in self.retriever(datapoint):
                if "queries" not in document.metadata:
                    document.metadata["queries"] = []
                document.metadata["queries"].append(subquery)
                documents.append(document)

            datapoint.claim = original_claim
            retriever = FAISS.from_documents(documents, embedding=self.retriever.embeddings)
            results = retriever.max_marginal_relevance_search(
                datapoint.claim, k=self.k, fetch_k=self.fetch_k, lambda_mult=self.lambda_mult
            )
        return RetrievalResult(results, metadata={"queries": queries})
