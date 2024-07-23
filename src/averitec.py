from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple, Union
from utils.chat import SimpleJSONChat
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from anthropic import AnthropicVertex

claim = str


@dataclass
class RetrievalResult:
    documents: List[Document] = field(default_factory=list)
    metadata: Any = None

    def __iter__(self):
        return iter(self.documents)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]


@dataclass
class Evidence:
    question: str = None
    answer: str
    url: str


@dataclass
class EvidenceGenerationResult:
    evidences: List[Evidence] = field(default_factory=list)
    proposed_label: str = None
    metadata: Any = None

    def __iter__(self):
        return iter(self.evidences)

    def __len__(self):
        return len(self.evidences)

    def __getitem__(self, index):
        return self.evidences[index]


@dataclass
class ClassificationResult:
    label: str
    metadata: Any = None


@dataclass
class Datapoint:
    claim: str
    claim_id: int
    claim_date: str = None
    speaker: str = None
    original_claim_url: str = None
    reporting_source: str = None
    location_ISO_code: str = None
    label: str = None
    split: str = "dev"
    metadata: dict = None

    @classmethod
    def from_dict(cls, json_data: dict, claim_id: int = None):
        json_data = json_data.copy()
        return cls(
            claim=json_data.pop("claim"),
            claim_id=json_data.pop("claim_id", claim_id),
            claim_date=json_data.pop("claim_date", None),
            speaker=json_data.pop("speaker", None),
            original_claim_url=json_data.pop("original_claim_url", None),
            reporting_source=json_data.pop("reporting_source", None),
            location_ISO_code=json_data.pop("location_ISO_code", None),
            label=json_data.pop("label", None),
            split=json_data.pop("split", "dev"),
            metadata=json_data,
        )

    def to_dict(self):
        return {
            "claim": self.claim,
            "claim_id": self.claim_id,
            "claim_date": self.claim_date,
            "speaker": self.speaker,
            "original_claim_url": self.original_claim_url,
            "reporting_source": self.reporting_source,
            "location_ISO_code": self.location_ISO_code,
            "label": self.label,
            "split": self.split,
            **self.metadata,
        }


@dataclass
class PipelineResult:
    datapoint: Datapoint = None
    evidence_generation_result: EvidenceGenerationResult = None
    retrieval_result: RetrievalResult = None
    classification_result: ClassificationResult = None


class Retriever:
    def __call__(self, datapoint: Datapoint, *args, **kwargs) -> RetrievalResult:
        raise NotImplementedError


class SimpleFaissRetriever(Retriever):
    def __init__(self, path: str, embeddings: Embeddings):
        self.path = path
        self.embeddings = embeddings
        self.k = 10

    def __call__(self, datapoint: Datapoint, *args, **kwargs) -> RetrievalResult:
        vecstore = FAISS.load_local(
            f"{self.path}/{datapoint.claim_id}",
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        documents = vecstore.similarity_search(datapoint.claim, self.k)
        return RetrievalResult(documents=documents)


class EvidenceGenerator:
    def __call__(
        self, datapoint: Datapoint, retrieval_result: RetrievalResult, *args, **kwargs
    ) -> EvidenceGenerationResult:
        raise NotImplementedError


class GptEvidenceGenerator(EvidenceGenerator):
    def __init__(self, client: SimpleJSONChat = None, model="gpt-4o"):
        if client is None:
            self.client = SimpleJSONChat(model=model)

    def __call__(
        self, datapoint: Datapoint, retrieval_results: List[RetrievalResult], *args, **kwargs
    ) -> EvidenceGenerationResult:
        return EvidenceGenerationResult(
            evidences=[Evidence(answer="I don't know", url="")],
            metadata={
                "suggested_label": "Supported",
                "label_confidences": {
                    "Supported": 0.5,
                    "Refuted": 0.5,
                    "Not Enough Info": 0.0,
                    "Cherrypicking": 0.0,
                },
            },
        )


class Classifier:
    def __call__(
        self,
        datapoint: Datapoint,
        evidence_generation_result: EvidenceGenerationResult,
        retrieval_result: RetrievalResult,
        *args,
        **kwargs,
    ) -> ClassificationResult:
        raise NotImplementedError


class DefaultClassifier(Classifier):
    def __call__(
        self,
        datapoint: Datapoint,
        evidence_generation_result: EvidenceGenerationResult,
        retrieval_result: RetrievalResult,
        *args,
        **kwargs,
    ) -> ClassificationResult:
        if evidence_generation_result.metadata and "suggested_label" in evidence_generation_result.metadata:
            return ClassificationResult(label=evidence_generation_result.proposed_label)
        elif (
            evidence_generation_result.metadata and "label_confidences" in evidence_generation_result.metadata
        ):
            return ClassificationResult(
                label=max(
                    evidence_generation_result.metadata["label_confidences"],
                    key=evidence_generation_result.metadata["label_confidences"].get,
                )
            )
        return ClassificationResult(label="Not Enough Info")


class Pipeline:
    retriever: Retriever = None
    evidence_generator: EvidenceGenerator = None
    classifier: Classifier = None

    def __init__(
        self,
        retriever: Retriever = None,
        evidence_generator: EvidenceGenerator = None,
        classifier: Classifier = None,
    ):
        self.retriever = retriever
        self.evidence_generator = evidence_generator
        if classifier is None:
            classifier = DefaultClassifier()
        self.classifier = classifier

    def __call__(self, datapoint, *args, **kwargs) -> PipelineResult:
        retrieval_result = self.retriever(datapoint, *args, **kwargs)
        evidence_generation_result = self.evidence_generator(datapoint, retrieval_result, *args, **kwargs)
        classification_result = self.classifier(
            datapoint, evidence_generation_result, retrieval_result, *args, **kwargs
        )

        return PipelineResult(
            datapoint=datapoint,
            evidence_generation_result=evidence_generation_result,
            retrieval_result=retrieval_result,
            classification_result=classification_result,
        )
