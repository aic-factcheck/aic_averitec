import numpy as np
import json
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple, Union, Dict
from utils.chat import SimpleJSONChat
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from anthropic import AnthropicVertex
from scipy.special import softmax

label2id = {"Refuted": 0, "Supported": 1, "Not Enough Evidence": 2, "Conflicting Evidence/Cherrypicking": 3}
id2label = {v: k for k, v in label2id.items()}


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


@dataclass
class Evidence:
    question: str = None
    answer: str
    url: str
    scraped_text: str = None


@dataclass
class EvidenceGenerationResult:
    evidences: List[Evidence] = field(default_factory=list)
    metadata: Dict[str, Any] = None

    def __iter__(self):
        return iter(self.evidences)

    def __len__(self):
        return len(self.evidences)

    def __getitem__(self, index):
        return self.evidences[index]


@dataclass
class ClassificationResult:
    probs: np.ndarray[float] = None
    metadata: Dict[str, Any] = None

    def to_dict(self):
        probs_dict = {id2label[i]: prob for i, prob in enumerate(self.probs)}
        result = {"probs": probs_dict}
        if self.metadata is not None:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict):
        probs = np.zeros(len(label2id))
        for label, prob in data["probs"].items():
            probs[label2id[label]] = prob
        return cls(probs=probs, metadata=data.get("metadata", None))

    def __str__(self) -> str:
        return id2label[np.argmax(self.probs)]


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
    @classmethod
    def parse_label(cls, label: str) -> str:
        if "sup" in label.lower():
            return "Supported"
        elif "ref" in label.lower():
            return "Refuted"
        elif "conf" in label.lower() or "cherr" in label.lower():
            return "Conflicting Evidence/Cherrypicking"
        elif "not" in label.lower():
            return "Not Enough Evidence"
        return "Refuted"

    @classmethod
    def parse_likert(cls, likert_string: str) -> float:
        if "1" in likert_string or ("strong" in likert_string and "disagree" in likert_string):
            return -2
        if "5" in likert_string or ("strong" in likert_string and "agree" in likert_string):
            return 2
        if "2" in likert_string or ("disagree" in likert_string):
            return -1
        if "3" in likert_string or "neutral" in likert_string:
            return 0
        if "4" in likert_string or ("agree" in likert_string):
            return 1
        return 0

    @classmethod
    def parse_label_probabilities(cls, data: dict) -> np.ndarray:
        result = np.zeros(4)
        for label, likert in data.items():
            result[cls.parse_label(label2id[label])] = cls.parse_likert(likert)
        return softmax(result)

    @classmethod
    def parse_json(cls, message):
        try:
            result = message
            # trim message before first ```
            if "```json" in message:
                message = message.split("```json")[1]
            if "```" in message:
                message = message.split("```")[0]
            result = message.replace("```json", "").replace("```", "")
            return json.loads(result)
        except:
            print("Error parsing JSON for EvidenceGenerator.\n", message)
            return []

    @classmethod
    def parse_evidence(cls, input_data, retrieval_result) -> List[Evidence]:
        result = []
        for e in input_data:
            evidence = Evidence(question=e.get("question", None), answer=e.get("answer", None))
            try:
                id = int(str(e["source"]).split(",")[0]) - 1
                evidence.url = retrieval_result[id].metadata["url"]
                evidence.scraped_text = "\n".join(
                    [
                        retrieval_result[id].metadata["context_before"],
                        retrieval_result[id].page_content,
                        retrieval_result[id].metadata["context_after"],
                    ]
                )
            except:
                evidence.url = None
                evidence.scraped_text = None
            result.append(evidence)
        return result

    def __call__(
        self, datapoint: Datapoint, retrieval_result: RetrievalResult, *args, **kwargs
    ) -> EvidenceGenerationResult:
        raise NotImplementedError


class GptEvidenceGenerator(EvidenceGenerator):
    def __init__(self, client: SimpleJSONChat = None, model="gpt-4o"):
        if client is None:
            self.client = SimpleJSONChat(model=model, parse_output=False)

    def format_system_prompt(self, retrieval_result: RetrievalResult) -> str:
        result = "You are a professional fact checker, formulate up to 10 questions that cover all the facts needed to validate whether the factual statement (in User message) is true, false, uncertain or a matter of opinion.\nAfter formulating Your questions and their answers using the provided sources, You evaluate the possible veracity verdicts (Supported claim, Refuted claim, Not enough evidence, or Conflicting evidence/Cherrypicking) given your claim and evidence on a Likert scale (1 - Strongly disagree, 2 - Disagree, 3 - Neutral, 4 - Agree, 5 - Strongly agree).\nThe facts must be coming from these sources, please refer them using assigned IDs:"
        for i, e in enumerate(retrieval_result):
            result += f"\n---\n## Source ID: {i+1} ({e.metadata['url']})\n"
            result += "\n".join([e.metadata["context_before"], e.page_content, e.metadata["context_after"]])
        result += """\n---\n## Output formatting\nPlease, you MUST only print the output in the following output format:
```json
{
    "questions":
        [
            {"question": "<Your first question>", "answer": "<The answer to the Your first question>", "source": "<Single numeric source ID backing the answer for Your first question>"},
            {"question": "<Your second question>", "answer": "<The answer to the Your second question>", "source": "<Single numeric Source ID backing the answer for Your second question>"}
        ],
    "claim_veracity": {
        "Supported": "<Likert-scale rating of how much You agree with the 'Supported' veracity classification>",
        "Refuted": "<Likert-scale rating of how much You agree with the 'Refuted' veracity classification>",
        "Not Enough Evidence": "<Likert-scale rating of how much You agree with the 'Not Enough Evidence' label>",
        "Conflicting Evidence/Cherrypicking": "<Likert-scale rating of how much You agree with the 'Conflicting Evidence/Cherrypicking' label>"
    }
}
```"""

    def __call__(
        self, datapoint: Datapoint, retrieval_result: RetrievalResult, *args, **kwargs
    ) -> EvidenceGenerationResult:
        gpt_result = self.client(
            system_prompt=self.format_system_prompt(retrieval_result), user_prompts=[datapoint.claim]
        )
        gpt_data = self.parse_json(gpt_result)
        return EvidenceGenerationResult(
            evidences=self.parse_evidence(gpt_data["questions"], retrieval_result),
            metadata={
                "suggested_label": self.parse_label_probabilities(gpt_data["claim_veracity"]),
                "llm_type": self.client.model,
                "llm_output": gpt_data,
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
    """Passes on the label suggested by evidence generator"""

    def __call__(
        self,
        datapoint: Datapoint,
        evidence_generation_result: EvidenceGenerationResult,
        retrieval_result: RetrievalResult,
        *args,
        **kwargs,
    ) -> ClassificationResult:
        if evidence_generation_result.metadata and "suggested_label" in evidence_generation_result.metadata:
            suggested = evidence_generation_result.metadata["suggested_label"]
            if isinstance(suggested, str):
                return ClassificationResult.from_dict({"probs": {suggested: 1.0}})
            if isinstance(suggested, dict):
                return ClassificationResult.from_dict({"probs": suggested})
            if isinstance(suggested, np.ndarray) or isinstance(suggested, list):
                return ClassificationResult(probs=suggested)
            if isinstance(suggested, ClassificationResult):
                return suggested
        return None


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
            retrieval_result=retrieval_result,
            evidence_generation_result=evidence_generation_result,
            classification_result=classification_result,
        )
