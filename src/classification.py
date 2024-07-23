import numpy as np
from dataclasses import dataclass
from typing import Any, Dict
from averitec import Datapoint
from evidence_generation import EvidenceGenerationResult
from retrieval import RetrievalResult
from labels import label2id, id2label

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
                return ClassificationResult(probs=np.array(suggested))
            if isinstance(suggested, ClassificationResult):
                return suggested
        return None
