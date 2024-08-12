from dataclasses import dataclass
from typing import Dict, List, Union
from classification import ClassificationResult, Classifier
from evidence_generation import EvidenceGenerationResult, EvidenceGenerator
from retrieval import RetrievalResult, Retriever
from averitec import Datapoint
import pickle


@dataclass
class PipelineResult:
    datapoint: Datapoint = None
    evidence_generation_result: EvidenceGenerationResult = None
    retrieval_result: RetrievalResult = None
    classification_result: ClassificationResult = None

    def to_submission(self):
        return {
            "claim_id": self.datapoint.claim_id,
            "claim": self.datapoint.claim,
            "evidence": [e.to_dict() for e in self.evidence_generation_result if True or e.url is not None],
            "pred_label": str(self.classification_result),
        }


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


class MockPipeline(Pipeline):
    """Skips the pipeline steps not specified in the constructor and mocks their results using the provided PipelineResult data (as path or list)."""

    def __init__(
        self,
        dumps: Union[str, List[PipelineResult], Dict[int, PipelineResult]],
        retriever: Retriever = None,
        evidence_generator: EvidenceGenerator = None,
        classifier: Classifier = None,
    ):
        if isinstance(dumps, str):
            with open(dumps, "rb") as f:
                dumps = pickle.load(f)
        if isinstance(dumps, list):
            dumps = {dp.datapoint.claim_id: dp for dp in dumps}
        self.dumps = dumps

        self.retriever, self.evidence_generator, self.classifier = retriever, evidence_generator, classifier

    def __call__(self, datapoint, *args, **kwargs) -> PipelineResult:
        dump = self.dumps[datapoint.claim_id]

        retrieval_result = (
            self.retriever(datapoint, *args, **kwargs) if self.retriever else dump.retrieval_result
        )

        evidence_generation_result = (
            self.evidence_generator(datapoint, retrieval_result, *args, **kwargs)
            if self.evidence_generator
            else dump.evidence_generation_result
        )

        classification_result = (
            self.classifier(datapoint, evidence_generation_result, retrieval_result, *args, **kwargs)
            if self.classifier
            else dump.classification_result
        )

        return PipelineResult(
            datapoint=datapoint,
            retrieval_result=retrieval_result,
            evidence_generation_result=evidence_generation_result,
            classification_result=classification_result,
        )
