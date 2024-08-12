import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from averitec import Datapoint
from evidence_generation import EvidenceGenerationResult
from retrieval import RetrievalResult
from labels import label2id, id2label
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import scipy.optimize as opt
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import json
import pickle

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
    def __init__(self, evidence_generation_results=None) -> None:
        super().__init__()
        self.evidence_generation_results = evidence_generation_results 

    def __call__(
        self,
        datapoint: Datapoint,
        evidence_generation_result: EvidenceGenerationResult,
        retrieval_result: RetrievalResult,
        *args,
        **kwargs,
    ) -> ClassificationResult:
        #if own evidence generation results are provided, use them
        if self.evidence_generation_results is not None:
            evidence_generation_result = self.evidence_generation_results[datapoint.claim_id]

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

class NoTiebreakClassifier(DefaultClassifier):
    """Passes on the label suggested by evidence generator without tiebreak"""
    def __call__(
        self,
        datapoint: Datapoint,
        evidence_generation_result: EvidenceGenerationResult,
        retrieval_result: RetrievalResult,
        *args,
        **kwargs,
    ) -> ClassificationResult:
        #if own evidence generation results are provided, use them
        if self.evidence_generation_results is not None:
            evidence_generation_result = self.evidence_generation_results[datapoint.claim_id]

        if evidence_generation_result.metadata and "label_confidences" in evidence_generation_result.metadata:
            suggested = evidence_generation_result.metadata["label_confidences"]
            if isinstance(suggested, str):
                return ClassificationResult.from_dict({"probs": {suggested: 1.0}})
            if isinstance(suggested, dict):
                return ClassificationResult.from_dict({"probs": suggested})
            if isinstance(suggested, np.ndarray) or isinstance(suggested, list):
                return ClassificationResult(probs=np.array(suggested))
            if isinstance(suggested, ClassificationResult):
                return suggested
        return super().__call__(datapoint, evidence_generation_result, retrieval_result, *args, **kwargs)

class HuggingfaceClassifier(Classifier):
    """Uses a Huggingface text classification model to classify the datapoint"""
    def __init__(self, model_path:str, device:Optional[str]=None, max_length:int=1024, rand_order_evidence:bool=False, num_orders:int = 10, seed:int=42) -> None:
        super().__init__()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        #maximum length of input sequence
        self.max_length = max_length

        #if the evidence should be permuted
        self.rand_order_evidence = rand_order_evidence
        self.num_orders =num_orders

        self.seed = seed

        #load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        #put model into evaluation mode
        self.model.eval()

    def __call__(
        self,
        datapoint: Datapoint,
        evidence_generation_result: EvidenceGenerationResult,
        retrieval_result: RetrievalResult,
        *args,
        **kwargs,
    ) -> ClassificationResult:
        claim = datapoint.claim

        #concatenate all evidence into one string
        qas = []
        for e in evidence_generation_result.evidences:
            qas.append(e.question + " " + e.answer)

        evidences = []
        if self.rand_order_evidence:
            #set seed for reproducibility
            np.random.seed(self.seed)
            #all permutations are not feasible -> randomly select
            for _ in range(min(self.num_orders, np.math.factorial(len(qas)))):
                order = np.random.choice(len(qas), size=len(qas), replace=False) #one order

                #create evidence string
                evidence = " ".join([qas[j] for j in order])

                evidences.append(evidence)

            claims = [claim] * len(evidences)
 
        else:
            evidences = [" ".join(qas)]
            claims = [claim]

        #tokenize claim and evidence (without last whitespace)
        inputs = self.tokenizer(claims, evidences, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()} #move inputs to device

        #get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        #convert logits to probabilities
        mean_logits = torch.mean(logits, dim=0)
        probs = torch.softmax(mean_logits, dim=-1).cpu().numpy().squeeze()

        return ClassificationResult(probs=probs, metadata={"logits": logits.cpu().numpy(), "mean_logits": mean_logits.cpu().numpy()})

class RandomForestClassifier(Classifier):
    def __init__(self) -> None:
        super().__init__()
        # Load the main model and preprocessing objects
        with open('/mnt/data/factcheck/averitec-data/bryce_data/all_data_model.pkl', 'rb') as model_file:
            self.main_rf = pickle.load(model_file)

        with open('/mnt/data/factcheck/averitec-data/bryce_data/all_data_tfidf_vectorizer.pkl', 'rb') as tfidf_file:
            self.main_vectorizer = pickle.load(tfidf_file)

        with open('/mnt/data/factcheck/averitec-data/bryce_data/all_data_label_map.pkl', 'rb') as label_map_file:
            self.main_label_map = pickle.load(label_map_file)

        # Load the secondary model and preprocessing objects
        with open('/mnt/data/factcheck/averitec-data/bryce_data/nee_cp_model.pkl', 'rb') as secondary_model_file:
            self.secondary_rf = pickle.load(secondary_model_file)

        with open('/mnt/data/factcheck/averitec-data/bryce_data/nee_cptfidf_vectorizer.pkl', 'rb') as secondary_tfidf_file:
            self.secondary_vectorizer = pickle.load(secondary_tfidf_file)

        with open('/mnt/data/factcheck/averitec-data/bryce_data/nee_cplabel_map.pkl', 'rb') as secondary_label_map_file:
            self.secondary_label_map = pickle.load(secondary_label_map_file)

        # Inverse label maps
        self.main_label_map_inverse = {v: k for k, v in self.main_label_map.items()}
        self.secondary_label_map_inverse = {v: k for k, v in self.secondary_label_map.items()}

    # Preprocess claims using the respective vectorizers
    def preprocess_claim(self, claim, vectorizer):
        return vectorizer.transform([claim]).toarray()
    
    # First step: Classify claims using the main model
    def classify_main(self, claim):
        features = self.preprocess_claim(claim, self.main_vectorizer)
        prediction = self.main_rf.predict(features)
        probability_distribution = self.main_rf.predict_proba(features)
        confidence = np.max(probability_distribution)
        return prediction[0], confidence, probability_distribution[0]
    
    # Second step: Classify claims using the secondary model
    def classify_secondary(self, claim):
        features = self.preprocess_claim(claim, self.secondary_vectorizer)
        prediction = self.secondary_rf.predict(features)
        probability_distribution = self.secondary_rf.predict_proba(features)
        return prediction[0], probability_distribution[0]

    # Two-step classification process for a single claim
    def predict_single_claim(self, claim):
        predicted_label, confidence, probability_distribution = self.classify_main(claim)

        if self.main_label_map_inverse[predicted_label] in ['Supported', 'Refuted'] and confidence >= 0.65:
            # High confidence in main model's prediction
            predicted_label_name = self.main_label_map_inverse[predicted_label]
            class_probabilities = {self.main_label_map_inverse[i]: prob for i, prob in enumerate(probability_distribution)}
        else:
            # Use secondary model for further classification
            predicted_label, probability_distribution = self.classify_secondary(claim)
            predicted_label_name = self.secondary_label_map_inverse[predicted_label]
            class_probabilities = {self.secondary_label_map_inverse[i]: prob for i, prob in enumerate(probability_distribution)}

        return predicted_label_name, class_probabilities
    
    def __call__(self, datapoint: Datapoint, evidence_generation_result: EvidenceGenerationResult, retrieval_result: RetrievalResult, *args, **kwargs) -> ClassificationResult:
        claim = datapoint.claim
        predicted_label, class_probabilities = self.predict_single_claim(claim)
        #remap class probabilities to np array
        probs = np.zeros(len(label2id))
        for label, prob in class_probabilities.items():
            probs[label2id[label]] = prob

        return ClassificationResult(probs=probs, metadata={"predicted_label": predicted_label})

class AverageEnsembleClassifier(Classifier):
    """Ensemble classifier that averages the predictions of multiple classifiers, when `weights` are provided, the predictions are weighted accordingly"""
    def __init__(self, classifiers: List[Classifier], weights: np.ndarray=None):
        self.classifiers = classifiers

        assert(len(classifiers) == len(weights) if weights is not None else True) #check if number of classifiers and weights match

        #when no weigths are provided, all classifiers are weighted equally
        if weights is None:
            self.weights = np.ones(len(classifiers))
        else:
            self.weights = weights

    def __call__(self, datapoint, evidence_generation_result, retrieval_result):
        clf_probs = [c(datapoint, evidence_generation_result, retrieval_result).probs for c in self.classifiers]
        return ClassificationResult(
            probs=np.average(clf_probs, axis=0, weights=self.weights),
            metadata={"clf_probs": clf_probs, "weights": self.weights}
        )
    

    def fit_weights(self, datapoints, evidence_generation_results, retrieval_results, metric:str="cross-entropy"):
        """fit weights using a validation set"""
        labels = [label2id[datapoint.label] for datapoint in datapoints]

        #labels in one hot representation
        one_hot_labels = np.zeros((len(labels), len(label2id)))
        for i, label in enumerate(labels):
            one_hot_labels[i, label] = 1
        
        predictions = []
        for clf in self.classifiers:
            clf_predictions = []
            for datapoint, evidence_generation_result, retrieval_result in zip(datapoints, evidence_generation_results, retrieval_results):
                clf_predictions.append(clf(datapoint, evidence_generation_result, retrieval_result).probs)

            predictions.append(clf_predictions)

        #convert to numpy array of shape (#num_classifiers, #num_datapoints, #num_classes)
        predictions = np.array(predictions)

        def opt_func_multivariate(weights):
            #calculate weighted averages for each data_point
            weighted_avg = np.average(predictions, axis=0, weights=weights)

            if metric == "cross-entropy":
                #calculate cross entropy loss
                loss = np.average(-np.sum(one_hot_labels * np.log(weighted_avg), axis=1))
                return loss
            elif metric == "f1":
                #get predicted classes
                pred_classes = np.argmax(weighted_avg, axis=-1)

                #calculate macro F1 score
                f1 = f1_score(labels, pred_classes, average='macro')

                #we minize the negative F1 score
                return -f1
            elif metric == "accuracy":
                #get predicted classes
                pred_classes = np.argmax(weighted_avg, axis=-1)

                #calculate accuracy
                acc = np.mean(labels == pred_classes)

                #we minize the negative accuracy
                return -acc
            else:
                raise ValueError("Metric not supported")
            
        def opt_func_univariate(weight):
            #calculate weighted averages for each data_point
            weighted_avg = np.average(predictions, axis=0, weights=np.array([weight, 1-weight]))

            if metric == "cross-entropy":
                #calculate cross entropy loss
                loss = np.average(-np.sum(one_hot_labels * np.log(weighted_avg), axis=1))
                return loss
            elif metric == "f1":
                #get predicted classes
                pred_classes = np.argmax(weighted_avg, axis=-1)

                #calculate macro F1 score
                f1 = f1_score(labels, pred_classes, average='macro')

                #we minize the negative F1 score
                return -f1
            elif metric == "accuracy":
                #get predicted classes
                pred_classes = np.argmax(weighted_avg, axis=-1)

                #calculate accuracy
                acc = np.mean(labels == pred_classes)

                #we minize the negative accuracy
                return -acc
            else:
                raise ValueError("Metric not supported")


        if len(self.classifiers) == 1:
            self.weights = np.array([1.0])
        elif len(self.classifiers) == 2:
            res = opt.minimize_scalar(opt_func_univariate, bounds=(0, 1), method="bounded")
            print(res)
            self.weights = np.array([res.x, 1-res.x])

        else:
            #define constraints
            cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})

            #define bounds
            bounds = [(0, 1)] * len(self.classifiers)

            res = opt.minimize(opt_func_multivariate, self.weights, method="SLSQP", bounds=bounds, constraints=cons)

            print(res)
            self.weights = res.x


class LogRegEnsembleClassifier(Classifier):
    """Ensemble classifier that utilize stacking with meta model logistic regression"""
    def __init__(self, classifiers: List[Classifier]):
        self.classifiers = classifiers
        self.logreg = LogisticRegression()

    def __call__(self, datapoint, evidence_generation_result, retrieval_result):
        clf_probs = [c(datapoint, evidence_generation_result, retrieval_result).probs for c in self.classifiers]

        #stack the classfier probabilities into one input (they serve as features for the logreg model)
        input = np.hstack(clf_probs).reshape(1, -1) 

        return ClassificationResult(
            probs=self.logreg.predict_proba(input).squeeze(),
            metadata={"clf_probs": clf_probs}
        )
    

    def fit(self, datapoints, evidence_generation_results, retrieval_results, metric:str="cross-entropy"):
        """fit logreg"""
        labels = [label2id[datapoint.label] for datapoint in datapoints]

        predictions = []
        for clf in self.classifiers:
            clf_predictions = []
            for datapoint, evidence_generation_result, retrieval_result in zip(datapoints, evidence_generation_results, retrieval_results):
                clf_predictions.append(clf(datapoint, evidence_generation_result, retrieval_result).probs)

            predictions.append(clf_predictions)

        #convert to numpy array of shape (#num_classifiers, #num_datapoints, #num_classes)
        predictions = np.array(predictions)

        #fit logreg
        input = np.hstack(predictions)

        self.logreg.fit(input, labels)