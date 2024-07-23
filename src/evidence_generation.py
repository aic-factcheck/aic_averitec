import numpy as np
import json
from dataclasses import dataclass, field
from typing import Any, List, Dict
from averitec import Datapoint
from retrieval import RetrievalResult
from utils.chat import SimpleJSONChat
from scipy.special import softmax
from labels import label2id


@dataclass
class Evidence:
    question: str = None
    answer: str = None
    url: str = None
    scraped_text: str = None

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "url": self.url,
            "scraped_text": self.scraped_text,
        }


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
            result[label2id[cls.parse_label(label)]] = cls.parse_likert(likert)
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
    def __init__(self, model="gpt-4o", client: SimpleJSONChat = None):
        if client is None:
            client = SimpleJSONChat(model=model, parse_output=False)
        self.model = model
        self.client = client
        self.last_llm_output = None

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
        "Not Enough Evidence": "<Likert-scale rating of how much You agree with the 'Not Enough Evidence' veracity classification>",
        "Conflicting Evidence/Cherrypicking": "<Likert-scale rating of how much You agree with the 'Conflicting Evidence/Cherrypicking' veracity classification>"
    }
}
```"""
        return result

    def __call__(
        self, datapoint: Datapoint, retrieval_result: RetrievalResult, *args, **kwargs
    ) -> EvidenceGenerationResult:
        gpt_result = self.client(
            system_prompt=self.format_system_prompt(retrieval_result), user_prompts=[datapoint.claim]
        )
        self.last_llm_output = gpt_result
        gpt_data = self.parse_json(gpt_result)
        return EvidenceGenerationResult(
            evidences=self.parse_evidence(gpt_data["questions"], retrieval_result),
            metadata={
                "suggested_label": self.parse_label_probabilities(gpt_data["claim_veracity"]),
                "llm_type": self.client.model,
                "llm_output": gpt_data,
            },
        )
