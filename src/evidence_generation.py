import numpy as np
import json
import os
import time
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any, List, Dict
from averitec import Datapoint
from retrieval import RetrievalResult
from utils.chat import SimpleJSONChat
from scipy.special import softmax
from labels import label2id
from openai import OpenAI
from rank_bm25 import BM25Okapi
import nltk

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
    
class FixedFewShotEvidenceGenerator(GptEvidenceGenerator):
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
        result+= """\n---\n## Few-shot learning\nYou have access to the following few-shot learning examples for questions and answers.:\n
            question: When was Donald trump elected? answer: November 2016\n
            question: Was the U.S economy stagnant in 2016? anser: The IMF marked down its forecast for the United States in 2016 to 1.6 percent, from 2.2 percent in July 2016.\n
            question: Did Joe Biden vote for the Iraq War? answer: Yes. When Joe Biden was a U.S. Senator, he voted in favor of a resolution on Iraq in October 2022.\n
            question: When did President Buhari sign a bill repealing the Police Act Cap. P19. Laws of the Federation, 2004? answer: The president in a memo dated September 16, 2020, communicated his assent to the Bill to the National Assembly, through the Clerk of the legislature.
        """   
        return result

class DynamicFewShotEvidenceGenerator(GptEvidenceGenerator):
    def __init__(self, model="gpt-4o", client: SimpleJSONChat = None, reference_corpus_path="/mnt/data/factcheck/averitec-data/data/train.json", k=10):
        #load reference (train) corpus
        with open(reference_corpus_path, "r") as f:
            self.reference_corpus = json.load(f)

        #prepare tokenized bm25 corpus
        tokenized_corpus = []
        for example in self.reference_corpus:
            tokenized_corpus.append(nltk.word_tokenize(example['claim']))

        #initialize bm25 model
        self.bm25 = BM25Okapi(tokenized_corpus)

        #number of retrieved few shot examples
        self.k = k

        super().__init__(model, client)
        

    def format_system_prompt(self, retrieval_result: RetrievalResult, few_shot_examples) -> str:
        #alternative for not outputing 10 every time - maybe better for classfiers (not problem now): (There is no need to output all 10 questions if you know that the questions contain all necessary information for fact-checking of the claim)
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

        #add few shot examples
        result += """\n---\n## Few-shot learning\nYou have access to the following few-shot learning examples for questions and answers.:\n"""
        for example in few_shot_examples:
            for q in example['questions']:
                question = q['question']
                for a in q['answers']:
                    if a['answer_type'] == "Boolean":
                        answer = a['answer'] + ", because " + a['boolean_explanation']
                    elif a['answer_type'] in ["Extractive", "Abstractive"]:
                        answer = a['answer']


                    result += f'\n#Example for claim "{example["claim"]}": "question": "{question}", "answer": "{answer}"\n'
        return result
    
    def __call__(
        self, datapoint: Datapoint, retrieval_result: RetrievalResult, *args, **kwargs
    ) -> EvidenceGenerationResult:
        #get top k sentences
        claim = datapoint.claim
        scores = self.bm25.get_scores(nltk.word_tokenize(claim))
        top_n = np.argsort(scores)[::-1][:self.k]
        few_shot_examples = [self.reference_corpus[i] for i in top_n]
        #get system prompt
        system_prompt = self.format_system_prompt(retrieval_result, few_shot_examples)
        #call gpt
        gpt_result = self.client(
            system_prompt=system_prompt, user_prompts=[claim]
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

class GptBatchedEvidenceGenerator(GptEvidenceGenerator):
    def __init__(self, model="gpt-4o", client=None):
        super().__init__(model, client)
        self.batch = []
        self.fallback_gpt_generator = GptEvidenceGenerator()

    def get_batch_dict(self, datapoint: Datapoint, retrieval_result: RetrievalResult):
        system_prompt = self.format_system_prompt(retrieval_result)
        user_prompt = datapoint.claim
        return {
            "custom_id": f"averitec-{datapoint.claim_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                # "model": "gpt-3.5-turbo-0125",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
            },
        }

    def get_batch_files(self, batch_size=100, path="data_store/batch"):
        batches = [self.batch[i : i + batch_size] for i in range(0, len(self.batch), batch_size)]
        filenames = []
        j = 0
        if not os.path.exists(path):
            os.makedirs(path)
        for batch in batches:
            j += 1
            filenames.append(f"{path}/batch_{j}.jsonl")
            with open(f"{path}/batch_{j}.jsonl", "w") as f:
                for item in batch:
                    f.write(json.dumps(item) + "\n")
        return filenames

    def submit_and_await_batches(self, files, outfile, sleep=10):
        # if outfile already exists, read it
        if os.path.exists(outfile):
            with open(outfile, "r") as f:
                print("!!!!! existing outfile found, skipping computation")
                concat_text = f.read()
        else:
            client = OpenAI()
            i = 1
            concat_text = ""
            for file in tqdm(files):
                batch_input_file = client.files.create(file=open(file, "rb"), purpose="batch")

                batch = client.batches.create(
                    input_file_id=batch_input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": f"dev-set job, batch {i}",
                    },
                )
                print(batch)
                while True:
                    batch = client.batches.retrieve(batch.id)
                    if batch.status == "completed":
                        break
                    time.sleep(sleep)
                    print("waiting for batch to complete", batch.request_counts, batch.id)
                print(f"batch {i} completed")
                i += 1
                file_response = client.files.content(batch.output_file_id)
                concat_text += file_response.text
                with open(outfile, "w") as f:
                    f.write(concat_text)

        result = []
        for line in concat_text.split("\n"):
            if not line:
                continue
            #print(json.loads(line))
            result.append(json.loads(line)["response"]["body"]["choices"][0]["message"]["content"])
        return result

    def __call__(
        self, datapoint: Datapoint, retrieval_result: RetrievalResult, *args, **kwargs
    ) -> EvidenceGenerationResult:
        self.batch.append(self.get_batch_dict(datapoint, retrieval_result))
        return EvidenceGenerationResult(evidences=[], metadata={"suggested_label": [0, 0, 0, 0]})

    def update_pipeline_result(self, pipeline_result, gpt_result, classifier):
        from pipeline import PipelineResult

        self.last_llm_output = gpt_result
        gpt_data = self.parse_json(gpt_result)
        try:
            evidence_generation_result = EvidenceGenerationResult(
                evidences=self.parse_evidence(gpt_data["questions"], pipeline_result.retrieval_result),
                metadata={
                    "suggested_label": self.parse_label_probabilities(gpt_data["claim_veracity"]),
                    "llm_type": self.client.model,
                    "llm_output": gpt_data,
                },
            )
        except:
            print("failed, using fallback gpt")
            evidence_generation_result = self.fallback_gpt_generator(
                pipeline_result.datapoint, pipeline_result.retrieval_result
            )
        return PipelineResult(
            datapoint=pipeline_result.datapoint,
            retrieval_result=pipeline_result.retrieval_result,
            evidence_generation_result=evidence_generation_result,
            classification_result=classifier(
                pipeline_result.datapoint, evidence_generation_result, pipeline_result.retrieval_result
            ),
        )
