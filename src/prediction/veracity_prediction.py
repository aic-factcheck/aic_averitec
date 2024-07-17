import argparse
import json
import tqdm
import torch
import pytorch_lightning as pl
from transformers import BertTokenizer, BertForSequenceClassification, BitsAndBytesConfig
from models.SequenceClassificationModule import SequenceClassificationModule
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

LABEL = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


class SequenceClassificationDataLoader(pl.LightningDataModule):
    def __init__(self, tokenizer, data_file, batch_size, add_extra_nee=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_file = data_file
        self.batch_size = batch_size
        self.add_extra_nee = add_extra_nee

    def tokenize_strings(
        self,
        source_sentences,
        max_length=400,
        pad_to_max_length=False,
        return_tensors="pt",
    ):
        encoded_dict = self.tokenizer(
            source_sentences,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else "longest",
            truncation=True,
            return_tensors=return_tensors,
        )

        input_ids = encoded_dict["input_ids"]
        attention_masks = encoded_dict["attention_mask"]

        return input_ids, attention_masks

    def quadruple_to_string(self, claim, question, answer, bool_explanation=""):
        if bool_explanation is not None and len(bool_explanation) > 0:
            bool_explanation = ", because " + bool_explanation.lower().strip()
        else:
            bool_explanation = ""
        return (
            "[CLAIM] "
            + claim.strip()
            + " [QUESTION] "
            + question.strip()
            + " "
            + answer.strip()
            + bool_explanation
        )
    

#rewrite the main function to a function

def veracity_prediction(claim_with_evidence_file="data_store/dev_top_3_rerank_qa.json", output_file="data_store/dev_veracity_prediction.json", best_checkpoint="pretrained_models/bert_veracity.ckpt"):
    examples = []
    with open(claim_with_evidence_file) as f:
        for line in f:
            examples.append(json.loads(line))

    bert_model_name = "bert-base-uncased"

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = BertForSequenceClassification.from_pretrained(
        bert_model_name, num_labels=4, problem_type="single_label_classification"
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trained_model = SequenceClassificationModule.load_from_checkpoint(
        best_checkpoint, tokenizer=tokenizer, model=bert_model
    ).to(device)

    dataLoader = SequenceClassificationDataLoader(
        tokenizer=tokenizer,
        data_file="this_is_discontinued",
        batch_size=32,
        add_extra_nee=False,
    )

    predictions = []

    for example in tqdm.tqdm(examples):
        example_strings = []
        for evidence in example["evidence"]:
            example_strings.append(
                dataLoader.quadruple_to_string(
                    example["claim"], evidence["question"], evidence["answer"], ""
                )
            )

        if (
            len(example_strings) == 0
        ):  # If we found no evidence e.g. because google returned 0 pages, just output NEI.
            example["label"] = "Not Enough Evidence"
            continue

        tokenized_strings, attention_mask = dataLoader.tokenize_strings(example_strings)
        example_support = torch.argmax(
            trained_model(
                tokenized_strings.to(device), attention_mask=attention_mask.to(device)
            ).logits,
            axis=1,
        )

        has_unanswerable = False
        has_true = False
        has_false = False

        for v in example_support:
            if v == 0:
                has_true = True
            if v == 1:
                has_false = True
            if v in (
                2,
                3,
            ):  # TODO another hack -- we cant have different labels for train and test so we do this
                has_unanswerable = True

        if has_unanswerable:
            answer = 2
        elif has_true and not has_false:
            answer = 0
        elif not has_true and has_false:
            answer = 1
        else:
            answer = 3

        json_data = {
            "claim_id": example["claim_id"],
            "claim": example["claim"],
            "evidence": example["evidence"],
            "pred_label": LABEL[answer],
        }
        predictions.append(json_data)

    with open(output_file, "w", encoding="utf-8") as output_file:
        json.dump(predictions, output_file, ensure_ascii=False, indent=4)


class SequenceClassificationDataLoaderNLI(pl.LightningDataModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def tokenize_strings(
        self,
        source_sentences,
        max_length=400,
        pad_to_max_length=False,
        return_tensors="pt",
    ):
        encoded_dict = self.tokenizer(
            source_sentences,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else "longest",
            truncation=True,
            return_tensors=return_tensors,
        )

        input_ids = encoded_dict["input_ids"]
        attention_masks = encoded_dict["attention_mask"]

        return input_ids, attention_masks

    def quadruple_to_string(self, claim, question, answer):
        return [
            claim.strip(), answer.strip()
        ]

def veracity_prediction_v2(claim_with_evidence_file:str, output_file:str, best_checkpoint:str="models/averitec/nli/deberta-v3-large/checkpoint-687", nei_new_eval=True):

    examples = []
    with open(claim_with_evidence_file) as f:
        for line in f:
            examples.append(json.loads(line))

    id2correct_id = {0: 1, 1: 0, 2: 2}

    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(best_checkpoint)

    dataLoader_nli = SequenceClassificationDataLoaderNLI(tokenizer)

    model.eval()

    predictions = []

    for example in tqdm.tqdm(examples):
        example_strings = []
        for evidence in example["evidence"]:
            example_strings.append(
                dataLoader_nli.quadruple_to_string(
                    example["claim"], evidence["question"], evidence["answer"]
                )
            )

        if (
            len(example_strings) == 0
        ):  # If we found no evidence e.g. because google returned 0 pages, just output NEI.
            example["label"] = "Not Enough Evidence"
            continue

        tokenized_strings, attention_mask = dataLoader_nli.tokenize_strings(example_strings)
        with torch.no_grad():
            logits = model(tokenized_strings.to(device), attention_mask=attention_mask.to(device)).logits

        example_support = torch.argmax(
            logits,
            axis=1,
        )

        #remap the labels in example support tensor using the id2correct_id
        example_support = torch.tensor([id2correct_id[i.item()] for i in example_support])

        has_unanswerable = False
        has_true = False
        has_false = False

        for v in example_support:
            if v == 0:
                has_true = True
            if v == 1:
                has_false = True
            if v in (
                2,
                3,
            ):  # TODO another hack -- we cant have different labels for train and test so we do this
                has_unanswerable = True

        if not nei_new_eval:
            if has_unanswerable:
                answer = 2
            elif has_true and not has_false:
                answer = 0
            elif not has_true and has_false:
                answer = 1
            else:
                answer = 3
        else:
            if has_true and has_false:
                answer = 3
            elif has_true and not has_false:
                answer = 0
            elif not has_true and has_false:
                answer = 1
            else:
                answer = 2 #otherwise NEI


        json_data = {
            "claim_id": example["claim_id"],
            "claim": example["claim"],
            "evidence": example["evidence"],
            "pred_label": LABEL[answer],
        }
        predictions.append(json_data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Given a claim and its 3 QA pairs as evidence, we use another pre-trained BERT model to predict the veracity label."
    )
    parser.add_argument(
        "-i",
        "--claim_with_evidence_file",
        default="data_store/dev_top_3_rerank_qa.json",
        help="Json file with claim and top question-answer pairs as evidence.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        default="data_store/dev_veracity_prediction.json",
        help="Json file with the veracity predictions.",
    )
    parser.add_argument(
        "-ckpt",
        "--best_checkpoint",
        type=str,
        default="pretrained_models/bert_veracity.ckpt",
    )
    args = parser.parse_args()

    examples = []
    with open(args.claim_with_evidence_file) as f:
        for line in f:
            examples.append(json.loads(line))

    bert_model_name = "bert-base-uncased"

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = BertForSequenceClassification.from_pretrained(
        bert_model_name, num_labels=4, problem_type="single_label_classification"
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trained_model = SequenceClassificationModule.load_from_checkpoint(
        args.best_checkpoint, tokenizer=tokenizer, model=bert_model
    ).to(device)

    dataLoader = SequenceClassificationDataLoader(
        tokenizer=tokenizer,
        data_file="this_is_discontinued",
        batch_size=32,
        add_extra_nee=False,
    )

    predictions = []

    for example in tqdm.tqdm(examples):
        example_strings = []
        for evidence in example["evidence"]:
            example_strings.append(
                dataLoader.quadruple_to_string(
                    example["claim"], evidence["question"], evidence["answer"], ""
                )
            )

        if (
            len(example_strings) == 0
        ):  # If we found no evidence e.g. because google returned 0 pages, just output NEI.
            example["label"] = "Not Enough Evidence"
            continue

        tokenized_strings, attention_mask = dataLoader.tokenize_strings(example_strings)
        example_support = torch.argmax(
            trained_model(
                tokenized_strings.to(device), attention_mask=attention_mask.to(device)
            ).logits,
            axis=1,
        )

        has_unanswerable = False
        has_true = False
        has_false = False

        for v in example_support:
            if v == 0:
                has_true = True
            if v == 1:
                has_false = True
            if v in (
                2,
                3,
            ):  # TODO another hack -- we cant have different labels for train and test so we do this
                has_unanswerable = True

        if has_unanswerable:
            answer = 2
        elif has_true and not has_false:
            answer = 0
        elif not has_true and has_false:
            answer = 1
        else:
            answer = 3

        json_data = {
            "claim_id": example["claim_id"],
            "claim": example["claim"],
            "evidence": example["evidence"],
            "pred_label": LABEL[answer],
        }
        predictions.append(json_data)

    with open(args.output_file, "w", encoding="utf-8") as output_file:
        json.dump(predictions, output_file, ensure_ascii=False, indent=4)


class SequenceClassificationDataLoaderNLI4concat(pl.LightningDataModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def tokenize_strings(
        self,
        source_sentences,
        max_length=1024,
        pad_to_max_length=False,
        return_tensors="pt",
    ):
        encoded_dict = self.tokenizer(
            source_sentences,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else "longest",
            truncation=True,
            return_tensors=return_tensors,
        )

        input_ids = encoded_dict["input_ids"]
        attention_masks = encoded_dict["attention_mask"]

        return input_ids, attention_masks

    def quadruple_to_string(self, claim, question, answer):
        if question is None:
            return [
                claim.strip(), answer.strip()
            ]
        else:
            return [
                claim.strip(), question.strip() + " " + answer.strip()
            ]
        
def veracity_prediction_4concat(claim_with_evidence_file:str, output_file:str, best_checkpoint:str="models/averitec/nli_4concat/deberta-v3-large/checkpoint-576", peft:bool = False):
    examples = []
    with open(claim_with_evidence_file) as f:
        for line in f:
            examples.append(json.loads(line))

    id2correct_id = {0: 1, 1: 0, 2: 2, 3:3}

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if not peft:
        model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint).to(device)
        tokenizer = AutoTokenizer.from_pretrained(best_checkpoint)
    else:
        bnb_config = BitsAndBytesConfig(
                        load_in_4bit= True,
                        bnb_4bit_quant_type= "nf4",
                        bnb_4bit_compute_dtype= torch.bfloat16,
                        bnb_4bit_use_double_quant= False,
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
                best_checkpoint,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                num_labels=4,
        )
        tokenizer = AutoTokenizer.from_pretrained(best_checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        model = PeftModel.from_pretrained(base_model, best_checkpoint)
    

    dataLoader_nli = SequenceClassificationDataLoaderNLI4concat(tokenizer)

    model.eval()

    predictions = []

    for example in tqdm.tqdm(examples):
        ev_string = ""
        for evidence in example["evidence"]:
            ev_string += evidence["question"] + " " + evidence["answer"] + " "

        example_strings = [dataLoader_nli.quadruple_to_string(example["claim"], None, ev_string[:-1])]

        tokenized_strings, attention_mask = dataLoader_nli.tokenize_strings(example_strings, max_length=1024)
        with torch.no_grad():
            logits = model(tokenized_strings.to(device), attention_mask=attention_mask.to(device)).logits

        example_support = torch.argmax(
            logits,
            axis=1,
        )

        label = id2correct_id[example_support.item()]


        json_data = {
            "claim_id": example["claim_id"],
            "claim": example["claim"],
            "evidence": example["evidence"],
            "pred_label": LABEL[label],
        }
        predictions.append(json_data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
