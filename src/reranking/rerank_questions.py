import argparse
import json
import torch
import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from models.DualEncoderModule import DualEncoderModule


def triple_to_string(x):
    return " </s> ".join([item.strip() for item in x])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rerank the QA paris and keep top 3 QA paris as evidence using a pre-trained BERT model."
    )
    parser.add_argument(
        "-i",
        "--top_k_qa_file",
        default="data_store/dev_top_k_qa.json",
        help="Json file with claim and top k generated question-answer pairs.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        default="data_store/dev_top_3_rerank_qa.json",
        help="Json file with the top3 reranked questions.",
    )
    parser.add_argument(
        "-ckpt",
        "--best_checkpoint",
        type=str,
        default="pretrained_models/bert_dual_encoder.ckpt",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=3,
        help="top_n question answer pairs as evidence to keep.",
    )
    args = parser.parse_args()

    examples = []
    with open(args.top_k_qa_file) as f:
        for line in f:
            examples.append(json.loads(line))

    bert_model_name = "bert-base-uncased"

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = BertForSequenceClassification.from_pretrained(
        bert_model_name, num_labels=2, problem_type="single_label_classification"
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trained_model = DualEncoderModule.load_from_checkpoint(
        args.best_checkpoint, tokenizer=tokenizer, model=bert_model
    ).to(device)

    with open(args.output_file, "w", encoding="utf-8") as output_file:
        for example in tqdm.tqdm(examples):
            strs_to_score = []
            values = []

            bm25_qau = example["bm25_qau"] if "bm25_qau" in example else []
            claim = example["claim"]

            for question, answer, url in bm25_qau:
                str_to_score = triple_to_string([claim, question, answer])

                strs_to_score.append(str_to_score)
                values.append([question, answer, url])

            if len(bm25_qau) > 0:
                encoded_dict = tokenizer(
                    strs_to_score,
                    max_length=512,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                input_ids = encoded_dict["input_ids"]
                attention_masks = encoded_dict["attention_mask"]

                scores = torch.softmax(
                    trained_model(input_ids, attention_mask=attention_masks).logits,
                    axis=-1,
                )[:, 1]

                top_n = torch.argsort(scores, descending=True)[: args.top_n]
                evidence = [
                    {
                        "question": values[i][0],
                        "answer": values[i][1],
                        "url": values[i][2],
                    }
                    for i in top_n
                ]
            else:
                evidence = []

            json_data = {
                "claim_id": example["claim_id"],
                "claim": claim,
                "evidence": evidence,
            }
            output_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
            output_file.flush()
