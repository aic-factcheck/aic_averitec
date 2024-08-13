import argparse
import time
import json
import nltk
from rank_bm25 import BM25Okapi
import numpy as np
import torch
from transformers import BloomTokenizerFast, BloomForCausalLM


def claim2prompts(example):
    claim = example["claim"]

    # claim_str = "Claim: " + claim + "||Evidence: "
    claim_str = "Evidence: "

    for question in example["questions"]:
        q_text = question["question"].strip()
        if len(q_text) == 0:
            continue

        if not q_text[-1] == "?":
            q_text += "?"

        answer_strings = []

        for a in question["answers"]:
            if a["answer_type"] in ["Extractive", "Abstractive"]:
                answer_strings.append(a["answer"])
            if a["answer_type"] == "Boolean":
                answer_strings.append(
                    a["answer"]
                    + ", because "
                    + a["boolean_explanation"].lower().strip()
                )

        for a_text in answer_strings:
            if not a_text[-1] in [".", "!", ":", "?"]:
                a_text += "."

            # prompt_lookup_str = claim + " " + a_text
            prompt_lookup_str = a_text
            this_q_claim_str = (
                claim_str + " " + a_text.strip() + "||Question answered: " + q_text
            )
            yield (
                prompt_lookup_str,
                this_q_claim_str.replace("\n", " ").replace("||", "\n"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use a prompt to generate questions that could be answered by top-k retrieved evidence. Output generated questions."
    )
    parser.add_argument("--reference_corpus", default="data/train.json", help="")
    parser.add_argument("--target_file", default="data/dev.json", help="")
    parser.add_argument(
        "-i",
        "--top_k_target_knowledge",
        default="data_store/dev_top_k_sentences.json",
        help="Directory where the sentences for the scraped data is saved.",
    )
    parser.add_argument(
        "-o",
        "--output_questions",
        default="data_store/dev_top_k_qa.json",
        help="Directory where the sentences for the scraped data is saved.",
    )
    parser.add_argument(
        "--top_k",
        default=100,
        type=int,
        help="How many documents should we pick out with BM25",
    )
    args = parser.parse_args()

    # few-shot learning from the training set
    with open(args.reference_corpus, "r", encoding="utf-8") as json_file:
        train_examples = json.load(json_file)

    prompt_corpus, tokenized_corpus = [], []

    for example in train_examples:
        for lookup_str, prompt in claim2prompts(example):
            entry = nltk.word_tokenize(lookup_str)
            tokenized_corpus.append(entry)
            prompt_corpus.append(prompt)

    prompt_bm25 = BM25Okapi(tokenized_corpus)

    # Load the bloom model:
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-7b1")
    model = BloomForCausalLM.from_pretrained(
        "bigscience/bloom-7b1",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        offload_folder="./offload",
    )

    with open(args.output_questions, "w", encoding="utf-8") as output_file:
        with open(args.top_k_target_knowledge, "r", encoding="utf-8") as json_file:
            for i, line in enumerate(json_file):
                data = json.loads(line)
                top_k_sentences_urls = data[f"top_{args.top_k}"]
                claim = data["claim"]
                claim_id = data["claim_id"]

                bm25_qau = []  # question, answer, url
                # Generate questions for those top k:
                for sent_i, sentences_urls in enumerate(top_k_sentences_urls):

                    prompt_lookup_str = sentences_urls["sentence"]
                    url = sentences_urls["url"]

                    prompt_s = prompt_bm25.get_scores(
                        nltk.word_tokenize(prompt_lookup_str)
                    )
                    prompt_n = 10
                    prompt_top_n = np.argsort(prompt_s)[::-1][:prompt_n]
                    prompt_docs = [prompt_corpus[i] for i in prompt_top_n]

                    claim_prompt = (
                        "Evidence: "
                        + prompt_lookup_str.replace("\n", " ")
                        + "\nQuestion answered: "
                    )

                    prompt = "\n\n".join(prompt_docs + [claim_prompt])

                    inputs = tokenizer([prompt], padding=True, return_tensors="pt").to(
                        model.device
                    )
                    st = time.time()
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_length=5000,
                        num_beams=2,
                        no_repeat_ngram_size=2,
                        early_stopping=True,
                    )
                    print(
                        f"Generated QA for sent {sent_i} in file {i}. Time elapsed: {time.time() - st}"
                    )

                    tgt_text = tokenizer.batch_decode(
                        outputs[:, inputs["input_ids"].shape[-1] :],
                        skip_special_tokens=True,
                    )[0]

                    # We are not allowed to generate more than 250 characters:
                    tgt_text = tgt_text[:250]

                    qau_pair = [
                        tgt_text.strip().split("?")[0].replace("\n", " ") + "?",
                        prompt_lookup_str.replace("\n", " "),
                        url,
                    ]

                    bm25_qau.append(qau_pair)

                json_data = {
                    "claim_id": claim_id,
                    "claim": claim,
                    "bm25_qau": bm25_qau,
                }
                output_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                output_file.flush()
