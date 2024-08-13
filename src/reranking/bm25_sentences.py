import argparse
import json
import os
import time
import numpy as np
import nltk
from rank_bm25 import BM25Okapi


def combine_all_sentences(knowledge_file):
    sentences, urls = [], []

    with open(knowledge_file, "r", encoding="utf-8") as json_file:
        for i, line in enumerate(json_file):
            data = json.loads(line)
            sentences.extend(data["url2text"])
            urls.extend([data["url"] for i in range(len(data["url2text"]))])
    return sentences, urls, i + 1


def retrieve_top_k_sentences(query, document, urls, top_k):
    tokenized_docs = [nltk.word_tokenize(doc) for doc in document]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(nltk.word_tokenize(query))
    top_k_idx = np.argsort(scores)[::-1][:top_k]

    return [document[i] for i in top_k_idx], [urls[i] for i in top_k_idx]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get top 100 sentences with BM25 in the knowledge store."
    )
    parser.add_argument(
        "-k",
        "--knowledge_store_dir",
        type=str,
        default="data_store/output_dev",
        help="The path of the knowledge_store_dir containing json files with all the retrieved sentences.",
    )
    parser.add_argument(
        "-c",
        "--claim_file",
        type=str,
        default="data/dev.json",
        help="The path of the file that stores the claim.",
    )
    parser.add_argument(
        "-o",
        "--json_output",
        type=str,
        default="data_store/dev_top_k.json",
        help="The output dir for JSON files to save the top 100 sentences for each claim.",
    )
    parser.add_argument(
        "--top_k",
        default=100,
        type=int,
        help="How many documents should we pick out with BM25.",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=0,
        help="Staring index of the files to process.",
    )
    parser.add_argument(
        "-e", "--end", type=int, default=-1, help="End index of the files to process."
    )

    args = parser.parse_args()

    with open(args.claim_file, "r", encoding="utf-8") as json_file:
        target_examples = json.load(json_file)

    if args.end == -1:
        args.end = len(os.listdir(args.knowledge_store_dir))
        print(args.end)

    files_to_process = list(range(args.start, args.end))
    total = len(files_to_process)

    with open(args.json_output, "w", encoding="utf-8") as output_json:
        done = 0
        for idx, example in enumerate(target_examples):
            # Load the knowledge store for this example
            if idx in files_to_process:
                print(f"Processing claim {idx}... Progress: {done + 1} / {total}")
                document_in_sentences, sentence_urls, num_urls_this_claim = (
                    combine_all_sentences(
                        os.path.join(args.knowledge_store_dir, f"{idx}.json")
                    )
                )

                print(
                    f"Obtained {len(document_in_sentences)} sentences from {num_urls_this_claim} urls."
                )

                # Retrieve top_k sentences with bm25
                st = time.time()
                top_k_sentences, top_k_urls = retrieve_top_k_sentences(
                    example["claim"], document_in_sentences, sentence_urls, args.top_k
                )
                print(f"Top {args.top_k} retrieved. Time elapsed: {time.time() - st}.")

                json_data = {
                    "claim_id": idx,
                    "claim": example["claim"],
                    f"top_{args.top_k}": [
                        {"sentence": sent, "url": url}
                        for sent, url in zip(top_k_sentences, top_k_urls)
                    ],
                }
                output_json.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                done += 1
                output_file.flush()
