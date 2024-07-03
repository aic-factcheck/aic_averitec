## reranking of higher number of sentences retrieved
from sentence_transformers import CrossEncoder
from typing import List
import json
import os
import time

def rerank_sentences(claim: str, sentences: List[str], rerank_model="mixedbread-ai/mxbai-rerank-xsmall-v1", top_k=10):
    #load model
    model = CrossEncoder(rerank_model, device="cuda")
    query = claim

    docs = [query] + sentences

    # # 2. Encode
    # embeddings = model.encode(docs)

    # similarities = cos_sim(embeddings[0], embeddings[1:]).flatten()

    # #get indices of sorted similarities
    # sorted_indices = similarities.argsort(descending=True)

    results = model.rank(query, sentences, top_k=top_k)
    
    scores = []
    sorted_indices = []
    for res in results:
        scores.append(res['score'])
        sorted_indices.append(res['corpus_id'])

    return scores, sorted_indices

def combine_all_documents(knowledge_file):
    documents, urls = [], []

    with open(knowledge_file, "r", encoding="utf-8") as json_file:
        for i, line in enumerate(json_file):
            data = json.loads(line)
            #concat strings (sentences) in the list data["url2text"] to one string with \n as separator
            document = "\n".join(data["url2text"])

            documents.append(document)
            urls.append(data["url"])

    return documents, urls, i + 1


def retrieve_top_k_sentences(query, documents, urls, top_k, rerank_model="mixedbread-ai/mxbai-rerank-xsmall-v1"):
    _, top_k_idx = rerank_sentences(claim=query, sentences=documents, rerank_model=rerank_model, top_k=top_k)
    return [documents[i] for i in top_k_idx], [urls[i] for i in top_k_idx]


# modified original __main__ to function
def get_top_k_sentences_nn(knowledge_store_dir:str = "data_store/output_dev", claim_file: str = "data/dev.json", json_output:str = "data_store/dev_top_k.json", top_k:int = 10, start:int = 0, end:int = -1, rerank_model="mixedbread-ai/mxbai-rerank-xsmall-v1"):
    with open(claim_file, "r", encoding="utf-8") as json_file:
        target_examples = json.load(json_file)

    if end == -1:
        end = len(os.listdir(knowledge_store_dir))
        print(end)

    files_to_process = list(range(start, end))
    total = len(files_to_process)

    with open(json_output, "w", encoding="utf-8") as output_json:
        done = 0
        for idx, example in enumerate(target_examples):
            # Load the knowledge store for this example
            if idx in files_to_process:
                print(f"Processing claim {idx}... Progress: {done + 1} / {total}")
                documents, document_urls, num_urls_this_claim = combine_all_documents(os.path.join(knowledge_store_dir, f"{idx}.json"))


                print(
                    f"Obtained {len(documents)} documents from {num_urls_this_claim} urls."
                )

                # Retrieve top_k sentences with bm25
                st = time.time()
                top_k_documents, top_k_urls = retrieve_top_k_sentences(
                    example["claim"], documents, document_urls, top_k, rerank_model=rerank_model
                )
                print(f"Top {top_k} retrieved. Time elapsed: {time.time() - st}.")

                json_data = {
                    "claim_id": idx,
                    "claim": example["claim"],
                    f"top_{top_k}": [
                        {"sentence": sent, "url": url}
                        for sent, url in zip(top_k_documents, top_k_urls)
                    ],
                }
                output_json.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                done += 1


if __name__ == "__main__":
    get_top_k_sentences_nn(knowledge_store_dir="/mnt/data/factcheck/averitec-data/data_store/output_dev", claim_file="/mnt/data/factcheck/averitec-data/data/dev.json", json_output="./aic_averitec/data_store/dev_top_k_nn.json", rerank_model = "mixedbread-ai/mxbai-rerank-xsmall-v1")

