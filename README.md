---
license: cc-by-nc-4.0
---

# AVeriTeC

Data, knowledge store and source code to reproduce the baseline experiments for the [AVeriTeC](https://arxiv.org/abs/2305.13117) dataset, which will be used for the 7th [FEVER](https://fever.ai/) workshop co-hosted at EMNLP 2024.


## NEWS:
 - 19.04.2024: The submisstion page (with eval.ai) for the shared-task is alive, you can participate by submitting your predictions [here](https://eval.ai/web/challenges/challenge-page/2285/overview)!

## Dataset
The training and dev dataset can be found under [data](https://huggingface.co/chenxwh/AVeriTeC/tree/main/data). Test data will be released at a later date. Each claim follows the following structure:
```json
{
    "claim": "The claim text itself",
    "required_reannotation": "True or False. Denotes that the claim received a second round of QG-QA and quality control annotation.",
    "label": "The annotated verdict for the claim",
    "justification": "A textual justification explaining how the verdict was reached from the question-answer pairs.",
    "claim_date": "Our best estimate for the date the claim first appeared",
    "speaker": "The person or organization that made the claim, e.g. Barrack Obama, The Onion.",
    "original_claim_url": "If the claim first appeared on the internet, a url to the original location",
    "cached_original_claim_url": "Where possible, an archive.org link to the original claim url",
    "fact_checking_article": "The fact-checking article we extracted the claim from",
    "reporting_source": "The website or organization that first published the claim, e.g. Facebook, CNN.",
    "location_ISO_code": "The location most relevant for the claim. Highly useful for search.",
    "claim_types": [
            "The types of the claim",
    ],
    "fact_checking_strategies": [
        "The strategies employed in the fact-checking article",
    ],
    "questions": [
        {
            "question": "A fact-checking question for the claim",
            "answers": [
                {
                    "answer": "The answer to the question",
                    "answer_type": "Whether the answer was abstractive, extractive, boolean, or unanswerable",
                    "source_url": "The source url for the answer",
                    "cached_source_url": "An archive.org link for the source url"
                    "source_medium": "The medium the answer appeared in, e.g. web text, a pdf, or an image.",
                }
            ]
        },
    ]
}
```

## Reproduce the baseline 

Below are the steps to reproduce the baseline results. The main difference from the reported results in the paper is that, instead of requiring direct access to the paid Google Search API, we provide such search results for up to 1000 URLs per claim using different queries, and the scraped text as a knowledge store for retrieval for each claim. This is aimed at reducing the overhead cost of participating in the Shared Task. Another difference is that we also added text scraped from pdf URLs to the knowledge store.


### 0. Set up environment

You will need to have [Git LFS](https://git-lfs.com/) installed:
```bash
git lfs install
git clone https://huggingface.co/chenxwh/AVeriTeC
```
You can also skip the large files in the repo and selectively download them later:
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/chenxwh/AVeriTeC
```
Then create `conda` environment and install the libs.

```bash
conda create -n averitec python=3.11
conda activate averitec

pip install -r requirements.txt
python -m spacy download en_core_web_lg
python -m nltk.downloader punkt
python -m nltk.downloader wordnet
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 1. Scrape text from the URLs obtained by searching queries with the Google API

The URLs of the search results and queries used for each claim can be found [here](https://huggingface.co/chenxwh/AVeriTeC/tree/main/data_store/urls).

 Next, we scrape the text from the URLs and parse the text to sentences. The processed files are also provided and can be found [here](https://huggingface.co/chenxwh/AVeriTeC/tree/main/data_store/knowledge_store). You can use your own scraping tool to extract sentences from the URLs.

```bash
bash script/scraper.sh <split> <start_idx> <end_idx> 
# e.g., bash script/scraper.sh dev 0 500
```

### 2. Rank the sentences in the knowledge store with BM25
Then, we rank the scraped sentences for each claim using BM25 (based on the similarity to the claim), keeping the top 100 sentences per claim.
See [bm25_sentences.py](https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/reranking/bm25_sentences.py) for more argument options. We provide the output file for this step on the dev set [here](https://huggingface.co/chenxwh/AVeriTeC/blob/main/data_store/dev_top_k_sentences.json).
```bash
python -m src.reranking.bm25_sentences
```

### 3. Generate questions-answer pair for the top sentences
We use [BLOOM](https://huggingface.co/bigscience/bloom-7b1) to generate QA paris for each of the top 100 sentence, providing 10 closest claim-QA-pairs from the training set as in-context examples. See [question_generation_top_sentences.py](https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/reranking/question_generation_top_sentences.py) for more argument options. We provide the output file for this step on the dev set [here](https://huggingface.co/chenxwh/AVeriTeC/blob/main/data_store/dev_top_k_qa.json).
```bash
python -m src.reranking.question_generation_top_sentences
```

### 4. Rerank the QA pairs
Using a pre-trained BERT model [bert_dual_encoder.ckpt](https://huggingface.co/chenxwh/AVeriTeC/blob/main/pretrained_models/bert_dual_encoder.ckpt), we rerank the QA paris and keep top 3 QA paris as evidence. See [rerank_questions.py](https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/reranking/rerank_questions.py) for more argument options. We provide the output file for this step on the dev set [here](https://huggingface.co/chenxwh/AVeriTeC/blob/main/data_store/dev_top_3_rerank_qa.json).
```bash
python -m src.reranking.rerank_questions
```


### 5. Veracity prediction
Finally, given a claim and its 3 QA pairs as evidence, we use another pre-trained BERT model [bert_veracity.ckpt](https://huggingface.co/chenxwh/AVeriTeC/blob/main/pretrained_models/bert_veracity.ckpt) to predict the veracity label. See [veracity_prediction.py](https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/prediction/veracity_prediction.py) for more argument options. We provide the prediction file for this step on the dev set [here](https://huggingface.co/chenxwh/AVeriTeC/blob/main/data_store/dev_veracity_prediction.json).
```bash
python -m src.prediction.veracity_prediction
```

Then evaluate the veracity prediction performance with (see [evaluate_veracity.py](https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/prediction/evaluate_veracity.py) for more argument options):
```bash
python -m src.prediction.evaluate_veracity
```

The result for dev and the test set below. We recommend using 0.25 as cut-off score for evaluating the relevance of the evidence. 

| Model             | Split	| Q only | Q + A | Veracity @ 0.2 | @ 0.25 | @ 0.3 |
|-------------------|-------|--------|-------|----------------|--------|-------|
| AVeriTeC-BLOOM-7b | dev	|  0.240 | 0.185 | 	    0.186     |  0.092 | 0.050 |
| AVeriTeC-BLOOM-7b | test	|  0.248 | 0.185 |  	0.176     |  0.109 | 0.059 |

## Citation
If you find AVeriTeC useful for your research and applications, please cite us using this BibTeX:
```bibtex
@inproceedings{
  schlichtkrull2023averitec,
  title={{AV}eriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web},
  author={Michael Sejr Schlichtkrull and Zhijiang Guo and Andreas Vlachos},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023},
  url={https://openreview.net/forum?id=fKzSz0oyaI}
}
```

## 🌌 Slurm cluster usage
If you have access to a Slurm cluster, you can use the provided scripts to run the experiments. The scripts are located in the `script` directory. You can launch a simple jupyter notebook using:
```bash
sbatch script/_slurm_job.sh
```

and get its url using `script/jupyter_url.py`.
```bash
python script/jupyter_url.py
```