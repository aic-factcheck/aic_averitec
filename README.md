# AIC CTU system at AVeriTeC: Re-framing automated fact-checking as a simple RAG task

![AIC logo](logo.svg "AIC logo")

This repository is the official implementation of [AIC CTU system at AVeriTeC: Re-framing automated fact-checking as a simple RAG task](https://arxiv.org/abs/2410.11446). This system description paper describes our 3rd place submission in the [AVeriTeC Shared Task](https://fever.ai/task.html) which was organised as a part of the 7th FEVER workshop co-located with [EMNLP 2024](https://2024.emnlp.org/program/workshops/) conference.

## Abstract & Diagram
> This paper describes our 3rd place submission in the AVeriTeC shared task in which we attempted to address the challenge of fact-checking with evidence retrieved in the wild using a simple scheme of Retrieval-Augmented Generation (RAG) designed for the task, leveraging the predictive power of Large Language Models. We release our codebase and explain its two modules - the Retriever and the Evidence & Label generator - in detail, justifying their features such as MMR-reranking and Likert-scale confidence estimation. We evaluate our solution on AVeriTeC dev and test set and interpret the results, picking the GPT-4o as the most appropriate model for our pipeline at the time of our publication, with Llama 3.1 70B being a promising open-source alternative. We perform an empirical error analysis to see that faults in our predictions often coincide with noise in the data or ambiguous fact-checks, provoking further research and data augmentation.

![Diagram of our pipeline.](pipeline.png "Pipeline diagram")

## Citation
```bibtex
@inproceedings{ullrich2024aicctuaveritecreframing,
      title={AIC CTU system at AVeriTeC: Re-framing automated fact-checking as a simple RAG task}, 
      author={Herbert Ullrich and Tomáš Mlynář and Jan Drchal},
      year={2024},
      eprint={2410.11446},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.11446}, 
}
```

## Requirements

To install requirements using pip, run:

```setup
pip install -r requirements.txt
```

The datasets and baseline system are available at [AVeriTeC 🤗 Hugging Face repository](https://huggingface.co/chenxwh/AVeriTeC).

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our system achieved the 3rd place on AVeriTeC test set evaluation performed on the [EvalAI platform](https://eval.ai/web/challenges/challenge-page/2285/leaderboard/5655). The test set results are also provided in the following table (with our submission hihglighted in bold):

| Rank | Participant team               | Q only (↑) | Q + A (↑) | AVeriTeC Score (↑) |
|------|---------------------------------|------------|-----------|--------------------|
| 1    | TUDA_MAI (InFact)               | 0.45       | 0.34      | 0.63               |
| 2    | HUMANE (HerO)                   | 0.48       | 0.35      | 0.57               |
| **3**    | **CTU AIC**                         | **0.46**       | **0.32**      | **0.50**               |
| 4    | Dunamu-ml                       | 0.49       | 0.35      | 0.50               |
| 5    | Papelo                          | 0.44       | 0.30      | 0.48               |
| 6    | UHH                             | 0.48       | 0.32      | 0.45               |
| 7    | SynApSe                         | 0.41       | 0.30      | 0.42               |
| 8    | arioriAveri (FactCheckFusion)   | 0.38       | 0.29      | 0.39               |
| 9    | Data-Wizards                    | 0.35       | 0.27      | 0.33               |
| 10   | MA-Bros-H                       | 0.38       | 0.24      | 0.27               |
| 11   | mitchelldehaven                 | 0.27       | 0.23      | 0.25               |
| 12   | SK_DU                           | 0.40       | 0.26      | 0.22               |
| 13   | UPS                             | 0.31       | 0.27      | 0.21               |
| 14   | FZI-WIM                         | 0.32       | 0.21      | 0.20               |
| 15   | KnowComp                        | 0.32       | 0.21      | 0.18               |
| 16   | IKR3-UNIMIB (AQHS)              | 0.32       | 0.24      | 0.18               |
| 17   | IKR3-Bicocca                    | 0.32       | 0.24      | 0.18               |
| 18   | ngetach                         | 0.37       | 0.21      | 0.14               |
| 19   | VGyasi                          | 0.38       | 0.22      | 0.12               |
| 20   | Host_81793_Team (AVeriTeC)      | 0.24       | 0.20      | 0.11               |
| 21   | InfinityScalers!                | 0.26       | 0.19      | 0.08               |
| 22   | AYM                             | 0.13       | 0.12      | 0.06               |
| 23   | Factors                         | 0.20       | 0.14      | 0.05               |

## Contributing
Our system is licensed by CC BY-NC 4.0 license, defined in the [LICENSE](LICENSE.md) file. To contribute to this repository, please reach out to the authors via contacts in our paper.

## Usage of AI assistants
While writing code in this repository, we used the help of GitHub Copilot.



# Old version of the README.md

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
To obtain data faster than via git-lfs, feel free to use `script/copy_data.sh` to copy the data to your project dir.
```bash
bash script/copy_data.sh
```


If you have access to a Slurm cluster, you can use the provided scripts to run the experiments. The scripts are located in the `script` directory. You can launch a simple jupyter notebook using:
```bash
sbatch script/_slurm_job.sh
```

and get its url using `script/jupyter_url.py`.
```bash
python script/jupyter_url.py
```


# Orig scheme:
Claim -> (Query generation -> Search) -> BM25 reranking -> Evidence (Q+A) extraction -> Evidence Reranking -> Veracity prediction

# New scheme 
Claim -> (Query generation -> Search) -> reranking based on claim -> resolution generation (CoT)


