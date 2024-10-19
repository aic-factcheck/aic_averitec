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

## System Running

Our system is composed of retriever, evidence generation and optional classifier components. These components are implemented in the corresponding files in the `src` directory. The system can be run using a Jupyter notebook `TODO`.

The directory `notebooks` then contains more detailed notebooks used for creating the vector stores, evaluation of the pipeline and error analysis.

The experiments mentioned in our paper in the section 4 (Other examined approaches) are implemented in the `other_examined_approaches` directory. The pretrained model from the other examined approaches is available at our [🤗 Hugging Face repository](https://huggingface.co/collections/ctu-aic/averitec-shared-task-66b60f5571fdc534d8358156).

### Slurm Cluster Usage
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