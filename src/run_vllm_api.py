import os
import pickle
import json
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    base_url="http://g12:8095/v1",
    api_key="token-abc123",
)

#files in directory
DIR = '/mnt/data/factcheck/averitec-data/data_store/batch_jobs/test_mmr+gpt4o-dfewshot-tiebrk-atype'
OUT_DIR = "/mnt/data/factcheck/averitec-data/data_store/batch_jobs_llama_res/test_mmr+gpt4o-dfewshot-tiebrk-atype"
files = os.listdir(DIR)
sorted_files = sorted(files)
#leave only files with batch prefix
sorted_files = [f for f in sorted_files if f.startswith("batch")]

#take only second half
sorted_files = sorted_files[12:]


for f in sorted_files:
    results = {}
    with open(os.path.join(DIR, f), "r") as file:
        for line in tqdm(file):
            data = json.loads(line)
            completion = client.chat.completions.create(model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", 
                                                        messages=data["body"]["messages"], 
                                                        temperature=data["body"]["temperature"])
            results[data["custom_id"]] = completion        

    #save results to pickle
    with open(os.path.join(OUT_DIR, f[:-6] + "_" + "results.pickle"), "wb") as out_f:
        pickle.dump(results, out_f)

    #save results to jsonl
    with open(os.path.join(OUT_DIR, f[:-6] + "_" + "results.jsonl"), "w") as out_f:
        for k, v in results.items():
            out_f.write(json.dumps({"custom_id": k, "response": v.choices[0].message.content}) + "\n")