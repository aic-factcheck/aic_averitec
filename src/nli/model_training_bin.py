from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
import evaluate
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import os
from torch import nn

dotenv_path = Path('aic_averitec/.env')

DATASET_PATH = "/mnt/data/factcheck/averitec-data/data"

#SEED
SEED = 42 #Answer to the Ultimate Question of Life, the Universe, and Everything

#model_id = "cross-encoder/nli-deberta-v3-large"
model_id = "microsoft/deberta-v3-large"

model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_id)

data_collator = DataCollatorWithPadding(tokenizer)


#prepare dataset
dataset = load_dataset("json", data_files = {"train": os.path.join(DATASET_PATH, "train_nli_bin_concat.jsonl"), "dev": os.path.join(DATASET_PATH, "dev_nli_bin_concat.jsonl")})


#tokenize dataset
def tokenize_function(examples):
    example = tokenizer(examples["claim"], examples["evidence"], truncation=True, max_length=1024)

    return example

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset=tokenized_dataset.remove_columns(["claim", "evidence"])

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def compute_metrics(eval_pred):
   labels = eval_pred.label_ids
   preds = eval_pred.predictions.argmax(-1)

   # Calculate accuracy
   accuracy = accuracy_score(labels, preds)
   
   # Calculate precision, recall, and F1-score
   precision = precision_score(labels, preds, average="binary")
   recall = recall_score(labels, preds, average='binary')
   f1 = f1_score(labels, preds, average='binary')
   return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


training_args = TrainingArguments(
   output_dir=f"/home/mlynatom/models/averitec/nli_bin/{model_id.split('/')[-1]}_custom_loss",
   learning_rate=1e-6,
   per_device_train_batch_size=32,
   per_device_eval_batch_size=32,
   num_train_epochs=12,
   weight_decay=0.01,
   eval_strategy="epoch",
   save_strategy="epoch",
   load_best_model_at_end=True,
   metric_for_best_model="f1",
   warmup_ratio=0.1,
   gradient_checkpointing=True,
   report_to="wandb",
   fp16=True,
   logging_steps=10,
   logging_strategy="steps",
   save_total_limit=1
)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 5]).to(self.args.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_dataset["train"].shuffle(seed=SEED),
   eval_dataset=tokenized_dataset["dev"],
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

trainer.train()