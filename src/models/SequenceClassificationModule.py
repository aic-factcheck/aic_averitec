import pytorch_lightning as pl
import torch
from transformers.optimization import AdamW
import torchmetrics
from torchmetrics.classification import F1Score


class SequenceClassificationModule(pl.LightningModule):

    def __init__(
        self, tokenizer, model, use_question_stance_approach=True, learning_rate=1e-3
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.learning_rate = learning_rate

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=model.num_labels
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=model.num_labels
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=model.num_labels
        )

        self.train_f1 = F1Score(
            task="multiclass", num_classes=model.num_labels, average="macro"
        )
        self.val_f1 = F1Score(
            task="multiclass", num_classes=model.num_labels, average=None
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=model.num_labels, average=None
        )

        self.use_question_stance_approach = use_question_stance_approach

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, x_mask, y = batch

        outputs = self(x, attention_mask=x_mask, labels=y)
        logits = outputs.logits
        loss = outputs.loss

        preds = torch.argmax(logits, axis=1)

        self.log("train_loss", loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, x_mask, y = batch

        outputs = self(x, attention_mask=x_mask, labels=y)
        logits = outputs.logits
        loss = outputs.loss

        preds = torch.argmax(logits, axis=1)

        if not self.use_question_stance_approach:
            self.val_acc(preds, y)
            self.log("val_acc_step", self.val_acc)

            self.val_f1(preds, y)
            self.log("val_loss", loss)

        return {"val_loss": loss, "src": x, "pred": preds, "target": y}

    def validation_epoch_end(self, outs):
        if self.use_question_stance_approach:
            self.handle_end_of_epoch_scoring(outs, self.val_acc, self.val_f1)

        self.log("val_acc_epoch", self.val_acc)

        f1 = self.val_f1.compute()
        self.val_f1.reset()

        self.log("val_f1_epoch", torch.mean(f1))

        class_names = ["supported", "refuted", "nei", "conflicting"]
        for i, c_name in enumerate(class_names):
            self.log("val_f1_" + c_name, f1[i])

    def test_step(self, batch, batch_idx):
        x, x_mask, y = batch

        outputs = self(x, attention_mask=x_mask)
        logits = outputs.logits

        preds = torch.argmax(logits, axis=1)

        if not self.use_question_stance_approach:
            self.test_acc(preds, y)
            self.log("test_acc_step", self.test_acc)
            self.test_f1(preds, y)

        return {"src": x, "pred": preds, "target": y}

    def test_epoch_end(self, outs):
        if self.use_question_stance_approach:
            self.handle_end_of_epoch_scoring(outs, self.test_acc, self.test_f1)

        self.log("test_acc_epoch", self.test_acc)

        f1 = self.test_f1.compute()
        self.test_f1.reset()
        self.log("test_f1_epoch", torch.mean(f1))

        class_names = ["supported", "refuted", "nei", "conflicting"]
        for i, c_name in enumerate(class_names):
            self.log("test_f1_" + c_name, f1[i])

    def handle_end_of_epoch_scoring(self, outputs, acc_scorer, f1_scorer):
        gold_labels = {}
        question_support = {}
        for out in outputs:
            srcs = out["src"]
            preds = out["pred"]
            tgts = out["target"]

            tokens = self.tokenizer.batch_decode(
                srcs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            for src, pred, tgt in zip(tokens, preds, tgts):
                claim_id = src.split("[ question ]")[0]

                if claim_id not in gold_labels:
                    gold_labels[claim_id] = tgt
                    question_support[claim_id] = []

                question_support[claim_id].append(pred)

        for k, gold_label in gold_labels.items():
            support = question_support[k]

            has_unanswerable = False
            has_true = False
            has_false = False

            for v in support:
                if v == 0:
                    has_true = True
                if v == 1:
                    has_false = True
                if v in (
                    2,
                    3,
                ):  # TODO very ugly hack -- we cant have different numbers of labels for train and test so we do this
                    has_unanswerable = True

            if has_unanswerable:
                answer = 2
            elif has_true and not has_false:
                answer = 0
            elif has_false and not has_true:
                answer = 1
            elif has_true and has_false:
                answer = 3

            # TODO this is very hacky and wont work if the device is literally anything other than cuda:0
            acc_scorer(
                torch.as_tensor([answer]).to("cuda:0"),
                torch.as_tensor([gold_label]).to("cuda:0"),
            )
            f1_scorer(
                torch.as_tensor([answer]).to("cuda:0"),
                torch.as_tensor([gold_label]).to("cuda:0"),
            )
