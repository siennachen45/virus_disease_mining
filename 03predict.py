import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def tokenize_batch(tokenizer, examples, max_length):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=max_length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/pubmedbert_association_classifier")
    parser.add_argument("--data", default="data/candidate_sentences.tsv")
    parser.add_argument("--out", default="results/sentence_predictions.tsv")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--threshold", type=float, default=0.74)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.data, sep="\t", header=None)
    sentence_col = df.columns[-1]
    df["sentence"] = df[sentence_col].astype(str)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    dataset = Dataset.from_pandas(df[["sentence"]], preserve_index=False)
    dataset = dataset.map(
        lambda x: tokenize_batch(tokenizer, x, args.max_length),
        batched=True,
        remove_columns=["sentence"],
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True)

    training_args = TrainingArguments(
        output_dir="results/prediction_tmp",
        per_device_eval_batch_size=args.batch_size,
        fp16=args.fp16,
        report_to=[],
        disable_tqdm=False,
    )

    trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer)

    predictions = trainer.predict(dataset)
    logits = torch.tensor(predictions.predictions)
    prob = torch.softmax(logits, dim=-1)[:, 1].numpy()
    label = (prob >= args.threshold).astype(int)

    df["prob"] = prob
    df["label"] = label
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["sentence"]).to_csv(args.out, sep="\t", header=False, index=False)


if __name__ == "__main__":
    main()
