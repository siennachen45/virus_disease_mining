import argparse
import os

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed


def tokenize_batch(tokenizer, examples, max_length):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/association_training_examples.tsv")
    parser.add_argument("--model_dir", default="models/pubmedbert_base")
    parser.add_argument("--out_dir", default="models/pubmedbert_association_classifier")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    df = pd.read_csv(args.data, sep="\t", dtype=str)
    df = df.rename(columns={df.columns[0]: "text", df.columns[1]: "label"})
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    dataset = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    dataset = dataset.map(
        lambda x: tokenize_batch(tokenizer, x, args.max_length),
        batched=True,
        remove_columns=["text"],
    )
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        num_labels=2,
        local_files_only=True,
    )

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_dir=os.path.join(args.out_dir, "tensorboard"),
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=["tensorboard"],
        fp16=args.fp16,
        seed=args.seed,
        disable_tqdm=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)


if __name__ == "__main__":
    main()
