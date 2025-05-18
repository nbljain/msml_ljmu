import os
import shutil

import evaluate
import nltk
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from google.colab import files
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, T5ForConditionalGeneration,
                          T5Tokenizer)

os.environ["WANDB_DISABLED"] = "true"

# Load nltk for texts
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
metric = evaluate.load("rouge")

# Global Parameters
L_RATE = 3e-4
BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH = 4
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 3
NUM_EPOCHS = 3
LOG_STEPS = 100

# Load the tokenizer, model, and data collator
MODEL_NAME = "google/flan-t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Acquire the training data from Hugging Face
dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
train_dataset = dataset["train"].select(range(50000)).shuffle(seed=20)
val_dataset = dataset["validation"].select(range(5000)).shuffle(seed=20)
test_dataset = dataset["test"].select(range(5000)).shuffle(seed=20)

# We prefix our tasks with "answer the question"
prefix = "Give the summary of the article: "


# Define the preprocessing function
def preprocess_function(examples):
    """Add prefix to the sentences, tokenize the text, and set the labels"""
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(
        text_target=examples["highlights"], max_length=512, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Map the preprocessing function across our dataset
train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True)
test_tokenized_dataset = test_dataset.map(preprocess_function, batched=True)
val_tokenized_dataset = val_dataset.map(preprocess_function, batched=True)


# Define compute metrics function to get the rouge score while training and validation
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    return result


# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./flant-t5-finetuned-cnn-dailymail_50k",
    eval_strategy="epoch",
    learning_rate=L_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
    weight_decay=WEIGHT_DECAY,
    save_total_limit=SAVE_TOTAL_LIM,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=LOG_STEPS,
    predict_with_generate=True,
    push_to_hub=False,
    report_to="none",
)

# Define model trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=val_tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# We see last epoch gives the best Rouge scores, so we will use this for the testing
last_checkpoint = "./flant-t5-finetuned-cnn-dailymail_50k/checkpoint-36500"
finetuned_model = T5ForConditionalGeneration.from_pretrained(last_checkpoint)
tokenizer = T5Tokenizer.from_pretrained(last_checkpoint)

# Fine-tuned model
finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(
    "./flant-t5-finetuned-cnn-dailymail_50k/checkpoint-37500"
)
finetuned_tokenizer = AutoTokenizer.from_pretrained(
    "./flant-t5-finetuned-cnn-dailymail_50k/checkpoint-37500"
)

# Original model
original_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
original_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")


def generate_summaries(
    model, tokenizer, inputs, max_input_length=512, max_target_length=128
):
    model.eval()
    inputs_tokenized = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    input_ids = inputs_tokenized["input_ids"].to(model.device)
    attention_mask = inputs_tokenized["attention_mask"].to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_target_length,
        )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


# Generate summaries from the finetuned and original model
original_preds = generate_summaries(
    original_model, original_tokenizer, val_dataset["article"]
)
finetuned_preds = generate_summaries(
    finetuned_model, finetuned_tokenizer, val_dataset["article"]
)
references = val_dataset["highlights"]

# Compute Rouge
rouge = load("rouge")
original_scores = rouge.compute(
    predictions=original_preds, references=references, use_stemmer=True
)
finetuned_scores = rouge.compute(
    predictions=finetuned_preds, references=references, use_stemmer=True
)

# Build DataFrame
df = pd.DataFrame(
    {
        "original_summary": original_preds,
        "finetuned_summary": finetuned_preds,
        "reference_summary": references,
    }
)

# Save to CSV
df.to_csv("model_comparison_finetune_50k_results.csv", index=False)

# Path to your model checkpoint directory
model_dir = "./flant-t5-finetuned-cnn-dailymail_50k/checkpoint-37500"

# Output zip file name
zip_file = "finetuned_model_cnn_news_50k"

# Zip the directory
shutil.make_archive(zip_file, "zip", model_dir)

files.download(f"{zip_file}.zip")
