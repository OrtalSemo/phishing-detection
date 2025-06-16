from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import os

# Load dataset from local CSV file
dataset = load_dataset("csv", data_files="combined_balanced_data.csv", split="train")


# Load tokenizer and tokenize the dataset
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], padding=True, truncation=True)

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split into training and evaluation sets
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load pre-trained DistilBERT model with binary classification head
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Define optimized training arguments for GPU
training_args = TrainingArguments(
    output_dir="./results",                  # Directory to save checkpoints
    evaluation_strategy="steps",             # Evaluate every N steps
    eval_steps=250,
    save_strategy="steps",                   # Save checkpoint every N steps
    save_steps=250,
    save_total_limit=3,                      # Keep only last 3 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    per_device_train_batch_size=64,          # Optimized for A100 GPU
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    seed=42,
    report_to="none",
    no_cuda=not torch.cuda.is_available(),
    fp16=True                                # Use mixed precision to reduce memory & speed up
)

# Initialize Trainer with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Resume training if checkpoint exists
checkpoint_path = os.path.join(training_args.output_dir, "checkpoint-500")
if os.path.exists(checkpoint_path):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

# Evaluate the trained model
eval_results = trainer.evaluate()

# Save evaluation metrics to text file
with open("results.txt", "w") as f:
    f.write("===== Evaluation Results =====\n")
    for key, value in eval_results.items():
        f.write(f"{key}: {value:.4f}\n")

# Save the final model and tokenizer
trainer.save_model("saved_model/")
tokenizer.save_pretrained("saved_model/tokenizer/")
