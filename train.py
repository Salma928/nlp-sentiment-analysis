from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 1. Load dataset
print("Loading dataset...")
dataset = load_dataset("sst2", trust_remote_code=True)

# 2. Load tokenizer
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Tokenize
def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize, batched=True)

# 4. Load model
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 5. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].select(range(2000)),
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

# 8. Train
print("Training...")
trainer.train()

# 9. Save model
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
print("Model saved!")