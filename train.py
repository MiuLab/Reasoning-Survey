import json
import random

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer

# Configuration
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
output_dir = "./tts-model"
hub_repo = "voidful/tts-qwen2.5-math"

qwen_path = "tgt_Qwen2.5-Math-1.5B_math500_data_length_500_max512_n_250.json"
deepseek_path = "tgt_DeepSeek-R1-Distill-Qwen-1.5B_math500_data_length_500_max512_n_250.json"

custom_prompt_templates = {
    "default": "{question} Please provide the initial step towards resolving the question. This step may serve as a foundation but might not encompass the entire solution.\n"
}

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

# JSONL loader
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

qwen_data = load_jsonl(qwen_path)
deepseek_data = load_jsonl(deepseek_path)

# Dataset construction
dataset = []

for item in qwen_data:
    question = item["problem"]
    for resp in item["response"]:
        template = custom_prompt_templates["default"]
        dataset.append({
            "text": template.format(question=question, response=resp),
            "source": "qwen"
        })

for item in deepseek_data:
    question = item["problem"]
    for resp in item["response"]:
        messages = [{"role": "user", "content": question}, {"role": "assistant", "content": resp}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        dataset.append({"text": chat_text, "source": "deepseek"})

# Train/test split
random.shuffle(dataset)
n_total = len(dataset)
n_test = int(n_total * 0.10)

test_data = dataset[:n_test]
train_data = dataset[n_test:]

eval_data = test_data[:len(test_data) // 2]
real_test_data = test_data[len(test_data) // 2:]

dataset_dict = DatasetDict({
    "train": Dataset.from_list(train_data),
    "eval": Dataset.from_list(eval_data),
})

# Tokenization
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding=True,
        truncation=True,
    )
    tokenized["labels"] = [
        [token if token != tokenizer.pad_token_id else -100 for token in seq]
        for seq in tokenized["input_ids"]
    ]
    return tokenized

tokenized_dataset = dataset_dict.map(tokenize_function, batched=True, remove_columns=["text", "source"])

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=3,
    save_strategy="epoch",
    save_total_limit=3,
    evaluation_strategy="epoch",
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    bf16=True,
    max_grad_norm=1.0,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id=hub_repo,
    hub_strategy="every_save",
)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.push_to_hub()
tokenizer.push_to_hub(hub_repo)

print("Training and upload complete. Best model pushed to:", hub_repo)
