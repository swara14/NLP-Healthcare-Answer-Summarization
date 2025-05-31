import os
import torch
from math import ceil
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from safetensors.torch import load_file

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Setup
PERSPECTIVES = ["INFORMATION", "SUGGESTION", "EXPERIENCE", "QUESTION", "CAUSE"]

def clean_summary(text):
    if not text or not isinstance(text, str):
        return ""
    stripped = text.strip()
    if stripped.lower() in ["false", "true", "not_duplicate", "duplicate", "n/a", "", "no summary available."]:
        return ""
    return stripped

def format_example(example):
    input_text = example["input"].strip()
    output_sections = example["output"].strip().split("\n")
    formatted_lines = []
    for label in PERSPECTIVES:
        matches = [line for line in output_sections if line.startswith(f"{label} SUMMARY:")]
        if matches:
            summary = clean_summary(matches[0].split("SUMMARY:", 1)[-1])
            if summary:
                formatted_lines.append(f"{label} SUMMARY: {summary}")
    return {"input": input_text, "output": "\n".join(formatted_lines)}

# Load tokenizer and base model
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
base_model = BartForConditionalGeneration.from_pretrained(model_name)

# Apply LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(base_model, peft_config)

# Load the saved LoRA weights
lora_weights_path = "./bart-optimized/adapter_model.safetensors"
lora_weights = load_file(lora_weights_path)
model.load_state_dict(lora_weights, strict=False)
model.train()

# Load and format hard examples
raw_data = load_dataset("json", data_files={"train": "./hard_examples.json"})["train"]
formatted_data = [format_example(ex) for ex in tqdm(raw_data, desc="Formatting hard examples")]
dataset = Dataset.from_list(formatted_data)

# Preprocessing
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input"],
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["output"],
            max_length=256,
            padding="max_length",
            truncation=True,
        )
    labels_input_ids = labels["input_ids"]
    labels_input_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label_seq]
        for label_seq in labels_input_ids
    ]
    model_inputs["labels"] = labels_input_ids
    return model_inputs

# Tokenize and prepare dataset
tokenized = dataset.map(preprocess_function, batched=True)
tokenized = tokenized.remove_columns(dataset.column_names)

# Training args
training_args = TrainingArguments(
    output_dir="./hard_example_lora",
    do_eval=True,
    eval_steps=ceil(len(tokenized) / 4),
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_total_limit=2,
    logging_steps=10,
    fp16=False,
    local_rank=-1,
    label_names=["labels"],
    logging_dir="./logs",
    report_to=[],
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

# Train and save
trainer.train()
model.save_pretrained("./bart-lora-hard")
tokenizer.save_pretrained("./bart-lora-hard")
torch.save(model.state_dict(), "bart-lora-hard.pt")
# from safetensors.torch import save_file
# save_file(model.state_dict(), "bart-lora-hard.safetensors")
