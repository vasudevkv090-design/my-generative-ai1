import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

model_name = "gpt2"
dataset = load_dataset("json", data_files={"train": "data/train.jsonl"})["train"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["c_attn"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

def preprocess(examples):
    return tokenizer(examples["prompt"] + examples["completion"], truncation=True, max_length=256)
tokenized = dataset.map(preprocess, remove_columns=dataset.column_names)

training_args = TrainingArguments(output_dir="./gpt2-lora", per_device_train_batch_size=4, num_train_epochs=1, fp16=torch.cuda.is_available())
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
trainer.train()
model.save_pretrained("./gpt2-lora")
