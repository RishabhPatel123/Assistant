import torch ,os
from pathlib import Path
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType

# ✅ Load tokenizer and model
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = GPT2LMHeadModel.from_pretrained(model_name)

# ✅ Apply LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base_model, lora_config)

# ✅ Load dataset
def load_baalak_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().split("<|endoftext|>")
        data = [x.strip() for x in data if x.strip()]
    return {"text": data}

raw_dataset = load_baalak_dataset(os.path.join("data","data.txt"))

# ✅ Convert to HuggingFace Dataset & shuffle
ds = Dataset.from_dict(raw_dataset).shuffle(seed=42)

# ✅ Train/test split (optional, for eval)
split = ds.train_test_split(test_size=0.1)
train_ds = split["train"]
eval_ds = split["test"]

# ✅ Tokenization
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_train = train_ds.map(tokenize_function, batched=True)
tokenized_eval = eval_ds.map(tokenize_function, batched=True)

# ✅ Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ✅ Training Arguments
training_args = TrainingArguments(
    output_dir="./baalak_adapter",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=100,
    #evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
    logging_dir="./logs",
    logging_first_step=True
)

# ✅ Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ✅ Train
trainer.train()

# ✅ Save only the LoRA adapter (not full model)
model.save_pretrained("baalak_adapter", save_adapter=True)
tokenizer.save_pretrained("baalak_adapter")

# ✅ Evaluate and log final loss
eval_metrics = trainer.evaluate()
print(f"📉 Final Evaluation Loss: {eval_metrics['eval_loss']:.4f}")
print("✅ LoRA fine-tuned adapter saved at ./baalak_adapter")
