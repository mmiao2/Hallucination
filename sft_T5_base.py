import math, copy, torch, gc
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, TaskType
from utils import miscalibration_analysis, hallucination_analysis, create_powerlaw_p, batch_log_probability
from utils import tokenize_function, custom_data_collator, sample

##set up cuda device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## define lora target modules to be all attention block and q v layers, easier for prob
target_modules = []
# Encoder: 12 layers, each with SelfAttention
for i in range(12):
    base = f"encoder.block.{i}.layer.0.SelfAttention"
    target_modules.append(f"{base}.q")
    target_modules.append(f"{base}.v")

# Decoder: 12 layers, each with SelfAttention and EncDecAttention
for i in range(12):
    # Self-attention
    base_sa = f"decoder.block.{i}.layer.0.SelfAttention"
    target_modules.append(f"{base_sa}.q")
    target_modules.append(f"{base_sa}.v")

    # Cross-attention
    base_ca = f"decoder.block.{i}.layer.1.EncDecAttention"
    target_modules.append(f"{base_ca}.q")
    target_modules.append(f"{base_ca}.v")
# print(len(target_modules))

##lora set up 
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=target_modules
)

##load data
np.random.seed(1217) 
data_path = "/home1/m/miaom/hallucination/llm_data_2_prompt_generation_T5.csv"
data = pd.read_csv(data_path)
length_data = 5000
dataset = data[0:length_data]
train_datasets = {}
p_datasets = {}
sample_size = length_data
for alpha in [1, 1.5, 2]:
  powerlaw_p = create_powerlaw_p(dataset, alpha)
  training_data = sample(powerlaw_p, sample_size)
  train_datasets[alpha] = training_data
  p_datasets[alpha] = powerlaw_p

##device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(device)
tokenizer = AutoTokenizer.from_pretrained("t5-base")
# Use the first training dataset (DataFrame with columns "x" and "y")
alpha, train_texts = list(train_datasets.items())[1]
print(f"alpha is: {alpha}")
print(f"\n{'='*20} Fine-tuning for Pareto α = {alpha} {'='*20}")
print(train_texts)

# Build HF Dataset
train_dataset = Dataset.from_dict({
    "x": train_texts["x"].tolist(),
    "y": train_texts["y"].tolist()
})

# Tokenize and split
tokenized_train = train_dataset.map(lambda example: tokenize_function(tokenizer, example), batched=False)
split_dataset = tokenized_train.train_test_split(test_size=0.1, seed=1217)
train_dataset_final = split_dataset["train"]
eval_dataset_final = split_dataset["test"]

##set training arguments
training_args = TrainingArguments(
    output_dir=f"cache/model_lora_alpha{alpha}",
    overwrite_output_dir=True,
    num_train_epochs=64,
    per_device_train_batch_size=128,
    gradient_accumulation_steps=8,
    learning_rate=5e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    eval_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset_final,
    eval_dataset=eval_dataset_final,
    data_collator=lambda features: custom_data_collator(features, tokenizer),
    args=training_args
)

trainer.train()
##save the model weights for later testing
trainer.save_model(f"/home1/m/miaom/hallucination/model_alpha{alpha}_2_5000")
eval_results = trainer.evaluate()
print("Validation loss:", eval_results.get("eval_loss"))

exact_h_rate, output_table = hallucination_analysis(model, train_dataset_final["y"], tokenizer, 128, device)
miscal_rate, miscal_output_table = miscalibration_analysis(p_datasets, model, tokenizer, alpha, 0.1, device)
print(f"Hallucination rate is: {exact_h_rate} and miscalibration is: {miscal_rate}")