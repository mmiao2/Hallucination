import math, copy, torch, gc
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, TaskType
from utils import miscalibration_analysis, create_powerlaw_p, tokenize_function, custom_data_collator, sample
from utils_callback import CallBackTrainer


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
data_path = "TBU" 
data = pd.read_csv(data_path)
length_data = 10000
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
alpha, train_texts = list(train_datasets.items())[0] #options are 0, 1, 2
print(f"alpha is: {alpha}")
print(f"\n{'='*20} Fine-tuning for Pareto α = {alpha} {'='*20}")
print(train_texts)

# Build HF Dataset
train_dataset = Dataset.from_dict({
    "x": train_texts["x"].tolist(),
    "y": train_texts["y"].tolist(),
    "names": train_texts["names"].tolist(),
    "gold": train_texts["gold"].tolist()
})

# Tokenize and split
tokenized_train = train_dataset.map(lambda example: tokenize_function(tokenizer, example), batched=False)
split_dataset = tokenized_train.train_test_split(test_size=0.1, seed=1217)
train_dataset_final = split_dataset["train"]
eval_dataset_final = split_dataset["test"]

callback = CallBackTrainer(
    train_dataset_texts=train_dataset_final,
    train_datasets = train_datasets, 
    p_datasets=p_datasets,
    tokenizer=tokenizer,
    alpha=alpha,
    device=device,
    output_csv_path=f"TBU",
    epsilon=0.1,
    batch_size=48
)

######NORMAL TRAINING#####

##set training arguments
training_args = TrainingArguments(
    output_dir=f"cache/model_lora_alpha{alpha}",
    overwrite_output_dir=True,
    num_train_epochs=64,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=32,
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
    args=training_args,
    callbacks=[callback] 
)

trainer.train()

# ######DUPLICATION SECTION#####
#first train the duplicate data 
subset_size = int(0.05 * len(train_dataset_final)) #toggle
train_subset = train_dataset_final.select(range(subset_size))

# Duplicate the subset three times
duplications = 10 #toggle
train_subset_duplicated = concatenate_datasets([train_subset] * duplications)

# Define new training arguments for the additional 64 epochs
upweight_training_args = TrainingArguments(
    output_dir=f"cache/model_lora_alpha{alpha}_dup",  # new output directory to avoid conflicts
    overwrite_output_dir=True,
    num_train_epochs=64,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=32,
    learning_rate=5e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    eval_strategy="epoch",
    report_to="none"
)

# Create a new Trainer instance using the model and the duplicated subset
trainer_dup = Trainer(
    model=model, 
    train_dataset=train_subset_duplicated,
    eval_dataset=eval_dataset_final,
    data_collator=lambda features: custom_data_collator(features, tokenizer),
    args=upweight_training_args,
    callbacks=[callback] 
)

# Continue training on the duplicated data
trainer_dup.train()

#######END OF TRAININIG####
tvd, regret, miscal_output_table = miscalibration_analysis(p_datasets, train_datasets, model, tokenizer, alpha, 0.1, device)
print(f"Loss rate is: miscalibration is: {tvd} and regret is: {regret}")
miscal_output_table.to_csv(f"TBU", index=False)

