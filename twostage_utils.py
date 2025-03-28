import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Trainer

def get_tokens_from_first_n(dataset, n):
    """
    Returns a set of all token IDs from the "labels" field of the first n examples.
    Assumes the dataset is already tokenized and each example has a "labels" field.
    """
    tokens = set()
    for i in range(n):
        example = dataset[i]
        tokens.update(example["labels"])
    return tokens

def custom_data_collator_weight(tokenizer, features):
    # 1) Separate labels (unchanged)
    labels_list = [f["labels"] for f in features]
    features_no_labels = [{k: v for k, v in f.items() if k != "labels"} for f in features]

    # 2) Pad input (unchanged)
    batch = tokenizer.pad(features_no_labels, padding=True, return_tensors="pt")
    input_max_len = batch["input_ids"].size(1)

    # 3) Compute label max length and pad labels (unchanged)
    label_max_len = max(len(lbl) for lbl in labels_list)
    final_max_len = max(input_max_len, label_max_len)

    padded_labels = []
    for labels in labels_list:
        needed = final_max_len - len(labels)
        padded = labels + [-100]*needed
        padded_labels.append(padded)
    batch["labels"] = torch.tensor(padded_labels)

    # 4) Include sample_weight if it's present
    if "sample_weight" in features[0]:
        weights = [f["sample_weight"] for f in features]
        batch["sample_weight"] = torch.tensor(weights, dtype=torch.float)

    return batch

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop off sample_weight so we don't pass it into the model
        sample_weight = inputs.pop("sample_weight", None)

        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        # Flatten for cross-entropy
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        # CrossEntropyLoss with no averaging
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        loss_per_token = loss_fct(logits_flat, labels_flat)

        # Reshape to [batch_size, seq_len] and sum over seq_len
        loss_per_example = loss_per_token.view(logits.size(0), -1).sum(dim=1)

        # If we have sample_weight, multiply each example's loss by that weight
        if sample_weight is not None:
            loss_per_example = loss_per_example * sample_weight

        # Average across the batch
        loss = loss_per_example.mean()

        return (loss, outputs) if return_outputs else loss
