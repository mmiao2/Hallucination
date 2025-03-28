import pandas as pd
import numpy as np
import random
from collections import Counter
from scipy.stats import pareto
import scipy.stats as stats
import math
import torch
from tqdm import tqdm

random.seed(1217)
def create_powerlaw_p(df, pareto_alpha):
    new_rows = []
    for _, row in df.iterrows():
        reps = int(np.floor(pareto.rvs(b=pareto_alpha, scale=1)))
        for _ in range(reps):
            new_rows.append((row['x'], row['y']))
    power_df = pd.DataFrame(new_rows, columns=['x', 'y'])
    print(f"Monofact % in p is: {mono_calc(power_df)}")
    return power_df

def sample(power_df, size):
    sampled_df = power_df.sample(n=size, replace=True, random_state=1217)
    print(f"Monofact in % sample is: {mono_calc(sampled_df)}")
    return sampled_df

def mono_calc(df):
    pairs = list(zip(df['x'], df['y']))
    pair_counts = Counter(pairs)
    num_mono = sum(1 for c in pair_counts.values() if c == 1)
    return num_mono / len(df) if len(df) else 0.0

## parallel log prob calculation, need to preset model to model.eval()
def batch_log_probability(model, tokenizer, input_texts, target_texts, device, batch_size=32):
    results = []
    # input_texts = len(target_texts) * ["[PROMPT]"]
    for start in range(0, len(input_texts), batch_size):
        end = start + batch_size
        batch_inputs = tokenizer(input_texts[start:end], padding=True, truncation=True, return_tensors="pt").to(device)
        batch_targets = tokenizer(target_texts[start:end], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**batch_inputs, labels=batch_targets["input_ids"])
        # outputs.logits: [B, target_seq_len, vocab_size]
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        # Sum log probabilities for each token in the target (ignore padding)
        for i in range(batch_targets["input_ids"].size(0)):
            tgt_ids = batch_targets["input_ids"][i]
            mask = (tgt_ids != tokenizer.pad_token_id)
            lp = log_probs[i].gather(1, tgt_ids.unsqueeze(1)).squeeze(1)
            total_lp = lp[mask].sum().item()
            results.append(total_lp)
    return results

def tokenize_statement(s):
    """Split the statement on commas and strip whitespace.
    Returns a set of tokens."""
    return set(token.strip() for token in s.split(",") if token.strip())

def check_exact_hallucination_batch(statements, truth_set):
    return [statement not in truth_set for statement in statements]

def hallucination_analysis(model, truth_list, tokenizer, batch_size=32):
    sample_size = 1  # one generation per prompt
    total_prompts = 0
    total_exact_hallucinated = 0
    generated_info = []  # will store tuples: (prompt, generated statement)

    fixed_prompt = "[PROMPT]"
    num_examples = len(truth_list)
    truth_set = set(truth_list)

    for i in tqdm(range(0, num_examples, batch_size), desc="Processing hallucination"):
        current_batch_size = min(batch_size, num_examples - i)
        # Use the fixed prompt for every example in this batch.
        input_texts = [fixed_prompt] * current_batch_size
        total_prompts += current_batch_size

        # Tokenize the batch.
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        # Generate outputs in batch.
        outputs = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=sample_size,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode outputs.
        decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Group outputs so that each prompt gets its corresponding generated text.
        grouped_generated = [decoded_texts[j:j+sample_size] for j in range(0, len(decoded_texts), sample_size)]

        # For each prompt in the batch, compare the generated output against the entire truth set.
        for prompt_text, gen_list in zip(input_texts, grouped_generated):
            generated_info.extend([(prompt_text, gen) for gen in gen_list])

            # Use updated helper functions (which compare each generated output against the entire truth list).
            exact_flags = check_exact_hallucination_batch(gen_list, truth_set)
            total_exact_hallucinated += sum(exact_flags)

    exact_hallucination_rate = total_exact_hallucinated / total_prompts
    return exact_hallucination_rate, generated_info

def create_epsilon_induced_bins(epsilon):
   if epsilon < 0:
       raise ValueError("Epsilon must be non-negative")
   if epsilon == 0:
       # For epsilon = 0, we create a bin for each unique probability in g
       # This will be handled in the main calibration function
       return "finest"
   if epsilon >= 1:
       # When epsilon = 1, return single bin for all probabilities
       return [(0, 1)]
   bins = []
   i = 0
   while True:
       upper = (1-epsilon)**i
       lower = (1-epsilon)**(i+1)
       # If lower bound gets very small, make it 0 and make this our last bin
       if lower < 1e-10:
           bins.append((0, upper))
           break
       if upper - lower > 1e-10:
           bins.append((lower, upper))
       i += 1
   ##append final bin for edge case where everything is 1
   bins.append((1.0, 1.0))
   return bins


def miscalibration_calc(p_probs_normalized, g_probs_normalized, epsilon):
    bins = create_epsilon_induced_bins(epsilon)

    if isinstance(bins, str) and bins == "finest":
       unique_probs = sorted(set(g_probs_normalized), reverse=True)
       bins = [(p, p) for p in unique_probs]

    # Assign facts to bins and calculate sums
    binned_facts = [[] for _ in range(len(bins))]
    binned_p_sums = [0.0] * len(bins)
    binned_g_sums = [0.0] * len(bins)

    # Bin assignment
    for i, (p_val, g_val) in enumerate(zip(p_probs_normalized, g_probs_normalized)):
       assigned = False
       for bin_idx, (low, high) in enumerate(bins):
          if epsilon != 0:
               if low <= g_val < high:
                   binned_p_sums[bin_idx] += p_val
                   binned_g_sums[bin_idx] += g_val
                   assigned = True
                   break
          else:
               if low <= g_val <= high:
                   binned_p_sums[bin_idx] += p_val
                   binned_g_sums[bin_idx] += g_val
                   assigned = True
                   break
       if not assigned:
           last_idx = len(bins) - 1
           binned_p_sums[last_idx] += p_val
           binned_g_sums[last_idx] += g_val

    # Calculate miscalibration
    miscalibration = 0.5 * sum(abs(binned_p_sums[i] - binned_g_sums[i]) for i in range(len(bins)))
    return miscalibration


def miscalibration_analysis(model, alpha, epsilon):
# Evaluate miscalibration using p_datasets
##p calcs
  p_list = []
  for stmt in tqdm(p_datasets[alpha]["y"], desc = "Processing miscalibration"):
      # core = stmt.replace(special_token, "").strip().strip(',')
      p_list.append(stmt)
  p_counts = Counter(p_list)
  N = len(p_list)
  p_probs = [p_counts[stmt]/N for stmt in p_list]
  total_p = sum(p_probs)
  p_probs_normalized = [p / total_p for p in p_probs]
  counts = [p_counts[stmt] for stmt in p_list]
  print("p_probs_normalized completed calc")

##g calcs
  g_probs = []

  x_list = p_datasets[alpha]["x"].tolist()
  y_list = p_datasets[alpha]["y"].tolist()

  log_probs = batch_log_probability(model, tokenizer, x_list, y_list, device, 64)

  for log_prob in log_probs:
    g_probs.append(math.exp(log_prob))
  total_g = sum(g_probs)
  g_probs_normalized = [g / total_g for g in g_probs]
  print("g_probs_normalized completed calc")

  miscal_table = pd.DataFrame({
        "stmt": p_datasets[alpha]["y"],
        "counts": counts,
        "p": p_probs,
        "g": g_probs,
        "p_norm": p_probs_normalized,
        "g_norm": g_probs_normalized
    })

  miscal_table = miscal_table.sort_values(by="counts", ascending=False)
  ##miscal
  tvd = miscalibration_calc(p_probs_normalized, g_probs_normalized, epsilon)
  print(f"Miscalibration (TVD): {tvd}")
  return tvd, miscal_table


def tokenize_function(tokenizer, example):
    # Convert your BERT-style input into T5 format:
    # For one masked word, do:
    #   x: "Kpaveda, <extra_id_0>"
    #   y: "<extra_id_0> Word2"
    input_text = example["x"]
    target_text = example["y"]
    inputs = tokenizer(input_text, truncation=True)
    targets = tokenizer(target_text, truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

def custom_data_collator(features):
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
