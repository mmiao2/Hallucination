import pandas as pd
import numpy as np
import random
import os 
from collections import Counter
from scipy.stats import pareto
import scipy.stats as stats
import math
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import ast

# random.seed(1217)
def create_powerlaw_p(df, pareto_alpha, random_state=1217):
    base = df[['x','y','names','gold']].copy()
    rng = np.random.default_rng(random_state)
    # draw Pareto samples, floor them, and force at least 1 repetition
    reps = pareto.rvs(b=pareto_alpha, scale=1, size=len(base), random_state=rng)
    reps = np.floor(reps).astype(int)
    reps = np.maximum(reps, 1)
    return base.loc[base.index.repeat(reps)].reset_index(drop=True)

def sample(power_df, size):
    sampled_df = power_df.sample(n=size, replace=True, random_state=1217)
    print(f"Monofact in % sample is: {mono_calc(sampled_df)}")
    return sampled_df

def mono_calc(df):
    pairs = list(zip(df['x'], df['y'], df['names'], df['gold']))
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

def create_epsilon_induced_bins(epsilon):
   if epsilon < 0:
       raise ValueError("Epsilon must be non-negative")
   if epsilon == 0:
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

def tokenize_function(tokenizer, example):
    input_text = example["x"]
    target_text = example["y"]
    inputs = tokenizer(input_text, truncation=True)
    targets = tokenizer(target_text, truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

def custom_data_collator(features, tokenizer):
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

##############ANALYSIS FUNCTIONS##############
def miscalibration_calc(p_probs, g_probs, epsilon, alpha):
    # 1. Build epsilon-induced bins
    bins = create_epsilon_induced_bins(epsilon)
    if bins == "finest":
        unique_vals = sorted(set(g_probs), reverse=True)
        bins = [(v, v) for v in unique_vals]

    # 2. Initialize per-bin sums
    binned_p = [0.0] * len(bins)
    binned_g = [0.0] * len(bins)

    # 3. Assign each (p,g) to a bin (with fall-through to last bin)
    for p, g in zip(p_probs, g_probs):
        placed = False
        for idx, (low, high) in enumerate(bins):
            if epsilon != 0:
                if low <= g < high:
                    binned_p[idx] += p
                    binned_g[idx] += g
                    placed = True
                    break
            else:
                # when epsilon == 0, make upper bound inclusive
                if low <= g <= high:
                    binned_p[idx] += p
                    binned_g[idx] += g
                    placed = True
                    break
        if not placed:
            # dump into last bin
            binned_p[-1] += p
            binned_g[-1] += g

    # 4. Compute per-bin |Δ| and total miscalibration
    miscal_bins = [abs(p_sum - g_sum) for p_sum, g_sum in zip(binned_p, binned_g)]
    total_miscal = 0.5 * sum(miscal_bins)

    # 5. Build summary DataFrame
    df = pd.DataFrame({
        "alpha":        [alpha] * len(bins),
        "bin":    list(range(len(bins))),
        "p_sum":        binned_p,
        "g_sum":        binned_g,
        "miscal_bin":   miscal_bins,
        "total_miscal": [total_miscal] * len(bins),
    })

    return df

def regret_calc(p_probs, g_probs, epsilon):
    print("Conducting KL divergence analysis...")
    bins = create_epsilon_induced_bins(epsilon)
    if bins == "finest":
        unique = sorted(set(g_probs), reverse=True)
        bins = [(u,u) for u in unique]
    bin_edges = np.array([(low, high) for low,high in bins])
    gs = np.array(g_probs)
    ps = np.array(p_probs)
    regrets = []
    for i,(low,high) in enumerate(bin_edges):
        mask = (gs>=low) & (gs<(high if epsilon!=0 else high+1e-12))
        p_sum = ps[mask].sum()
        g_sum = gs[mask].sum()
        if p_sum == 0:
            r = 0.0
        elif g_sum == 0:
            r = float('inf')
        else:
            r = p_sum * math.log(p_sum/g_sum)
        regrets.append(r)
    total = sum(regrets)
    df = pd.DataFrame({
        "bin":    list(range(len(bins))),
        "regret": regrets,
        "total_regret": [total]*len(bins)
    })
    return df

def batched(it, n):
    """Yield successive n-sized chunks from iterable it."""
    it = iter(it)
    from itertools import islice
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


def miscalibration_analysis(p_datasets, train_datasets, model, tokenizer, alpha, epsilon, device):
    print("Conducting miscalibration analysis...")
    # normalize true distributions
    def compute_norm_probs(stmts):
        counts = Counter(stmts)
        N = len(stmts)
        probs = np.array([counts[s] / N for s in stmts])
        return probs / probs.sum()

    p_stmts = p_datasets[alpha]["y"].tolist()
    train_p_stmts = train_datasets[alpha]["y"].tolist()
    p_probs = compute_norm_probs(p_stmts)
    train_p_probs = compute_norm_probs(train_p_stmts)

    x_list, y_list = p_datasets[alpha]["x"].tolist(), p_datasets[alpha]["y"].tolist()
    train_x, train_y = train_datasets[alpha]["x"].tolist(), train_datasets[alpha]["y"].tolist()

    model.eval()
    torch.cuda.empty_cache()
    pbar = tqdm(total=2, desc="Log-probs")
    with torch.no_grad():
        g_log_probs = batch_log_probability(model, tokenizer, x_list, y_list, device, batch_size=16)
        pbar.update(1)
        train_g_log_probs = batch_log_probability(model, tokenizer, train_x, train_y, device, batch_size=16)
        pbar.update(1)
    pbar.close()
    # model.train()
    g_probs       = np.exp(g_log_probs);       g_probs       /= g_probs.sum()
    train_g_probs = np.exp(train_g_log_probs); train_g_probs /= train_g_probs.sum()
    miscal_df  = miscalibration_calc(p_probs, g_probs, epsilon, alpha)
    regret_df  = regret_calc( train_p_probs, train_g_probs, epsilon)
    merged_df  = pd.merge(miscal_df, regret_df, on="bin", how="left")

    tvd    = miscal_df["total_miscal"].iat[0]
    regret = regret_df["total_regret"].iat[0]
    return tvd, regret, merged_df

def dedupe_by_y(ds):
    """
    ds: either a dict-like {'x','y','gold',...} or a Dataset
    returns: a dict with only unique 'y' entries kept,
             and parses 'gold' strings into real lists.
    """
    # determine columns
    if hasattr(ds, "column_names"):
        cols = ds.column_names
    else:
        cols = list(ds.keys())

    seen = set()
    out = {k: [] for k in cols}
    for row in zip(*[ds[k] for k in cols]):
        y = row[cols.index("y")]
        if y in seen:
            continue
        seen.add(y)
        for k, val in zip(cols, row):
            if k == "gold" and isinstance(val, str):
                # parse the list‐literal string into an actual Python list
                out[k].append(ast.literal_eval(val))
            else:
                out[k].append(val)
    return out

def inaccuracy_analysis(model, train_dataset_final, tokenizer, batch_size, alpha, device):
    """
    Token‐level attribute eval over *unique* y’s:
      • returns (avg attribute‐loss, avg attribute‐inaccuracy)
      • writes per‐example CSV aligned to the deduped set
    """
    print("Conducting token‐level attribute analysis…")
    mini = dedupe_by_y(train_dataset_final)
    xs, ys, golds = mini['x'], mini['y'], mini['gold']
    n = len(ys)

    # Pre‐tokenize all gold attributes (skip the name at index 0)
    gold_vids = [
        [tokenizer(val, add_special_tokens=False)["input_ids"]
         for val in gold_list[1:]]
        for gold_list in golds
    ]

    per_loss = [None] * n
    per_acc  = [None] * n

    model.to(device).eval()
    for i in tqdm(range(0, n, batch_size), desc="Attr eval", unit="batch"):
        bs      = min(batch_size, n - i)
        batch_x = xs[i : i + bs]
        batch_y = ys[i : i + bs]

        inp = tokenizer(batch_x, return_tensors="pt",
                        padding=True, truncation=True).to(device)
        tgt = tokenizer(batch_y, return_tensors="pt",
                        padding=True, truncation=True).to(device)

        with torch.no_grad():
            logits   = model(**inp, labels=tgt["input_ids"]).logits
            logp     = torch.log_softmax(logits, dim=-1)  # [bs, seq, vocab]
            pred_ids = logp.argmax(dim=-1)                # [bs, seq]

        tgt_ids = tgt["input_ids"]
        for b in range(bs):
            idx       = i + b
            ids_list  = tgt_ids[b].tolist()
            lp_tensor = logp[b]
            vids_list = gold_vids[idx]

            # attribute‐loss
            total_lp, total_tok = 0.0, 0
            for vid in vids_list:
                L = len(vid)
                for k in range(len(ids_list) - L + 1):
                    if ids_list[k:k+L] == vid:
                        for j, tok in enumerate(vid):
                            total_lp += lp_tensor[k+j, tok].item()
                        total_tok += L
            ex_loss = -total_lp / total_tok if total_tok > 0 else None

            # attribute‐accuracy
            corr = 0
            for vid in vids_list:
                L = len(vid)
                ok = any(
                    ids_list[k:k+L]    == vid and
                    pred_ids[b, k:k+L].tolist() == vid
                    for k in range(len(ids_list) - L + 1)
                )
                corr += ok
            ex_acc = corr / len(vids_list) if vids_list else 0.0

            per_loss[idx] = ex_loss
            per_acc[idx]  = ex_acc

    # compute averaged metrics
    losses    = [l for l in per_loss if l is not None]
    accs      = [a for a in per_acc  if a is not None]
    avg_loss  = sum(losses) / len(losses) if losses else 0.0
    avg_acc   = sum(accs)   / len(accs)   if accs    else 0.0
    avg_inacc = 1.0 - avg_acc

    # save per‐example CSV
    out_df = pd.DataFrame({
        "x":               xs,
        "y":               ys,
        "gold":            golds,
        "attr_loss":       per_loss,
        "attr_accuracy":   per_acc,
        "attr_inaccuracy":[None if a is None else 1.0-a for a in per_acc]
    })
    print(f"Saving attribute analysis to CSV for alpha {alpha}...")
    out_df.to_csv(f"TBU", index=False)

    return avg_loss, avg_inacc

def hallucination_analysis(model, train_dataset_final, tokenizer, batch_size, alpha, device):
    """
    Free-run hallucination metric with speedups:
      - cache gold lookup & prompt tokens
      - use torch.inference_mode & AMP
      - reuse prompt_ids/attention_mask
      - vectorized post-processing
    """
    # cache name→gold mapping
    if not hasattr(hallucination_analysis, "_name_to_gold") or hallucination_analysis._cached_ds is not train_dataset_final:
        raw = [ast.literal_eval(g) if isinstance(g, str) else g for g in train_dataset_final["gold"]]
        hallucination_analysis._name_to_gold = {g[0]: g for g in raw}
        hallucination_analysis._cached_ds = train_dataset_final
    name_to_gold = hallucination_analysis._name_to_gold

    # cache prompt tokens
    prompt = "<BIOGRAPHY>"
    if not hasattr(hallucination_analysis, "_prompt"):
        toks = tokenizer(prompt, return_tensors="pt", truncation=True)
        hallucination_analysis._prompt = {
            "input_ids": toks.input_ids.to(device),
            "attention_mask": toks.attention_mask.to(device)
        }
    prompt_inputs = hallucination_analysis._prompt

    model.to(device).eval()
    gens = []
    n = len(train_dataset_final["gold"])
    for i in range(0, n, batch_size):
        bs = min(batch_size, n - i)
        inp = {
            "input_ids": prompt_inputs["input_ids"].repeat(bs, 1),
            "attention_mask": prompt_inputs["attention_mask"].repeat(bs, 1)
        }
        with torch.inference_mode(), torch.cuda.amp.autocast():
            outs = model.generate(
                **inp,
                do_sample=True,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=128
            )
        gens += tokenizer.batch_decode(outs, skip_special_tokens=True)

    # extract names, lookup golds, compute hall rates
    names = [(" ".join(g.split()[:2])).rstrip("'s") for g in gens]
    golds_list = [name_to_gold.get(n, [None]*7)[1:] for n in names]
    hall_rates = []
    total_incorrect = total_attrs = 0
    for gen, attrs in zip(gens, golds_list):
        correct = sum(1 for a in attrs if a and a in gen)
        incorrect = len(attrs) - correct
        hall_rates.append(incorrect / len(attrs))
        total_incorrect += incorrect
        total_attrs += len(attrs)

    # save results
    df = pd.DataFrame({
        "predicted_name": names,
        "generation":     gens,
        "gold":           golds_list,
        "hall_rate":      hall_rates
    })
    print(f"Saving hallucination analysis to CSV for alpha {alpha}...")
    df.to_csv(f"TBU", index=False)
    hall_rate = total_incorrect / total_attrs if total_attrs else 0.0
    print(f"hallucination rate is:{hall_rate}")
    return hall_rate

