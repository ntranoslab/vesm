
import torch, os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.data_ops import partition, shift_variant
from utils.helpers import load_model
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import argparse
parser = argparse.ArgumentParser(description='ESM')
parser.add_argument("-g", "--gpu_id", default=0, type=int)
parser.add_argument("-m", "--model_name", default='esm1b', type=str)
parser.add_argument("-c", "--ckt", default='vesm1', type=str)
parser.add_argument("-d", "--data", default='DMS', type=str)
args = parser.parse_args()

save_dir = "checkpoints"
data_dir = "data"
result_dir = "results"

device = torch.device(f"cuda:{args.gpu_id}")
esm_model, tokenizer, _ = load_model(args.model_name)
esm_model = esm_model.to(device)
sequence_vocabs = tokenizer.get_vocab()

ckt_fpath = f"{save_dir}/{args.model_name}/{args.ckt}.pth"
if os.path.isfile(ckt_fpath):
    esm_model.load_state_dict(torch.load(ckt_fpath), map_location=f"cuda:{args.gpu_id}")
    print("=== Loading checkpoint")
    save_path = f"{result_dir}/{args.data}_{args.model_name}_{args.ckt}"
else:
    save_path = f"{result_dir}/{args.data}_{args.model_name}_base"

def scoring(sequence, mutants):
    tokens = tokenizer([sequence], return_tensors="pt").to(device)
    with torch.no_grad():
        outs = esm_model(**tokens)
    logits = outs["logits"][0]
    probs = torch.log_softmax(logits, dim=-1)
    wt_probs = probs[range(len(logits)), tokens["input_ids"][0]]
    negllrs = probs - wt_probs.reshape(-1, 1)
    pred_scores = []
    for mutant in mutants:
        mutant_score = 0
        for mut in mutant.split(":"):
            pos = int(mut[1:-1]) # 1-based
            mt = sequence_vocabs[mut[-1]]
            pred = negllrs[pos, mt] 
            mutant_score += pred.item()
        pred_scores.append(mutant_score)
    return pred_scores

# DATA
data_prefix = f"{data_dir}/{args.data}/{args.data}"
if args.data == 'DMS':
    seq_id = "DMS_id"
    seq_name = "target_seq"
    eval_fn = lambda targets, preds: spearmanr(targets, -np.asarray(preds))[0]
    position_df = pd.read_csv(f"{data_prefix}_references.csv")
    variant_df = pd.read_csv(f"{data_prefix}_variants.csv.gz")
elif 'ClinVar' in args.data:
    seq_id = "protein"
    seq_name = "wt_seq"
    eval_fn = lambda targets, preds: roc_auc_score(targets, np.asarray(preds))
    variant_df = pd.read_csv(f"{data_prefix}_variants.csv")
    position_df = pd.read_csv(f"{data_prefix}_sequences.csv")
else:
    raise ValueError(f"Unknown dataset: {args.data}")

# partitioning sequences
proteins = variant_df[seq_id].unique()
seq_dct = {row[seq_id]: row[seq_name] for _, row in position_df.iterrows()}
position_df["seq_len"] = position_df[seq_name].apply(len)
protein_lst = {
    "short": position_df[position_df.seq_len <= 1022][seq_id].values,
    "long": position_df[position_df.seq_len > 1022][seq_id].values
} 
data_dfs = {
    "short": variant_df[variant_df[seq_id].isin(protein_lst["short"])],
    "long": variant_df[variant_df[seq_id].isin(protein_lst["long"])]
}

score_dct = {}
predictions = []
for protein in tqdm(protein_lst["short"]):
    df = variant_df[variant_df[seq_id] == protein]
    mutants = df.mutant.values
    pred_scores = scoring(seq_dct[protein], mutants)
    for mutant, score in zip(mutants, pred_scores):
        predictions.append([protein, mutant, score])

data_dfs["long"]["position"] = data_dfs["long"].mutant.apply(lambda x: int(x[1:-1]))
for protein in tqdm(protein_lst["long"]):
    df = data_dfs["long"][data_dfs["long"][seq_id] == protein]
    positions = sorted(df['position'].unique())
    partitions = partition(seq_dct[protein], positions)
    mutants = []
    pred_scores = []
    for start_pos, dct in partitions.items():
        sub_df = df[df.position.isin(dct['positions'])]
        shifted_mutants = sub_df.mutant.apply(lambda x: shift_variant(x, start_pos + 1))
        preds = scoring(dct["seq"], shifted_mutants)
        mutants += list(sub_df.mutant.values)
        pred_scores += list(preds)
        
    for mutant, score in zip(mutants, pred_scores):
        predictions.append([protein, mutant, score])

score_df = pd.DataFrame(predictions, columns=["protein", "mutant", args.model_name])
score_df["variant"] = score_df.protein + ':' + score_df.mutant
score_df.to_csv(f"{save_path}.csv", index=False)