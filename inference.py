import torch, os, sys
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils import constants as C
from utils.helpers import load_model, try_makedir
from utils.data_utils import ProteinSeqDataset
from utils.seq_ops import get_interval
from torch.utils.data import DataLoader
from torch.nn import functional as F

import argparse
parser = argparse.ArgumentParser(description='ESM')
parser.add_argument("-g", "--gpu_id", default=0, type=int)
parser.add_argument("-m", "--model_name", default='esm2_8m', type=str)
parser.add_argument("-c", "--ckt", default='esm', type=str)
parser.add_argument("-b", "--batch_size", default=64, type=int)
parser.add_argument("-d", "--data", default='ClinVar', type=str)
args = parser.parse_args()

save_dir = "checkpoints"
result_dir = "results"
data_dir = "data/benchmarks"

##### Models ###########
device = torch.device(f"cuda:{args.gpu_id}")
esm_model, tokenizer, _ = load_model(args.model_name)
esm_model = esm_model.to(device)
sequence_vocabs = tokenizer.get_vocab()

ckt_fpath = f"{save_dir}/{args.ckt}"
if os.path.isfile(ckt_fpath):
    esm_model.load_state_dict(torch.load(ckt_fpath), strict=False)
    print("=== Loading checkpoint")
    save_path = f"{result_dir}/{args.data}_v{args.model_name}"
else:
    print("No checkpoint, running on base model!!")
    save_path = f"{result_dir}/{args.data}_base_{args.model_name}"

def inference_fn(sequence):
    tokens = tokenizer([sequence], return_tensors="pt").to(device)
    with torch.no_grad():
        outs = esm_model(**tokens)
    return outs["logits"][0], tokens["input_ids"][0]

def get_batch_llrs(logits, input_ids, num_classes=33):
    wt_positions = F.one_hot(input_ids, num_classes=num_classes)
    wt_probs = logits * wt_positions
    wt_probs = wt_probs.sum(dim=-1, keepdim=True)
    batch_llrs = logits - wt_probs.expand(logits.shape)
    return batch_llrs

def score_mutations(llrs, mutations):
    pred_scores = []
    for mutation in mutations:
        mutation_score = 0
        for mut in mutation.split(":"):
            pos = int(mut[1:-1]) # 1-based
            mt = C.sequence_vocabs[mut[-1]]
            pred = llrs[pos, mt] 
            mutation_score += pred.item()
        pred_scores.append(mutation_score)
    return pred_scores

def scoring(sequence, mutations):
    logits, input_ids = inference_fn(sequence)
    probs = torch.log_softmax(logits, dim=-1)
    wt_probs = probs[range(len(logits)),input_ids]
    llrs = probs - wt_probs.reshape(-1, 1)
    return score_mutations(llrs, mutations)

    
def partition(sequence, positions):
    L = len(sequence)
    partitions = {}
    for p in positions:
        s, e = get_interval(p, L)
        if s not in partitions:
            partitions[s] = {
                'seq': sequence[s:e], 
                'positions': [], 
                'start_index': s
            }
        partitions[s]['positions'].append(p)
    return partitions


def shift_mutation(mutation, start_pos):
    return f"{mutation[0]}{int(mutation[1:-1]) - start_pos + 1}{mutation[-1]}"

def shift_variant(variant, start_pos):
    if ':' in variant:
        mutations = variant.split(':')
        return ":".join([shift_mutation(mutation, start_pos) for mutation in mutations])
    else:
        return shift_mutation(variant, start_pos)
    
##### DATA ###########
if args.data == C.__DMS__:
    cfgs = {
        "seq_id": "DMS_id",
        "seq_name": "target_seq",
        "label": "DMS_score"
    }
    ref_fname= "DMS_ref_positions.csv"
    data_fname = 'DMS_variant_data.csv'
elif "ClinVar" in args.data:
    cfgs = {
        "seq_id": "protein",
        "seq_name": "wt_sequence",
        "label": "clinvar_label"
    }
    ref_fname = "ClinVar_sequences.csv"
    data_fname = "ClinVar_variants.csv"
    if args.data == C.__BalancedClinVar__:
        data_fname = "Balanced" + data_fname
        ref_fname = "Balanced" + ref_fname
else:
    print("Not found data")
    sys.exit()

# loading data
variant_df = pd.read_csv(f"{data_dir}/{data_fname}")
ref_df = pd.read_csv(f"{data_dir}/{ref_fname}")
protein_list = variant_df[cfgs["seq_id"]].unique() 

seq_dct = {row[cfgs["seq_id"]]: row[cfgs["seq_name"]] for _, row in ref_df.iterrows()}
max_length = 1022
short_seq_ids = [k for k, seq in seq_dct.items() if len(seq) <= max_length]
long_seq_ids = [k for k in list(seq_dct.keys()) if k not in short_seq_ids]

##### inference ###########
score_dct = {}
predictions = []

seq_lens = [len(seq_dct[p]) for p in short_seq_ids]
indices = np.argsort(seq_lens)
short_seq_ids = np.asarray(short_seq_ids)[indices]

loader = DataLoader(ProteinSeqDataset(short_seq_ids, seq_dct), 
    batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

try:
    short_proteins = []
    with torch.inference_mode():
        for batch in tqdm(loader):
            batch_proteins, seqs = batch
            tokens = tokenizer(list(seqs), truncation=True, padding='longest', max_length=1024, return_tensors='pt').to(device)
            outs = esm_model(**tokens)
            # If get_llrs can be vectorized, do it here:
            batch_llrs = get_batch_llrs(outs["logits"], tokens["input_ids"]).cpu().numpy()
            for k, p in enumerate(batch_proteins):
                l = len(seqs[k])
                llrs = batch_llrs[k][:l+1]
                mutations = variant_df[variant_df[cfgs["seq_id"]] == p].mutation.values
                pred_scores = score_mutations(llrs, mutations)
                for mutation, score in zip(mutations, pred_scores):
                    predictions.append([p, mutation, score])

    for protein in tqdm(long_seq_ids):
        protein_df = variant_df[variant_df[cfgs["seq_id"]] == protein].copy()
        wt_seq = seq_dct[protein]
        postision_values = protein_df.mutation.apply(lambda x: int(x[1:-1])).values
        protein_df.loc[:, "position"] = postision_values
        positions = sorted(protein_df['position'].unique())
        partitions = partition(wt_seq, positions)
        mutations = []
        pred_scores = []
        for start_pos, partition_dct in partitions.items():
            paritition_df = protein_df[protein_df["position"].isin(partition_dct['positions'])]
            shifted_mutations = paritition_df["mutation"].apply(lambda x: shift_variant(x, start_pos + 1))
            preds = scoring(partition_dct["seq"], shifted_mutations)
            mutations += list(paritition_df["mutation"].values)
            pred_scores += list(preds)
        # append
        for mutation, score in zip(mutations, pred_scores):
            predictions.append([protein, mutation, score])
except Exception as e:
    print("Running error", e)
    from IPython import embed; embed()


try:
    score_df = pd.DataFrame(predictions, columns=["protein", "mutation", args.model_name])
    score_df["variant"] = score_df.protein + ':' + score_df.mutation
    score_df.to_csv(f"{save_path}.csv.gz", compression='gzip' , index=False)
except Exception as e:
    print("Saving error", e)
    from IPython import embed; embed()