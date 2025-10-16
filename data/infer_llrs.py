
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

from tqdm import tqdm
import numpy as np
from utils.helpers import load_model
from utils.helpers import write_pkl, try_makedir, read_pkl
from utils.seq_ops import *
from utils.data_utils import ProteinSeqDataset
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

import argparse
parser = argparse.ArgumentParser(description='ESM')
parser.add_argument("-g", "--gpu_id", default=0, type=int)
parser.add_argument("-m", "--model_name", default='esm2_8m', type=str)
parser.add_argument("-d", "--data", default='UPh', type=str, choices=["UPh", "UPnh"])
parser.add_argument("-r", "--round_name", default='', type=str)
parser.add_argument("-b", "--batch_size", default=64, type=int)
parser.add_argument("-p", "--partition", default=0, type=int)
# over 4
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu_id}")
esm_model, tokenizer, _ = load_model(args.model_name)
esm_model = esm_model.to(device)

if args.round_name == "base":
    print("=== BASE ===")
    save_dir = "base_llrs"
else:
    save_dir = f"{args.data}/{args.round_name}"
    ckt = f"../checkpoints/{args.round_name}/v{args.model_name}.pth"
    if os.path.isfile(ckt):
        esm_model.load_state_dict(torch.load(ckt), strict=False)
        print("Load checkpoint")
    else:
        print("No checkpoint", ckt)
        sys.exit(0)
    print(args.model_name, ckt)

try_makedir(save_dir)
fname = f"{args.model_name}_{args.data}"
data_dct = read_pkl(f"train/ESM11_{args.data}_min.pkl") 
if args.data == "UPnh": # nohuman data
    range_lst = [0, 6000, 10000, 13000]
else:
    range_lst = [0, 7000, 13000, 17576]

seq_dct = {p: d["sequence"] for p, d in data_dct.items()}
proteins = list(seq_dct.keys())
seq_lens = [len(seq_dct[p]) for p in proteins]
indices = np.argsort(seq_lens)
proteins = np.asarray(proteins)[indices]

partitions = {}
for i, s in enumerate(range_lst):
    if i < len(range_lst) - 1:
        e = range_lst[i+1]
        partitions[i + 1] = proteins[s:e]
    else:
        partitions[i + 1] = proteins[s:]
if args.partition > 0:
    partitions = {args.partition: partitions[args.partition]}
    fname += f"_p{args.partition}"

def get_batch_llrs(logits, input_ids, num_classes=33):
    wt_positions = F.one_hot(input_ids, num_classes=num_classes)
    wt_probs = logits * wt_positions
    wt_probs = wt_probs.sum(dim=-1, keepdim=True)
    batch_llrs = logits - wt_probs.expand(logits.shape)
    return batch_llrs

def process_proteins(proteins):
    llr_dct = {}
    loader = DataLoader(ProteinSeqDataset(proteins, seq_dct), 
                        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    with torch.inference_mode():
        for batch in tqdm(loader):
            batch_proteins, seqs = batch
            tokens = tokenizer(list(seqs), truncation=True, padding='longest', max_length=1024, return_tensors='pt').to(device)
            outs = esm_model(**tokens)
            batch_llrs = get_batch_llrs(outs["logits"], tokens["input_ids"]).cpu().numpy()
            for k, p in enumerate(batch_proteins):
                l = len(seqs[k])
                llr_dct[p] = batch_llrs[k][1:l+1, 4:24]
    return llr_dct

llr_dct = {}
for p, seqs in partitions.items():
    print(f"Proccessing parition {p}: {len(seqs)} proteins .... ")
    dct = process_proteins(seqs)
    llr_dct.update(dct)

print("Writing ...")
write_pkl(llr_dct, f"{save_dir}/llrs_{fname}.pkl")
