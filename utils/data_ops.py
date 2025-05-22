import torch
import numpy as np
from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, data_arr, tok_vocabs, seq_length=1024, with_masked_data=False, normalization=False):
        self.data_arr = data_arr
        self.seq_length = seq_length
        self.mask_token = '<mask>'
        self.pad_token = '<pad>'
        self.tok_vocabs = tok_vocabs
        self.vocabs = list(self.tok_vocabs.keys())[4:24]
        self.with_masked_data = with_masked_data
        self.normalization = normalization

    def __len__(self):
        return len(self.data_arr)
    
    def __getitem__(self, item):
        wt_seq, llrs = self.data_arr[item]
        N, L = llrs.shape
        gap_size = self.seq_length - L
        prefix_zeros = np.zeros((N, 1))
        suffix_zeros = np.zeros((N, gap_size - 1))
        output = {"sequence": wt_seq}
        if self.normalization:
            normfunc = self.normfunc()
            llrs = normfunc(llrs)
        adjusted_llrs = -np.hstack([prefix_zeros, llrs, suffix_zeros]) # negate 
        output[f"llrs"] = torch.tensor(np.transpose(adjusted_llrs, (1, 0))).float()
        if self.with_masked_data:
            masked_seq, llm_labels = self.random_masking(wt_seq)
            output["masked_seq"] = masked_seq
            output["masked_label"] = torch.Tensor(llm_labels).long()  
        return output
    

def get_interval(one_based_position, seq_length, model_window = 1022):
    half_window = model_window // 2
    if seq_length <= model_window:
        return [0, model_window]
    p = one_based_position - 1
    k = (p // half_window) * half_window    
    if k < half_window:
        return [0, model_window]
    elif k + half_window > seq_length:
        return [max(0, k - half_window), seq_length]
    else:
        if p - k < k + half_window - p:
            s = max(0, k - half_window)
            e = min(seq_length, k + half_window)
        else:
            s = k
            e = min(seq_length, k + model_window)
        return [s, e]
    
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

def shift_mutant(mutant, start_pos):
    return f"{mutant[0]}{int(mutant[1:-1]) - start_pos + 1}{mutant[-1]}"

def shift_variant(variant, start_pos):
    if ':' in variant:
        mutants = variant.split(':')
        return ":".join([shift_mutant(mutant, start_pos) for mutant in mutants])
    else:
        return shift_mutant(variant, start_pos)
