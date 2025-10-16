import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

class ProteinSeqDataset(Dataset):
    def __init__(self, proteins, seq_dct):
        self.proteins = proteins
        self.seq_dct = seq_dct
    def __len__(self):
        return len(self.proteins)
    def __getitem__(self, idx):
        p = self.proteins[idx]
        return p, self.seq_dct[p]
    
class SeqDataset(Dataset):
    def __init__(self, data_dct, max_length=1024, protein_order=0):
        self.data_dct = data_dct
        self.seq_ids = list(self.data_dct.keys())
        self.seq_length = max_length
        if protein_order:
            mean_llrs = [self.data_dct[p]["llrs"].mean() for p in self.seq_ids]
            indices = np.argsort(mean_llrs)
            self.seq_ids = np.asarray(self.seq_ids)[indices]

    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, item):
        seq_id = self.seq_ids[item]
        dct = self.data_dct[seq_id]
        wt_seq, llrs = dct["sequence"], dct["llrs"]
        L, n_AAs = llrs.shape
        top_rows = torch.zeros((1, n_AAs))
        bottom_rows = torch.zeros((self.seq_length - L - 1, n_AAs))
        adjusted_llrs = torch.cat([top_rows, torch.Tensor(llrs), bottom_rows]).float()
        output = {
            "row_id": seq_id, 
            "sequence": wt_seq,
            "llrs": adjusted_llrs
        }
        return output


class TokenDataset(Dataset):
    def __init__(self, data_dct, max_length=1024, protein_order=0, using_structure = False):
        self.seq_ids = list(data_dct.keys())
        self.data_dct = data_dct
        self.max_length = max_length
        self.using_structure = using_structure
        VQVAE_CODEBOOK_SIZE = 4096
        self.STRUCTURE_PAD_TOKEN = VQVAE_CODEBOOK_SIZE + 3
        self.SEQUENCE_PAD_TOKEN = 1
        if protein_order:
            mean_llrs = [self.data_dct[p]["vesm_nllrs"].mean() for p in self.seq_ids]
            indices = np.argsort(mean_llrs)
            self.seq_ids = np.asarray(self.seq_ids)[indices]

    def __len__(self):
        return len(self.seq_ids)

    def pad_score(self, llrs):
        L, N = llrs.shape
        gap_size = self.max_length - L
        prefix_zeros = np.zeros((1, N))
        suffix_zeros = np.zeros((gap_size - 1, N))
        return np.vstack([prefix_zeros, llrs, suffix_zeros])

    def pad_token(self, tokens, pad_token_id):
        if len(tokens) < self.max_length:
            right_pad = self.max_length - len(tokens)
            padded_tokens = np.pad(
                tokens,
                (0, right_pad),
                'constant',
                constant_values=pad_token_id,
            )
            return padded_tokens[:self.max_length]
        return tokens
    
    def __getitem__(self, item):
        seq_id = self.seq_ids[item]
        dct = self.data_dct[seq_id]
        # sequence tokens
        seq_tokens = dct["esm3_tokens"]["sequence"]
        padded_seq_tokens = self.pad_token(seq_tokens, pad_token_id=self.SEQUENCE_PAD_TOKEN)
        # attention_masks
        attention_mask = np.zeros(len(padded_seq_tokens))
        attention_mask[:len(seq_tokens)] = 1
        output = {
            "row_id": seq_id,
            "llrs": torch.Tensor(self.pad_score(dct["vesm_nllrs"])).float(),
            "sequence_tokens": torch.Tensor(padded_seq_tokens).long(),
            "attention_masks": torch.Tensor(attention_mask).long()
        }
        if self.using_structure:
            struct_tokens = self.pad_token(dct["esm3_tokens"]["masked_structure"], pad_token_id=self.STRUCTURE_PAD_TOKEN)
            output["structure_tokens"] = torch.Tensor(struct_tokens).long()
        return output

def load_validation(valid_dct, batch_size):
    valid_dct["sequence_loader"] = DataLoader(
            ProteinSeqDataset(list(valid_dct["mutation_dict"].keys()), valid_dct["seq_dict"]), 
            batch_size=batch_size, 
            shuffle=False, num_workers=4, pin_memory=True)
    return valid_dct

def get_loaders(data_dct, dev_proteins, batch_size, val_batch_size, val_frac=0.1,  protein_order=0, min_proteins=200, token_data=False):
    _DS = TokenDataset if token_data else SeqDataset
    _load_data = lambda dct, bs, shuffle: DataLoader(
            _DS(dct, protein_order=protein_order), 
            batch_size=bs, 
            shuffle=shuffle,
            num_workers=4,  
            pin_memory=True,  
            persistent_workers=True  
        )  
    
    n = len(dev_proteins)
    nvals = max(min(min_proteins, int(n * val_frac)), min(10, n // 2)) 
    ntrains = n - nvals
    train_proteins = dev_proteins[:ntrains]
    valid_proteins = dev_proteins[ntrains:]
    train_dct = {uid: data_dct[uid] for uid in train_proteins}
    valid_dct = {uid: data_dct[uid] for uid in valid_proteins}
    return {
        'train': _load_data(train_dct, batch_size, protein_order == 0), 
        'valid': _load_data(valid_dct, val_batch_size, False)}



