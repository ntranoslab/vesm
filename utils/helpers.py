import torch
import os, json
import pickle
import random
import numpy as np

from esm.models.esmc import ESMC
from esm.models.esm3 import ESM3
from transformers import AutoTokenizer, EsmForMaskedLM

def set_all_seeds(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_model(model_name, inference=False):
    if model_name.startswith("esmc"):
        model = ESMC.from_pretrained(model_name)
        tokenizer = model.tokenizer
        embed_dim = model.embed.weight.shape[-1]
    elif model_name.startswith("esm3"):
        model = ESM3.from_pretrained("esm3_sm_open_v1")
        model = model.to(torch.float) 
        tokenizer = model.tokenizers.sequence
        embed_dim = model.encoder.sequence_embed.weight.shape[1]
    else:
        esm_name_dct = {
            'esm1b': 'esm1b_t33_650M_UR50S',
            'esm1v_1': 'esm1v_t33_650M_UR90S_1',
            'esm1v_2': 'esm1v_t33_650M_UR90S_2',
            'esm1v_3': 'esm1v_t33_650M_UR90S_3',
            'esm1v_4': 'esm1v_t33_650M_UR90S_4',
            'esm1v_5': 'esm1v_t33_650M_UR90S_5',
            'esm2_650m': 'esm2_t33_650M_UR50D',
            'esm2_3b': 'esm2_t36_3B_UR50D',
            'esm2_150m': 'esm2_t30_150M_UR50D',
            'esm2_8m': 'esm2_t6_8M_UR50D',
            'esm2_35m': 'esm2_t12_35M_UR50D',
            'esm2_15b': 'esm2_t48_15B_UR50D'
        }
        checkpoint = f'facebook/{esm_name_dct[model_name]}'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = EsmForMaskedLM.from_pretrained(checkpoint)
        embed_dim = model.esm.embeddings.word_embeddings.embedding_dim
    return model, tokenizer, embed_dim

def read_json(fpath):
    with open(fpath, 'r') as file:
        data = json.load(file)
    return data

def read_pkl(fpath):
    with open(fpath, 'rb') as fp:
        dct = pickle.load(fp)
    return dct

def dict2json(json_pth, dct):
    with open(json_pth, "w") as outfile:
        json.dump(dct, outfile, indent=2)

def try_makedir(fdir):
    if not os.path.exists(fdir):
        os.makedirs(fdir)

def write_pkl(dct, fpath):
    with open(fpath, 'wb') as fp:
        pickle.dump(dct, fp)
        

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeterList(object):
    def __init__(self, measure_dct):
        # measures: "avg", "cat", "list"
        self.measure_dct = measure_dct
        self.meter_lst = {}           
        self.reset()
    def reset(self):
        for m, metric_type in self.measure_dct.items():
            if metric_type == "avg":
                self.meter_lst[m] = AverageMeter()
            else:
                self.meter_lst[m] = []
    def update(self, val_dct, n=1):
        for m, metric_type in self.measure_dct.items():
            if metric_type == "avg":
                x = val_dct[m]
                x = x.item() if torch.is_tensor(x) else x
                self.meter_lst[m].update(x, n)
            elif metric_type == "cat":
                self.meter_lst[m].append(val_dct[m].cpu())
            elif metric_type == "list":
                self.meter_lst[m] += val_dct[m]
    def get_avg(self):
        return_dct = {}
        for m, metric_type in self.measure_dct.items():
            if metric_type == "avg":
                return_dct[m] = self.meter_lst[m].avg
            elif metric_type == "cat":
                return_dct[m] = torch.cat(self.meter_lst[m]).numpy()
            elif metric_type == "list":
                return_dct[m] = self.meter_lst[m]
        return return_dct
    
def get_stats(data_dct):
    n = 0; mu = 0; ss = 0
    min_e = 1000; max_e = -1000
    for _, arr in data_dct.items():
        size_protein = arr.shape[0] * arr.shape[1] # keep 0
        mu_protein = arr.mean()
        std_protein = arr.std()
        n += size_protein
        mu += mu_protein * size_protein
        ss_protein = (std_protein**2 + mu_protein**2) * size_protein
        ss += ss_protein
    # aggregate
    mu = mu / n
    variance = ss/n - mu**2
    return mu, np.sqrt(variance) #, min_e, max_e)

