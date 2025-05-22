import torch
import os, json
from esm.models.esmc import ESMC
from esm.models.esm3 import ESM3
from transformers import AutoTokenizer, EsmForMaskedLM

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

def dict2json(json_pth, dct):
    with open(json_pth, "w") as outfile:
        json.dump(dct, outfile)

def logging(logf, msg, console_print=True):
    logf.write(msg + '\n')
    if console_print:
        print(msg)

def try_makedir(fdir):
    if not os.path.exists(fdir):
        os.makedirs(fdir)