import json
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.EsmDistiller import ESMDistiller
from utils.data_ops import SeqDataset
from utils.helpers import load_model
import argparse

parser = argparse.ArgumentParser(description='VESM')
parser.add_argument("-g", "--gpu_id", default=0, type=int)
parser.add_argument("-t", "--testing", default=0, type=int)
parser.add_argument("-b", "--batch_size", default=16, type=int)
args = parser.parse_args()

##################### Helpers #####################
rng = np.random.RandomState(seed=123)
def get_loader(data_dct, tok_vocabs, batch_size):
    proteins = rng.permutation(list(data_dct.keys()))
    ntrains = int(len(proteins) * 0.9)
    def _load_data(protein_lst, shuffle=False):
        data_arr = [data_dct[uid] for uid in protein_lst]
        seq_dataset = SeqDataset(data_arr, tok_vocabs)
        return DataLoader(
            seq_dataset, batch_size=batch_size, shuffle=shuffle
        )  
    return {'train': _load_data(proteins[:ntrains], True), 
               'valid': _load_data(proteins[ntrains:])}

##################### Setting #####################
with open(f"configs.json") as f:
    configs = json.load(f)
model_name = configs["model"]["model_name"]
configs.update({
    "model_dir": f'{configs["save_dir"]}/{model_name}',
    "task_name": configs["task"]
})

##################### Model #####################
device = torch.device(f"cuda:{args.gpu_id}")
esm_model, tokenizer, _ = load_model(model_name)
trainer = ESMDistiller({"esm": esm_model.to(device)}, tokenizer, configs)

with open(configs["data"]["esmin"], 'rb') as fp:
    data_dct = pickle.load(fp) 
loaders = get_loader(data_dct, trainer.tokenizer.get_vocab(), args.batch_size)
# sample testing dictionary
trainer.set_loaders(loaders)
if trainer.best_scores is not None:
    score_dct = trainer.best_scores
else:
    score_dct = trainer.evaluate(loaders["valid"])
min_loss = score_dct['loss']

trainer.log_fn("Training: {} ---- loss {:.3f}".format(model_name, min_loss))
# train
trainer.init_training(len(loaders['train']))
for epoch in range(configs["training"]["nepochs"]):
    min_loss = trainer.train_epoch(epoch, min_loss)
trainer.log_fn("Min loss {:.3f} ".format(min_loss))
