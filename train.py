
import os
import torch
import numpy as np
from utils import constants as C
from utils.helpers import *
from utils.data_utils import get_loaders, load_validation
import argparse

parser = argparse.ArgumentParser(description='VESM')
parser.add_argument("-g", "--gpu_id", default=0, type=int)
parser.add_argument("-m", "--model_name", default="esm2_650m", type=str)
parser.add_argument("-c", "--config", default="esmin", type=str)
parser.add_argument("-b", "--batch_size", default=8, type=int)
parser.add_argument("-v", "--val_batch_size_factor", default=4, type=int)
args = parser.parse_args()

if args.model_name == "esm3":
    from modelling.EsmCoDistiller import ESM3Distiller as ESMCoDistiller
else:
    from modelling.EsmCoDistiller import ESMCoDistiller

##################### Settings #####################
data_dir = "data"
configs = read_json(f"configs/{args.config}.json")
seed = configs['setting']['seed']
set_all_seeds(seed)
rng = np.random.RandomState(seed=seed)

data_version = configs["setting"]["data_name"]
configs.update({
    "model_name": args.model_name,
    "model_dir": f"checkpoints/{args.config}", 
    "testing": 0,
})

if data_version in C.llrs_stats_dcts:
    stat_dct = C.llrs_stats_dcts[data_version]
    if args.model_name in stat_dct:
        configs["setting"].update({
            "mu_base": stat_dct[args.model_name],
            "mu_target": stat_dct["mu"],
            "sig": stat_dct["sig"]
        })
try_makedir(configs["model_dir"])
##################### Model #####################
torch.backends.cudnn.benchmark = True
device = torch.device(f"cuda:{args.gpu_id}")
esm_model, tokenizer, _ = load_model(args.model_name)


if len(configs["setting"]["pretrained_dir"]) > 0:
    pretrained_fpath = f'checkpoints/{configs["setting"]["pretrained_dir"]}/v{args.model_name}.pth'
    print("Load pretrained model: ", pretrained_fpath)
    missing, unexpected = esm_model.load_state_dict(torch.load(pretrained_fpath), strict=False)
    print("Unexpected (should be empty):", unexpected)
trainer = ESMCoDistiller(esm_model.to(device), tokenizer, configs)

##################### DATA #####################
data_dct = read_pkl(f"{data_dir}/train/{data_version}.pkl")

proteins = rng.permutation(list(data_dct.keys()))
val_bs = args.batch_size * args.val_batch_size_factor
loaders = get_loaders(data_dct, proteins, args.batch_size, val_bs, val_frac=0.1,  protein_order=configs["setting"]["data_order"], min_proteins=200, token_data=args.model_name == "esm3")
if args.model_name == "esm3":
    loaders["val"] = read_pkl(f"{data_dir}/train/Token_ESM11_UPh_base_val.pkl")
else:
    valid_dct = read_pkl(f"{data_dir}/train/ESM11_UPh_base_val.pkl")
    loaders["val"] = load_validation(valid_dct, val_bs)
trainer.set_loaders(loaders)

##################### Training #####################
trainer.log_fn(f"Training {args.model_name} on {data_version} for {configs['training']['nepochs']} epochs")
trainer.log_fn("Evaluating base model ...")
score_dct = trainer.eval_epoch(trainer.evaluate(loaders["valid"]))
min_loss = score_dct['loss']
best_score = score_dct["score"]

trainer.log_fn(f"\t loss {min_loss:.3f} approx-corr {best_score:.3f}")
trainer.init_training(len(loaders['train']))
for epoch in range(configs["training"]["nepochs"]):
    min_loss, best_score = trainer.train_epoch(epoch, [min_loss, best_score], log_freq=configs["training"]["log_freq"])
    if trainer.early_stops['counter'] >= trainer.early_stops['patience'] and epoch >= trainer.train_cfgs["minimum_epochs"]:
        break
    print(f"End of epoch {epoch} training")

trainer.log_fn(f"Best score {best_score:.3f} ")
