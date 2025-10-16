import os, sys 
sys.path.append(os.path.dirname(os.getcwd()))
import logging
import torch
from torch.optim import AdamW
from tqdm import tqdm
import math
from utils.helpers import AverageMeterList, dict2json
from utils import constants as C

def split_decay(named_params):
    decay, no_decay = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if n.endswith("bias") or "LayerNorm.weight" in n or "layernorm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay

class ESMTrainerObject():
    def __init__(self, model, tokenizer, configs):
        self.model = model
        self.tokenizer = tokenizer
        self.device = self.model.device  
        self.configs = configs
        self.model_cfgs = configs["model"]
        self.train_cfgs = configs["training"]
        self.model_name = configs["model_name"]
        self.early_stops = self.train_cfgs["early_stopping"]
        self.measures = [m for m, t in self.model_cfgs["measure_dct"].items() if t == "avg"]
        self.model_type = self.model_name.split('_')[0].upper()
        self.model_path = f'{configs["model_dir"]}/v{self.model_name}.pth'
        log_fpath = f'{configs["model_dir"]}/log_{self.model_name}.out'
        score_fpath = f'{configs["model_dir"]}/score_{self.model_name}.json'

        logging.basicConfig(
            filename=log_fpath,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            force=True,
        )
        logging.getLogger().addHandler(logging.StreamHandler())
        self.log_fn = lambda message: logging.info(message)
        self.log_score_fn = lambda d: dict2json(score_fpath, d)
        self.save_score_fn = lambda d, suffix: dict2json(f'{configs["model_dir"]}/score_{self.model_name}_{suffix}.json', d)
            
    def set_loaders(self, loaders):
        self.loaders = loaders

    def get_param_dct(self):
        for param in self.model.parameters():
            param.requires_grad = False

        if self.model_type == 'ESM3':
            param_dct = {
                "head": self.model.output_heads.sequence_head,
                "nlayers": self.model.transformer.blocks,
                "norm": self.model.transformer.norm
            }
        elif self.model_type == 'ESMC':
            param_dct = {
                "head": self.model.sequence_head,
                "nlayers": self.model.transformer.blocks
            }
        else:
            param_dct = {
                "head": self.model.lm_head,
                "nlayers": self.model.esm.encoder.layer
            }
        return param_dct
    
    def get_trainable_parameters(self, param_dct, base_lr=2e-5, decay_factor=0.8, weight_decay=1e-2):
        groups = []
        for param_name, params in param_dct.items():
            if self.model_cfgs["esm_params"][param_name]:
                if param_name == "nlayers":
                    L = len(params)
                    print("Number of layers:", L)
                    k = self.model_cfgs["esm_params"]["nlayers"]
                    for i in range(L - k, L):  # last k layers
                        for param in params[i].parameters(): 
                            param.requires_grad = True
                        lr_i = base_lr * (decay_factor ** (L - 1 - i))
                        groups += [
                            {"params": params[i].parameters(), "lr": lr_i, "weight_decay": weight_decay},
                        ]
                else:
                    for param in params.parameters():
                        param.requires_grad = True
                    groups += [{"params": params.parameters(), "lr": base_lr, "weight_decay": weight_decay}]
        return groups
    
    def get_trainable_parameters_with_decay(self, param_dct, base_lr=2e-5, head_lr=5e-5, decay_factor=0.8, weight_decay=1e-2):
        groups = []
        for param_name, params in param_dct.items():
            if self.model_cfgs["esm_params"][param_name]:
                if param_name == "nlayers":
                    L = len(params)
                    k = self.model_cfgs["esm_params"]["nlayers"]
                    for i in range(L - k, L):  # last k layers
                        for param in params[i].parameters(): 
                            param.requires_grad = True
                        lr_i = base_lr * (decay_factor ** (L - 1 - i))
                        dec, nd = split_decay(params[i].named_parameters())
                        groups += [
                            {"params": dec, "lr": lr_i, "weight_decay": weight_decay},
                            {"params": nd,  "lr": lr_i, "weight_decay": 0.0},
                        ]
                else:
                    if param_name == "head":
                        for name, param  in params.named_parameters():
                            param.requires_grad = True

                        dec, nd = split_decay(params.named_parameters())
                        groups += [
                            {"params": dec, "lr": head_lr, "weight_decay": weight_decay},
                            {"params": nd,  "lr": head_lr, "weight_decay": 0.0},
                        ]
                    else:
                        for param in params.parameters():
                            param.requires_grad = True
                        groups += [{"params": list(params.parameters()), "lr": base_lr, "weight_decay": weight_decay}]
        return groups
    
    def get_llrd_groups(self, base_lr=2e-5, decay_rate=0.9, head_lr_mult=2.0, weight_decay=0.01):
        groups = []
        seen = set()

        def add_group(named_params, lr, wd):
            dec, nd = split_decay(named_params)
            dec = [p for p in dec if id(p) not in seen]
            nd  = [p for p in nd  if id(p) not in seen]
            seen.update(id(p) for p in dec)
            seen.update(id(p) for p in nd)
            if dec:
                groups.append({"params": dec, "lr": lr, "weight_decay": wd})
            if nd:
                groups.append({"params": nd,  "lr": lr, "weight_decay": 0.0})

        layers = self.model.esm.encoder.layer
        L = len(layers)

        # Embeddings (own the tied weight)
        emb_lr = base_lr * (decay_rate ** L)
        add_group(self.model.esm.embeddings.named_parameters(), emb_lr, weight_decay)

        # Encoder layers 0..L-1 (increasing LR toward last)
        for i in range(L):
            lr_i = base_lr * (decay_rate ** (L - 1 - i))
            add_group(layers[i].named_parameters(), lr_i, weight_decay)

        # LM head (dedup skips the tied weight if already added)
        head_lr = base_lr * head_lr_mult
        add_group(self.model.lm_head.named_parameters(), head_lr, weight_decay)

        return groups
    
    def init_training(self, niters, decay_rate = 0.9):
        base_lr = self.train_cfgs["optim"]['lr']
        weight_decay = self.train_cfgs["optim"]["weight_decay"]
        head_lr_mult = self.train_cfgs["optim"]["head_lr_mult"] 
        if self.model_cfgs["esm_params"]["full"]:
            groups = self.get_llrd_groups(base_lr, decay_rate, head_lr_mult, weight_decay)
        else:
            param_dct = self.get_param_dct()
            if self.train_cfgs["optim"]["split_decay"]:
                groups = self.get_trainable_parameters_with_decay(
                    param_dct, base_lr, base_lr * head_lr_mult, decay_rate, weight_decay)
            else:
                groups = self.get_trainable_parameters(
                    param_dct, base_lr, decay_rate, weight_decay)
        self.optimizer = AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

        total_steps = int(niters *  self.train_cfgs["nepochs"] * self.train_cfgs["optim"]["scheduler_factor"])
        warmup_steps = int(self.train_cfgs["optim"]["warmup_rate"] * total_steps)
        def cosine(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=cosine)

    def logit_fn_esm3(self, batch):
        input_src = self.train_cfgs["input_src"]
        seq_tokens = batch["sequence_tokens"].to(self.device)
        attention_masks = batch["attention_masks"].to(self.device) if "attention_masks" in batch else None
        results = {
            "sequence_logits": None,
            "input_ids": seq_tokens,
            "attenion_masks": attention_masks,
            "embeddings": None
        }
        if C.SEQ == input_src: # seq only
            outs = self.model.forward(sequence_tokens=seq_tokens, sequence_id=attention_masks)
        else:
            struct_tokens = batch["structure_tokens"].to(self.device)
            outs = self.model.forward(
                sequence_tokens=seq_tokens, 
                structure_tokens=struct_tokens, sequence_id=attention_masks)
            results["structure_logits"] = outs.structure_logits
        # sequence logits
        results["sequence_logits"] = outs.sequence_logits
        return results
    
    def logit_fn_esmc(self, batch):
        tokens = self.tokenizer(batch["sequence"], truncation=True, padding='longest', max_length=1024, return_tensors='pt')
        outputs = self.model.forward(**tokens.to(self.device))
        results = {
            "attention_masks": tokens["attention_mask"].to(self.device),
            "input_ids": tokens["input_ids"].to(self.device),
            "sequence_logits": outputs.sequence_logits.float(),
            "embeddings": outputs.embeddings.float()
        }
        return results

    def logit_fn_esm2(self, batch):
        tokens = self.tokenizer(batch["sequence"], truncation=True, padding='longest', max_length=1024, return_tensors='pt')
        outputs = self.model(**tokens.to(self.device), output_hidden_states=False)
        results = {
            "attention_masks": tokens["attention_mask"].to(self.device),
            "input_ids": tokens["input_ids"].to(self.device),
            "sequence_logits": outputs["logits"],
            "embeddings": None
        }
        return results

    def logits(self, batch):
        if 'ESM3' == self.model_type:
            return self.logit_fn_esm3(batch)
        elif 'ESMC' == self.model_type:
            return self.logit_fn_esmc(batch)
        else:
            return self.logit_fn_esm2(batch)

    def loss_fn(self, batch, training_scaler=False):
        pass

    def optimize(self, batch):
        self.model.train()  
        self.optimizer.zero_grad()
        loss_dct = self.loss_fn(batch)
        loss_dct['loss'].backward()
        if not(self.model_type == 'ESM3'):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) 
        self.optimizer.step()
        self.scheduler.step()
        return loss_dct
    
    def eval_epoch(self, loss_dct, return_score=False):
        pass

    def train_epoch(self, epoch, best_scores, log_freq):            
        min_loss, best_score = best_scores
        loss_train_dct = AverageMeterList(self.model_cfgs['measure_dct'])
        high_loss_count = 0
        for batch_idx, batch in enumerate(self.loaders['train']):
            self.model.train()
            loss_batch_dct = self.optimize(batch)
            loss_train_dct.update(loss_batch_dct, len(batch["row_id"]))

            if batch_idx % log_freq == 0:
                self.model.eval()
                self.log_fn("[{:02}/{}] ".format(epoch, batch_idx))
                loss_dct = {
                    "train": loss_train_dct.get_avg(),
                    "valid": self.evaluate(self.loaders['valid'])
                }
                for split, dct in loss_dct.items():
                    self.log_fn(f"\t {split}: " + " ".join(["{} {:.3f}".format(m, dct[m]) for m in self.measures]))
                
                val_loss = loss_dct["valid"]["loss"]
                post_fix = self.eval_epoch(loss_dct["valid"], return_score=False)
                score = post_fix["score"]
                post_fix.update({"epoch": epoch, "batch_idx": batch_idx}) 

                if val_loss < min_loss:
                    min_loss = val_loss
                    self.log_fn("== best validation loss!")
                    self.save_model("val")
                    self.save_score_fn(post_fix, "val")
                    high_loss_count = 0
                else:
                    if val_loss > loss_dct["train"]["loss"]:
                        gap = (val_loss - loss_dct["train"]["loss"])/loss_dct["train"]["loss"]
                        if gap >= 0.5:
                            self.log_fn("high val/train loss ratio")
                            high_loss_count += 1

                if score >= best_score + self.early_stops["max_delta"]:
                    best_score = score
                    self.log_fn("== best score!")
                    self.save_model()
                    self.log_score_fn(post_fix)
                    self.early_stops['counter'] = 0
                elif score < best_score - self.early_stops["min_delta"]:
                    self.early_stops['counter'] += 1

            if self.early_stops['counter'] >= self.early_stops['patience'] and epoch >= self.train_cfgs["minimum_epochs"]:
                self.log_fn("Early stopping triggered.")
                break
            if high_loss_count > self.early_stops['patience']:
                self.log_fn("Early stopping triggered with high validation loss.")
                break
            
        return min_loss, best_score

    def evaluate(self, val_loader):
        self.model.eval()
        losses_dct = AverageMeterList(self.model_cfgs['measure_dct'])
        with torch.no_grad():
            for batch in val_loader:
                loss_dct = self.loss_fn(batch)
                losses_dct.update(loss_dct, len(batch['row_id']))
        return losses_dct.get_avg()
    
    def load_model(self, model_path = None):
        if model_path is None:
            model_path = self.model_path

        if os.path.isfile(model_path):
            self.model.load_state_dict(torch.load(model_path), strict=False)
            print(f"===> Load model")
        else:
            print(f"===> No Ckt")

    def save_model(self, suffix=""):
        saved_dct = self.model.state_dict()
        if len(suffix) > 0:
            model_path = self.model_path.replace(".pth", f"_{suffix}.pth")
        else:
            model_path = self.model_path
        torch.save(saved_dct, model_path)