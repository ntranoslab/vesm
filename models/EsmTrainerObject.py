import os
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from utils.helpers import dict2json, read_json, logging
from utils import constants as C

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
    def __init__(self, measures):
        self.measures = measures
        self.meter_lst = {}           
        self.reset()

    def reset(self):
        for m in self.measures:
            self.meter_lst[m] = AverageMeter()
            
    def update(self, val_dct, n=1):
        for m in self.measures:
            self.meter_lst[m].update(val_dct[m].item(), n)

    def get_avg(self):
        return_dct = {}
        for m in self.measures:
            return_dct[m] = self.meter_lst[m].avg
        return return_dct
    
class ESMTrainerObject():
    def __init__(self, models, tokenizer, configs):
        self.models = models 
        self.tokenizer = tokenizer
        self.model_cfgs = configs["model"]
        self.train_cfgs = configs["training"]
        self.device = self.models["esm"].device  

        self.model_type = "ESM" + self.model_cfgs["name"].split('_')[0][-1].upper()
        self.measures = [m for m, t in self.model_cfgs["measure_dct"].items() if t == "avg"]
        self.AA_indices = range(4, 24)

        fpath_fn = lambda s, ext: f'{configs["model_dir"]}/{s}_{configs["task_name"]}.{ext}'
        self.model_paths = {m: fpath_fn(m, "pth") for m in self.models.keys()}
        logf = open(fpath_fn("log", "json"), "a" if self.train_cfgs["testing"] else "w")
        score_fpath = fpath_fn("score", "json")
        self.log_fn = lambda message: logging(logf, message)
        self.log_score_fn = lambda d: dict2json(score_fpath, d)

        self.best_scores = None
        if self.train_cfgs["use_ckt"]:
            saved_ckt = self.train_cfgs["saved_ckt"]
            if len(saved_ckt) > 0:
                model_paths = {m: f'{configs["model_dir"]}/{m}_{saved_ckt}.pth' for m in self.models.keys()}
                self.load_model(model_paths)
                print("Load", saved_ckt)
            else:
                print("Loading checkpoint === ")
                self.load_model()
                if os.path.isfile(score_fpath):
                    self.best_scores = read_json(score_fpath)
                else:
                    self.log_fn("No score found at " + score_fpath)
    
    def set_AAorder(self, AA_order):
        self.AA_indices = [self.tokenizer.encode(v)[1] for v in AA_order]

    def set_eval_fn(self, eval_fn):
        self.eval_fn = eval_fn
    
    def set_loaders(self, loaders):
        self.loaders = loaders

    def set_log_fn(self, log_fn):
        self.log_fn = log_fn

    def get_esm_trainable_parameters(self):
        if self.model_type == 'ESM3':
            param_dct = {
                "head": self.models['esm'].output_heads.sequence_head,
                "nlayers": self.models['esm'].transformer.blocks,
                "norm": self.models['esm'].transformer.norm
            }
        elif self.model_type == 'ESMC':
            param_dct = {
                "head": self.models['esm'].sequence_head,
                "nlayers": self.models['esm'].transformer.blocks
            }
        else:
            param_dct = {
                "head": self.models['esm'].lm_head,
                "nlayers": self.models['esm'].esm.encoder.layer
            }
        model_parameters = []
        for param_name, params in param_dct.items():
            if self.model_cfgs["esm_params"][param_name]:
                if param_name == "nlayers":
                    for k in range(1, 1 + self.model_cfgs["esm_params"]["nlayers"]):
                        for param in params[-k].parameters(): 
                            param.requires_grad = True
                            model_parameters.append(param)
                else:
                    for param in params.parameters():
                        param.requires_grad = True
                        model_parameters.append(param)
        return model_parameters
    
    def init_training(self, niters):
        if self.model_cfgs["esm_params"]["full"]:
            model_parameters = self.models['esm'].parameters()
        else:
            for param in self.models['esm'].parameters():
                param.requires_grad = False
            model_parameters = self.get_esm_trainable_parameters()

        if "decoder" in self.models:
            decoder_parameters = self.models["decoder"].parameters()
            self.optimizer = AdamW([
                {'params': model_parameters, 
                 'lr': self.train_cfgs["optim"]['lr_model']}, 
                {'params': decoder_parameters, 
                 'lr': self.train_cfgs["optim"]['lr_decoder']}],
                weight_decay=self.train_cfgs["optim"]["weight_decay"], 
                betas=(0.9, 0.98)
            )
        else:
            self.optimizer = AdamW([
                {'params': model_parameters, 
                'lr': self.train_cfgs["optim"]['lr_model']}], 
                weight_decay=self.train_cfgs["optim"]["weight_decay"], 
                betas=(0.9, 0.98)
            )
        # scheduler
        if self.train_cfgs["optim"]["scheduler"] == 'linear':
            total_steps = niters * self.train_cfgs["nepochs"]
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=int(total_steps * 10), num_training_steps=total_steps)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(self.train_cfgs["nepochs"]), last_epoch=-1)

    def logit_fn_esm3(self, batch):
        input_src = self.train_cfgs["input_src"]
        seq_tokens = batch["sequence_tokens"].to(self.device)
        # attention_masks = batch["attention_masks"].to(self.device)
        results = {
            "sequence_logits": None,
            "input_ids": seq_tokens,
            "attenion_masks": None,
            "embeddings": None
        }
        if C.SEQ == input_src: # seq only
            outs = self.models['esm'].forward(sequence_tokens=seq_tokens)
        else:
            struct_tokens = batch["structure_tokens"].to(self.device)
            if C.SEQ_STRUCT == input_src: # both
                outs = self.models['esm'].forward(
                    sequence_tokens=seq_tokens, 
                    structure_tokens=struct_tokens)
            elif C.SEQ_STRUCT_2 == input_src: # both
                per_res_plddt = batch["per_res_plddt"].to(self.device)
                coordinates = batch["coordinates"].to(self.device)
                outs = self.models['esm'].forward(
                    sequence_tokens=seq_tokens, 
                    structure_tokens=struct_tokens, 
                    per_res_plddt=per_res_plddt,
                    structure_coords=coordinates,
                    # sequence_id=attention_masks,
                    chain_id=None)
            else: # 1
                outs = self.models['esm'].forward(structure_tokens=struct_tokens)
            # structure logits
            results["structure_logits"] = outs.structure_logits
        # sequence logits
        results["sequence_logits"] = outs.sequence_logits
        return results
    
    def logit_fn_esmc(self, batch):
        tokens = self.tokenizer(batch["sequence"], truncation=True, padding='longest', max_length=1024, return_tensors='pt')
        outputs = self.models['esm'].forward(**tokens.to(self.device))
        results = {
            "attention_masks": tokens["attention_mask"].to(self.device),
            "input_ids": tokens["input_ids"].to(self.device),
            "sequence_logits": outputs.sequence_logits.float(),
            "embeddings": outputs.embeddings.float()
        }
        return results

    def logit_fn_esm2(self, batch):
        tokens = self.tokenizer(batch["sequence"], truncation=True, padding='longest', max_length=1024, return_tensors='pt')
        outputs = self.models['esm'](**tokens.to(self.device), output_hidden_states=True)
        results = {
            "attention_masks": tokens["attention_mask"].to(self.device),
            "input_ids": tokens["input_ids"].to(self.device),
            "sequence_logits": outputs["logits"],
            "embeddings": outputs['hidden_states'][-1]
        }
        return results

    def logits(self, batch):
        if 'ESM3' == self.model_type:
            return self.logit_fn_esm3(batch)
        elif 'ESMC' == self.model_type:
            return self.logit_fn_esmc(batch)
        else:
            return self.logit_fn_esm2(batch)

    def hinge_loss(self, scores):
        ranks = {}
        for m in ['target', 'pred']:
            s = scores[m].unsqueeze(-1).expand(-1, 20, 20)
            s_t = s.transpose(2, 1).flatten()
            s = s.flatten()
            r = s - s_t
            if m == 'target':
                masks = (s > s_t) * (s > -1000) 
            ranks[m] = r[masks]
        signs = torch.sign(ranks['target'] * ranks['pred'])
        gaps = F.sigmoid(ranks['pred']) * signs
        loss = F.gelu(1 - gaps).mean()
        return loss.mean()

    def loss_fn(self, targets, predictions=None):
        pass

    def optimize(self, batch):
        for m in self.models.keys():
            self.models[m].train()  
        self.optimizer.zero_grad()
        outputs = self.logits(batch)
        loss_dct = self.loss_fn(batch, outputs)
        loss_dct['loss'].backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss_dct
    
    def train_epoch(self, epoch, best_scores):
        min_loss = best_scores
        loss_train_dct = AverageMeterList(self.model_cfgs['measure_dct'])
        for j, batch in enumerate(self.loaders['train']):
            loss_batch_dct = self.optimize(batch)
            loss_train_dct.update(loss_batch_dct, len(batch["row_id"]))
            if j % self.train_cfgs["log_freq"] == 0:
                self.log_fn("[{:02}/{}] ".format(epoch, j))
                loss_dct = {
                    "train": loss_train_dct.get_avg(),
                    "valid": self.evaluate(self.loaders['valid'])
                }
                for split, dct in loss_dct.items():
                    self.log_fn(f"\t {split}: " + " ".join(["{} {:.3f}".format(m, dct[m]) for m in self.measures]))
                val_loss = loss_dct["valid"]["loss"]
                if  val_loss < min_loss:
                    self.log_fn("== best model!")
                    min_loss = val_loss
                    self.save_model()
                    post_fix = self.eval_epoch(loss_dct["valid"])
                    post_fix.update({"epoch": epoch, "iter": j}) 
                    self.log_score_fn(post_fix)
        return min_loss

    def evaluate(self, val_loader):
        for m in self.models.keys():
            self.models[m].eval()
        losses_dct = AverageMeterList(self.model_cfgs['measure_dct'])
        with torch.no_grad():
            for batch in val_loader:
                outputs = self.logits(batch)
                loss_dct = self.loss_fn(batch, outputs)
                losses_dct.update(loss_dct, len(batch['row_id']))
        return losses_dct.get_avg()
    
    def load_model(self, model_paths = None):
        if model_paths is None:
            model_paths = self.model_paths
        for m in self.models.keys():
            if os.path.isfile(model_paths[m]):
                self.models[m].load_state_dict(torch.load(model_paths[m]))
                print(f"== Load {m}")
            else:
                print(f"== No checkpoint at ", model_paths[m])

    def save_model(self):
        for m in self.models.keys():
            torch.save(self.models[m].state_dict(), self.model_paths[m])