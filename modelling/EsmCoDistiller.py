import os, sys 
sys.path.append(os.path.dirname(os.getcwd()))

import torch
from torch.nn import functional as F
import numpy as np
from scipy.stats import spearmanr
from modelling.EsmTrainerObject import ESMTrainerObject
from utils import constants as C

class ESMCoDistiller(ESMTrainerObject):
    def __init__(self, model, tokenizer, params = None):
        super().__init__(model, tokenizer, params)
    
    def get_llrs(self, sequence_logits, input_ids):
        token_probs = torch.log_softmax(sequence_logits, dim=-1)
        wt_positions = F.one_hot(input_ids, num_classes=token_probs.shape[-1])
        wt_probs = token_probs * wt_positions
        wt_probs = wt_probs.sum(dim=-1, keepdim=True)
        llrs = token_probs - wt_probs.expand(token_probs.shape)
        return llrs, token_probs, wt_probs
    
    def loss_fn(self, batch):
        outputs = self.logits(batch)
        input_ids = outputs["input_ids"]
        llrs, AA_probs, wt_probs = self.get_llrs(outputs["sequence_logits"], input_ids)
        non_wt_positions = 1 - F.one_hot(input_ids, num_classes=llrs.shape[-1])[..., 4:24] 
        llrs = llrs[:, :, 4:24]
        AA_probs = AA_probs[:, :, 4:24]
        wt_probs = wt_probs[:, :, 0]
        target_llrs = batch["llrs"][:, :llrs.shape[1], :].to(self.device) # tokenizer takes the longest length

        AA_masks = (input_ids >= 4) * (input_ids < 24)
        num_valid = torch.sum(AA_masks)
        masked_mean_fn = lambda t: torch.sum(t * AA_masks.float()) / num_valid

        shifted_llrs =  llrs - self.configs["setting"]['mu_base'] 
        shifted_targets = target_llrs - self.configs["setting"]['mu_target']
        sig = self.configs["setting"]["sig"]
        mse = F.mse_loss(shifted_targets/sig, shifted_llrs/sig, reduction='none') * non_wt_positions
        losses = {"loss": masked_mean_fn(mse.mean(dim=-1))}
        return losses
    
    def infer_batch(self, sequence_loader, mutation_dct):
        score_dct = {}
        self.model.eval() 
        with torch.no_grad(): 
            for batch_proteins, seqs in sequence_loader:
                outs = self.logits({"sequence": seqs})
                batch_llrs = self.get_llrs(outs["sequence_logits"], outs["input_ids"])[0]
                for k, p in enumerate(batch_proteins):
                    llrs = batch_llrs[k][:len(seqs[k]) + 2].cpu().numpy()
                    score_dct.update({mut_id: llrs[int(mut[1:-1]), C.sequence_vocabs[mut[-1]]] for mut_id, mut in mutation_dct[p]})
        return score_dct
    
    def eval_epoch(self, loss_dct, return_score=False, threshold=0.1):
        fn_t = lambda preds, targets, inds: spearmanr(preds[inds], targets[inds])[0]
        valid_dct = self.loaders["val"]
        score_dct = self.infer_batch(valid_dct["sequence_loader"], valid_dct["mutation_dict"])
        pred_scores = np.asarray([score_dct[k] for k in range(len(score_dct))])
        pseudo_target_scores = np.asarray(valid_dct["base_scores"]["esm11_avg"])
        base_scores = np.asarray(valid_dct["base_scores"][self.model_name])

        mu_p = np.mean(pseudo_target_scores) 
        mu_o = np.mean(base_scores)
        indices_tp = pseudo_target_scores < mu_p
        indices_fn = base_scores < mu_o

        corr_tp = fn_t(pred_scores, pseudo_target_scores, indices_tp)
        corr_fn = fn_t(pred_scores, base_scores, indices_fn)
        
        corr = corr_tp - threshold* corr_fn # round 1
        self.log_fn(f"\t approx-corr {corr:.4f}  ")
        post_fix = {
            "loss": loss_dct["loss"],
            "score": corr,
            "losses": loss_dct
        }
        if return_score:
            return post_fix, pred_scores
        return post_fix
    


class ESM3Distiller(ESMCoDistiller):
    def __init__(self, models, tokenizer, params = None):
        super().__init__(models, tokenizer, params)

    def logit_fn(self, dct, models=None):
        t_tensor = lambda x: torch.Tensor(x).long().unsqueeze(0).to(self.device)
        batch = {
            "sequence_tokens": t_tensor(dct["esm3_tokens"]["sequence"]),
            "structure_tokens": t_tensor(dct["esm3_tokens"]["masked_structure"])
        }
        if "structure" in dct:
            batch["per_res_plddt"] = t_tensor(dct["structure"]["masked_per_res_plddt"])
            batch["coordinates"] = t_tensor(dct["structure"]["masked_coordinates"])

        outputs = self.logit_fn_esm3(batch)
        return outputs
    
    def score_mutations_from_llrs(self, llrs, mutations, sequence_vocabs):
        pred_scores = []
        for mutation in mutations:
            mutation_score = 0
            for mut in mutation.split(":"):
                pos = int(mut[1:-1]) # 1-based
                mt = sequence_vocabs[mut[-1]]
                pred = llrs[pos, mt] 
                mutation_score += pred.item()
            pred_scores.append(mutation_score)
        return list(pred_scores)
    
    def eval_epoch(self, loss_dct, return_score=False, threshold=0.1):
        fn_t = lambda preds, targets, inds: spearmanr(preds[inds], targets[inds])[0]
        valid_dct = self.loaders["val"]
        pred_scores = []
        pseudo_target_scores = []
        base_scores = []
        with torch.no_grad():
            for protein, dct in valid_dct.items():
                outs = self.logit_fn(dct)
                llrs = self.get_llrs(outs["sequence_logits"], outs["input_ids"])[0]
                pred_scores += self.score_mutations_from_llrs(llrs[0], dct["mutations"], C.sequence_vocabs)
                pseudo_target_scores += list(dct["base_scores"]["esm11_avg"])
                base_scores += list(dct["base_scores"]["esm3"])
        pred_scores = np.asarray(pred_scores)
        pseudo_target_scores = np.asarray(pseudo_target_scores)
        base_scores = np.asarray(base_scores)

        mu_p = np.mean(pseudo_target_scores) 
        mu_o = np.mean(base_scores)
        indices_tp = pseudo_target_scores < mu_p
        indices_fn = base_scores < mu_o

        corr_tp = fn_t(pred_scores, pseudo_target_scores, indices_tp)
        corr_fn = fn_t(pred_scores, base_scores, indices_fn)
        
        corr = corr_tp - threshold* corr_fn # round 1
        self.log_fn(f"\t approx-corr {corr:.4f}  ")
        post_fix = {
            "loss": loss_dct["loss"],
            "score": corr,
            "losses": loss_dct
        }
        if return_score:
            return post_fix, pred_scores
        return post_fix
   