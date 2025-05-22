import torch
from torch.nn import functional as F
from .EsmTrainerObject import ESMTrainerObject

class ESMDistiller(ESMTrainerObject):
    def __init__(self, models, tokenizer, params = None):
        super().__init__(models, tokenizer, params)
    
    def estimate_entropy(self, log_probs):
        probs = torch.exp(log_probs)
        p_log_p = log_probs * probs
        entropy = -p_log_p.sum(dim=-1)
        return entropy
    
    def get_nllr(self, sequence_logits, input_ids):
        token_probs = torch.log_softmax(sequence_logits, dim=-1)
        wt_positions = F.one_hot(input_ids, num_classes=token_probs.shape[-1])
        wt_probs = token_probs * wt_positions
        wt_probs = wt_probs.sum(dim=-1, keepdim=True)
        # add alpha 
        neg_llrs = -(token_probs - wt_probs.expand(token_probs.shape))
        return neg_llrs, token_probs
    
    def loss_fn(self, batch, outputs):
        input_ids = outputs["input_ids"]
        seq_masks = [x >=4 and x < 24 for x in input_ids.flatten()]   
        attend_fn = lambda x: x.view(-1, 20)[seq_masks, :]
        masked_mean_fn = lambda x: x.flatten()[seq_masks].mean()

        neg_llrs, token_probs = self.get_nllr(outputs["sequence_logits"], input_ids)
        neg_llrs = neg_llrs[:, :, self.AA_indices]
        target_llrs = batch["llrs"].to(self.device)

        losses = {"loss": 0}
        loss_mse = F.mse_loss(target_llrs, neg_llrs, reduction='none').mean(dim=-1)
        losses["mse"] = masked_mean_fn(loss_mse)
        losses["hinge"] = self.hinge_loss({
            "pred": attend_fn(neg_llrs), 
            "target": attend_fn(target_llrs)
        })
        AA_probs = token_probs[:, :, self.AA_indices]
        losses["reg"] = masked_mean_fn(self.estimate_entropy(AA_probs))
        for k, alpha in self.model_cfgs["alphas"].items():
            losses["loss"] += alpha * losses[k]
        
        return losses
    
    def set_logit_for_test_dct(self, logit_func):
        self.logit_for_test_dct = logit_func
    