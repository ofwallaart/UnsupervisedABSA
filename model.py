import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from config import *

class LQLoss(nn.Module):
    
    def __init__(self, q, weight, alpha=0.0):
        super().__init__()
        self.q = q ## parameter in the paper
        self.alpha = alpha ## hyper-parameter for trade-off between weighted and unweighted GCE Loss
        self.weight = nn.Parameter(F.softmax(torch.log(1. / torch.tensor(weight, dtype=torch.float32)), dim=-1), requires_grad=False).to(config['device']) ## per-class weights

    def forward(self, input, target, *args, **kwargs):
        bsz, _ = input.size()

        Yq = torch.gather(input, 1, target.unsqueeze(1))
        lq = (1 - torch.pow(Yq, self.q)) / self.q

        _weight = self.weight.repeat(bsz).view(bsz, -1)
        _weight = torch.gather(_weight, 1, target.unsqueeze(1))
    
        return torch.mean(self.alpha * lq + (1 - self.alpha) * lq * _weight)

class BERTLinear(nn.Module):
    def __init__(self, bert_type, num_cat, num_pol):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            bert_type, output_hidden_states=True)
        self.ff_cat = nn.Linear(768, num_cat)
        self.ff_pol = nn.Linear(768, num_pol)
        self.aspect_weights = [164, 31, 73, 182, 78, 45, 173, 87, 66, 13, 50] # [0, 319, 179, 95, 150, 203, 184] #[345, 67, 201]
        self.sentiment_weights = [927, 35] #[231, 382]

    def forward(self, labels_cat, labels_pol, **kwargs):
        outputs = self.bert(**kwargs)
        x = outputs[2][11]  # (bsz, seq_len, 768)

        mask = kwargs['attention_mask']  # (bsz, seq_len)
        se = x * mask.unsqueeze(2)
        den = mask.sum(dim=1).unsqueeze(1)
        se = se.sum(dim=1) / den  # (bsz, 768)

        logits_cat = self.ff_cat(se)  # (bsz, num_cat)
        logits_pol = self.ff_pol(se)  # (bsz, num_pol)
        loss = LQLoss(0.4, self.aspect_weights)(F.softmax(logits_cat), labels_cat)
        loss = loss + LQLoss(0.4, self.sentiment_weights)(F.softmax(logits_pol), labels_pol)
        return loss, logits_cat, logits_pol

