import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel
from config import *

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class LQLoss(nn.Module):
    
    def __init__(self, q, weight, alpha=1.0):
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
        self.aspect_weights = weights['aspect_weights'][config['domain']]
        self.sentiment_weights = weights['sentiment_weights'][config['domain']]

    def forward(self, labels_cat, labels_pol, **kwargs):
        outputs = self.bert(**kwargs)
        x = outputs[2][11]  # (bsz, seq_len, 768)

        mask = kwargs['attention_mask']  # (bsz, seq_len)
        se = x * mask.unsqueeze(2)
        den = mask.sum(dim=1).unsqueeze(1)
        se = se.sum(dim=1) / den  # (bsz, 768)

        logits_cat = self.ff_cat(se)  # (bsz, num_cat)
        logits_pol = self.ff_pol(se)  # (bsz, num_pol)
        #loss = LQLoss(0.4, self.aspect_weights)(F.softmax(logits_cat), labels_cat)
        #loss = loss + LQLoss(0.4, self.sentiment_weights)(F.softmax(logits_pol), labels_pol)
        loss = FocalLoss(gamma=3)(logits_cat, labels_cat) + FocalLoss(gamma=3)(logits_pol, labels_pol)
        return loss, logits_cat, logits_pol

