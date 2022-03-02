# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %sh python -m spacy download nl_core_news_sm

# COMMAND ----------

# MAGIC %sh python -m spacy download en_core_web_sm

# COMMAND ----------

import torch

with torch.no_grad():
    torch.cuda.empty_cache()

# COMMAND ----------

import numpy as np
def labels_to_class_weights(labels, nc=2): 
  # Get class weights (inverse frequency) from training labels 
  classes = np.array(labels).astype(np.int32)  # labels = [class xywh] 
  weights = np.bincount(classes, minlength=nc)  # occurences per class 
  weights[weights == 0] = 1  # replace empty bins with 1 
  weights = 1 / weights  # number of targets per class 
  weights /= weights.sum()  # normalize 
  return weights.tolist()
    

# COMMAND ----------

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from config_sent import *

class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, scale=1.0, gamma=0, num_classes=10, alpha=None, size_average=True):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class ReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.device = config['device']
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()

class NFLandRCE(torch.nn.Module):
    def __init__(self, num_classes, alpha=1.0, beta=1.0, gamma=0.5):
        super(NFLandRCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(scale=alpha, gamma=gamma, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.rce(pred, labels)

# COMMAND ----------

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel
from config_sent import *

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

# class BERTLinear(nn.Module):
#     def __init__(self, bert_type, num_cat, num_pol):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(
#             bert_type, output_hidden_states=True)

#         self.aspect_weights = weights['aspect_weights'][config['domain']]
#         self.sentiment_weights = weights['sentiment_weights'][config['domain']]
        
#         self.bert2 = BertModel.from_pretrained(
#             bert_type, output_hidden_states=True)
#         self.dropcat = nn.Dropout(p=0.1)
#         self.droppol = nn.Dropout(p=0.1)

#         self.num_cat = num_cat
#         self.num_pol = num_pol

#         self.outcat = nn.Linear(768, num_cat)
#         self.outpol = nn.Linear(768, num_pol)

#     def forward(self, labels_cat, labels_pol, **kwargs):
#         outputs = self.bert(**kwargs)
#         outputs2 = self.bert2(**kwargs)
# #         x = outputs[2][12]  # (bsz, seq_len, 768)

# #         mask = kwargs['attention_mask']  # (bsz, seq_len)
# #         se = x * mask.unsqueeze(2)
# #         den = mask.sum(dim=1).unsqueeze(1)
# #         se = se.sum(dim=1) / den  # (bsz, 768)
        
#         output_cat = F.relu(outputs.pooler_output)
#         output_cat = self.dropcat(output_cat)
        
#         output_pol = F.relu(outputs2.pooler_output)
#         output_pol = self.dropcat(output_pol)
        
#         logits_cat = self.outcat(output_cat)
#         logits_pol = self.outpol(output_pol)

# #         logits_cat = self.ff_cat(se)  # (bsz, num_cat)
# #         logits_pol = self.ff_pol(se)  # (bsz, num_pol)
# #         loss = LQLoss(0.8, self.aspect_weights)(F.softmax(logits_cat, dim=-1), labels_cat)
# #         loss = loss + LQLoss(0.8, self.sentiment_weights)(F.softmax(logits_pol, dim=-1), labels_pol)
#         loss = loss = FocalLoss(gamma=2)(logits_cat, labels_cat) + FocalLoss(gamma=1)(logits_pol, labels_pol)
#         return loss, logits_cat, logits_pol


class BERTLinear(nn.Module):
    def __init__(self, bert_type, num_cat, num_pol):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            bert_type, output_hidden_states=True)
        self.ff_cat = nn.Linear(768, num_cat)
        self.ff_pol = nn.Linear(768, num_pol)
        self.aspect_weights = weights['aspect_weights'][config['domain']]
        self.sentiment_weights = weights['sentiment_weights'][config['domain']]
        self.num_cat = num_cat
        self.num_pol = num_pol
    
    def set_weigths(self, aspect_weights, sentiment_weights):
        self.aspect_weights = aspect_weights
        self.sentiment_weights = sentiment_weights

    def forward(self, labels_cat, labels_pol, **kwargs):
        outputs = self.bert(**kwargs)
        x = outputs[2][11]  # (bsz, seq_len, 768)

        mask = kwargs['attention_mask']  # (bsz, seq_len)
        se = x * mask.unsqueeze(2)
        den = mask.sum(dim=1).unsqueeze(1)
        se = se.sum(dim=1) / den  # (bsz, 768)

        logits_cat = self.ff_cat(se)  # (bsz, num_cat)
        logits_pol = self.ff_pol(se)  # (bsz, num_pol)
#         loss = LQLoss(0.4, self.aspect_weights)(F.softmax(logits_cat), labels_cat)
#         loss = loss + LQLoss(0.4, self.sentiment_weights)(F.softmax(logits_pol), labels_pol)
#         loss = torch.nn.KLDivLoss()(torch.log(F.softmax(logits_cat, dim=-1)), F.one_hot(labels_cat, num_classes=8).float()) + torch.nn.KLDivLoss()(torch.log(F.softmax(logits_pol, dim=-1)), F.one_hot(labels_pol, num_classes=2).float())
        loss = FocalLoss(gamma=3)(logits_cat, labels_cat) + FocalLoss(gamma=3)(logits_pol, labels_pol)
#         loss = NFLandRCE(self.num_cat, gamma=2)(logits_cat, labels_cat) + NFLandRCE(self.num_pol, gamma=1)(logits_pol, labels_pol)
#         loss = NormalizedFocalLoss(gamma=2, num_classes=self.num_cat)(logits_cat, labels_cat) + NormalizedFocalLoss(gamma=1, num_classes=self.num_pol)(logits_pol, labels_pol)
        return loss, logits_cat, logits_pol


# COMMAND ----------

from transformers import AutoTokenizer, BertForMaskedLM
from config_sent import *
from filter_words import filter_words
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import re
import time

EPOCHS = 10
DOMAIN = 'laptop'

class Trainer:

    def __init__(self):
        self.domain = DOMAIN
        self.bert_type = bert_mapper[self.domain]
        self.device = config['device']
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.root_path = path_mapper[self.domain]

        categories = aspect_category_mapper[self.domain]
        polarities = sentiment_category_mapper[self.domain]

        self.model = BERTLinear(self.bert_type, len(
            categories), len(polarities)).to(self.device)
        self.weights_cat = None
        self.weights_pol = None

        aspect_dict = {}
        inv_aspect_dict = {}
        for i, cat in enumerate(categories):
            aspect_dict[i] = cat
            inv_aspect_dict[cat] = i

        polarity_dict = {}
        inv_polarity_dict = {}
        for i, pol in enumerate(polarities):
            polarity_dict[i] = pol
            inv_polarity_dict[pol] = i

        self.aspect_dict = aspect_dict
        self.inv_aspect_dict = inv_aspect_dict
        self.polarity_dict = polarity_dict
        self.inv_polarity_dict = inv_polarity_dict

    def load_training_data(self):
        sentences = []
        cats = []
        pols = []
        with open(f'{self.root_path}/label-sentences.txt', 'r', encoding="utf8") as f:
            for idx, line in enumerate(f):
                if idx % 2 == 1:
                    cat, pol = line.strip().split()
                    cats.append(self.inv_aspect_dict[cat])
                    pols.append(self.inv_polarity_dict[pol])
                else:
                    sentences.append(line.strip())
        encoded_dict = self.tokenizer(
            sentences,
            padding=True,
            return_tensors='pt',
            max_length=128,
            return_attention_mask=True,
            truncation=True)
        self.weights_cat = labels_to_class_weights(cats)
        self.weights_pol = labels_to_class_weights(pols)
        labels_cat = torch.tensor(cats)
        labels_pol = torch.tensor(pols)
        dataset = TensorDataset(
            labels_cat, labels_pol, encoded_dict['input_ids'], encoded_dict['token_type_ids'], encoded_dict['attention_mask'])
        return dataset

    def set_seed(self, value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    def train_model(self, dataset, epochs=EPOCHS):
        """Train the BertClassifier model.
            """
        self.set_seed(0)
        
        # Prepare dataset
        train_data, val_data = torch.utils.data.random_split(
            dataset, [len(dataset) - validation_data_size, validation_data_size])
        dataloader = DataLoader(train_data, batch_size=batch_size)
        val_dataloader = DataLoader(val_data, batch_size=batch_size)

        model = self.model
        device = self.device

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Start training loop
        print("Start training...\n")
        for epoch_i in range(epochs):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-" * 70)
            
            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()
            
            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0
            
            # Put the model into the training mode
            model.train()
            
            for step, (labels_cat, labels_pol, input_ids, token_type_ids, attention_mask) in enumerate(dataloader):
                batch_counts += 1
              
                optimizer.zero_grad()
                encoded_dict = {
                    'input_ids': input_ids.to(device),
                    'token_type_ids': token_type_ids.to(device),
                    'attention_mask': attention_mask.to(device)
                }
                loss, _, _ = model(labels_cat.to(device),
                                   labels_pol.to(device), **encoded_dict)
                
                # Perform a backward pass to calculate gradients
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
                total_loss += loss.item()
                
                # Print the loss values and time elapsed for every 20 batches
                if (step % 50 == 0 and step != 0) or (step == len(dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()
            
            avg_train_loss = total_loss / len(dataloader)

            print("-" * 70)

            model.eval()
            
            # Tracking variables
            val_accuracy = []
            val_loss = []
                
            with torch.no_grad():

                for labels_cat, labels_pol, input_ids, token_type_ids, attention_mask in val_dataloader:
                    encoded_dict = {
                        'input_ids': input_ids.to(device),
                        'token_type_ids': token_type_ids.to(device),
                        'attention_mask': attention_mask.to(device)
                    }
                    loss, logits_cat, logits_pol = model(labels_cat.to(
                        device), labels_pol.to(device), **encoded_dict)
                    val_loss.append(loss.item())
                    
                    # Get the predictions
                    preds_cat = torch.argmax(logits_cat, dim=1).flatten()
                    preds_col = torch.argmax(logits_pol, dim=1).flatten()
                    
                     # Calculate the accuracy rate
                    accuracy = (preds_cat == labels_cat.to(device)).cpu().numpy().mean() * 100
                    val_accuracy.append(accuracy)

            # Compute the average accuracy and loss over the validation set.
            val_loss = np.mean(val_loss)
            val_accuracy = np.mean(val_accuracy)
                
            # Display the epoch training loss and validation loss
            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
        print("\n")

    def save_model(self, name):
        torch.save(self.model, f'{self.root_path}/{name}.pth')

    def load_model(self, name):
        self.model = torch.load(f'{self.root_path}/{name}.pth')

    def evaluate(self):
        test_sentences = []
        test_cats = []
        test_pols = []

        with open(f'{self.root_path}/test.txt', 'r', encoding="utf8") as f:
            for line in f:
                _, cat, pol, sentence = line.strip().split('\t')
                cat = int(cat)
                pol = int(pol)
                test_cats.append(cat)
                test_pols.append(pol)
                test_sentences.append(sentence)

        df = pd.DataFrame(columns=(
            ['sentence', 'actual category', 'predicted category', 'actual polarity', 'predicted polarity']))

        model = self.model
        model.eval()
        device = self.device

        actual_aspect = []
        predicted_aspect = []

        actual_polarity = []
        predicted_polarity = []

        iters = 0
        with torch.no_grad():
            for input, cat, pol in tqdm(zip(test_sentences, test_cats, test_pols)):

                encoded_dict = self.tokenizer([input],
                                              padding=True,
                                              return_tensors='pt',
                                              return_attention_mask=True,
                                              truncation=True).to(device)

                loss, logits_cat, logits_pol = model(torch.tensor([cat]).to(
                    device), torch.tensor([pol]).to(device), **encoded_dict)

                actual_aspect.append(self.aspect_dict[cat])
                actual_polarity.append(self.polarity_dict[pol])

                predicted_aspect.append(
                    self.aspect_dict[torch.argmax(logits_cat).item()])
                predicted_polarity.append(
                    self.polarity_dict[torch.argmax(logits_pol).item()])
                df.loc[iters] = [input, actual_aspect[-1], predicted_aspect[-1],
                                 actual_polarity[-1], predicted_polarity[-1]]
                iters += 1

        df.to_csv(f'{self.root_path}/predictions.csv')

        predicted_pol = np.array(predicted_polarity)
        actual_pol = np.array(actual_polarity)
        print("Polarity")
        print(classification_report(actual_pol, predicted_pol, digits=4))
        print()

        predicted = np.array(predicted_aspect)
        actual = np.array(actual_aspect)
        print("Aspect")
        print(classification_report(actual, predicted, digits=4))
        print()
        
        return classification_report(actual_pol, predicted_pol, digits=6, output_dict=True), classification_report(actual, predicted, digits=6, output_dict=True)

# COMMAND ----------

from labeler_sentence import Labeler

def SBERTNNrun(i = 0):
  with torch.no_grad():
    torch.cuda.empty_cache()
  labeler = Labeler()
  label_pols, label_cats = [], []
  labeler(evaluate=False, load=True)

  trainer = Trainer()
  dataset = trainer.load_training_data()
  trainer.set_seed(i)
  trainer.train_model(dataset)
  return trainer.evaluate(), label_pols, label_cats

# COMMAND ----------

SBERTNNrun()

# COMMAND ----------

RUNS = 5
polarity_list, aspect_list, label_polarity_list, label_aspect_list = [], [] ,[],[]
for i in range(RUNS):
    print('RUN: ', i)
    (polarity, aspect), label_pols, label_cats = SBERTNNrun(i)
    polarity_list.append(polarity)
    aspect_list.append(aspect)
    label_polarity_list.append(label_pols)
    label_aspect_list.append(label_cats)
    
acc, prec, rec, f1 = 0, 0, 0, 0
for item in polarity_list:
    acc += item['accuracy']
    prec += item['macro avg']['precision']
    rec += item['macro avg']['recall']
    f1 += item['macro avg']['f1-score']

print(f"accuracy: {acc/len(polarity_list)},\t precision: {prec/len(polarity_list)},\t recall: {rec/len(polarity_list)},\t f1-score: {f1/len(polarity_list)}")

acc, prec, rec, f1 = 0, 0, 0, 0
for item in aspect_list:
    acc += item['accuracy']
    prec += item['macro avg']['precision']
    rec += item['macro avg']['recall']
    f1 += item['macro avg']['f1-score']

print(
    f"accuracy: {acc / len(aspect_list)},\t precision: {prec / len(aspect_list)},\t recall: {rec / len(aspect_list)},\t f1-score: {f1 / len(aspect_list)}")

# COMMAND ----------

acc, prec, rec, f1 = 0, 0, 0, 0
for item in label_aspect_list:
    acc += item['accuracy']
    prec += item['macro avg']['precision']
    rec += item['macro avg']['recall']
    f1 += item['macro avg']['f1-score']

print(
    f"accuracy: {acc / len(label_aspect_list)},\t precision: {prec / len(label_aspect_list)},\t recall: {rec / len(label_aspect_list)},\t f1-score: {f1 / len(label_aspect_list)}")

# COMMAND ----------

from pyspark.sql import functions as F

df = spark.read.options(header='True').option("quote", "\"").option("escape", "\"").csv(r'dbfs:/FileStore/kto/laptop/predictions.csv')
display(df)

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

classes = df.select('actual category').distinct().rdd.flatMap(lambda x: x).collect()

cm = confusion_matrix(df.select('actual category').rdd.flatMap(lambda x: x).collect(), df.select('predicted category').rdd.flatMap(lambda x: x).collect(), labels=classes)
cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

fig, ax = plt.subplots(figsize=(15,15))
cmp.plot(ax=ax, xticks_rotation='vertical')

# COMMAND ----------

path = "dbfs:/FileStore/kto/restaurant/scores-test.txt"

df1 = spark.read.text(path)
display(df1)
