# Databricks notebook source
import time
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from config import *
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import random
import numpy as np
import torch.nn as nn
from transformers import BertModel

# COMMAND ----------

# Specify loss function
loss_fn = nn.CrossEntropyLoss()

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """

    def __init__(self, freeze_bert=False, D_out=2):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H = 768, 50

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(H, D_out).to(config['device'])
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, **kwargs):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(**kwargs)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


class BertBaseline:
    def __init__(self):
        self.domain = config['domain']
        self.root_path = path_mapper[self.domain]
        self.bert_type = 'bert-base-uncased'
        self.categories = aspect_category_mapper[self.domain]
        self.polarities = sentiment_category_mapper[self.domain]
        self.device = config['device']
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.batch_size = 16

    def find_sentences(self, categories, seeds):
        inv_cat_dict = {}
        for i, cat in enumerate(categories):
            inv_cat_dict[cat] = i

        sentences, cats = [], []

        with open(f'{self.root_path}/train.txt', 'r', encoding="utf8") as f:
            for idx, line in enumerate(f):
                found_cats = []
                for category in categories:
                    text = line.strip()
                    if not set(seeds[category]).isdisjoint(text.split()):
                        found_cats.append(category)
                if len(found_cats) == 1:
                    sentences.append(text)
                    cats.append(inv_cat_dict[found_cats[0]])

        encoded_dict = self.tokenizer(
            sentences,
            padding=True,
            return_tensors='pt',
            max_length=128,
            return_attention_mask=True,
            truncation=True)
        labels_cat = torch.tensor(cats)

        dataset = TensorDataset(
            labels_cat, encoded_dict['input_ids'], encoded_dict['token_type_ids'],
            encoded_dict['attention_mask'])
        return dataset

    def load_training_data(self):
        aspect_seeds = aspect_seed_mapper[self.domain]
        sentiment_seeds = sentiment_seed_mapper[self.domain]

        cat_dataset = self.find_sentences(self.categories, aspect_seeds)
        pol_dataset = self.find_sentences(self.polarities, sentiment_seeds)
        return cat_dataset, pol_dataset

    def initialize_model(self, dataloader, epochs=4, cats='polarities'):
        """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
        """
        if cats == 'polarities':
            out_len = len(self.polarities)
        else:
            out_len = len(self.categories)

        # Instantiate Bert Classifier and Tell PyTorch to run the model on GPU
        self.model = BertClassifier(freeze_bert=False, D_out=out_len).to(self.device)

        # Create the optimizer
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=5e-5,  # Default learning rate
                                     eps=1e-8  # Default epsilon value
                                     )

        # Total number of training steps
        total_steps = len(dataloader) * epochs

        # Set up the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value
                                                         num_training_steps=total_steps)

    def set_seed(self, value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    def train_model(self, dataset, epochs=epochs, evaluation=True, cats='categories'):
        """Train the BertClassifier model.
            """
        
        device = self.device

        # Prepare dataset
        train_data, val_data = torch.utils.data.random_split(
            dataset, [len(dataset) - validation_data_size, validation_data_size])
        dataloader = DataLoader(train_data, batch_size=self.batch_size)
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size)

        # Initialize the Bert Classifier
        self.initialize_model(dataloader, epochs=2, cats=cats)

        model = self.model

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

            for step, (labels, input_ids, token_type_ids, attention_mask) in enumerate(dataloader):
                batch_counts += 1
                # Zero out any previously calculated gradients
                model.zero_grad()

                encoded_dict = {
                    'input_ids': input_ids.to(device),
                    'token_type_ids': token_type_ids.to(device),
                    'attention_mask': attention_mask.to(device)
                }

                # Perform a forward pass. This will return logits.
                logits = model(**encoded_dict)

                # Compute loss and accumulate the loss values
                loss = loss_fn(logits, labels.to(device))
                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and the learning rate
                self.optimizer.step()
                self.scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 100 == 0 and step != 0) or (step == len(dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

                # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(dataloader)

            print("-" * 70)
            # =======================================
            #               Evaluation
            # =======================================
            if evaluation == True:
                # Put the model into the evaluation mode. The dropout layers are disabled during
                # the test time.
                model.eval()

                # Tracking variables
                val_accuracy = []
                val_loss = []

                for labels, input_ids, token_type_ids, attention_mask in val_dataloader:
                    encoded_dict = {
                        'input_ids': input_ids.to(device),
                        'token_type_ids': token_type_ids.to(device),
                        'attention_mask': attention_mask.to(device)
                    }

                    # Compute logits
                    with torch.no_grad():
                        logits = model(**encoded_dict)

                    # Compute loss
                    loss = loss_fn(logits, labels.to(device))
                    val_loss.append(loss.item())

                    # Get the predictions
                    preds = torch.argmax(logits, dim=1).flatten()

                    # Calculate the accuracy rate
                    accuracy = (preds == labels.to(device)).cpu().numpy().mean() * 100
                    val_accuracy.append(accuracy)

                # Compute the average accuracy and loss over the validation set.
                val_loss = np.mean(val_loss)
                val_accuracy = np.mean(val_accuracy)

                # Display the epoch training loss and validation loss
                print(
                    f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                print("-" * 70)
            print("\n")

        print("Training complete!")

    def evaluate(self, cats='categories'):
        test_sentences = []
        test_cats = []

        cat_dict = {}
        if cats == 'categories' :
            for i, cat in enumerate(self.categories):
                cat_dict[i] = cat
        else:
            for i, cat in enumerate(self.polarities):
                cat_dict[i] = cat

        with open(f'{self.root_path}/test.txt', 'r', encoding="utf8") as f:
            for line in f:
                _, cat, pol, sentence = line.strip().split('\t')
                if cats=='categories':
                    cat = int(cat)
                else:
                    cat = int(pol)

                test_cats.append(cat)
                test_sentences.append(sentence)

        model = self.model
        model.eval()
        device = self.device

        actual_category = []
        predicted_category= []

        iters = 0
        with torch.no_grad():
            for input, cat in tqdm(zip(test_sentences, test_cats)):

                encoded_dict = self.tokenizer([input],
                                              padding=True,
                                              return_tensors='pt',
                                              return_attention_mask=True,
                                              truncation=True).to(device)
                # Compute logits
                logits = model(**encoded_dict)

                actual_category.append(cat_dict[cat])
                predicted_category.append(
                    cat_dict[torch.argmax(logits).item()])
                iters += 1

            predicted = np.array(predicted_category)
            actual = np.array(actual_category)
            print(cats)
            print(classification_report(actual, predicted, digits=4))
            print()

            return classification_report(
                actual, predicted, digits=4, zero_division=0,
                output_dict=True
            )


# COMMAND ----------

RUNS = 5
polarity_list, aspect_list = [], []

for i in range(RUNS):
    bert_baseline = BertBaseline()
    cat_dataset, pol_dataset = bert_baseline.load_training_data()
    bert_baseline.train_model(pol_dataset, epochs=4, cats='polarities')
    polarity = bert_baseline.evaluate(cats='polarities')
    polarity_list.append(polarity)

acc, prec, rec, f1 = 0, 0, 0, 0
for item in polarity_list:
    acc += item['accuracy']
    prec += item['macro avg']['precision']
    rec += item['macro avg']['recall']
    f1 += item['macro avg']['f1-score']

print(
    f"accuracy: {acc / len(polarity_list)},\t precision: {prec / len(polarity_list)},\t recall: {rec / len(polarity_list)},\t f1-score: {f1 / len(polarity_list)}")


for i in range(RUNS):
    bert_baseline = BertBaseline()
    cat_dataset, pol_dataset = bert_baseline.load_training_data()
    bert_baseline.train_model(cat_dataset, epochs=4)
    aspect = bert_baseline.evaluate()
    aspect_list.append(aspect)

acc, prec, rec, f1 = 0, 0, 0, 0
for item in aspect_list:
    acc += item['accuracy']
    prec += item['macro avg']['precision']
    rec += item['macro avg']['recall']
    f1 += item['macro avg']['f1-score']

print(f"accuracy: {acc/len(aspect_list)},\t precision: {prec/len(aspect_list)},\t recall: {rec/len(aspect_list)},\t f1-score: {f1/len(aspect_list)}")

# COMMAND ----------

acc, prec, rec, f1 = 0, 0, 0, 0
for item in polarity_list:
    acc += item['accuracy']
    prec += item['macro avg']['precision']
    rec += item['macro avg']['recall']
    f1 += item['macro avg']['f1-score']

print(
    f"accuracy: {acc / len(polarity_list)},\t precision: {prec / len(polarity_list)},\t recall: {rec / len(polarity_list)},\t f1-score: {f1 / len(polarity_list)}")
