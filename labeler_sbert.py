import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from config import *
import re
import pickle


def load_training_data(file_path):
    sentences = []
    for line in open(file_path, encoding="utf-8"):
        split_lines = list(filter(None, re.split('; |\. |\! |\n| \?', line.lower())))
        for split_line in split_lines:
            sentences.append(split_line.strip())
    return sentences


def get_rep_sentences(self, embeddings, cosine_scores_train, aspect_seed, aspect_category):
    train_sentences = self.sentences
    topk_scores = []
    topk = torch.topk(cosine_scores_train, cosine_scores_train.shape[1], dim=1).indices

    for idx, top in enumerate(topk):

        # Make sure sentences contain at least one seed word and no seed words from other aspects
        seeds_in_sent = []
        seeds_not_in_sent = []
        for i in top.tolist():
            seeds_in_sent.append(
                [ele for ele in list(aspect_seed[aspect_category[idx]]) if
                 (" " + ele + " " in train_sentences[i])])

            lists = []
            for not_list_item in aspect_category:
                if not_list_item != aspect_category[idx]:
                    lists.extend(list(aspect_seed[not_list_item]))
            seeds_not_in_sent.append([ele for ele in lists if (ele in train_sentences[i])])

        # Do the checks
        final_top = []

        # Try to make at least each seed word appear in a sentence
        for seed in aspect_seed[aspect_category[idx]]:
            g = [i for i, e in enumerate(seeds_in_sent) if seed in e]
            for g_item in g:
                sentence_index = top.tolist()[g_item]
                if seeds_not_in_sent[g_item] \
                        or sentence_index in final_top \
                        or len(train_sentences[sentence_index].split()) <= 1 \
                        or train_sentences[sentence_index] in [train_sentences[i] for i in final_top]:
                    continue
                else:
                    final_top.append(sentence_index)
                    break

        # Fill top sentences to size N with most relevant sentences
        for i, t_item in enumerate(top.tolist()):
            if seeds_not_in_sent[i] or t_item in final_top or len(train_sentences[t_item].split()) <= 1:
                continue
            else:
                if len(final_top) > N[self.domain]:
                    break
                final_top.append(t_item)

        for i in final_top:
            print(train_sentences[i])
        print()

        sent_embeddings = torch.stack([embeddings[i] for i in final_top])

        # Also include the average of top K sentences
        topk_embeddings = torch.cat((sent_embeddings, torch.mean(sent_embeddings, dim=0).unsqueeze(0)))

        # Compute cosine-similarities between top representing sentences and all other train data
        topk_scores.append(torch.max(util.cos_sim(topk_embeddings, embeddings), dim=0)[0].unsqueeze(dim=-1))

    return torch.t(torch.cat(topk_scores, 1))


class Labeler:
    def __init__(self):
        self.domain = config['domain']
        self.model = SentenceTransformer(sbert_mapper[self.domain], device=config['device'])
        self.cat_threshold = 0.4
        self.pol_threshold = 0.3
        self.root_path = path_mapper[self.domain]
        self.categories = aspect_category_mapper[self.domain]
        self.polarities = sentiment_category_mapper[self.domain]
        self.labels = None
        self.sentences = None

    def __call__(self, use_two_step=True):

        category_seeds = aspect_seed_mapper[self.domain]
        polarity_seeds = sentiment_seed_mapper[self.domain]

        split = [len(self.categories), len(self.polarities)]

        # Seeds
        seeds = {}
        for cat in self.categories:
            seeds[cat] = " ".join(category_seeds[cat])
        for pol in self.polarities:
            seeds[pol] = " ".join(polarity_seeds[pol])

        seed_embeddings = self.model.encode(list(seeds.values()), convert_to_tensor=True, show_progress_bar=True)

        # Load and encode the train set
        self.sentences = load_training_data(f'{self.root_path}/train.txt')
        embeddings = self.model.encode(self.sentences, convert_to_tensor=True, show_progress_bar=True)

        # Compute cosine-similarities
        cosine_category_scores, cosine_polarity_scores = torch.split(util.cos_sim(seed_embeddings, embeddings), split)

        if use_two_step:
            # Get top most representable sentences for each aspect
            cosine_category_scores = get_rep_sentences(self, embeddings, cosine_category_scores,
                                                       category_seeds, self.categories)
            cosine_polarity_scores = get_rep_sentences(self, embeddings, cosine_polarity_scores,
                                                       polarity_seeds, self.polarities)

        category_argmax = torch.argmax(cosine_category_scores, dim=0).tolist()
        category_max = torch.max(cosine_category_scores, dim=0)[0].tolist()

        polarity_argmax = torch.argmax(cosine_polarity_scores, dim=0).tolist()
        polarity_max = torch.max(cosine_polarity_scores, dim=0)[0].tolist()

        labels = np.array(
            [category_argmax, category_max, polarity_argmax, polarity_max, np.arange(0, len(self.sentences))])

        self.labels = labels

        # No conflict (avoid multi-class sentences)
        labels = np.transpose(labels[:, (labels[1, :] >= self.cat_threshold) & (labels[3, :] >= self.pol_threshold)])

        nf = open(f'{self.root_path}/label-sbert.txt', 'w', encoding="utf8")
        cnt = {}

        for label in labels:
            sentence = self.sentences[int(label[4])]
            aspect = self.categories[int(label[0])]
            sentiment = self.polarities[int(label[2])]
            nf.write(f'{sentence}\n')
            nf.write(f'{aspect} {sentiment}\n')
            keyword = f'{aspect}-{sentiment}'
            cnt[keyword] = cnt.get(keyword, 0) + 1

        nf.close
        # Labeled data statistics
        print('Labeled data statistics:')
        print(cnt)

    def update_labels(self, cat_threshold, pol_threshold):
        labels = self.labels
        sentences = self.sentences

        # No conflict (avoid multi-class sentences)
        labels = np.transpose(labels[:, (labels[1, :] >= cat_threshold) & (labels[3, :] >= pol_threshold)])

        nf = open(f'{self.root_path}/label-sbert.txt', 'w', encoding="utf8")
        cnt = {}

        for label in labels:
            sentence = sentences[int(label[4])]
            aspect = self.categories[int(label[0])]
            sentiment = self.polarities[int(label[2])]
            nf.write(f'{sentence}\n')
            nf.write(f'{aspect} {sentiment}\n')
            keyword = f'{aspect}-{sentiment}'
            cnt[keyword] = cnt.get(keyword, 0) + 1

        nf.close
        # Labeled data statistics
        print('Labeled data statistics:')
        print(cnt)


if __name__ == '__main__':
    labeler = Labeler()
    labeler()

