import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from config import *
import re


def load_training_data(file_path):
    sentences = []
    for line in open(file_path, encoding="utf-8"):
        split_lines = list(filter(None, re.split('; |\. |\! |\n|\? ', line.lower())))
        for split_line in split_lines:
          sentences.append(split_line.strip())
    return sentences


class Labeler:
    def __init__(self):
        self.domain = config['domain']
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=config['device']) # paraphrase-multilingual-mpnet-base-v2
        self.root_path = path_mapper[self.domain]
        self.cat_threshold = 0.4
        self.pol_threshold = 0.3

    def __call__(self):
        categories = aspect_category_mapper[self.domain]
        polarities = sentiment_category_mapper[self.domain]

        category_seeds = aspect_seed_mapper[self.domain]
        polarity_seeds = sentiment_seed_mapper[self.domain]

        split = [len(categories), len(polarities)]

        # Seeds
        seeds = {}
        for cat in categories:
            seeds[cat] = " ".join(category_seeds[cat])
        for pol in polarities:
            seeds[pol] = " ".join(polarity_seeds[pol])

        seed_embeddings = self.model.encode(list(seeds.values()), convert_to_tensor=True, show_progress_bar=True)

        # Load and encode the train set
        sentences = load_training_data(f'{self.root_path}/train.txt')
        embeddings = self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)

        # Compute cosine-similarities
        cosine_category_scores, cosine_polarity_scores = torch.split(util.cos_sim(seed_embeddings, embeddings), split)

        category_argmax = torch.argmax(cosine_category_scores, dim=0).tolist()
        category_max = torch.max(cosine_category_scores, dim=0)[0].tolist()

        polarity_argmax = torch.argmax(cosine_polarity_scores, dim=0).tolist()
        polarity_max = torch.max(cosine_polarity_scores, dim=0)[0].tolist()

        labels = np.array([category_argmax, category_max, polarity_argmax, polarity_max, np.arange(0, len(sentences))])

        # No conflict (avoid multi-class sentences)
        labels = np.transpose(labels[:, (labels[1, :] >= self.cat_threshold) & (labels[3, :] >= self.pol_threshold)])

        nf = open(f'{self.root_path}/label-sbert.txt' , 'w', encoding="utf8")
        cnt = {}

        for label in labels:
            sentence = sentences[int(label[4])]
            aspect = categories[int(label[0])]
            sentiment = polarities[int(label[2])]
            nf.write(f'{sentence}\n')
            nf.write(f'{aspect} {sentiment}\n')
            keyword = f'{aspect}-{sentiment}'
            cnt[keyword] = cnt.get(keyword, 0) + 1

        nf.close
        # Labeled data statistics
        print('Labeled data statistics:')
        print(cnt)

if __name__ == '__main__':
    path = r".\train.txt"
    labeler = Labeler()
    labeler()
