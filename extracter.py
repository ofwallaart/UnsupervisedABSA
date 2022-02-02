from tqdm import tqdm
from config import *
from bertopic import BERTopic
import spacy
import numpy as np
from sklearn.metrics import pairwise

class Extracter:
    '''
    Extract potential-aspects and potential-opinion words
    '''

    def __init__(self):
        spacy.prefer_gpu()
        self.smodel = spacy.load('en_core_web_sm') #nl_core_news_sm
        self.topic_model = BERTopic(verbose=True, calculate_probabilities=True)
        self.domain = config['domain']
        self.root_path = path_mapper[self.domain]

    def __call__(self):
        # Extract potential-aspects and potential-opinions
        sentences = []
        aspects = []
        opinions = []

        with open(f'{self.root_path}/train.txt', encoding="utf8") as f:
            for line in tqdm(f):
                text = line.strip()
                sentences.append(text)
                words = self.smodel(text)
                o = []
                a = []
                for word in words:
                    if word.tag_.startswith('JJ') or word.tag_.startswith('RR'): #ADJ| BW
                        # Adjective or Adverb
                        o.append(word.text)
                    if word.tag_.startswith('NN'): #N|
                        # Noun
                        a.append(word.text)
                opinions.append(' '.join(o) if len(o) > 0 else '##')
                aspects.append(' '.join(a) if len(a) > 0 else '##')

        print('Training topic model:')
        aspect_categories = aspect_category_mapper[self.domain]
        aspect_seeds = aspect_seed_mapper[self.domain]

        sentence_aspects = []
        cat_table = {}
        cat_vec = []

        topics, probs = self.topic_model.fit_transform(sentences)

        for category in aspect_categories:
            data = self.topic_model.find_topics(" ".join(list(aspect_seeds[category])), top_n=10)
            cat_vec.append(" ".join(list(aspect_seeds[category])))
            cat_table[category] = [data[0][i] for i in [i for i, e in enumerate(data[1]) if e >= 0.4]]

        cat_vec_table = self.topic_model.transform(cat_vec)[1]

        vecs = pairwise.cosine_similarity(cat_vec_table, probs)

        vecs_list = np.argmax(vecs, axis=0).tolist()

        for vec in vecs_list:
            sentence_aspects.append(aspect_categories[vec])

        # for topic in topics:
        #     for key, value in cat_table.items():
        #         if topic in value:
        #             sentence_aspects.append(key)
        #             break

        return sentences, aspects, opinions, sentence_aspects
