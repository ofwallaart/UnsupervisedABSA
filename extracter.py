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
        self.domain = config['domain']
        if self.domain == 'kto':
            self.smodel = spacy.load('nl_core_news_sm')
        else:
            self.smodel = spacy.load('en_core_web_sm')

        self.topic_model = BERTopic(verbose=True, calculate_probabilities=True)
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
                    if word.tag_.startswith('ADJ' if self.domain == 'kto' else 'JJ') \
                            or word.tag_.startswith('BW' if self.domain == 'kto' else 'RR'):
                        # Adjective or Adverb
                        o.append(word.text)
                    if word.tag_.startswith('N' if self.domain == 'kto' else 'NN'):
                        # Noun
                        a.append(word.text)
                opinions.append(' '.join(o) if len(o) > 0 else '##')
                aspects.append(' '.join(a) if len(a) > 0 else '##')

        return sentences, aspects, opinions
