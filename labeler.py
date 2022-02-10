from config import *
import numpy as np
from sklearn.metrics import classification_report

class Labeler:

    def __init__(self):
        self.domain = config['domain']
        self.root_path = path_mapper[self.domain]
    
    def __call__(self, evaluate=False):
        categories = aspect_category_mapper[self.domain]
        polarities = sentiment_category_mapper[self.domain]

        # Distributions
        dist = {}
        for cat in categories:
            dist[cat] = []
        for pol in polarities:
            dist[pol] = []

        # Read scores
        file = 'scores' if not evaluate else 'scores-test'

        with open(f'{self.root_path}/{file}.txt', 'r', encoding="utf8") as f:
            for idx, line in enumerate(f):
                if idx % 2 == 1:
                    values = line.strip().split()
                    for j, val in enumerate(values):
                        if j % 2 == 1:
                            dist[values[j-1][:-1]].append(float(val))
        
        # Compute mean and sigma for each category
        means = {}
        sigma = {}
        for key in dist:
            means[key] = np.mean(dist[key])
            sigma[key] = np.std(dist[key])

        if not evaluate:
            nf = open(f'{self.root_path}/label.txt', 'w', encoding="utf8")
            cnt = {}
            with open(f'{self.root_path}/{file}.txt', 'r', encoding="utf8") as f:
                sentence = None
                for idx, line in enumerate(f):
                    if idx % 2 == 1:
                        aspect = []
                        sentiment = []
                        key = None
                        for j, val in enumerate(line.strip().split()):
                            if j % 2 == 1:
                                # Normalise score
                                dev = (float(val) - means[key]) / sigma[key]
                                if dev >= lambda_threshold:
                                    if key in categories:
                                        aspect.append(key)
                                    else:
                                        sentiment.append(key)
                            else:
                                key = val[:-1]
                        # No conflict (avoid multi-class sentences)
                        if len(aspect) == 1 and len(sentiment) == 1:
                            nf.write(sentence)
                            nf.write(f'{aspect[0]} {sentiment[0]}\n')
                            keyword = f'{aspect[0]}-{sentiment[0]}'
                            cnt[keyword] = cnt.get(keyword, 0) + 1
                    else:
                        sentence = line
            nf.close()
            # Labeled data statistics
            print('Labeled data statistics:')
            print(cnt)
        else:
            cnt = {}
            aspects = []
            sentiments = []

            with open(f'{self.root_path}/{file}.txt', 'r', encoding="utf8") as f:
                for idx, line in enumerate(f):
                    if idx % 2 == 1:
                        aspect = []
                        sentiment = []
                        aspect_high = 0
                        sentiment_high = 0
                        key = None
                        for j, val in enumerate(line.strip().split()):
                            if j % 2 == 1:
                                # Normalise score
                                dev = (float(val) - means[key]) / sigma[key]
                                if key in categories and dev > aspect_high:
                                    aspect.append(key)
                                elif dev > sentiment_high:
                                    sentiment.append(key)
                            else:
                                key = val[:-1]
                        # No conflict (avoid multi-class sentences)
                        if len(aspect) == 1 and len(sentiment) == 1:
                            keyword = f'{aspect[0]}-{sentiment[0]}'
                            cnt[keyword] = cnt.get(keyword, 0) + 1
                            aspects.append(aspect[0])
                            sentiments.append(sentiment[0])

            test_cats = []
            test_pols = []

            with open(f'{self.root_path}/test.txt', 'r', encoding="utf8") as f:
                for line in f:
                    _, cat, pol, sentence = line.strip().split('\t')
                    cat = int(cat)
                    pol = int(pol)
                    test_cats.append(list(categories)[cat])
                    test_pols.append(list(polarities)[pol])

            predicted = np.array(sentiments)
            actual = np.array(test_pols)
            print("Polarity")
            print(classification_report(actual, predicted, digits=4))
            print()

            predicted = np.array(aspects)
            actual = np.array(test_cats)
            print("Aspect")
            print(classification_report(actual, predicted, digits=4))
            print()
