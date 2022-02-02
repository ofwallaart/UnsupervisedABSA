from config import *
import numpy as np

class Labeler:

    def __init__(self):
        self.domain = config['domain']
        self.root_path = path_mapper[self.domain]
    
    def __call__(self, sentence_aspects):
        categories = aspect_category_mapper[self.domain]
        polarities = sentiment_category_mapper[self.domain]

        # Distributions
        dist = {}
        for cat in categories:
            dist[cat] = []
        for pol in polarities:
            dist[pol] = []

        # Read scores
        with open(f'{self.root_path}/scores.txt', 'r', encoding="utf8") as f:
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
        
        nf = open(f'{self.root_path}/label.txt', 'w', encoding="utf8")
        cnt = {}
        with open(f'{self.root_path}/scores.txt', 'r', encoding="utf8") as f:
            sentence = None
            sentence_aspect = None
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
                        # nf.write(f'{aspect[0]} {sentiment[0]}\n')
                        # keyword = f'{aspect[0]}-{sentiment[0]}'
                        nf.write(f'{sentence_aspect} {sentiment[0]}\n')
                        keyword = f'{sentence_aspect}-{sentiment[0]}'
                        cnt[keyword] = cnt.get(keyword, 0) + 1
                else:
                    sentence = line
                    sentence_aspect = sentence_aspects[int(idx/2)]
        nf.close()
        # Labeled data statistics
        print('Labeled data statistics:')
        print(cnt)