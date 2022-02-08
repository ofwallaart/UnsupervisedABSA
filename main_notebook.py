# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

pip install -r requirements.txt

# COMMAND ----------

# MAGIC %pip install -U tokenizers tqdm
# MAGIC %pip install -U sentence-transformers
# MAGIC %pip install -U bertopic

# COMMAND ----------

# MAGIC %sh python -m spacy download nl_core_news_sm

# COMMAND ----------

from vocab_generator import VocabGenerator
from extracter import Extracter
from score_computer import ScoreComputer
from labeler import Labeler
from trainer import Trainer
import pickle

def save_obj(obj, name ):
    with open(r'/dbfs/FileStore/kto/kto/store/' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(r'/dbfs/FileStore/kto/kto/store/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# vocabGenerator = VocabGenerator()
# aspect_vocabularies, sentiment_vocabularies = vocabGenerator()

# extracter = Extracter()
# sentences, aspects, opinions = extracter()

# save_obj(aspect_vocabularies, 'aspect_vocabularies')
# save_obj(sentiment_vocabularies, 'sentiment_vocabularies')
# save_obj(sentences, 'sentences')
# save_obj(aspects, 'aspects')
# save_obj(opinions, 'opinions')

# aspect_vocabularies = load_obj('aspect_vocabularies')
# sentiment_vocabularies = load_obj('sentiment_vocabularies')
# sentences = load_obj('sentences')
# aspects = load_obj('aspects')
# opinions = load_obj('opinions')

# scoreComputer = ScoreComputer(aspect_vocabularies, sentiment_vocabularies)
# scoreComputer(sentences, aspects, opinions)

# labeler = Labeler()
# labeler()

trainer = Trainer()
dataset = trainer.load_training_data()
trainer.train_model(dataset)
trainer.save_model('model')
trainer.load_model('model')
trainer.evaluate()

# COMMAND ----------

from labeler_sbert import Labeler
from trainer import Trainer
import pickle

labeler = Labeler()
labeler()

trainer = Trainer()
dataset = trainer.load_training_data(sbert=True)
trainer.train_model(dataset)
trainer.save_model('model')
trainer.load_model('model')
trainer.evaluate()

# COMMAND ----------

df = spark.read.options(header='True').option("quote", "\"").option("escape", "\"").csv(r'dbfs:/FileStore/kto/kto/predictions.csv')
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

classes = df.select('actual polarity').distinct().rdd.flatMap(lambda x: x).collect()

cm = confusion_matrix(df.select('actual polarity').rdd.flatMap(lambda x: x).collect(), df.select('predicted polarity').rdd.flatMap(lambda x: x).collect(), labels=classes)
cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

fig, ax = plt.subplots(figsize=(5,5))
cmp.plot(ax=ax, xticks_rotation='vertical')
