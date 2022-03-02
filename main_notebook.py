# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %sh python -m spacy download nl_core_news_sm

# COMMAND ----------

# MAGIC %sh python -m spacy download en_core_web_sm

# COMMAND ----------

from labeler import Labeler
from vocab_generator import VocabGenerator
from extracter import Extracter
from score_computer import ScoreComputer
from trainer import Trainer
import pickle
import time

date = time.strftime("%Y%m%d_%H%M%S")

def save_obj(obj, name ):
    with open(r'/dbfs/FileStore/kto/kto/store/' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(r'/dbfs/FileStore/kto/kto/store/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def CASCrun():
  vocabGenerator = VocabGenerator()
  aspect_vocabularies, sentiment_vocabularies = vocabGenerator()

  # save_obj(aspect_vocabularies, 'aspect_vocabularies')
  # save_obj(sentiment_vocabularies, 'sentiment_vocabularies')

  # aspect_vocabularies = load_obj('aspect_vocabularies')
  # sentiment_vocabularies = load_obj('sentiment_vocabularies')

  extracter = Extracter()
  sentences, aspects, opinions = extracter()

  scoreComputer = ScoreComputer(aspect_vocabularies, sentiment_vocabularies)
  scoreComputer(sentences, aspects, opinions)

  labeler = Labeler()
  labeler()

  trainer = Trainer()
  dataset = trainer.load_training_data()
  trainer.train_model(dataset)
  trainer.save_model(f'model_single_{date}')
  trainer.load_model(f'model_single_{date}')
  return trainer.evaluate()

CASCrun()

# COMMAND ----------

trainer = Trainer()
dataset = trainer.load_training_data()
trainer.train_model(dataset)
trainer.evaluate()

# COMMAND ----------

# sentences, aspects, opinions = extracter(evaluate=True)
# scoreComputer(sentences, aspects, opinions, evaluate=True)
# labeler(evaluate=True)

# COMMAND ----------

RUNS = 5
polarity_list, aspect_list = [], []
for i in range(RUNS):
    print('RUN: ', i)
    polarity, aspect = CASCrun()
    polarity_list.append(polarity)
    aspect_list.append(aspect)
    
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

polarity_list

# COMMAND ----------

from pyspark.sql import functions as F

df = spark.read.options(header='True').option("quote", "\"").option("escape", "\"").csv(r'dbfs:/FileStore/kto/restaurant/predictions.csv')
display(df.where(F.col('predicted category') == 'place'))

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

classes = df.select('actual category').distinct().rdd.flatMap(lambda x: x).collect()

cm = confusion_matrix(df.select('actual category').rdd.flatMap(lambda x: x).collect(), df.select('predicted category').rdd.flatMap(lambda x: x).collect(), labels=classes)
cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

fig, ax = plt.subplots(figsize=(15,15))
cmp.plot(ax=ax, xticks_rotation='vertical')

# COMMAND ----------


