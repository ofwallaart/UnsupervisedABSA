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

# MAGIC %sh python -m spacy download en_core_web_sm

# COMMAND ----------

from labeler import Labeler
from vocab_generator import VocabGenerator
from extracter import Extracter
from score_computer import ScoreComputer
from trainer import Trainer
import pickle

vocabGenerator = VocabGenerator()
aspect_vocabularies, sentiment_vocabularies = vocabGenerator()

extracter = Extracter()
sentences, aspects, opinions = extracter()

scoreComputer = ScoreComputer(aspect_vocabularies, sentiment_vocabularies)
scoreComputer(sentences, aspects, opinions)

labeler = Labeler()
labeler()

trainer = Trainer()
dataset = trainer.load_training_data()
trainer.train_model(dataset)
trainer.save_model('model')
trainer.load_model('model')
trainer.evaluate()

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
