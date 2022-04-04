# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %sh python -m spacy download nl_core_news_sm

# COMMAND ----------

# MAGIC %sh python -m spacy download en_core_web_sm

# COMMAND ----------

from labeler_sbert import Labeler

labeler = Labeler()
labeler()

# COMMAND ----------

labeler.update_labels(0.3, 0.3)

# COMMAND ----------

from trainer import Trainer

trainer = Trainer()
dataset = trainer.load_training_data()
trainer.train_model(dataset)
trainer.save_model('model')
trainer.load_model('model')
trainer.evaluate()

# COMMAND ----------

from pyspark.sql import functions as F

df = spark.read.options(header='True').option("quote", "\"").option("escape", "\"").csv(r'dbfs:/FileStore/kto/restaurant/predictions.csv')
display(df.where(F.col('predicted polarity') == 'positive'))

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

classes = df.select('actual category').distinct().rdd.flatMap(lambda x: x).collect()

cm = confusion_matrix(df.select('actual category').rdd.flatMap(lambda x: x).collect(), df.select('predicted category').rdd.flatMap(lambda x: x).collect(), labels=classes)
cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

fig, ax = plt.subplots(figsize=(15,15))
cmp.plot(ax=ax, xticks_rotation='vertical')

# COMMAND ----------

path = "dbfs:/FileStore/kto/restaurant-nl/label-sentences.txt"

df1 = spark.read.text(path)
display(df1)
