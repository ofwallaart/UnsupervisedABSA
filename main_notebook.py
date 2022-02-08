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

# COMMAND ----------

# MAGIC %md ## Apply business rules

# COMMAND ----------

# MAGIC %run /Basics/imports

# COMMAND ----------

rules = [
  {
    'rule': 'App', 
    'map_condition': ['app', 'ah-app', 'ahapp'], 
    'map_to_aspect': 'app'
  },
  {
    'rule': 'Corona', 
    'map_condition': ['corona', 'covid', 'covid-19', 'mondkap', 'mondkapje', 'mondkapjes', 'mondmaskers', '1.5 meter', '1,5 meter', '1,5 mtr', '1.5 mtr', '1,5m', '1.5m', '1,5 m', '1.5 m'], 
    'map_to_aspect': 'corona'
  },
  {
    'rule': 'Indeling', 
    'map_condition': ['indeling van de winkel', 'klantentoilet', 'klanten wc', 'toilet', 'koffiecorner', 'koffiecorner', 'kleine winkel', 'winkelindeling', 'wc', 'koffiehoek', 'inpaktafel', 'inpak tafel', 'winkel uitbreiden', 'ruimere winkel', 'groter pand', 'groter winkeloppervlak', 'ruimer opzetten', 'ruimere opzet'], 
    'map_to_aspect': 'winkel'
  },
  {
    'rule': 'Muziek / Wifi', 
    'map_condition': ['muziek', 'wifi', 'geen internet', 'achtergrondmuziek'], 
    'map_to_aspect': 'winkel'},
  {
    'rule': 'Fiets / garage', 
    'map_condition': ['fietsenstalling', 'fietsenrekken', 'fiets', 'fietsen', 'buitenterrein', 'garage', 'parkeerterrein', 'parkeerplaatsen', 'parkeergarage', 'parkeerplaats'], 
    'map_to_aspect': 'winkel'
  },
  {
    'rule': 'Goed / Doorgaan', 
    'map_condition': ['ga zo door', 'geen tips', 'gewoon zo doorgaan', 'vooral zo doorgaan', 'zo door gaan', 'zo doorgaan', 'ga vooral zo door', 'doorgaan'], 
    'map_to_aspect': 'winkel', 
    'map_to_sentiment' : 'positive', 
    'max_length': 5 
  },
  {
    'rule': 'Overig', 
    'map_condition': ['bezorgdienst', 'bezorging', 'bezorgservice', 'bestelmogelijkheid', 'thuisbezorgen', 'duurzaam', 'duurzaamheid', 'duurzamer', 'bezorgen', 'plastic verpakking', 'plastic verpakkingen'], 
    'map_to_aspect': 'overig'
  },
  {
    'rule': 'Kwaliteit', 
    'map_condition': ['beschimmeld', 'rijp', 'fifo', 'versheid'], 
    'map_to_aspect': 'kwaliteit'},
  {
    'rule': 'Service', 
    'map_condition': ['creditscards', 'credit cards', 'creditcard', 'credit card', 'visa'], 
    'map_to_aspect': 'service' 
  },
  {
    'rule': 'Assortiment', 
    'map_condition': ['assortimentskeuze', 'meer vegetarische', 'meer vega', 'meer lokale producten', 'meer vegan'], 
    'map_to_aspect': 'assortiment'
  }
]

# COMMAND ----------

sentences = df.select(F.lower('sentence')).rdd.flatMap(lambda x: x).collect()
cat = df.select(F.lower('actual category')).rdd.flatMap(lambda x: x).collect()
rule_overwrite_aspect = [None for _ in range(len(sentences))]
rule_overwrite_polarity = [None for _ in range(len(sentences))]
for i, sentence in enumerate(sentences):
  for rule in rules:
    if 'max_length' in rule:
      if len(sentence.split()) > rule['max_length']:
        continue
    for condition in rule['map_condition']:
      if i == 218:
        print(condition, sentence)
      if " " + condition + " " in " " + sentence + " ":
        rule_overwrite_aspect[i] = rule['map_to_aspect']
        print(cat[i], '\t', rule['map_to_aspect'], ' ', sentence)
        if 'map_to_sentiment' in rule:
          rule_overwrite_polarity[i] = rule['map_to_sentiment']
        break
        
pd_df = df.toPandas()
pd_df['rule_overwrite_aspect'] = rule_overwrite_aspect
pd_df['rule_overwrite_polarity'] = rule_overwrite_polarity

pd_df['merged_aspect'] = pd_df['rule_overwrite_aspect'].combine_first(pd_df['predicted category'])
pd_df['merged_polarity'] = pd_df['rule_overwrite_polarity'].combine_first(pd_df['predicted polarity'])

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

classes = df.select('actual category').distinct().rdd.flatMap(lambda x: x).collect()

print(classification_report(pd_df['actual category'], pd_df['merged_aspect']))

cm = confusion_matrix(pd_df['actual category'], pd_df['merged_aspect'], labels=classes)
cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

fig, ax = plt.subplots(figsize=(15,15))
cmp.plot(ax=ax, xticks_rotation='vertical')
