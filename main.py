from trainer_sbert import TrainerSbert
from vocab_generator import VocabGenerator
from extracter import Extracter
from score_computer import ScoreComputer
from labeler import Labeler
from trainer import Trainer
import pickle

def save_obj(obj, name ):
    with open('/dbfs/FileStore/kto/kto/store/' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('/dbfs/FileStore/kto/kto/store/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def run():
  vocabGenerator = VocabGenerator()
  aspect_vocabularies, sentiment_vocabularies = vocabGenerator()

  extracter = Extracter()
  sentences, aspects, opinions = extracter()

  save_obj(aspect_vocabularies, 'aspect_vocabularies')
  save_obj(sentiment_vocabularies, 'sentiment_vocabularies')
  save_obj(sentences, 'sentences')
  save_obj(aspects, 'aspects')
  save_obj(opinions, 'opinions')

  # aspect_vocabularies = load_obj('aspect_vocabularies')
  # sentiment_vocabularies = load_obj('sentiment_vocabularies')
  # sentences = load_obj('sentences')
  # aspects = load_obj('aspects')
  # opinions = load_obj('opinions')

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

# print('-------')
# print('Start training model with Sentence Transformers')
#
# trainersbert = TrainerSbert()
# dataset = trainersbert.load_training_data()
# trainersbert.train_model(dataset)
# trainersbert.save_model('model_sbert')
# trainersbert.load_model('model_sbert')
# trainersbert.evaluate()








