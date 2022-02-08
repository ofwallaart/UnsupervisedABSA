from trainer_sbert import TrainerSbert
from vocab_generator import VocabGenerator
from extracter import Extracter
from score_computer import ScoreComputer
from labeler_sbert import Labeler
from trainer import Trainer
import pickle

def save_obj(obj, name ):
    with open('obj/kto/' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/kto/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# vocabGenerator = VocabGenerator()
# aspect_vocabularies, sentiment_vocabularies = vocabGenerator()
#
# extracter = Extracter()
# sentences, aspects, opinions, sentence_aspects = extracter()
#
# save_obj(aspect_vocabularies, 'aspect_vocabularies')
# save_obj(sentiment_vocabularies, 'sentiment_vocabularies')
# save_obj(sentences, 'sentences')
# save_obj(aspects, 'aspects')
# save_obj(opinions, 'opinions')
# save_obj(sentence_aspects, 'sentence_aspects')
#
# aspect_vocabularies = load_obj('aspect_vocabularies')
# sentiment_vocabularies = load_obj('sentiment_vocabularies')
# sentences = load_obj('sentences')
# aspects = load_obj('aspects')
# opinions = load_obj('opinions')
# sentence_aspects = load_obj('sentence_aspects')
#
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

# print('-------')
# print('Start training model with Sentence Transformers')
#
# trainersbert = TrainerSbert()
# dataset = trainersbert.load_training_data()
# trainersbert.train_model(dataset)
# trainersbert.save_model('model_sbert')
# trainersbert.load_model('model_sbert')
# trainersbert.evaluate()








