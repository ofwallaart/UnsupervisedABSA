from trainer import Trainer

test_sentences = [
    'Geen klanten zonder mondkapjes en zorgen dat producten niet op zijn .',
    'Controle op naleving mondkapjesplicht, en een betere WIFI verbinding, deze valt regelmatig weg.',
    'Doe iets aan de zooi en de klantvriendelijkheid.',
    'Bevoorrading kan beter. Amandelmelk was op van eigen merk en laatst de halfvolle melk ook van eigen merk',
    'Zorg dat het assortiment op voorraad is.  Bonusbox aanbiedingen functioneren vaak niet.'
    ]

trainer = Trainer()
trainer.load_model('model')

predicted_aspects, _, _, _ = trainer.predict_multiple(test_sentences, 2)
print(predicted_aspects)
