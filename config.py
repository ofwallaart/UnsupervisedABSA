config = {
    'domain': 'restaurant',
    'device': 'cuda'
}
bert_mapper = {
    'laptop': 'activebus/BERT-DK_laptop',
    'restaurant': 'activebus/BERT-DK_rest',
    'kto': './datasets/kto/BERT-DK_kto'
}

sbert_mapper = {
    'laptop': 'all-mpnet-base-v2',
    'restaurant': 'all-mpnet-base-v2',
    'kto': 'paraphrase-multilingual-mpnet-base-v2'
}

path_mapper = {
    'laptop': './datasets/laptop',
    'restaurant': './datasets/restaurant',
    'kto': './datasets/kto'
}
aspect_category_mapper = {
    'laptop': ['support', 'os', 'display', 'battery', 'company', 'mouse', 'software', 'keyboard'],
    'restaurant': ['food', 'place', 'service'],
    'kto': ['service', 'app', 'assortiment', 'beschikbaarheid', 'corona', 'kwaliteit', 'winkel', 'personeel',
            'opgeruimd', 'prijzen', 'overig']}
aspect_seed_mapper = {
    'laptop': {
        'support': {"support", "service", "warranty", "coverage", "replace"},
        'os': {"os", "windows", "ios", "mac", "system", "linux"},
        'display': {"display", "screen", "led", "monitor", "resolution"},
        'battery': {"battery", "life", "charge", "last", "power"},
        'company': {"company", "product", "hp", "toshiba", "dell", "apple", "lenovo"},
        'mouse': {"mouse", "touch", "track", "button", "pad"},
        'software': {"software", "programs", "applications", "itunes", "photo"},
        'keyboard': {"keyboard", "key", "space", "type", "keys"}
    },
    'restaurant': {
        'food': {"food", "spicy", "sushi", "pizza", "taste", "delicious", "bland", "drinks", "flavourful"},
        'place': {"ambience", "atmosphere", "seating", "surroundings", "environment", "location", "decoration", "spacious", "comfortable", "place"},
        'service': {"tips", "manager", "waitress", "rude", "forgetful", "host", "server", "service", "quick", "staff"}
    },
    'kto': {
        'service': {"scanner", "zelfscan", "handscanner", "houders", "afrekenen", "kassa", "rij", "wachttijd", "servicebalie", "wachttijd", "service", "balie", "servicedesk"},
        'kwaliteit': {"vers", "kwaliteit", "rot", "beschimmeld", "houdbaarheid"},
        'app': {"app", "looproute"},
        'winkel': {"Parkeren","fiets","buiten","parkeerplaats","fietsenstalling","hangjongeren","garage", "smal", "indeling", "toilet", "ruimte", "opzet", "uitbreiden", "ingang", "inpaktafel", "wifi", "internet", "muziek"},
        'assortiment': {"assortiment", "aanbod", "biologische", "vegan"},
        'beschikbaarheid': {"beschikbaarheid", "uitverkocht", "voorraad", "verkrijgbaar", "leeg", "aanvullen", "brood", "afbakken", "bonus"},
        'personeel': {"klantvriendelijk", "begroeten", "behulpzaam", "vriendelijk", "hulp", "personeel", "weinig", "vragen", "aanwezig", "aanspreken", "versperren", "gangpad", "blokkeren", "obstakels", "vakkenvullers"},
        'corona': {"corona", "covid", "mondkapje", "desinfecteren", "coronamaatregelen", "maatregelen", "winkelwagen"},
        'opgeruimd': {"rommelig", "smerig", "afval", "vies", "opruimen", "schoon", "opgeruimd", "spiegelen", "prullenbak"},
        'prijzen': {"prijs", "duur", "goedkoper", "35%"},
        'overig': {"duurzaam", "bloemen", "bezorgen", "reeds", "vermeld", "enquete", "eerdere", "opmerking"}
    }
}
sentiment_category_mapper = {
    'laptop': ['negative', 'positive'],
    'restaurant': ['negative', 'positive'],
    'kto': ['negative', 'positive']
}
sentiment_seed_mapper = {
    'laptop': {
        'positive': {"good", "great", 'nice', "excellent", "perfect", "impressed", "best", "thin", "cheap", "fast"},
        'negative': {"bad", "disappointed", "terrible", "horrible", "small", "slow", "broken", "complaint", "malware", "virus", "junk", "crap", "cramped", "cramp"}
    },
    'restaurant': {
        'positive': {"good", "great", 'nice', "excellent", "perfect", "fresh", "warm", "friendly", "delicious", "fast", "quick", "clean"},
        'negative': {"bad", "terrible", "horrible", "tasteless", "awful", "smelled", "unorganized", "gross", "disappointment", "spoiled", "vomit", "cold", "slow", "dirty", "rotten", "ugly"}
    },
    'kto': {
        'positive': {"goed", 'uitstekend', "excellent", "perfect"},
        'negative': {"slecht", "betere", "teleurgesteld", "verschrikkelijk", "langzaam", "kapot", "klacht", "vies"}
    }
}

weights = {
    'aspect_weights': {
        'laptop': [100, 100, 100, 100, 100],
        'restaurant': [345, 67, 201],
        'kto': [164, 31, 73, 182, 78, 45, 173, 87, 66, 13, 50]
    },
    'sentiment_weights': {
        'laptop': [100, 100],
        'restaurant': [231, 382],
        'kto': [927, 35]
    }
}

M = {
    'laptop': 150,
    'restaurant': 100,
    'kto': 100
}
K_1 = 10
K_2 = 30
lambda_threshold = 0.5
batch_size = 12
validation_data_size = 50
learning_rate = 1e-5
epochs = 2
