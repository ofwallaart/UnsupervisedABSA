config = {
    'domain': 'kto',
    'device': 'cuda'
}
bert_mapper = {
    'laptop': 'activebus/BERT-DK_laptop',
    'restaurant': 'activebus/BERT-DK_rest',
    'kto': '/dbfs/FileStore/kto/pt_runs/pt_bertselect-Bert' #'/dbfs/FileStore/kto/kto/BERT-DK_kto'
}
path_mapper = {
    'laptop': '/dbfs/FileStore/kto/laptop',
    'restaurant': '/dbfs/FileStore/kto/restaurant',
    'kto': '/dbfs/FileStore/kto/kto'
}
aspect_category_mapper = {
    'laptop': ['support', 'os', 'display', 'battery', 'company', 'mouse', 'software', 'keyboard'],
    'restaurant': ['food', 'place', 'service'],
    'kto': ['service', 'app', 'assortiment', 'beschikbaarheid', 'corona', 'kwaliteit', 'winkel', 'personeel', 'opgeruimd', 'prijzen', 'overig' ]
}
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
        'negative': {"slecht", "betere", "teleurgesteld", "verschrikkelijk", "langzaam", "kapot", "klacht", "vies", "onvriendelijk"}
    }
}
M = {
    'laptop': 150,
    'restaurant': 100,
    'kto': 80
}
K_1 = 5
K_2 = 20
lambda_threshold = 0.5
batch_size = 64
validation_data_size = 150
learning_rate = 1e-5
epochs = 15
