config = {
    'domain': 'restaurant',
    'device': 'cpu'
}
bert_mapper = {
    'laptop': 'activebus/BERT-DK_laptop',
    'restaurant': 'activebus/BERT-DK_rest',
    'kto': './datasets/kto/BERT-DK_kto'
}
path_mapper = {
    'laptop': './datasets/laptop',
    'restaurant': './datasets/restaurant',
    'kto': './datasets/kto'
}
aspect_category_mapper = {
    'laptop': ['support', 'os', 'display', 'battery', 'company', 'mouse', 'software', 'keyboard'],
    'restaurant': ['food', 'place', 'service'],
    'kto': ['zelfscan', 'kassa', 'servicebalie',
            'app', 'assortiment', 'beschikbaarheid', 'beschikbaarheid_bonus', ]
            # 'beschikbaarheid_brood', 'corona', 'kwaliteit_groenten', 'kwaliteit_fruit', 'kwaliteit_producten',
            # 'indeling_winkel', 'omgeving_winkel', 'indeling_winkel', 'internet', 'muziek', 'klantvriendelijkheid',
            # 'aanwezigheid_personeel', "medw_karren_indeweg", "prijzen", "schoon_opgeruimd", "reeds_vermeld"]
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
        'zelfscan': {"scanner", "zelfscan", "handscanner", "houders", "afrekenen"},
        'kassa': {"kassa", "afrekenen", "rij", "rolband", "open", "bemand"},
        'servicebalie': {"servicebalie", "wachttijd", "service", "balie", "servicedesk"},
        'app': {"app", "looproute", "boodschappenlijst", "gebruik"},
        'assortiment': {"assortiment", "aanbod", "biologische", "vegan"},
        'beschikbaarheid': {"beschikbaarheid", "uitverkocht", "voorraad", "verkrijgbaar", "leeg", "bijvullen"},
        'beschikbaarheid_bonus': {"bonus", "aanbieding", "persoonlijk", "beschikbaarheid", "uitverkocht", "voorraad", "verkrijgbaar", "leeg", "bijvullen", "bonusbox"},
        # 'beschikbaarheid_brood': {"brood", "afbakken", "beschikbaarheid", "bakken", "leeg"},
        # 'corona': {"corona", "covid", "mondkapje", "desinfecteren", "coronomaatregelen", "karretjes"},
        # 'kwaliteit_groenten': {"groenten", "vers", "kwaliteit", "rot", "beschimmeld", "houdbaarheid"},
        # 'kwaliteit_fruit': {"fruit", "vers", "kwaliteit", "beurs", "beschimmeld", "rijp", "houdbaarheid"},
        # 'kwaliteit_producten': {"kwaliteit", "producten", "houdbaarheid", "fifo", "houdbaarheidsdatum"},
        # 'indeling_winkel': {"smal", "krap", "indeling", "toilet", "ruimte", "opzet", "uitbreiden", "ingang", "inpaktafel"},
        # 'omgeving_winkel': {"parkeren", "fiets", "buiten", "parkeerplaats", "fietsenstalling", "hangjongeren", "garage"},
        # 'internet': {"internet", "wifi", "draadloos", "verbinding", "netwerk"},
        # 'muziek': {"muziek", "radio", "liedjes", "geluid", "afspeellijst"},
        # 'klantvriendelijkheid': {"klantvriendelijk", "personeel", "begroeten", "service", "behulpzaam", "vriendelijk", "hulp"},
        # 'aanwezigheid_personeel': {"personeel", "weinig", "geen", "vragen", "nauwelijks", "aanwezig", "aanspreken"},
        # 'medw_karren_indeweg': {"versperren", "gangpad", "blokkeren", "obstakels", "vakkenvullers"},
        # 'prijzen': {"prijs", "duur", "verrekend", "goedkoper", "35%"},
        # 'schoon_opgeruimd': {"rommelig", "smerig", "zwerfafval", "vies", "opruimen", "schoon", "opgeruimd", "spiegelen", "prullenbak"},
        # 'reeds_vermeld': {"reeds", "hiervoor", "vermeld", "aangegeven"},
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
        'positive': {"goed", "prima", 'uitstekend', "excellent", "perfect", "goedkoop", "best", "cheap", "snel"},
        'negative': {"slecht", "betere", "teleurgesteld", "verschrikkelijk", "klein", "langzaam", "kapot", "klacht", "onzin", "jammer", "vies"}
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
validation_data_size = 100
learning_rate = 1e-5
epochs = 10
