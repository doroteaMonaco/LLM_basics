
dictionary= {
    "D1": "Stupida",
    "D2": "Cogliona",
    "D3": "Idiota",
    "D4": "Imbecille",
    "Cervello": 0
}

#dentro il dizionario può essere inserito qualsiasi cosa, anche liste e set e altri dizionari

print(dictionary)
print(dictionary["D3"])

print(dictionary.keys())
print(dictionary.values())
print(dictionary.items())

dictionary["D5"] = "Ritardata"
dictionary.update({"D6": "Scema"})

print(dictionary)

copia = dictionary.copy()
print(copia)

nested = {
    "Motivo1": {
        "Scema": "scemina",
        "Stupida": "stupidina"
    },
    "Motivo2": {
        "Cogliona": "coglione",
        "Idiota": "idiotina"
    },
    "Motivo3": {
        "Imbecille": "imbecillina",
        "Ritardata": "ritardatina"
    }
}

for it in nested.items():
    print(it)

print(len(nested))