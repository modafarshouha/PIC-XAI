# https://www.datacamp.com/cheat-sheet/spacy-cheat-sheet-advanced-nlp-in-python

import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("A person is jumping over a mountain")

t = [token.dep_ for token in doc]
l = [(ent.text, ent.label_) for ent in doc.ents]

print(len(t))
print(t)
print([spacy.explain(i) for i in t])

print(len(l))
print(l)