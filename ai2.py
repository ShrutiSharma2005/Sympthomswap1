import spacy
from spacy.training.example import Example
import random

# Load base model
nlp = spacy.blank("en")  # Or use "en_core_web_sm" if you want to start from pretrained

# Create NER pipeline component
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Define training data (sample — expand this with more cases)
TRAIN_DATA = [
    ("Persistent bloating and abdominal pain", {"entities": [(11, 19, "SYMPTOM"), (24, 38, "SYMPTOM")]}),
    ("Sudden facial paralysis that resolves on its own", {"entities": [(7, 24, "SYMPTOM")]}),
    ("Hives, redness, itching after sun exposure", {"entities": [(0, 5, "SYMPTOM"), (7, 14, "SYMPTOM"), (16, 23, "SYMPTOM")]}),
    ("Papaya leaves juice helps", {"entities": [(0, 18, "REMEDY")]}),
    ("Sweet taste in the mouth with dry saliva", {"entities": [(0, 24, "SYMPTOM"), (30, 41, "SYMPTOM")]}),
    ("Burning mouth with no visible cause", {"entities": [(0, 14, "SYMPTOM")]}),
    ("Ehlers-Danlos Syndrome is a rare condition", {"entities": [(0, 23, "DISEASE")]}),
]

# Add labels to NER
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Disable other components
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for iteration in range(20):
        print(f"Iteration {iteration + 1}")
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.3, losses=losses)
        print(losses)

# Save model
nlp.to_disk("symptomswap_ner_model")
print("Model trained and saved.")
