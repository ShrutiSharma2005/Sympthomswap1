import spacy
from flask import Flask, request, jsonify

# Load spaCy model
nlp = spacy.load("en_core_web_md")  # Use medium model for better similarity matching

app = Flask(__name__)

# Sample ingredient database
medicine_ingredients = {
    "Papaya Juice": ["Papain", "Vitamin C"],
    "Paracetamol": ["Acetaminophen"],
    "Boiled Papaya Leaves": ["Papain", "Flavonoids"]
}

# Sample allergy-safe alternatives
medicine_alternatives = {
    "Papaya Juice": ["Boiled Papaya Leaves"],
    "Paracetamol": ["Ibuprofen"]
}

# Sample user allergy list
user_allergies = ["Papain"]

@app.route('/symptom_match', methods=['POST'])
def symptom_match():
    data = request.json
    symptom1 = nlp(data['symptom1'])
    symptom2 = nlp(data['symptom2'])
    similarity = symptom1.similarity(symptom2)

    return jsonify({
        "similarity_score": similarity,
        "matched": similarity > 0.85
    })


@app.route('/validate_medicine', methods=['POST'])
def validate_medicine():
    data = request.json
    symptom = data["symptom"]
    medicine = data["medicine"]

    # Mock validation logic
    is_valid = symptom.lower() in ["abdominal pain", "fever", "headache"] and medicine in medicine_ingredients
    return jsonify({
        "valid": is_valid,
        "message": "Medicine is appropriate." if is_valid else "Medicine is not recommended."
    })


@app.route('/check_allergy', methods=['POST'])
def check_allergy():
    data = request.json
    medicine = data["medicine"]
    ingredients = medicine_ingredients.get(medicine, [])
    allergens = [ing for ing in ingredients if ing in user_allergies]

    if allergens:
        # Suggest alternative
        alternatives = medicine_alternatives.get(medicine, [])
        return jsonify({
            "allergic": True,
            "allergens_found": allergens,
            "alternatives": alternatives
        })
    else:
        return jsonify({
            "allergic": False,
            "message": "No allergens found. Safe to use."
        })


if __name__ == '__main__':
    app.run(debug=True, port=5001)