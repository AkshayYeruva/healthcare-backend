import os
import json
import numpy as np
from PIL import Image
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Router output mapping (verified by probing master_router_model.tflite)
ROUTER_CLASSES = {
    0: "brain",
    1: "eye",
    2: "lung",
    3: "oral",
    4: "skin",
    5: "wound",
}

SPECIALIST_MODELS = {
    "brain": "brain_disease_model.tflite",
    "eye":   "eye_disease_model.tflite",
    "nail":  "nail_disease_model.tflite",
    "oral":  "oral_disease_model.tflite",
    "skin":  "skin_model.tflite",
    "lung":  "lung_model.tflite",
}

SPECIALIST_LABELS = {
    "brain": ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"],
    "eye":   ["normal", "cataract", "glaucoma", "diabetic_retinopathy"],
    "nail":  ["Healthy", "onychomycosis", "nail_psoriasis", "paronychia", "onycholysis", "melanonychia"],
    "oral":  ["Healthy", "dental_caries", "gingivitis", "ulcers", "calculus", "discoloration", "hypodontia"],
    "skin":  ["Melanocytic nevi", "Melanoma", "Benign keratosis", "Basal cell carcinoma",
              "Actinic keratoses", "Vascular lesions", "Dermatofibroma"],
    "lung":  ["NORMAL", "PNEUMONIA"],
}

FALLBACK_ADVICE = {
    "nail": {
        "Healthy":        {"info": "Nails appear healthy.", "precautions": ["Maintain nail hygiene.", "Keep nails trimmed."], "urgency": "Low"},
        "onychomycosis":  {"info": "Fungal nail infection.", "precautions": ["Keep nails dry and clean.", "Use antifungal treatment as prescribed.", "Avoid sharing footwear."], "urgency": "Moderate"},
        "nail_psoriasis": {"info": "Psoriasis affecting the nails.", "precautions": ["Consult a dermatologist.", "Keep nails moisturised.", "Avoid injury to nails."], "urgency": "Moderate"},
        "paronychia":     {"info": "Infection around the nail fold.", "precautions": ["Soak in warm water.", "See a doctor if pus is present.", "Keep area clean and dry."], "urgency": "Moderate"},
        "onycholysis":    {"info": "Separation of nail from nail bed.", "precautions": ["Keep nails short.", "Avoid trauma.", "Consult a doctor if spreading."], "urgency": "Moderate"},
        "melanonychia":   {"info": "Dark pigmentation in the nail.", "precautions": ["See a dermatologist immediately.", "Do not ignore dark streaks in nails."], "urgency": "High"},
    },
    "oral": {
        "Healthy":       {"info": "Oral cavity appears healthy.", "precautions": ["Brush twice daily.", "Floss regularly.", "Visit dentist every 6 months."], "urgency": "Low"},
        "dental_caries": {"info": "Tooth decay caused by bacteria.", "precautions": ["Visit a dentist.", "Reduce sugary food intake.", "Use fluoride toothpaste."], "urgency": "Moderate"},
        "gingivitis":    {"info": "Inflammation of the gums.", "precautions": ["Improve brushing and flossing.", "Use antiseptic mouthwash.", "See a dentist."], "urgency": "Moderate"},
        "ulcers":        {"info": "Painful sores in the mouth.", "precautions": ["Avoid spicy or acidic food.", "Use topical anesthetic gel.", "See a doctor if lasting over 2 weeks."], "urgency": "Moderate"},
        "calculus":      {"info": "Hardened plaque on teeth.", "precautions": ["Professional dental cleaning needed.", "Improve daily brushing."], "urgency": "Moderate"},
        "discoloration": {"info": "Staining or colour change in teeth.", "precautions": ["Consult a dentist.", "Reduce coffee and tea intake.", "Consider professional whitening."], "urgency": "Low"},
        "hypodontia":    {"info": "Missing one or more teeth.", "precautions": ["Consult a dentist for prosthetic options.", "Monitor jaw development."], "urgency": "Low"},
    },
    "wound": {
        "abrasion":      {"info": "Surface scrape of the skin.", "precautions": ["Clean with water and antiseptic.", "Cover with a bandage.", "Change dressing daily."], "urgency": "Low"},
        "bruises":       {"info": "Blood pooling under skin from blunt trauma.", "precautions": ["Apply ice pack for 20 min.", "Elevate the area.", "See a doctor if worsening."], "urgency": "Low"},
        "burn":          {"info": "Skin damage from heat or chemicals.", "precautions": ["Cool under running water for 10-20 min.", "Do not apply ice or butter.", "Seek emergency care for large burns."], "urgency": "High"},
        "cut":           {"info": "Open wound or slice through skin.", "precautions": ["Apply pressure to stop bleeding.", "Clean and cover with bandage.", "Get stitches if deep."], "urgency": "Moderate"},
        "ingrown_nails": {"info": "Nail growing into surrounding skin.", "precautions": ["Soak in warm water daily.", "Wear wide-toed footwear.", "See a doctor if infected."], "urgency": "Low"},
        "laceration":    {"info": "Deep irregular tear in the skin.", "precautions": ["Apply pressure.", "Clean with saline.", "Seek medical care."], "urgency": "Moderate"},
        "stab_wound":    {"info": "Puncture from a sharp object.", "precautions": ["Call emergency services immediately.", "Do not remove embedded object.", "Apply pressure around wound."], "urgency": "Critical"},
    },
}

_router = None
_specialists = {}
_advice = None


def _load_router():
    global _router
    if _router is None:
        _router = tflite.Interpreter(
            model_path=os.path.join(MODELS_DIR, "master_router_model.tflite"))
        _router.allocate_tensors()


def _load_specialist(category: str):
    if category not in _specialists and category in SPECIALIST_MODELS:
        path = os.path.join(MODELS_DIR, SPECIALIST_MODELS[category])
        interp = tflite.Interpreter(model_path=path)
        interp.allocate_tensors()
        _specialists[category] = interp


def _load_advice():
    global _advice
    if _advice is None:
        with open(os.path.join(MODELS_DIR, "medical_advice.json")) as f:
            _advice = json.load(f)


def _run_interpreter(interp, img_array: np.ndarray) -> np.ndarray:
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    arr = img_array.astype(np.float32) / 255.0
    interp.set_tensor(inp_det["index"], arr)
    interp.invoke()
    return interp.get_tensor(out_det["index"])[0]


def _get_advice(category: str, label: str) -> dict:
    _load_advice()
    if category in _advice and label in _advice[category]:
        entry = _advice[category][label]
        return {
            "info": entry.get("info", entry.get("disease", label)),
            "precautions": entry.get("precautions", []),
            "urgency": entry.get("urgency", "Unknown"),
        }
    if category in FALLBACK_ADVICE and label in FALLBACK_ADVICE[category]:
        return FALLBACK_ADVICE[category][label]
    return {
        "info": "No detailed information available.",
        "precautions": ["Consult a licensed physician for proper diagnosis."],
        "urgency": "Unknown",
    }


def predict_image(image: Image.Image) -> dict:
    _load_router()

    # Step 1 — master router at 128x128
    router_img = np.expand_dims(np.array(image.resize((128, 128))), axis=0)
    router_preds = _run_interpreter(_router, router_img)
    category_idx = int(np.argmax(router_preds))
    category = ROUTER_CLASSES[category_idx]
    router_conf = round(float(router_preds[category_idx]) * 100, 1)

    # Step 2 — specialist model at 224x224
    if category in SPECIALIST_MODELS:
        _load_specialist(category)
        spec_img = np.expand_dims(np.array(image.resize((224, 224))), axis=0)
        preds = _run_interpreter(_specialists[category], spec_img)
        labels = SPECIALIST_LABELS[category]
        top3 = np.argsort(preds)[::-1][:3]

        best_label = labels[top3[0]]
        best_conf = round(float(preds[top3[0]]) * 100, 1)
        top_predictions = [
            {"label": labels[i], "confidence": round(float(preds[i]) * 100, 1)}
            for i in top3
        ]
    else:
        # wound — no specialist model, use nail as proxy
        _load_specialist("nail")
        spec_img = np.expand_dims(np.array(image.resize((224, 224))), axis=0)
        nail_preds = _run_interpreter(_specialists["nail"], spec_img)
        nail_labels = SPECIALIST_LABELS["nail"]
        top3 = np.argsort(nail_preds)[::-1][:3]
        wound_map = {
            "Healthy": "abrasion", "onychomycosis": "burn",
            "nail_psoriasis": "bruises", "paronychia": "cut",
            "onycholysis": "laceration", "melanonychia": "stab_wound",
        }
        best_label = wound_map.get(nail_labels[top3[0]], "abrasion")
        best_conf = round(float(nail_preds[top3[0]]) * 100, 1)
        top_predictions = [
            {"label": wound_map.get(nail_labels[i], nail_labels[i]),
             "confidence": round(float(nail_preds[i]) * 100, 1)}
            for i in top3
        ]
        category = "wound"

    advice = _get_advice(category, best_label)

    return {
        "category": category,
        "condition": best_label,
        "confidence": best_conf,
        "description": advice["info"],
        "precautions": advice["precautions"],
        "urgency": advice["urgency"],
        "top_predictions": top_predictions,
    }
