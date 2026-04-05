import os
import json
from openai import OpenAI

_client = None

SYSTEM_PROMPT = """You are a medical AI assistant. Analyze the given symptoms and respond ONLY with a JSON object in this exact format, no extra text:

{
  "disease": "Disease name",
  "confidence": 80.0,
  "description": "One sentence description.",
  "precautions": ["Precaution 1", "Precaution 2", "Precaution 3", "Precaution 4", "Precaution 5"],
  "detected_symptoms": ["symptom1", "symptom2"],
  "top_predictions": [
    {"disease": "First disease", "confidence": 80.0},
    {"disease": "Second disease", "confidence": 12.0},
    {"disease": "Third disease", "confidence": 8.0}
  ]
}

Rules: confidence is 0-100 float, always 5 precautions, top_predictions always has 3 entries summing to 100."""


def predict_disease(symptom_text: str) -> dict:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        _client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Symptoms: {symptom_text}"},
        ],
        temperature=0.2,
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip().strip("```json").strip("```").strip()
    result = json.loads(raw)
    result.setdefault("disease", "Unknown")
    result.setdefault("confidence", 0.0)
    result.setdefault("description", "")
    result.setdefault("precautions", [])
    result.setdefault("detected_symptoms", [])
    result.setdefault("top_predictions", [])
    return result
