import os
import json
import re
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

SYSTEM_PROMPT = """
You are a medical AI assistant.

Analyze the given symptoms and predict the most likely disease.

Return ONLY a valid JSON object in this format:

{
  "disease": "Disease name",
  "confidence": 75,
  "description": "Short explanation of the disease",
  "precautions": ["p1","p2","p3","p4","p5"],
  "detected_symptoms": ["symptom1","symptom2"],
  "top_predictions": [
    {"disease":"d1","confidence":60},
    {"disease":"d2","confidence":25},
    {"disease":"d3","confidence":15}
  ]
}

Important rules:
- Output must start with { and end with }
- No explanations outside JSON
- Always include exactly 5 precautions
- Confidence values between 0 and 100
"""

def extract_json(text: str):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            cleaned = match.group()
            cleaned = cleaned.replace("\n", " ").replace("\r", " ")
            return json.loads(cleaned)
    except Exception as e:
        print("JSON parsing error:", e)

    return {
        "disease": "Unknown",
        "confidence": 0,
        "description": "Unable to analyze symptoms",
        "precautions": [],
        "detected_symptoms": [],
        "top_predictions": []
    }


def predict_disease(symptom_text: str) -> dict:

    print("Received symptoms:", symptom_text)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": symptom_text}
        ],
        temperature=0.6
    )

    raw = response.choices[0].message.content

    print("Model raw response:", raw)

    return extract_json(raw)