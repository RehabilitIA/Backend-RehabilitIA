import os
import json
import uuid
from typing import Dict, List, Any
from typing_extensions import TypedDict

from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from openai import AzureOpenAI

from prompts.prompts_sr import sr_agent_system_prompt, validate_card_prompt, generate_sr_prompt


# ============================================================
# Firebase
# ============================================================

KEY_PATH = "serviceAccountKey.json"

if not firebase_admin._apps:
    cred = credentials.Certificate(KEY_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()


# ============================================================
# Azure OpenAI
# ============================================================

load_dotenv("env.env")

AZURE_ENDPOINT = "https://invuniandesai-2.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-4.1"
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_VERSION = "2024-12-01-preview"


def get_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )


# ============================================================
# Tool definitions (OpenAI function calling schema)
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "validate_card",
            "description": (
                "Valida una tarjeta SR según las reglas de calidad. "
                "Retorna si es válida y, si no, describe el problema."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stimulus": {"type": "string", "description": "La pregunta de la tarjeta"},
                    "answer": {"type": "string", "description": "La respuesta de la tarjeta"},
                    "category": {"type": "string", "description": "Categoría: personal, familia, rutina, objetos"},
                },
                "required": ["stimulus", "answer", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_cards",
            "description": (
                "Envía las 10 tarjetas finales validadas. "
                "Llamar solo cuando se tengan exactamente 10 tarjetas válidas y sin duplicados."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cards": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "stimulus": {"type": "string"},
                                "answer": {"type": "string"},
                                "category": {"type": "string"},
                            },
                            "required": ["stimulus", "answer", "category"],
                        },
                        "description": "Lista de exactamente 10 tarjetas SR válidas",
                    }
                },
                "required": ["cards"],
            },
        },
    },
]


# ============================================================
# Helpers
# ============================================================

def parse_json(raw: str) -> Dict:
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1:
            s = s[start:end + 1]
    return json.loads(s)


def call_llm(messages: List[Dict], tools=None, tool_choice="auto") -> Any:
    client = get_client()
    kwargs = dict(
        model=AZURE_DEPLOYMENT,
        messages=messages,
        temperature=0.3,
        max_tokens=4000,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    return client.chat.completions.create(**kwargs)


# ============================================================
# Tool execution
# ============================================================

def execute_validate_card(args: Dict, patient_profile: Dict, existing_answers: List[str]) -> str:
    prompt = validate_card_prompt(
        stimulus=args["stimulus"],
        answer=args["answer"],
        category=args["category"],
        patient_profile=patient_profile,
        existing_answers=existing_answers,
    )
    resp = call_llm(
        messages=[
            {"role": "system", "content": "Eres un validador estricto de tarjetas SR."},
            {"role": "user", "content": prompt},
        ],
        tools=None,
    )
    return resp.choices[0].message.content.strip()


def execute_submit_cards(args: Dict) -> str:
    cards = args.get("cards", [])
    return json.dumps({"submitted": True, "count": len(cards)})


# ============================================================
# ReAct agent loop
# ============================================================

def run_sr_agent(patient_profile: Dict) -> List[Dict]:
    messages = [
        {"role": "system", "content": sr_agent_system_prompt(patient_profile)},
        {"role": "user", "content": "Genera las 10 tarjetas SR para este paciente."},
    ]

    final_cards: List[Dict] = []
    accepted_answers: List[str] = []
    max_iterations = 40

    for _ in range(max_iterations):
        response = call_llm(messages, tools=TOOLS, tool_choice="auto")
        msg = response.choices[0].message

        # Append assistant message (may contain tool_calls)
        messages.append(msg.model_dump(exclude_unset=False))

        # No tool calls → agent thinks it's done but didn't call submit_cards
        if not msg.tool_calls:
            break

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            if fn_name == "validate_card":
                result = execute_validate_card(fn_args, patient_profile, accepted_answers)
                # If valid, track the answer to detect future duplicates
                try:
                    parsed = parse_json(result)
                    if parsed.get("valid"):
                        answer = fn_args.get("answer", "")
                        if answer not in accepted_answers:
                            accepted_answers.append(answer)
                except Exception:
                    pass

            elif fn_name == "submit_cards":
                final_cards = fn_args.get("cards", [])
                result = execute_submit_cards(fn_args)

            else:
                result = json.dumps({"error": f"Herramienta desconocida: {fn_name}"})

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        if final_cards:
            break

    if not final_cards:
        raise ValueError("El agente SR no llamó a submit_cards con las tarjetas finales.")

    return final_cards


# ============================================================
# Firestore helpers
# ============================================================

def asignar_a_paciente(user_id: str, ejercicio_id: str):
    db.collection("pacientes") \
      .document(user_id) \
      .collection("ejercicios_asignados") \
      .document(ejercicio_id) \
      .set({
            "id_ejercicio": ejercicio_id,
            "tipo": "privado",
            "estado": "pendiente",
            "fecha_asignacion": firestore.SERVER_TIMESTAMP,
        })


def save_sr_cards(user_id: str, cards: List[Dict]):
    col_sr = db.collection("ejercicios_SR")
    col_general = db.collection("ejercicios")

    for card in cards:
        doc_id = f"E{uuid.uuid4().hex[:6].upper()}"

        sr_data = {
            "id_ejercicio_general": doc_id,
            "pregunta": card.get("stimulus", ""),
            "rta_correcta": card.get("answer", ""),
            "interval_index": 0,
            "intervals_sec": [15, 30, 60, 120, 300],
            "success_streak": 0,
            "lapses": 0,
            "next_due": 0,
            "status": "learning",
        }

        col_general.document(doc_id).set({
            "id": doc_id,
            "terapia": "SR",
            "revisado": True,
            "tipo": "privado",
            "creado_por": "IA",
            "personalizado": True,
            "referencia_base": None,
            "id_paciente": user_id,
            "descripcion_adaptado": "",
            "fecha_creacion": firestore.SERVER_TIMESTAMP,
        })

        col_sr.document(doc_id).set(sr_data)
        asignar_a_paciente(user_id, doc_id)


# ============================================================
# Entry point (unchanged signature)
# ============================================================

def main_langraph_sr(user_id: str, patient_profile: dict) -> dict:
    cards = run_sr_agent(patient_profile)
    save_sr_cards(user_id, cards)
    return {"user_id": user_id, "cards": cards}


# ============================================================
# Simple prompt entry (for baseline comparison)
# ============================================================

def main_simple_sr(user_id: str, patient_profile: dict) -> dict:
    client = get_client()
    resp = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "user", "content": generate_sr_prompt(patient_profile)},
        ],
        temperature=0.3,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    result = parse_json(resp.choices[0].message.content)
    cards = result.get("cards", [])
    return {"user_id": user_id, "cards": cards}


# ============================================================
# Manual test
# ============================================================

if __name__ == "__main__":
    sample_profile = {
        "personal": {"nombre": "María", "lugar_nacimiento": "Bogotá"},
        "familia": {"hijos": ["Daniel", "Laura"], "pareja": "Carlos"},
        "rutinas": {"comida_favorita": "Ajiaco", "actividad_favorita": "Caminar"},
        "objetos": {"mascota": {"nombre": "Rocky"}},
    }

    result = main_langraph_sr("paciente123", sample_profile)
    print(json.dumps(result, indent=2, ensure_ascii=False))
