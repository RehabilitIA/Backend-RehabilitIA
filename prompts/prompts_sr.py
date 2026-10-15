import json


def generate_sr_prompt(patient_profile: dict, n: int = 10) -> str:
    return (
        f"De acuerdo a este perfil de paciente:\n{json.dumps(patient_profile, ensure_ascii=False)}\n\n"
        f"Genera {n} ejercicios de spaced retrieval con preguntas según el perfil.\n"
        "Devuelve solo JSON: {\"cards\": [{\"stimulus\": \"string\", \"answer\": \"string\", \"category\": \"string\"}]}"
    )


def sr_agent_system_prompt(patient_profile: dict) -> str:
    return (
        "Eres un terapeuta del lenguaje experto en Spaced Retrieval para pacientes con afasia.\n"
        "Tu tarea es generar exactamente 10 tarjetas SR de alta calidad basadas en el perfil del paciente.\n\n"
        f"Perfil del paciente:\n{json.dumps(patient_profile, ensure_ascii=False)}\n\n"
        "PROCESO:\n"
        "1. Identifica todos los hechos del perfil que merezcan una tarjeta.\n"
        "2. Para cada hecho, genera una tarjeta (stimulus + answer + category).\n"
        "3. Llama a validate_card para cada tarjeta. Si regresa un problema, corrígela y valídala de nuevo.\n"
        "4. Cuando tengas exactamente 10 tarjetas válidas, llama a submit_cards.\n\n"
        "REGLAS DE PREGUNTAS:\n"
        "- Siempre en segunda persona ('tu', 'tu nombre', etc.).\n"
        "- Para el nombre del paciente (su propio nombre y apellido): la pregunta es EXACTAMENTE '¿Cuál es tu nombre y apellido?'. No la modifiques.\n"
        "- Para nombres de personas o mascotas: '¿Cuál es el nombre de tu madre/padre/hermana/perro...?' "
        "— nunca '¿Cómo se llama?' ni '¿Quién es?'.\n"
        "- La pregunta NO puede contener el verbo exacto de la respuesta en infinitivo ni en gerundio "
        "(ej: respuesta 'Cocinar' → prohibido 'cocinar', 'cocinando'; "
        "respuesta 'Leer' → prohibido 'leer', 'leyendo'). "
        "Sí puedes y DEBES mencionar el contexto natural para que la pregunta sea clara: "
        "'libros' para Leer, 'cocina' para Cocinar ('cocina' es un sustantivo/lugar, no el verbo).\n"
        "- Las preguntas deben ser claras y naturales — evita rodeos o frases rebuscadas.\n"
        "- No uses 'tú' como sujeto explícito en medio de la pregunta (incorrecto: '¿Qué haces tú?'; correcto: '¿Qué actividad realizas?').\n"
        "- Sin preguntas de sí/no.\n\n"
        "REGLAS DE RESPUESTAS:\n"
        "- Solo datos explícitos del perfil — no inventes nada.\n"
        "- Actividades en infinitivo: 'Leer', 'Cocinar' — nunca 'Lectura', 'Cocina'.\n"
        "- Si una rutina tiene nombre específico entre paréntesis (ej: 'Ver series (La Casa de Papel)'), "
        "la respuesta es ese nombre específico.\n"
        "- Respuestas concretas y breves.\n\n"
        "REGLAS DE COBERTURA:\n"
        "- Una tarjeta por cada familiar, una por el nombre de cada mascota.\n"
        "- Una tarjeta por CADA elemento de la lista de rutinas.\n"
        "- Categorías válidas: 'personal', 'familia', 'rutina', 'objetos'.\n"
        "- Las mascotas van en categoría 'objetos', no en 'familia'.\n"
        "- Cada tarjeta con respuesta distinta — sin duplicados.\n"
    )


def validate_card_prompt(stimulus: str, answer: str, category: str, patient_profile: dict, existing_answers: list) -> str:
    return (
        "Valida esta tarjeta SR según las reglas. Responde SOLO con JSON.\n\n"
        f"Tarjeta: {{\"stimulus\": \"{stimulus}\", \"answer\": \"{answer}\", \"category\": \"{category}\"}}\n"
        f"Perfil del paciente: {json.dumps(patient_profile, ensure_ascii=False)}\n"
        f"Respuestas ya existentes: {json.dumps(existing_answers, ensure_ascii=False)}\n\n"
        "Reglas a verificar:\n"
        "1. La pregunta está en segunda persona (tu, tu nombre, etc.).\n"
        "2. Para nombres de personas/mascotas usa '¿Cuál es el nombre de...?' — no '¿Cómo se llama?' ni '¿Quién es?'.\n"
        "3. Si la respuesta es el nombre y apellido del paciente (nombre completo de dos palabras), la pregunta DEBE ser exactamente '¿Cuál es tu nombre y apellido?'.\n"
        "4. La pregunta no contiene el verbo de la respuesta en infinitivo ni en gerundio "
        "(ej: 'Cocinar' → rechaza si la pregunta contiene 'cocinar' o 'cocinando', pero 'cocina' como lugar es VÁLIDO; "
        "'Leer' → rechaza si contiene 'leer' o 'leyendo', pero 'libros' es VÁLIDO).\n"
        "5. La pregunta es clara y natural — rechaza preguntas rebuscadas o con rodeos innecesarios como "
        "'¿Qué acción llevas a cabo con ingredientes?' o '¿Qué haces con materiales llenos de palabras?'.\n"
        "6. La respuesta no es sí/no.\n"
        "7. La respuesta existe explícitamente en el perfil.\n"
        "8. La respuesta no está duplicada en las respuestas existentes.\n"
        "9. Las actividades están en infinitivo (Leer, Cocinar — no Lectura, Cocina).\n"
        "10. Las mascotas usan categoría 'objetos', no 'familia'.\n"
        "11. La categoría debe ser exactamente una de: 'personal', 'familia', 'rutina', 'objetos' — sin variantes ni subtipos.\n"
        "12. La pregunta no usa 'tú' como sujeto explícito en medio de la oración (incorrecto: '¿Qué haces tú?'; correcto: '¿Qué actividad realizas?').\n\n"
        "Responde: {\"valid\": true} o {\"valid\": false, \"issue\": \"descripción del problema\"}"
    )
