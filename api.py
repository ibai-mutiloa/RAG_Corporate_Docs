#!/usr/bin/env python3
import os
import numpy as np
from flask import Flask, request, jsonify
import re
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from openai import AzureOpenAI
from langdetect import detect, DetectorFactory

# Fijar seed para garantizar consistencia en detección de idioma
DetectorFactory.seed = 0

# Cargar variables de entorno desde .env
load_dotenv()

# ===========================
# Configuración de DB
# ===========================
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "normativas_db")
DB_USER = os.getenv("DB_USER", "normativas_user")
DB_PASS = os.getenv("DB_PASS", "")

# ===========================
# Azure OpenAI Configuration
# ===========================
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "text-embedding-3-small")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

# Segundo modelo para generación de texto/respuestas
AZURE_DEPLOYMENT_NAME_TEXT = os.getenv("AZURE_DEPLOYMENT_NAME_TEXT", "")
AZURE_API_KEY_TEXT = os.getenv("AZURE_API_KEY_TEXT", "")  # Opcional, usa AZURE_API_KEY si no está definida
AZURE_ENDPOINT_TEXT = os.getenv("AZURE_ENDPOINT_TEXT", "")  # Opcional, usa AZURE_ENDPOINT si no está definida

# ===========================
# Configuración de búsqueda
# ===========================
TOP_K = int(os.getenv("TOP_K", "5"))  # Número de chunks similares a retornar
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.35"))  # Umbral mínimo de similitud
MIN_SIMILARITY_WARNING = float(os.getenv("MIN_SIMILARITY_WARNING", "0.30"))  # Umbral zona gris
MIN_SIMILARITY_ABSOLUTE = float(os.getenv("MIN_SIMILARITY_ABSOLUTE", "0.25"))  # Umbral de silencio semántico
MIN_SIMILARITY_SECOND_PASS = float(os.getenv("MIN_SIMILARITY_SECOND_PASS", "0.20"))  # Mínimo para segundo pase
RETRIEVAL_CANDIDATE_LIMIT = int(os.getenv("RETRIEVAL_CANDIDATE_LIMIT", "40"))
HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.82"))
HYBRID_LEXICAL_WEIGHT = float(os.getenv("HYBRID_LEXICAL_WEIGHT", "0.18"))
REWRITE_QUERY = os.getenv("REWRITE_QUERY", "True").lower() == "true"  # Reescribir preguntas
CONTEXT_SUMMARY_MODE = os.getenv("CONTEXT_SUMMARY_MODE", "heuristic").lower()  # model | heuristic
CONTEXT_MAX_SOURCES = int(os.getenv("CONTEXT_MAX_SOURCES", str(TOP_K)))
CONTEXT_MAX_BULLETS = int(os.getenv("CONTEXT_MAX_BULLETS", "2"))
DEFAULT_GENERATE_ANSWER = os.getenv("DEFAULT_GENERATE_ANSWER", "True").lower() == "true"
LANGUAGE_DETECTION_ENABLED = os.getenv("LANGUAGE_DETECTION_ENABLED", "True").lower() == "true"
ENABLE_NEIGHBOR_EXPANSION = os.getenv("ENABLE_NEIGHBOR_EXPANSION", "False").lower() == "true"

# Cliente de Azure OpenAI para embeddings
azure_client = None
if AZURE_ENDPOINT and AZURE_API_KEY:
    azure_client = AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY
    )

# Cliente de Azure OpenAI para generación de texto
azure_client_text = None
if AZURE_DEPLOYMENT_NAME_TEXT:
    text_endpoint = AZURE_ENDPOINT_TEXT if AZURE_ENDPOINT_TEXT else AZURE_ENDPOINT
    text_api_key = AZURE_API_KEY_TEXT if AZURE_API_KEY_TEXT else AZURE_API_KEY
    
    if text_endpoint and text_api_key:
        azure_client_text = AzureOpenAI(
            api_version=AZURE_API_VERSION,
            azure_endpoint=text_endpoint,
            api_key=text_api_key
        )

# ===========================
# Flask App
# ===========================
app = Flask(__name__)
CORS(app)  # Permitir CORS para todas las rutas

# ===========================
# Funciones auxiliares
# ===========================

def connect_db():
    """Conectar a la base de datos PostgreSQL"""
    return psycopg2.connect(
        host=DB_HOST, 
        port=DB_PORT, 
        dbname=DB_NAME, 
        user=DB_USER, 
        password=DB_PASS
    )

def calculate_embedding(text):
    """Calcula el embedding de un texto usando Azure OpenAI"""
    if not azure_client:
        raise Exception("Azure OpenAI no configurado correctamente")
    
    try:
        response = azure_client.embeddings.create(
            input=[text],
            model=AZURE_DEPLOYMENT_NAME
        )
        return response.data[0].embedding
    except Exception as e:
        raise Exception(f"Error calculando embedding: {e}")

def cosine_similarity(vec1, vec2):
    """Calcula la similitud del coseno entre dos vectores"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    return dot_product / (norm_vec1 * norm_vec2)

def _embedding_to_pgvector_literal(embedding):
    """Convierte un embedding a literal compatible con el tipo vector de pgvector."""
    return '[' + ','.join(f'{float(value):.8f}' for value in embedding) + ']'

def _normalize_search_query(text):
    """Normaliza una consulta para búsqueda lexical simple."""
    if not text:
        return ''
    cleaned = text.lower()
    cleaned = re.sub(r"[^\w\sáéíóúüñçàèìòùäëïöüâêîôû0-9%]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def _hybrid_score(vector_similarity, lexical_rank):
    """Combina similitud vectorial y lexical en una sola puntuación."""
    lexical_boost = min(float(lexical_rank) * 8.0, 1.0)
    return (HYBRID_VECTOR_WEIGHT * float(vector_similarity)) + (HYBRID_LEXICAL_WEIGHT * lexical_boost)

def rewrite_query(question):
    """
    Reescribe la pregunta para mejorar el matching semántico
    Convierte preguntas naturales a lenguaje jurídico/técnico
    """
    if not azure_client_text or not REWRITE_QUERY:
        return [question]  # Retornar lista con pregunta original
    
    try:
        prompt = f"""Eres un asistente legal. Dada la siguiente pregunta en lenguaje natural, 
genera 2-3 variantes que usen lenguaje jurídico/técnico para mejorar búsquedas semánticas.

Pregunta original: {question}

Formato: devuelve SOLO las preguntas reformuladas, una por línea, sin numeración ni comillas."""
        
        response = azure_client_text.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME_TEXT,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        rewritten = response.choices[0].message.content.strip().split('\n')
        rewritten = [q.strip() for q in rewritten if q.strip()]
        
        # Retornar pregunta original + reescritas
        return [question] + rewritten[:2]  # Max 3 preguntas
    
    except Exception as e:
        print(f"[WARN] Error reescribiendo pregunta: {e}")
        return [question]

def detect_language(text):
    """
    Detecta el idioma de un texto (español/euzkera).
    Retorna 'es' para español, 'eu' para euskera, 'es' por defecto si falla.
    """
    if not LANGUAGE_DETECTION_ENABLED or not text:
        return 'es'
    
    try:
        lang = detect(text[:200])  # Usar primeros 200 caracteres
        
        # Mapear a códigos conocidos
        lang_map = {
            'es': 'es',
            'eu': 'eu',
            'en': 'es',  # Si detecta inglés, asumir español por defecto
        }
        
        return lang_map.get(lang, 'es')
    except Exception as e:
        print(f"[WARN] Error detectando idioma: {e}, usando español por defecto")
        return 'es'

def apply_normative_synonyms(question):
    """
    Mapea términos coloquiales a sinónimos normativos para mejorar búsqueda.
    Ejemplo: 'denuncia' -> 'denuncia comisión ética canal reporte'
    """
    synonyms_map = {
        # Denuncias y quejas
        'denuncia': 'denuncia comisión ética canal reporte procedimiento',
        'canal denuncia': 'denuncia comisión ética código ético procedimiento',
        'queja': 'denuncia comisión reclamación procedimiento',
        'reclamación': 'denuncia comisión procedimiento recurso',
        'cómo reportar': 'denuncia comisión ética procedimiento canal',
        'reportar': 'denuncia comisión ética código ético',
        
        # Permisos y derechos
        'permiso': 'licencia autorización permiso procedimiento',
        'vacaciones': 'vacaciones permiso descanso prestaciones',
        'baja': 'baja licencia incapacidad descanso',
        'excedencia': 'excedencia baja permiso procedimiento',
        'año sabático': 'año sabático excedencia permiso',
        
        # Horarios y trabajos
        'horario': 'jornada horario flexible trabajo presencial',
        'teletrabajo': 'trabajo remoto teletrabajo presencial flexible',
        'turno': 'turno jornada horario flexible',
        
        # Compensaciones
        'dinero': 'compensación retribución paga salario prestación',
        'compensación': 'compensación indemnización resarcimiento retribución',
        'gastos': 'gastos dietas manutención desplazamiento reembolso',
        'desplazamiento': 'desplazamiento gastos dietas viático',
        
        # Procedimientos generales
        'cómo': 'procedimiento pasos proceso requisitos normativa',
        'debo': 'obligación deber requisito normativa',
        'puedo': 'derecho permiso autorización normativa',
        'se puede': 'derecho permiso autorización normativa',
    }
    
    question_lower = question.lower()
    expanded = [question]
    
    for key, value in synonyms_map.items():
        if key in question_lower:
            expanded.append(value)
            break
    
    return expanded

def rewrite_query_bilingual(question, detected_lang):
    """
    Reescribe la pregunta en el idioma detectado para mejorar matching semántico.
    Soporta español (es) y euskera (eu).
    Ahora incluye mapeo de sinónimos normativos.
    """
    # Primero aplicar sinónimos locales (más rápido, sin API)
    synonyms_variants = apply_normative_synonyms(question)
    
    if not azure_client_text or not REWRITE_QUERY:
        return synonyms_variants
    
    try:
        if detected_lang == 'eu':
            # Reformular en euskera
            prompt = f"""Asistente legala zara. Emandako galdera zuzenean ebatziko dituzu antzeko nola legezko bilaketan.

Jatorrizko galdera: {question}

Formatua: itzuli SOILIK reformulatutako galderak, batez bat, lineaz lineaz, zenbaketa eta kutxik gabe."""
        else:
            # Reformular en español (por defecto)
            prompt = f"""Eres un asistente legal. Dada la siguiente pregunta en lenguaje natural, 
genera 2-3 variantes que usen lenguaje jurídico/técnico para mejorar búsquedas semánticas.

Pregunta original: {question}

Formato: devuelve SOLO las preguntas reformuladas, una por línea, sin numeración ni comillas."""
        
        response = azure_client_text.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME_TEXT,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        rewritten = response.choices[0].message.content.strip().split('\n')
        rewritten = [q.strip() for q in rewritten if q.strip()]
        
        # Combinar sinónimos locales + LLM reescrituras
        return synonyms_variants + rewritten[:2]
    
    except Exception as e:
        print(f"[WARN] Error reescribiendo pregunta en {detected_lang}: {e}")
        return synonyms_variants

def _normalize_text(text):
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def _strip_chunk_metadata(text):
    if not text:
        return ""
    cleaned = text.strip()
    
    # Eliminar encabezados de metadatos del formato enriquecido
    # Patrón multilinea: captura "Documento: <filename>\nTexto:\n<contenido>"
    cleaned = re.sub(
        r"(?sim)^Documento\s*:\s*.*?\n(Texto\s*:\s*)?",
        "",
        cleaned,
        flags=re.IGNORECASE | re.MULTILINE
    )
    
    # Línea de seguridad: si sigue habiendo "Documento:" o "Texto:" al inicio, quitarla
    cleaned = re.sub(r"(?im)^[^\n]*Documento[^\n]*\n", "", cleaned)
    cleaned = re.sub(r"(?im)^[^\n]*Texto[^\n]*:\s*", "", cleaned)
    
    # También quitar si aparecen después (por si el modelo reintroduce)
    cleaned = re.sub(r"\n[^\n]*Documento[^\n]*\n", "\n", cleaned)
    cleaned = re.sub(r"\n[^\n]*Texto[^\n]*:\s*", "\n", cleaned)
    
    return _normalize_text(cleaned)

def _extract_article_title(text):
    if not text:
        return None
    match = re.search(r"\b(ART[IÍ]CULO|Art\.?|Artículo)\s+\d+[A-Za-zºª\-]*[^\n\.]*", text)
    if match:
        return match.group(0).strip()
    return None

def _heuristic_summary(text, max_bullets=2):
    if not text:
        return []
    sentences = re.split(r"(?<=[\.!\?])\s+", text)
    # Descartar líneas muy cortas y limitar longitud de bullet
    bullets = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    bullets = [b[:150] + "..." if len(b) > 150 else b for b in bullets]
    return bullets[:max_bullets]

def build_clean_context(question, chunks):
    """Construye contexto limpio y conciso para evitar que el modelo copie texto crudo."""
    sources = []
    for chunk in chunks[:CONTEXT_MAX_SOURCES]:
        cleaned_text = _strip_chunk_metadata(chunk.get('text', ''))
        # Truncar a máximo 600 caracteres para evitar exceso de contexto
        cleaned_text = cleaned_text[:600] if len(cleaned_text) > 600 else cleaned_text
        article = _extract_article_title(cleaned_text)
        sources.append({
            'file_name': chunk.get('file_name', 'Documento'),
            'article': article,
            'text': cleaned_text,
            'similarity': chunk.get('similarity', 0.0)
        })

    if not sources:
        return ""

    # Usar siempre modo heurístico para mantener resúmenes concisos
    # y evitar que el LLM copie fragmentos largos del PDF
    parts = []
    for i, s in enumerate(sources, start=1):
        header = f"Fuente {i}: {s['file_name']}"
        if s['article']:
            header += f" ({s['article']})"
        bullets = _heuristic_summary(s['text'], CONTEXT_MAX_BULLETS)
        if not bullets:
            bullets = ["Fragmento sin contenido textual útil."]
        bullets_text = "\n".join([f"- {b}" for b in bullets])
        parts.append(f"{header}\nResumen del fragmento:\n{bullets_text}")
    
    return "\n\n".join(parts)

def _looks_like_context_summary(text):
    if not text:
        return False
    patterns = [
        r"(?im)^\s*fuente\s+\d+\s*:",
        r"(?im)^\s*resumen\s+del\s+fragmento\s*:",
        r"(?im)^\s*-\s*fragmento",
        r"(?im)^\s*documento\s*:",
    ]
    return any(re.search(pattern, text) for pattern in patterns)

def _sanitize_answer_output(text):
    if not text:
        return text
    cleaned = text.strip()
    
    # Eliminar completamente encabezados de metadatos
    cleaned = re.sub(
        r"(?sim)^Documento\s*:\s*.*?\n(Texto\s*:\s*)?",
        "",
        cleaned,
        flags=re.IGNORECASE | re.MULTILINE
    )
    
    # Líneas de seguridad para cualquier aparición de prefijos
    cleaned = re.sub(r"(?im)^[^\n]*?(?:Documento|Artículo|Texto)\s*:\s*[^\n]*\n", "", cleaned)
    cleaned = re.sub(r"(?im)(?:Documento|Artículo|Texto)\s*:\s*[^\n]*\.pdf", "", cleaned)
    
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def generate_answer_from_chunks(question, chunks, max_similarity, detected_lang='es'):
    """
    Genera una respuesta usando el modelo de texto de Azure OpenAI
    basándose en los chunks relevantes encontrados.
    
    Parámetros:
    - detected_lang: 'es' para español (por defecto) o 'eu' para euskera
    
    Cuatro niveles de confianza:
    - max_similarity >= MIN_SIMILARITY (0.65): respuesta normal (alta confianza)
    - MIN_SIMILARITY_WARNING <= max_similarity < MIN_SIMILARITY (0.55-0.65): respuesta con aviso (zona gris)
    - MIN_SIMILARITY_ABSOLUTE <= max_similarity < MIN_SIMILARITY_WARNING (0.50-0.55): aviso de baja confianza
    - max_similarity < MIN_SIMILARITY_ABSOLUTE (< 0.50): silencio semántico, no hay información
    """
    if not azure_client_text:
        return None
    
    # Nivel 0: Silencio semántico (< 0.50) - CORTE SECO
    # "Eco semántico" - coincidencias accidentales, no es conocimiento real
    if max_similarity < MIN_SIMILARITY_ABSOLUTE:
        if detected_lang == 'eu':
            return "Ez da informazio erlaziondadun aurkitu erregutaletan. Kontsulta ez du sufizienterik semantikoarekin dokumentuekin."
        else:
            return "No se ha encontrado información relacionada en la normativa consultada. La consulta no tiene suficiente relación semántica con los documentos disponibles."
    
    # Nivel 1: Baja confianza (0.50-0.55) - respuesta muy cautelosa
    if max_similarity < MIN_SIMILARITY_WARNING:
        if detected_lang == 'eu':
            return f"Ez dut aurkitu dokumentazioan informazio garrantzitsua zure galderari (antzekotasuna: {max_similarity:.0%}). Gomendio duzu Baliabide Humanekin kontsultatzea edo erregutalaren osoa portalen behatzeagatik."
        else:
            return f"No he encontrado en la documentación disponible información suficientemente relevante sobre tu pregunta (similitud: {max_similarity:.0%}). Te recomiendo consultar con RRHH o revisar el reglamento completo en el portal."
    
    # Nivel 2: Zona gris - respuesta con aviso (0.55-0.65)
    is_gray_zone = max_similarity < MIN_SIMILARITY
    
    try:
        # Construir el contexto limpio a partir de los chunks
        clean_context = build_clean_context(question, chunks)

        # Crear el prompt para el modelo según idioma detectado
        if detected_lang == 'eu':
            # Prompts en euskera
            if is_gray_zone:
                system_prompt = """Zu gara corporizazioaren intraneta asistente ofiziala.
Erabiltzaile-informazioa soilik, iturrietatik emaniko ereduan.
Zuzenean erantzun galderauser galdera.
Idatzi amaiera garbi, osoa eta zorrotzean.
Baldin informazioa ez bada ziur, adieraz testunaren gabe.
Az jaso iturrietako laburpena ez duen."""
            else:
                system_prompt = """Zu gara corporizazioaren intraneta asistente ofiziala.
Erabiltzaile-informazioa soilik, iturrietatik emaniko ereduan.
Zuzenean erantzun galderauser galdera.
Idatzi amaiera garbi eta osoa.
Baldin informazioa ez bada ziur, adieraz honetan.
Az jaso iturrietako laburpena."""
        else:
            # Prompts en español - MEJORADOS PARA PRECISIÓN
            if is_gray_zone:
                system_prompt = """Eres el asistente oficial de la intranet corporativa. Tu tarea es responder preguntas sobre normativas y políticas.

PRIORIDADES (en orden):
1. Responde DIRECTAMENTE la pregunta sin rodeos.
2. Cita el artículo, sección o procedimiento aplicable si es relevante.
3. Usa lenguaje claro y profesional.
4. Si faltan datos críticos, indícalo de forma transparente.

PROHIBICIONES:
- No copies párrafos enteros del documento.
- No hagas resúmenes por fuente ni listas de fragmentos.
- No incluyas encabezados como "Fuente 1", "Resumen del fragmento" o "Documento".
- No inventes información no presente en las fuentes."""
            else:
                system_prompt = """Eres el asistente oficial de la intranet corporativa. Tu tarea es responder preguntas sobre normativas y políticas de forma precisa.

PRIORIDADES (en orden):
1. Responde DIRECTAMENTE la pregunta sin rodeos.
2. Cita el artículo, sección o procedimiento aplicable.
3. Usa lenguaje claro, profesional y conciso.
4. Si hay dudas, indícalo explícitamente.

PROHIBICIONES:
- No copies párrafos enteros del documento.
- No hagas resúmenes por fuente ni listas de fragmentos.
- No incluyas encabezados como "Fuente 1", "Resumen del fragmento" o "Documento".
- No inventes información no presente en las fuentes."""

            if detected_lang == 'eu':
                user_prompt = f"""Galdera:
{question}

Iturriak (laburnegoekin):
{clean_context}

Agindua:
- Zuzenean erantzun user-ren galdera 1-3 paragrafokoen batean.
- Aipatuz informazio ez bada gustatu beharrezko.
- Gomendio ariketa edo dokumentuaren zehaztarpena egitea dubiren batean.
- Astakatu ez datua, ehunekoa edo eskakizuna ez iturrieetan.
- Itzuli SOILIK amaierako erantzuna user-raren (iturrien hautapena ez, labur talaren erakundea ez)."""
            else:
                user_prompt = f"""Pregunta del usuario:
{question}

Documentación disponible (resúmenes):
{clean_context}

Responde la pregunta de forma DIRECTA en 1-2 párrafos. No hagas resúmenes ni listas. Sé precisx y profesional."""
        
        # Llamar al modelo de texto
        response = azure_client_text.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME_TEXT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        answer_text = (response.choices[0].message.content or "").strip()

        # Reparación: si el modelo devuelve un resumen por fuente, convertirlo a respuesta directa
        if _looks_like_context_summary(answer_text):
            repair_prompt = f"""Convierte el siguiente borrador en una RESPUESTA DIRECTA a la pregunta del usuario.

Pregunta del usuario:
{question}

Borrador actual (incorrecto porque resume fuentes):
{answer_text}

Contexto disponible:
{clean_context}

Reglas:
- Responde la pregunta en 1-3 párrafos, en lenguaje claro y profesional.
- No incluyas encabezados tipo 'Fuente 1', 'Resumen del fragmento' ni listas por fuente.
- No inventes información fuera del contexto disponible.
- Si falta evidencia concluyente, indícalo explícitamente.
- Devuelve SOLO la respuesta final para el usuario."""

            repaired = azure_client_text.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME_TEXT,
                messages=[
                    {"role": "system", "content": "Responde preguntas de forma directa y sin formato de resumen por fuentes."},
                    {"role": "user", "content": repair_prompt}
                ],
                temperature=0.2,
                max_tokens=900
            )
            answer_text = (repaired.choices[0].message.content or "").strip()

        return _sanitize_answer_output(answer_text)
    
    except Exception as e:
        print(f"[ERROR] Error generando respuesta: {e}")
        return None

def requires_governance_keywords(question):
    """
    Determina si la pregunta requiere términos de gobernanza (mayoría, quórum, aprobación...).
    Evita penalizar preguntas que no dependen de esas palabras clave.
    """
    if not question:
        return False

    q = question.lower()
    patterns = [
        r'\bmayor[ií]a\b',
        r'\bqu[óo]rum\b',
        r'\baprobaci[óo]n\b',
        r'\bmodificaci[óo]n\b',
        r'\breglamento\b',
        r'\bart[ií]culo\b',
        r'\bconsejo\s+rector\b',
    ]
    return any(re.search(p, q) for p in patterns)


def contains_expected_keywords(text):
    """
    Valida si el texto contiene palabras clave esperadas para respuestas sobre normativas.
    
    Palabras clave esperadas:
    - Mayoría (mayoría, dos tercios, absoluta, simple)
    - Porcentajes (%, porcentaje)
    - Quórum (quórum, quorum)
    - Aprobación (aprobación, aprobado)
    - Modificación (modificación, modificar, cambio)
    
    Retorna:
    - True: contiene al menos una palabra clave
    - False: no contiene palabras clave (requiere segundo pase)
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Palabras clave de diferentes categorías
    keywords = {
        'mayoría': [r'\bmayoría\b', r'\bmayor[ií]a\b', r'\bdos\s+tercios\b', r'\btercios\b', r'\babsoluta\b', r'\bsimple\b'],
        'porcentaje': [r'\b\d+\s*%\b', r'\bporcentaje', r'\bporcent'],
        'quórum': [r'\bqu[óo]rum\b', r'\bquorum\b'],
        'aprobación': [r'\baprobaci[óo]n\b', r'\baprobado', r'\baprueba'],
        'modificación': [r'\bmodificaci[óo]n\b', r'\bmodificar\b', r'\bcambio\b', r'\breglamento\b'],
    }
    
    for category, patterns in keywords.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
    
    return False

def check_answer_completeness(chunks, question):
    """
    Determina si la respuesta está completa verificando:
    1. Que los chunks contengan palabras clave esperadas
    2. Que haya suficiente contexto
    
    Retorna un dict con:
    - answer_complete: bool
    - missing_keywords: bool
    - completeness_score: float (0-1)
    """
    if not chunks:
        return {
            'answer_complete': False,
            'missing_keywords': True,
            'completeness_score': 0.0,
            'reason': 'No se encontraron chunks relevantes'
        }
    
    # Combinar todos los textos
    combined_text = ' '.join([c.get('text', '') for c in chunks])
    
    # Verificar palabras clave solo si la pregunta realmente las requiere.
    governance_required = requires_governance_keywords(question)
    has_keywords = contains_expected_keywords(combined_text) if governance_required else True
    
    # Verificar longitud del contexto (heurística simple)
    word_count = len(combined_text.split())
    has_sufficient_context = word_count >= 100  # Al menos 100 palabras
    
    # Calcular score de completitud
    completeness_score = 0.0
    if has_keywords:
        completeness_score += 0.6
    if has_sufficient_context:
        completeness_score += 0.4
    
    answer_complete = has_keywords and has_sufficient_context
    
    return {
        'answer_complete': answer_complete,
        'missing_keywords': governance_required and not has_keywords,
        'has_sufficient_context': has_sufficient_context,
        'completeness_score': completeness_score,
        'reason': 'OK' if answer_complete else ('Faltan palabras clave normativas' if (governance_required and not has_keywords) else 'Contexto insuficiente')
    }

def find_similar_chunks(query_embedding, query_text=None, top_k=TOP_K, candidate_limit=None):
    """
    Encuentra los chunks más similares usando un ranking híbrido.
    """
    if candidate_limit is None:
        candidate_limit = max(RETRIEVAL_CANDIDATE_LIMIT, top_k * 8)

    lexical_query = _normalize_search_query(query_text)
    vector_literal = _embedding_to_pgvector_literal(query_embedding)
    conn = connect_db()
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Pedir solo los candidatos más prometedores al motor de PostgreSQL
            cur.execute("""
                SELECT
                    id,
                    file_name,
                    file_path,
                    folder_name,
                    chunk_index,
                    text,
                    1 - (embedding <=> %s::vector(1536)) AS vector_similarity,
                    COALESCE(ts_rank_cd(search_tsv, websearch_to_tsquery('simple', NULLIF(%s, ''))), 0) AS lexical_rank
                FROM chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector(1536)
                LIMIT %s
            """, (vector_literal, lexical_query, vector_literal, candidate_limit))
            chunks = cur.fetchall()

        similarities = []
        for chunk in chunks:
            vector_similarity = float(chunk['vector_similarity'] or 0.0)
            lexical_rank = float(chunk['lexical_rank'] or 0.0)
            similarities.append({
                'id': chunk['id'],
                'file_name': chunk['file_name'],
                'file_path': chunk['file_path'],
                'folder_name': chunk['folder_name'],
                'chunk_index': chunk['chunk_index'],
                'text': chunk['text'],
                'vector_similarity': vector_similarity,
                'lexical_rank': lexical_rank,
                'similarity': _hybrid_score(vector_similarity, lexical_rank)
            })

        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    finally:
        conn.close()

def calculate_keyword_density(text, keywords=None):
    """
    Calcula la densidad ponderada de palabras clave en un texto.
    Palabras clave normativamente críticas tienen mayor peso.
    """
    # Palabras clave ponderadas: (keyword, weight)
    critical_keywords = [
        ('mayoría', 2.0),
        ('quórum', 2.0),
        ('aprobación', 1.8),
        ('artículo', 1.5),
        ('procedimiento', 1.5),
        ('modificación', 1.5),
        ('reglamento', 1.3),
        ('norma', 1.0),
        ('disposición', 1.0),
        ('inciso', 1.0),
        ('párrafo', 1.0),
        ('%', 1.2),
    ]
    
    if not text:
        return 0.0
    
    text_lower = text.lower()
    word_count = len(text_lower.split())
    
    if word_count == 0:
        return 0.0
    
    weighted_score = 0.0
    for keyword, weight in critical_keywords:
        count = text_lower.count(keyword.lower())
        weighted_score += count * weight
    
    # Normalizar a rango [0, 1] considerando que esperamos 1-2 keywords por chunk
    density = min(weighted_score / max(word_count / 5, 1), 1.0)
    return density

def deduplicate_chunks(chunks, similarity_threshold=0.90):
    """
    Elimina chunks casi-duplicados (sim de texto > threshold).
    Mantiene el chunk con mayor score cuando encuentra duplicados.
    """
    if len(chunks) <= 1:
        return chunks
    
    deduplicated = []
    seen_texts = {}
    
    for chunk in chunks:
        text = chunk.get('text', '').lower().strip()
        normalized = re.sub(r'[^\w\s]', '', text)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        is_duplicate = False
        for seen_norm, seen_chunk in seen_texts.items():
            seen_words = set(seen_norm.split())
            current_words = set(normalized.split())
            
            if len(seen_words) > 0 and len(current_words) > 0:
                intersection = len(seen_words & current_words)
                union = len(seen_words | current_words)
                jaccard = intersection / union if union > 0 else 0.0
                
                if jaccard >= similarity_threshold:
                    is_duplicate = True
                    if chunk.get('similarity', 0) > seen_chunk.get('similarity', 0):
                        deduplicated.remove(seen_chunk)
                        deduplicated.append(chunk)
                        seen_texts[normalized] = chunk
                    break
        
        if not is_duplicate:
            deduplicated.append(chunk)
            seen_texts[normalized] = chunk
    
    return deduplicated

def expand_context_with_neighbors(chunks):
    """
    Expande el contexto de cada chunk añadiendo vecinos (anterior y posterior)
    del mismo archivo para darle al LLM más información continua.
    
    Parámetros:
        chunks: lista de chunks encontrados (ya ordenados por similitud)
    
    Retorna:
        Lista expandida con vecinos insertados de forma inteligente.
    """
    if not chunks:
        return chunks
    
    # Agrupar por archivo y chunk_index para buscar vecinos
    chunk_ids_by_file = {}
    for chunk in chunks:
        file_path = chunk.get('file_path', '')
        chunk_index = chunk.get('chunk_index', -1)
        if file_path:
            if file_path not in chunk_ids_by_file:
                chunk_ids_by_file[file_path] = []
            chunk_ids_by_file[file_path].append((chunk_index, chunk))
    
    # Traer vecinos de la BD
    expanded_chunks = []
    neighbor_cache = {}  # Caché local para evitar queries repetidas
    
    conn = connect_db()
    
    try:
        for chunk in chunks:
            file_path = chunk.get('file_path', '')
            chunk_index = chunk.get('chunk_index', -1)
            
            expanded_chunks.append(chunk)
            
            # Traer vecino anterior (chunk_index - 1)
            if chunk_index > 0:
                cache_key = (file_path, chunk_index - 1)
                if cache_key not in neighbor_cache:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute("""
                            SELECT id, file_name, file_path, folder_name, chunk_index, text
                            FROM chunks
                            WHERE file_path = %s AND chunk_index = %s AND embedding IS NOT NULL
                            LIMIT 1
                        """, (file_path, chunk_index - 1))
                        result = cur.fetchone()
                        neighbor_cache[cache_key] = result
                
                prev_chunk = neighbor_cache[cache_key]
                if prev_chunk:
                    prev_chunk_dict = dict(prev_chunk)
                    prev_chunk_dict['similarity'] = chunk.get('similarity', 0.0) * 0.7  # Reducir score del vecino
                    prev_chunk_dict['is_neighbor'] = True
                    prev_chunk_dict['neighbor_type'] = 'anterior'
                    # Insertar ANTES del chunk principal
                    expanded_chunks.insert(len(expanded_chunks) - 1, prev_chunk_dict)
            
            # Traer vecino posterior (chunk_index + 1)
            cache_key = (file_path, chunk_index + 1)
            if cache_key not in neighbor_cache:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT id, file_name, file_path, folder_name, chunk_index, text
                        FROM chunks
                        WHERE file_path = %s AND chunk_index = %s AND embedding IS NOT NULL
                        LIMIT 1
                    """, (file_path, chunk_index + 1))
                    result = cur.fetchone()
                    neighbor_cache[cache_key] = result
            
            next_chunk = neighbor_cache[cache_key]
            if next_chunk:
                next_chunk_dict = dict(next_chunk)
                next_chunk_dict['similarity'] = chunk.get('similarity', 0.0) * 0.7  # Reducir score del vecino
                next_chunk_dict['is_neighbor'] = True
                next_chunk_dict['neighbor_type'] = 'posterior'
                expanded_chunks.append(next_chunk_dict)
    
    finally:
        conn.close()
    
    return expanded_chunks

def find_similar_chunks_with_keywords(query_embedding, forced_keywords=None, top_k=TOP_K):
    """
    Segunda pasada: refuerza resultados que contengan palabras clave específicas.
    Útil cuando la primera búsqueda no encuentra términos normativos esperados.
    
    Args:
        query_embedding: vector de embeddings de la pregunta
        forced_keywords: lista de palabras clave a buscar (ej: ['mayoría', 'quórum'])
        top_k: número de resultados a retornar
    
    Retorna: lista de chunks ordenados por similitud que contengan las keywords
    """
    if not forced_keywords:
        forced_keywords = ['mayoría', 'quórum', 'aprobación', 'modificación', 'reglamento', 'artículo', '%']

    lexical_query = _normalize_search_query(' '.join(kw for kw in forced_keywords if re.search(r'\w', kw)))
    vector_literal = _embedding_to_pgvector_literal(query_embedding)
    
    conn = connect_db()
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Traer candidatos por similitud vectorial (segunda pasada)
            cur.execute("""
                SELECT
                    id,
                    file_name,
                    file_path,
                    folder_name,
                    chunk_index,
                    text,
                    1 - (embedding <=> %s::vector(1536)) AS vector_similarity,
                    COALESCE(ts_rank_cd(search_tsv, websearch_to_tsquery('simple', NULLIF(%s, ''))), 0) AS lexical_rank
                FROM chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector(1536)
                LIMIT %s
            """, (vector_literal, lexical_query, vector_literal, max(RETRIEVAL_CANDIDATE_LIMIT * 2, top_k * 15)))
            chunks = cur.fetchall()

        filtered_chunks = []
        for chunk in chunks:
            text_lower = chunk['text'].lower()
            found_kw = [kw for kw in forced_keywords if kw.lower() in text_lower]
            
            # Ser más selectivo: requiere al menos una palabra clave
            if found_kw:
                vector_similarity = float(chunk['vector_similarity'] or 0.0)
                lexical_rank = float(chunk['lexical_rank'] or 0.0)
                
                # Bonus por cantidad de keywords encontradas
                keyword_bonus = min(len(found_kw) * 0.05, 0.15)
                combined_score = _hybrid_score(vector_similarity, lexical_rank) + keyword_bonus
                
                filtered_chunks.append({
                    'id': chunk['id'],
                    'file_name': chunk['file_name'],
                    'file_path': chunk['file_path'],
                    'folder_name': chunk['folder_name'],
                    'chunk_index': chunk['chunk_index'],
                    'text': chunk['text'],
                    'vector_similarity': vector_similarity,
                    'lexical_rank': lexical_rank,
                    'similarity': combined_score,
                    'found_keywords': found_kw
                })

        filtered_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        return filtered_chunks[:top_k]
    
    finally:
        conn.close()

# ===========================
# Rutas de la API
# ===========================

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de salud para verificar que la API está funcionando"""
    return jsonify({
        'status': 'ok',
        'message': 'API de búsqueda de normativas funcionando correctamente'
    }), 200

@app.route('/search', methods=['POST'])
def search():
    """
    Endpoint principal para buscar normativas similares a una pregunta
    
    Implementa heurística inteligente:
    - Primer pase: búsqueda semántica normal
    - Validación: verifica si hay palabras clave esperadas (mayoría, quórum, %, etc.)
    - Segundo pase: si no hay keywords, relanza con términos forzados
    - Confianza variable: ajusta según completitud de respuesta
    
    Body esperado:
    {
        "question": "¿Cuál es la normativa sobre...?",
        "top_k": 5,  # Opcional, por defecto usa TOP_K del .env
        "generate_answer": true  # Opcional, genera respuesta con el modelo de texto
    }
    
    Retorna:
    {
        "question": "pregunta original",
        "answer": "respuesta generada por el modelo (si generate_answer=true)",
        "answer_complete": boolean,  # Indica si la respuesta contiene info completa
        "confidence": "alta|media|baja",  # Nivel de confianza
        "search_passes": int,  # Número de pases realizados (1 o 2)
        "results": [
            {
                "id": 123,
                "file_name": "normativa.pdf",
                "file_path": "folder/normativa.pdf",
                "folder_name": "folder",
                "chunk_index": 0,
                "text": "texto del chunk",
                "similarity": 0.95
            },
            ...
        ]
    }
    """
    try:
        # Validar request
        if not request.json:
            return jsonify({
                'error': 'Request debe ser JSON'
            }), 400
        
        question = request.json.get('question')
        if not question:
            return jsonify({
                'error': 'Campo "question" es requerido'
            }), 400
        
        top_k = request.json.get('top_k', TOP_K)
        generate_answer = request.json.get('generate_answer', DEFAULT_GENERATE_ANSWER)
        
        # Validar Azure OpenAI
        if not azure_client:
            return jsonify({
                'error': 'Azure OpenAI no configurado. Verifica las variables de entorno.'
            }), 500
        
        # ==================== DETECTAR IDIOMA: PRIORIZAR COOKIE ====================
        # Leer cookie de idioma de la página si está disponible
        cookie_lang = request.cookies.get('idioma', '').lower().strip()
        
        # Mapear valores comunes de cookie a códigos ISO 639-1
        if cookie_lang in ['eu', 'eus', 'euskera', 'basque', 'euskeri']:
            detected_lang = 'eu'
        elif cookie_lang in ['es', 'es-es', 'spanish', 'español', 'cas', 'cast']:
            detected_lang = 'es'
        else:
            # Si no hay cookie válida, detectar automáticamente del texto de la pregunta
            detected_lang = detect_language(question)
        
        query_variants = rewrite_query_bilingual(question, detected_lang)
        
        # ==================== PRIMER PASE: BÚSQUEDA SEMÁNTICA NORMAL ====================
        
        # Buscar con todas las variantes y combinar resultados
        all_similar = {}
        for query_variant in query_variants:
            query_embedding = calculate_embedding(query_variant)
            similar_chunks = find_similar_chunks(query_embedding, query_variant, top_k)
            
            for chunk in similar_chunks:
                chunk_id = chunk['id']
                if chunk_id not in all_similar:
                    all_similar[chunk_id] = chunk
                else:
                    # Si el mismo chunk aparece en múltiples búsquedas, aumentar su similitud
                    all_similar[chunk_id]['similarity'] = max(all_similar[chunk_id]['similarity'], chunk['similarity'])
        
        # Ordenar por similitud
        similar_chunks = sorted(all_similar.values(), key=lambda x: x['similarity'], reverse=True)[:top_k]
        
        # Aplicar densidad de palabras clave para re-ranking
        for chunk in similar_chunks:
            keyword_density = calculate_keyword_density(chunk.get('text', ''))
            chunk['keyword_density'] = keyword_density
            # Si hay keywords fuertes, amplificar la similitud
            if keyword_density > 0.3:
                chunk['similarity'] = (chunk['similarity'] * 0.85) + (keyword_density * 0.15)
            else:
                chunk['similarity'] = (chunk['similarity'] * 0.95) + (keyword_density * 0.05)
        
        similar_chunks = sorted(similar_chunks, key=lambda x: x['similarity'], reverse=True)
        similar_chunks = deduplicate_chunks(similar_chunks, similarity_threshold=0.82)
        
        max_similarity = similar_chunks[0]['similarity'] if similar_chunks else 0
        
        # ==================== VALIDACIÓN: VERIFICAR COMPLETITUD ====================
        completeness = check_answer_completeness(similar_chunks, question)
        search_passes = 1
        second_pass_performed = False
        
        # ==================== SEGUNDO PASE: BÚSQUEDA CON KEYWORDS FORZADAS ====================
        # Si la similitud es >= 0.45 pero faltan palabras clave, relanzar con términos forzados
        # Entre 0.45-0.50: último intento antes del silencio semántico
        # >= 0.50: búsqueda mejorada si faltan keywords
        if (max_similarity >= MIN_SIMILARITY_SECOND_PASS and 
            not completeness['answer_complete'] and 
            completeness['missing_keywords']):
            
            print(f"[INFO] Segundo pase activado: similitud {max_similarity:.2f} pero faltan keywords")
            
            # Términos forzados para búsqueda dirigida (prioriza términos presentes en la pregunta)
            question_lower = question.lower()
            forced_keywords = []
            for kw in ['mayoría', 'quórum', 'aprobación', 'modificación', 'reglamento', 'artículo', 'procedimiento']:
                if kw in question_lower:
                    forced_keywords.append(kw)
            if not forced_keywords:
                forced_keywords = ['reglamento', 'artículo', 'procedimiento']
            
            try:
                # Buscar con primera variante + keywords
                base_query = query_variants[0]
                # Crear query con términos forzados
                enhanced_query = f"{base_query} mayoría quórum aprobación modificación"
                query_embedding = calculate_embedding(enhanced_query)
                
                # Segunda pasada con keywords forzadas
                second_pass_chunks = find_similar_chunks_with_keywords(
                    query_embedding, 
                    forced_keywords=forced_keywords,
                    top_k=top_k
                )
                
                if second_pass_chunks:
                    # Combinar resultados del segundo pase
                    for chunk in second_pass_chunks:
                        chunk_id = chunk['id']
                        if chunk_id not in all_similar:
                            all_similar[chunk_id] = chunk
                        else:
                            # Dar más peso al resultado del segundo pase si es diferente
                            if chunk['similarity'] > all_similar[chunk_id]['similarity'] * 1.1:
                                all_similar[chunk_id] = chunk
                    
                    # Re-ordenar después del segundo pase
                    similar_chunks = sorted(all_similar.values(), key=lambda x: x['similarity'], reverse=True)[:top_k]
                    
                    # Re-validar completitud con nuevos resultados
                    completeness = check_answer_completeness(similar_chunks, question)
                    search_passes = 2
                    second_pass_performed = True
                    
                    print(f"[INFO] Segundo pase completado. Keywords encontradas: {completeness['missing_keywords'] == False}")
            
            except Exception as e:
                print(f"[WARN] Error en segundo pase: {e}")
                # Continuar con los resultados del primer pase
        
        # Actualizar max_similarity después de ambos pases
        max_similarity = similar_chunks[0]['similarity'] if similar_chunks else 0
        
        # ==================== EXPANDIR CONTEXTO CON VECINOS ====================
        # El contexto vecino puede bajar la precisión; se deja configurable y desactivado por defecto.
        if ENABLE_NEIGHBOR_EXPANSION:
            similar_chunks_expanded = expand_context_with_neighbors(similar_chunks)
        else:
            similar_chunks_expanded = similar_chunks
        
        # ==================== DETERMINAR CONFIANZA FINAL ====================
        # Lógica de confianza (colores semafóricos):
        # - alta (verde 🟢): similitud >= 0.65 Y respuesta completa
        # - media (amarillo 🟡): similitud >= 0.55 pero faltan keywords
        # - baja (naranja 🟠): similitud 0.50-0.55 (cerca del silencio semántico)
        # - muy-baja (rojo 🔴): similitud < 0.50 (silencio semántico - no responder)
        
        if max_similarity >= MIN_SIMILARITY:
            if completeness['answer_complete']:
                final_confidence = 'alta'
            else:
                final_confidence = 'media'  # Keywords no encontrados pero similitud alta
        elif max_similarity >= MIN_SIMILARITY_WARNING:
            final_confidence = 'media'
        elif max_similarity >= MIN_SIMILARITY_ABSOLUTE:
            final_confidence = 'baja'
        else:
            final_confidence = 'muy-baja'  # Silencio semántico - eco accidental
        
        # Preparar respuesta
        # Determinar zona y color UI
        if max_similarity >= MIN_SIMILARITY:
            zone = 'high'
            ui_color = 'green'
            ui_message = 'Información encontrada con alta confianza'
        elif max_similarity >= MIN_SIMILARITY_WARNING:
            zone = 'gray'
            ui_color = 'yellow'
            ui_message = 'Información encontrada con confianza media - verificar con fuente oficial'
        elif max_similarity >= MIN_SIMILARITY_ABSOLUTE:
            zone = 'low'
            ui_color = 'orange'
            ui_message = 'Información de baja confianza - se recomienda consultar directamente'
        else:
            zone = 'very-low'
            ui_color = 'red'
            ui_message = 'No se ha encontrado información relevante para esta consulta'
        
        response_results = []
        for item in similar_chunks:
            clean_item = dict(item)
            clean_item['text'] = _strip_chunk_metadata(item.get('text', ''))
            response_results.append(clean_item)

        # Construir lista de fuentes con archivo y similitud (solo chunks principales, sin vecinos)
        sources = []
        for item in similar_chunks:
            source_entry = {
                'file': item.get('file_name', 'Documento desconocido'),
                'similarity': round(float(item.get('similarity', 0)) * 100, 1)  # Porcentaje
            }
            sources.append(source_entry)

        response_data = {
            'question': question,
            'detected_language': detected_lang,
            'query_variants_used': len(query_variants),
            'max_similarity': float(max_similarity),
            'min_similarity_threshold': MIN_SIMILARITY,
            'min_similarity_warning': MIN_SIMILARITY_WARNING,
            'min_similarity_absolute': MIN_SIMILARITY_ABSOLUTE,
            'min_similarity_second_pass': MIN_SIMILARITY_SECOND_PASS,
            'zone': zone,
            'ui_confidence_color': ui_color,
            'ui_message': ui_message,
            'answer_complete': completeness['answer_complete'],
            'completeness_score': completeness['completeness_score'],
            'completeness_reason': completeness['reason'],
            'missing_keywords': completeness['missing_keywords'],
            'search_passes': search_passes,
            'second_pass_performed': second_pass_performed,
            'confidence': final_confidence,
            'sources': sources,
            'results': response_results,
            'count': len(response_results)
        }
        
        # Generar respuesta con el modelo de texto si se solicita
        # Usar chunks expandidos para darle al LLM más contexto continuo
        if generate_answer:
            if azure_client_text:
                answer = generate_answer_from_chunks(question, similar_chunks_expanded, max_similarity, detected_lang)
                response_data['answer'] = answer
                response_data['answer_generated'] = True
                # Si se generó respuesta, indicar al frontend que la use
                response_data['display_mode'] = 'answer' if answer else 'results'
            else:
                response_data['answer'] = None
                response_data['answer_generated'] = False
                response_data['answer_error'] = 'Modelo de texto no configurado. Configure AZURE_DEPLOYMENT_NAME_TEXT.'
                response_data['display_mode'] = 'results'
        else:
            response_data['display_mode'] = 'results'
        
        # Retornar resultados
        return jsonify(response_data), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def stats():
    """
    Endpoint para obtener estadísticas de la base de datos
    
    Retorna:
    {
        "total_chunks": 1000,
        "chunks_with_embeddings": 950,
        "total_files": 50
    }
    """
    try:
        conn = connect_db()
        with conn.cursor() as cur:
            # Total de chunks
            cur.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cur.fetchone()[0]
            
            # Chunks con embeddings
            cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
            chunks_with_embeddings = cur.fetchone()[0]
            
            # Total de archivos únicos
            cur.execute("SELECT COUNT(DISTINCT file_path) FROM chunks")
            total_files = cur.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'total_chunks': total_chunks,
            'chunks_with_embeddings': chunks_with_embeddings,
            'total_files': total_files
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

# ===========================
# Main
# ===========================
if __name__ == '__main__':
    port = int(os.getenv('API_PORT', '5000'))
    debug = os.getenv('API_DEBUG', 'False').lower() == 'true'
    
    print(f"[INFO] Iniciando API en puerto {port}")
    print(f"[INFO] Azure OpenAI (Embeddings) configurado: {'Sí' if azure_client else 'No'}")
    print(f"[INFO] Azure OpenAI (Texto) configurado: {'Sí' if azure_client_text else 'No'}")
    if azure_client_text:
        print(f"[INFO] Modelo de texto: {AZURE_DEPLOYMENT_NAME_TEXT}")
    print(f"[INFO] Umbrales de similitud (sistema semafórico):")
    print(f"       🟢 Alta confianza (verde): >= {MIN_SIMILARITY}")
    print(f"       🟡 Zona gris (amarillo): {MIN_SIMILARITY_WARNING} - {MIN_SIMILARITY}")
    print(f"       🟠 Baja confianza (naranja): {MIN_SIMILARITY_ABSOLUTE} - {MIN_SIMILARITY_WARNING}")
    print(f"       🔴 Silencio semántico (rojo): < {MIN_SIMILARITY_ABSOLUTE}")
    print(f"[INFO] Segundo pase activado para: >= {MIN_SIMILARITY_SECOND_PASS}")
    print(f"[INFO] Reescritura de preguntas activa: {'Sí' if REWRITE_QUERY else 'No'}")
    print(f"[INFO] Base de datos: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
