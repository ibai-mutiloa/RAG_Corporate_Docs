#!/usr/bin/env python3
"""
API simplificada — corpus solo en castellano (es/).
El LLM responde en el idioma de la pregunta (detección automática).
Un único FAQ en español; el LLM traduce si la pregunta llega en euskera.
"""
import os
import re
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from openai import AzureOpenAI
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0
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
# Azure OpenAI
# ===========================
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "text-embedding-3-small")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

AZURE_DEPLOYMENT_NAME_TEXT = os.getenv("AZURE_DEPLOYMENT_NAME_TEXT", "")
AZURE_API_KEY_TEXT = os.getenv("AZURE_API_KEY_TEXT", "")
AZURE_ENDPOINT_TEXT = os.getenv("AZURE_ENDPOINT_TEXT", "")

# ===========================
# Configuración de búsqueda
# ===========================
TOP_K = int(os.getenv("TOP_K", "8"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))
MIN_SIMILARITY_WARNING = float(os.getenv("MIN_SIMILARITY_WARNING", "0.18"))
MIN_SIMILARITY_ABSOLUTE = float(os.getenv("MIN_SIMILARITY_ABSOLUTE", "0.12"))
MIN_SIMILARITY_SECOND_PASS = float(os.getenv("MIN_SIMILARITY_SECOND_PASS", "0.08"))
RETRIEVAL_CANDIDATE_LIMIT = int(os.getenv("RETRIEVAL_CANDIDATE_LIMIT", "120"))
HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.75"))
HYBRID_LEXICAL_WEIGHT = float(os.getenv("HYBRID_LEXICAL_WEIGHT", "0.25"))
REWRITE_QUERY = os.getenv("REWRITE_QUERY", "True").lower() == "true"
DEFAULT_GENERATE_ANSWER = os.getenv("DEFAULT_GENERATE_ANSWER", "True").lower() == "true"
ENABLE_NEIGHBOR_EXPANSION = os.getenv("ENABLE_NEIGHBOR_EXPANSION", "False").lower() == "true"
CONTEXT_MAX_SOURCES = int(os.getenv("CONTEXT_MAX_SOURCES", str(TOP_K)))

# FAQ — único fichero en castellano
FAQ_FILE = os.getenv("FAQ_FILE", "es/faq_castellano.md")
FAQ_PRIORITY_BOOST = float(os.getenv("FAQ_PRIORITY_BOOST", "0.25"))
# Umbral de similitud para considerar que el FAQ responde directamente
FAQ_MIN_SIMILARITY = float(os.getenv("FAQ_MIN_SIMILARITY", "0.50"))

# ===========================
# Clientes Azure OpenAI
# ===========================
azure_client = None
if AZURE_ENDPOINT and AZURE_API_KEY:
    azure_client = AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    )

azure_client_text = None
if AZURE_DEPLOYMENT_NAME_TEXT:
    text_endpoint = AZURE_ENDPOINT_TEXT or AZURE_ENDPOINT
    text_api_key = AZURE_API_KEY_TEXT or AZURE_API_KEY
    if text_endpoint and text_api_key:
        azure_client_text = AzureOpenAI(
            api_version=AZURE_API_VERSION,
            azure_endpoint=text_endpoint,
            api_key=text_api_key,
        )

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = True
CORS(app)
# Forzar escape ASCII en JSON para compatibilidad con clientes Windows/IIS


# ===========================
# Helpers DB / embeddings
# ===========================

def connect_db():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )


def calculate_embedding(text):
    if not azure_client:
        raise Exception("Azure OpenAI no configurado")
    response = azure_client.embeddings.create(input=[text], model=AZURE_DEPLOYMENT_NAME)
    return response.data[0].embedding


def _embedding_to_pgvector_literal(embedding):
    return '[' + ','.join(f'{float(v):.8f}' for v in embedding) + ']'


def _normalize_search_query(text):
    if not text:
        return ''
    cleaned = text.lower()
    cleaned = re.sub(r"[^\w\sáéíóúüñçàèìòùäëïöüâêîôû0-9%]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _hybrid_score(vector_similarity, lexical_rank):
    lexical_boost = min(float(lexical_rank) * 8.0, 1.0)
    return (HYBRID_VECTOR_WEIGHT * float(vector_similarity)) + (HYBRID_LEXICAL_WEIGHT * lexical_boost)


def _is_faq_chunk(chunk):
    """Detecta si un chunk proviene del FAQ."""
    for field in ('file_name', 'file_path'):
        v = str(chunk.get(field) or '').lower().replace('\\', '/')
        if 'faq' in os.path.basename(v):
            return True
        if FAQ_FILE and os.path.basename(v) == os.path.basename(FAQ_FILE).lower():
            return True
    return False


# ===========================
# Detección de idioma
# ===========================

def detect_language(text):
    """Detecta el idioma de la pregunta. Devuelve 'es' o 'eu'."""
    if not text:
        return 'es'
    try:
        lang = detect(text[:200])
        return 'eu' if lang == 'eu' else 'es'
    except Exception:
        return 'es'


# ===========================
# Búsqueda híbrida
# ===========================

def _fetch_scored_chunks(query_embedding, query_text=None, top_k=TOP_K, candidate_limit=None, file_name_filter=None):
    """Búsqueda híbrida vectorial + lexical sin filtro por idioma (corpus monolingüe)."""
    if candidate_limit is None:
        candidate_limit = max(RETRIEVAL_CANDIDATE_LIMIT, top_k * 8)
    lexical_query = _normalize_search_query(query_text)
    vector_literal = _embedding_to_pgvector_literal(query_embedding)

    conn = connect_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = """
                SELECT
                    id, file_name, file_path, folder_name, chunk_index, text,
                    1 - (embedding <=> %s::vector(1536)) AS vector_similarity,
                    COALESCE(ts_rank_cd(search_tsv, websearch_to_tsquery('simple', NULLIF(%s, ''))), 0) AS lexical_rank
                FROM chunks
                WHERE embedding IS NOT NULL
            """
            params = [vector_literal, lexical_query]

            if file_name_filter:
                sql += " AND LOWER(file_name) = LOWER(%s)"
                params.append(file_name_filter)

            sql += " ORDER BY embedding <=> %s::vector(1536) LIMIT %s"
            params.extend([vector_literal, candidate_limit])

            cur.execute(sql, params)
            rows = cur.fetchall()

        scored = []
        for chunk in rows:
            vector_sim = float(chunk['vector_similarity'] or 0.0)
            lex_rank = float(chunk['lexical_rank'] or 0.0)
            faq_bonus = FAQ_PRIORITY_BOOST if _is_faq_chunk(chunk) else 0.0
            scored.append({
                'id': chunk['id'],
                'file_name': chunk['file_name'],
                'file_path': chunk['file_path'],
                'folder_name': chunk['folder_name'],
                'chunk_index': chunk['chunk_index'],
                'text': chunk['text'],
                'vector_similarity': vector_sim,
                'lexical_rank': lex_rank,
                'similarity': _hybrid_score(vector_sim, lex_rank) + faq_bonus,
            })

        scored.sort(key=lambda x: x['similarity'], reverse=True)
        return scored[:top_k]
    finally:
        conn.close()


# ===========================
# Reescritura de consulta
# ===========================

def rewrite_query(question, detected_lang):
    """Genera variantes de la pregunta en castellano para mejorar el retrieval.
    Si la pregunta viene en euskera, también añade una versión castellana directa.
    """
    # Sinónimos locales rápidos
    synonyms_map = {
        # Compliance / canal ético
        'denuncia':    'canal denuncias procedimiento interno comité cumplimiento penal',
        'queja':       'reclamación queja procedimiento interno',
        'delito':      'delito penal prevención riesgos penales cumplimiento',
        # Permisos y ausencias
        'permiso':     'licencia autorización permiso',
        'vacaciones':  'vacaciones permiso descanso prestaciones',
        'baja':        'baja licencia incapacidad descanso',
        'excedencia':  'excedencia baja permiso',
        'lactancia':   'lactancia permiso reducción jornada acumulación',
        'maternidad':  'nacimiento permiso maternidad semanas',
        'paternidad':  'nacimiento permiso paternidad semanas',
        # Jornada y horario
        'horario':     'jornada horario flexible reducida',
        'jornada':     'jornada horas anuales horario trabajo curso',
        'horas':       'horas anuales jornada horario curso trabajo',
        'teletrabajo': 'teletrabajo remoto flexible jornada',
        # Retribución
        'sueldo':      'pagas nómina retribución anticipo',
        'sueldos':     'pagas nómina retribución anticipo',
        'salario':     'pagas nómina retribución anticipo',
        'salarios':    'pagas nómina retribución anticipo',
        'nómina':      'pagas anticipo retribución',
        'irpf':        'IRPF retención fiscal porcentaje',
        'retención':   'retención fiscal descuento porcentaje',
        'canon':       'canon educación capitalización aportación',
        # Desplazamiento
        'gastos':      'gastos dietas manutención desplazamiento reembolso',
        'kilometraje': 'kilómetro vehículo desplazamiento compensación',
        'kilometros':  'kilómetro vehículo desplazamiento compensación',
        'km':          'kilómetro vehículo desplazamiento compensación',
        'campus':      'cambio campus base trabajador compensación kilómetros semestres',
        'socias':      'clases personas socias usuarias trabajo colaboradoras inactivas',
        'variable':    'retribución variable anticipo laboral porcentaje objetivos',
        # Prestaciones sociales
        'jubilación':  'jubilación LagunAro pensión cotización',
        'pensión':     'jubilación LagunAro pensión cotización',
        'médico':      'médico Osarten servicio sanitario baja',
        'accidente':   'accidente tráfico compensación vehículo',
    }
    question_lower = question.lower()
    variants = [question]
    for key, value in synonyms_map.items():
        if key in question_lower:
            variants.append(value)
            break

    if not azure_client_text or not REWRITE_QUERY:
        return variants

    try:
        # Si viene en euskera, pedir que también dé una versión castellana
        lang_note = (
            "La pregunta puede estar en euskera. Genera las variantes SIEMPRE en castellano."
            if detected_lang == 'eu' else ""
        )
        prompt = f"""Eres un asistente legal. Dada la siguiente pregunta, genera 2 variantes en lenguaje jurídico-técnico para mejorar la búsqueda semántica en documentos normativos.
{lang_note}

Pregunta: {question}

Devuelve SOLO las variantes, una por línea, sin numeración ni comillas."""
        response = azure_client_text.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME_TEXT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        rewritten = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
        return variants + rewritten[:2]
    except Exception as e:
        print(f"[WARN] Error reescribiendo pregunta: {e}")
        return variants


# ===========================
# Texto limpio / contexto
# ===========================

def _normalize_text(text):
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _strip_chunk_metadata(text):
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"(?sim)^Documento\s*:\s*.*?\n(Texto\s*:\s*)?", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"(?im)^[^\n]*Documento[^\n]*\n", "", cleaned)
    cleaned = re.sub(r"(?im)^[^\n]*Texto[^\n]*:\s*", "", cleaned)
    cleaned = re.sub(r"\n[^\n]*Documento[^\n]*\n", "\n", cleaned)
    cleaned = re.sub(r"\n[^\n]*Texto[^\n]*:\s*", "\n", cleaned)
    return _normalize_text(cleaned)


def _extract_faq_answer(text):
    """Extrae solo la respuesta de un chunk FAQ, sin el ## de la pregunta."""
    if not text:
        return ""
    cleaned = _strip_chunk_metadata(text)
    lines = cleaned.split('\n')
    if lines and lines[0].strip().startswith('##'):
        lines = lines[1:]
    answer = '\n'.join(lines).strip()
    if not answer:
        answer = re.sub(r'^##\s*', '', cleaned).strip()
    return _normalize_text(answer)


def _sanitize_answer_output(text):
    if not text:
        return text
    cleaned = text.strip()
    cleaned = re.sub(r"(?sim)^Documento\s*:\s*.*?\n(Texto\s*:\s*)?", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"(?im)^[^\n]*?(?:Documento|Artículo|Texto)\s*:\s*[^\n]*\n", "", cleaned)
    cleaned = re.sub(r"(?im)(?:Documento|Artículo|Texto)\s*:\s*[^\n]*\.pdf", "", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def build_clean_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks[:CONTEXT_MAX_SOURCES], start=1):
        text = _strip_chunk_metadata(chunk.get('text', ''))
        header = f"Fuente {i}: Documento interno" if _is_faq_chunk(chunk) else f"Fuente {i}: {chunk.get('file_name', 'Documento')}"
        # Añadir cabecera de artículo si la hay
        article_match = re.search(r"\b(ART[IÍ]CULO\s+\d+[A-Za-zºª\-]*[^\n\.]*)", text)
        if article_match:
            header += f" ({article_match.group(1).strip()})"
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def build_faq_context(chunks):
    """Contexto FAQ: conserva el ## para que el LLM pueda leer la pregunta original."""
    parts = []
    for chunk in chunks[:CONTEXT_MAX_SOURCES]:
        text = chunk.get('text', '').strip()
        text = re.sub(r"(?sim)^Documento\s*:\s*.*?\n(Texto\s*:\s*)?", "", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"(?im)^[^\n]*Documento[^\n]*\n", "", text)
        text = re.sub(r"(?im)^[^\n]*Texto[^\n]*:\s*\n?", "", text).strip()
        if text:
            parts.append(text)
    return "\n\n---\n\n".join(parts)


# ===========================
# Generación de respuesta
# ===========================

LANG_INSTRUCTIONS = {
    'es': "Responde en español.",
    'eu': "Erantzuna euskaraz idatzi. (Responde en euskera.)",
}

def _lang_instruction(detected_lang):
    return LANG_INSTRUCTIONS.get(detected_lang, LANG_INSTRUCTIONS['es'])


def generate_faq_answer(question, faq_chunks, detected_lang):
    """Genera respuesta desde el FAQ con el LLM. Responde en el idioma de la pregunta."""
    if not azure_client_text:
        # Fallback: devolver la respuesta extraída directamente
        answer = _extract_faq_answer(faq_chunks[0].get('text', '')) if faq_chunks else None
        return answer, False  # (texto, is_llm_generated)

    # Usar solo el chunk más relevante para evitar que el LLM elija el equivocado
    top_faq_chunk = faq_chunks[0] if faq_chunks else None
    if not top_faq_chunk:
        return None, False

    # Extraer solo la respuesta del chunk top (sin el ## de la pregunta)
    top_faq_answer = _extract_faq_answer(top_faq_chunk.get('text', ''))
    if not top_faq_answer:
        return None, False

    faq_context = top_faq_answer
    system_prompt = f"""Eres el asistente oficial de la intranet de MGEP.
Se te proporciona la respuesta exacta extraída de las FAQ internas.
Tu única tarea es reformularla de forma natural en el idioma indicado.
CRÍTICO: Reproduce TODOS los datos exactos (números, fechas, importes, nombres). NUNCA omitas ni cambies cifras.
No añadas información externa. No menciones la fuente.
{_lang_instruction(detected_lang)}"""

    user_prompt = f"""Reformula esta respuesta de forma natural para la pregunta "{question}":

{faq_context}"""

    try:
        response = azure_client_text.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME_TEXT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        answer = (response.choices[0].message.content or "").strip()
        answer = _sanitize_answer_output(answer)
        return (answer, True) if answer else (None, False)
    except Exception as e:
        print(f"[WARN] Error generando respuesta FAQ con LLM: {e}")
        return None, False


def generate_answer_from_chunks(question, chunks, detected_lang):
    """Genera respuesta desde documentos normativos. Responde en el idioma de la pregunta."""
    if not azure_client_text:
        return None

    clean_context = build_clean_context(chunks)
    system_prompt = f"""Eres el asistente oficial de la intranet de MGEP.
Se te proporciona contexto extraído de documentos normativos internos.
Reglas:
- Responde basándote exclusivamente en la información del contexto proporcionado.
- NUNCA inventes cifras, fechas, plazos o datos que no estén en el contexto.
- Si la respuesta implica varios pasos o elementos distribuidos en el contexto,
  sintetízalos en orden coherente.
- Lee TODOS los fragmentos del contexto antes de responder, no solo el primero.
  La información relevante puede estar en cualquier posición del contexto.
  Solo si ningún fragmento contiene información útil, indica que no tienes la respuesta.
- Sé preciso y directo. Si hay artículos o procedimientos relevantes, cítalos.
- No menciones el nombre del documento ni la fuente.
- {_lang_instruction(detected_lang)}"""

    user_prompt = f"""Pregunta: {question}

Contexto:
{clean_context}

Proporciona una respuesta directa y completa basándote en el contexto anterior. Si la respuesta está distribuida en varios fragmentos, combínalos. Si un fragmento describe el caso contrario al preguntado, ignóralo y céntrate en los fragmentos relevantes para la pregunta."""

    try:
        response = azure_client_text.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME_TEXT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        answer = (response.choices[0].message.content or "").strip()
        return _sanitize_answer_output(answer)
    except Exception as e:
        print(f"[ERROR] Error generando respuesta: {e}")
        return None


# ===========================
# Re-ranking y utilidades
# ===========================

def _extract_query_terms(question):
    stopwords = {
        'de', 'la', 'el', 'los', 'las', 'un', 'una', 'y', 'o', 'en', 'con',
        'por', 'para', 'del', 'al', 'que', 'como', 'es', 'se', 'su', 'sobre',
    }
    tokens = re.findall(r"[a-záéíóúüñ0-9%]{3,}", question.lower())
    return {t for t in tokens if t not in stopwords}


def calculate_keyword_density(text):
    critical_keywords = [
        ('mayoría', 2.0), ('quórum', 2.0), ('aprobación', 1.8), ('artículo', 1.5),
        ('procedimiento', 1.5), ('modificación', 1.5), ('reglamento', 1.3),
        ('norma', 1.0), ('disposición', 1.0), ('%', 1.2),
    ]
    if not text:
        return 0.0
    text_lower = text.lower()
    word_count = len(text_lower.split())
    if word_count == 0:
        return 0.0
    score = sum(text_lower.count(kw.lower()) * w for kw, w in critical_keywords)
    return min(score / max(word_count / 5, 1), 1.0)


def calculate_query_overlap(text, query_terms):
    if not text or not query_terms:
        return 0.0
    text_terms = set(re.findall(r"[a-záéíóúüñ0-9%]{3,}", text.lower()))
    hits = sum(1 for t in query_terms if t in text_terms)
    return min(hits / max(len(query_terms), 1), 1.0)


def rerank_chunks(chunks, question):
    """Reranking: combina score híbrido, densidad normativa y overlap. FAQ no se retoca."""
    if not chunks:
        return []
    query_terms = _extract_query_terms(question)
    reranked = []
    for chunk in chunks:
        if _is_faq_chunk(chunk):
            chunk.setdefault('keyword_density', 0.0)
            chunk.setdefault('query_overlap', 0.0)
            reranked.append(chunk)
            continue
        base = float(chunk.get('similarity', 0.0))
        kd = calculate_keyword_density(chunk.get('text', ''))
        qo = calculate_query_overlap(chunk.get('text', ''), query_terms)
        combined = (base * 0.78) + (kd * 0.12) + (qo * 0.10)
        if qo < 0.08:
            combined *= 0.92
        chunk['keyword_density'] = kd
        chunk['query_overlap'] = qo
        chunk['similarity'] = combined
        reranked.append(chunk)
    reranked.sort(key=lambda x: x['similarity'], reverse=True)
    return reranked


def requires_governance_keywords(question):
    q = question.lower()
    patterns = [r'\bmayor[ií]a\b', r'\bqu[óo]rum\b', r'\baprobaci[óo]n\b',
                r'\bmodificaci[óo]n\b', r'\breglamento\b', r'\bart[ií]culo\b', r'\bconsejo\s+rector\b']
    return any(re.search(p, q) for p in patterns)


def check_answer_completeness(chunks, question):
    if not chunks:
        return {'answer_complete': False, 'missing_keywords': True, 'completeness_score': 0.0}

    combined_text = ' '.join(c.get('text', '') for c in chunks)
    governance_required = requires_governance_keywords(question)

    governance_kws = [r'\bmayor[ií]a\b', r'\bqu[óo]rum\b', r'\baprobaci[óo]n\b',
                      r'\bmodificaci[óo]n\b', r'\b\d+\s*%\b', r'\breglamento\b']
    has_keywords = any(re.search(p, combined_text.lower()) for p in governance_kws) if governance_required else True
    has_context = len(combined_text.split()) >= 100

    score = (0.6 if has_keywords else 0.0) + (0.4 if has_context else 0.0)
    return {
        'answer_complete': has_keywords and has_context,
        'missing_keywords': governance_required and not has_keywords,
        'completeness_score': score,
    }


def is_factual_question(question):
    if not question:
        return False
    patterns = [r"\b\d{1,4}\b", r"%", r"\bpor ciento\b", r"\bfecha\b",
                r"\bdía[s]?\b", r"\bmes(es)?\b", r"\baño[s]?\b", r"\bplazo\b", r"\bhora[s]?\b"]
    return any(re.search(p, question.lower()) for p in patterns)


def verify_answer_against_chunks(answer_text, chunks):
    if not answer_text or not chunks:
        return False
    tokens = re.findall(r"\b\d+[\d\.]*\b|\d+\s*%|\b\d{2,4}[-\/]\d{1,2}[-\/]\d{1,2}\b", answer_text)
    if not tokens:
        return True
    combined = ' '.join(_strip_chunk_metadata(c.get('text', '')) for c in chunks).lower()
    return all(t.lower().strip() in combined for t in tokens)


def find_similar_chunks_with_keywords(query_embedding, forced_keywords, top_k=TOP_K):
    """Segunda pasada con keywords forzadas para consultas normativas."""
    lexical_query = _normalize_search_query(' '.join(kw for kw in forced_keywords if re.search(r'\w', kw)))
    vector_literal = _embedding_to_pgvector_literal(query_embedding)
    candidate_limit = max(RETRIEVAL_CANDIDATE_LIMIT * 2, top_k * 15)

    conn = connect_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = """
                SELECT id, file_name, file_path, folder_name, chunk_index, text,
                       1 - (embedding <=> %s::vector(1536)) AS vector_similarity,
                       COALESCE(ts_rank_cd(search_tsv, websearch_to_tsquery('simple', NULLIF(%s, ''))), 0) AS lexical_rank
                FROM chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector(1536) LIMIT %s
            """
            cur.execute(sql, [vector_literal, lexical_query, vector_literal, candidate_limit])
            rows = cur.fetchall()

        filtered = []
        query_terms = _extract_query_terms(' '.join(forced_keywords))
        for chunk in rows:
            text_lower = chunk['text'].lower()
            found_kw = [kw for kw in forced_keywords if kw.lower() in text_lower]
            if not found_kw:
                continue
            vs = float(chunk['vector_similarity'] or 0.0)
            lr = float(chunk['lexical_rank'] or 0.0)
            kw_bonus = min(len(found_kw) * 0.05, 0.15)
            ov_bonus = calculate_query_overlap(chunk['text'], query_terms) * 0.08
            faq_bonus = FAQ_PRIORITY_BOOST if _is_faq_chunk(chunk) else 0.0
            filtered.append({
                'id': chunk['id'],
                'file_name': chunk['file_name'],
                'file_path': chunk['file_path'],
                'folder_name': chunk['folder_name'],
                'chunk_index': chunk['chunk_index'],
                'text': chunk['text'],
                'vector_similarity': vs,
                'lexical_rank': lr,
                'similarity': _hybrid_score(vs, lr) + kw_bonus + ov_bonus + faq_bonus,
                'found_keywords': found_kw,
            })
        filtered.sort(key=lambda x: x['similarity'], reverse=True)
        return filtered[:top_k]
    finally:
        conn.close()


def expand_context_with_neighbors(chunks):
    """Añade chunks vecinos (anterior/posterior) para más contexto continuo."""
    if not chunks:
        return chunks
    conn = connect_db()
    expanded = []
    neighbor_cache = {}
    try:
        for chunk in chunks:
            file_path = chunk.get('file_path', '')
            chunk_index = chunk.get('chunk_index', -1)
            expanded.append(chunk)
            for offset in range(-3, 4):
                if offset == 0 or chunk_index + offset < 0:
                    continue
                label = 'anterior' if offset < 0 else 'posterior'
                cache_key = (file_path, chunk_index + offset)
                if cache_key not in neighbor_cache:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute(
                            "SELECT id, file_name, file_path, folder_name, chunk_index, text "
                            "FROM chunks WHERE file_path = %s AND chunk_index = %s AND embedding IS NOT NULL LIMIT 1",
                            (file_path, chunk_index + offset)
                        )
                        neighbor_cache[cache_key] = cur.fetchone()
                neighbor = neighbor_cache[cache_key]
                if neighbor:
                    nd = dict(neighbor)
                    nd['similarity'] = chunk.get('similarity', 0.0) * 0.85
                    nd['is_neighbor'] = True
                    nd['neighbor_type'] = label
                    if offset < 0:
                        expanded.insert(len(expanded) - 1, nd)
                    else:
                        expanded.append(nd)
    finally:
        conn.close()
    return expanded


# ===========================
# Búsqueda rápida en FAQ (sin embeddings)
# ===========================

def _quick_faq_lookup(question):
    """Busca la entrada del FAQ más relevante usando matching por palabras clave.

    Estrategias en orden de prioridad:
    1. Coincidencia exacta normalizada o una contiene a la otra.
    2. Overlap de palabras clave: >= 2 palabras en común O >= 50% de las palabras
       significativas de la pregunta aparecen en el título del FAQ.
    """
    faq_raw = os.path.expanduser(FAQ_FILE)
    if os.path.isabs(faq_raw):
        faq_abs = faq_raw
    else:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        faq_abs = os.path.join(repo_root, faq_raw)

    if not os.path.exists(faq_abs):
        return None

    stopwords = {
        'de', 'la', 'el', 'los', 'las', 'un', 'una', 'y', 'o', 'en', 'con',
        'por', 'para', 'del', 'al', 'que', 'como', 'es', 'se', 'su', 'sobre',
        'hay', 'qué', 'cuál', 'cual', 'cómo', 'donde', 'cuando', 'puedo',
        'debo', 'tengo', 'tiene', 'me', 'mi', 'nos', 'hacer',
        'pedir', 'solicitar', 'obtener', 'pido', 'solicito',
    }

    def _norm(s):
        return re.sub(r"[^a-z0-9áéíóúüñ]+", ' ', (s or '').lower()).strip()

    # Sinónimos: expanden las keywords de la pregunta antes del matching
    sinonimos = {
        'sueldos': ['pagas', 'nóminas', 'retribución'],
        'sueldo': ['paga', 'nómina', 'retribución'],
        'salarios': ['pagas', 'nóminas', 'retribución'],
        'salario': ['paga', 'nómina', 'retribución'],
        'nóminas': ['pagas', 'retribución'],
        'nómina': ['paga', 'retribución'],
        'cobrar': ['percibir', 'recibir', 'paga'],
        'cobran': ['perciben', 'pagas'],
        'cobro': ['percibo', 'paga'],
        'salario': ['pagas', 'paga', 'nómina', 'perciben', 'anuales', '14'],
        'salarios': ['pagas', 'paga', 'nómina', 'perciben', 'anuales', '14'],
        'recibe': ['percibe', 'paga', 'cobran'],
        'reciben': ['perciben', 'pagas', 'cobran'],
        'cobra': ['percibe', 'paga', 'cobran'],
        'km': ['kilómetro', 'kilómetros', 'desplazamiento'],
        'kilómetros': ['kilómetro', 'desplazamiento', 'compensación'],
        'coche': ['vehículo', 'desplazamiento'],
        'médico': ['médica', 'osarten', 'sanitario'],
        'doctora': ['médica', 'osarten'],
        'doctor': ['médico', 'osarten'],
        'jubilación': ['pensión', 'lagunaro', 'cotización'],
        'pensión': ['jubilación', 'lagunaro', 'cotización'],
        'retención': ['irpf', 'porcentaje', 'impuesto'],
        'impuesto': ['irpf', 'retención', 'porcentaje'],
        'maternidad': ['nacimiento', 'permiso', 'semanas'],
        'paternidad': ['nacimiento', 'permiso', 'semanas'],
        'bebé': ['nacimiento', 'lactancia', 'permiso'],
        'embarazo': ['nacimiento', 'maternidad', 'permiso'],
        'accidente': ['tráfico', 'vehículo', 'compensación'],
        'despido': ['baja', 'cese', 'voluntaria'],
        'despedir': ['baja', 'cese', 'voluntaria'],
        'contrato': ['sdd', 'sdi', 'incorporación'],
        'incorporación': ['contrato', 'sdd', 'trámites'],
        'nuevo': ['incorporación', 'contrato', 'trámites'],
        'nueva': ['incorporación', 'contrato', 'trámites'],
    }

    def _keywords(s):
        base = {w for w in _norm(s).split() if len(w) >= 2 and w not in stopwords}
        # Expandir con sinónimos
        expanded = set(base)
        for w in base:
            if w in sinonimos:
                expanded.update(sinonimos[w])
        return expanded

    try:
        with open(faq_abs, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return None

    q = None
    a_lines = []
    pairs = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('##') and not stripped.startswith('###'):
            if q:
                pairs.append((q, ' '.join(a_lines).strip()))
            q = stripped.lstrip('#').strip()
            a_lines = []
        elif q is not None:
            if stripped.startswith('###'):
                continue
            a_lines.append(stripped)
    if q:
        pairs.append((q, ' '.join(a_lines).strip()))

    nq = _norm(question)
    q_kw = _keywords(question)
    best_answer = None
    best_score = 0.0
    best_common_count = 0

    for fq, fa in pairs:
        if not fq or not fa:
            continue
        nfq = _norm(fq)

        # 1. Coincidencia exacta o contenido
        if nq == nfq or nq in nfq or nfq in nq:
            return fa

        # 2. Overlap contra el título del FAQ únicamente (sin body_preview).
        #    Usar el cuerpo de la respuesta para el matching causa falsos positivos:
        #    palabras genéricas del texto (mes, trabajador, plazo…) hacen que
        #    preguntas completamente distintas puntúen contra entradas incorrectas.
        if not q_kw:
            continue

        fq_kw = _keywords(fq)
        common = q_kw & fq_kw
        overlap_ratio = len(common) / len(q_kw)

        # Score único: ratio normalizado [0,1], sin mezclar con conteo absoluto.
        score = overlap_ratio
        if score > best_score:
            best_score = score
            best_answer = fa
            best_common_count = len(common)

    # Umbral estricto (AND): ratio >= 0.60 Y al menos 2 palabras en común.
    # El AND evita que preguntas de 2 KW con 1 coincidencia pasen (ratio=0.5,
    # common=1 → bloqueado). El ratio 0.60 exige mayoría de palabras significativas.
    if best_answer and best_score >= 0.60 and best_common_count >= 2:
        return best_answer
    return None


# ===========================
# Rutas API
# ===========================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'API normativas MGEP'}), 200


@app.route('/search', methods=['POST'])
def search():
    if not request.json:
        return jsonify({'error': 'Request debe ser JSON'}), 400
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'Campo "question" es requerido'}), 400

    # Idioma: site_language del frontend tiene prioridad, luego detección automática
    site_language = (
        request.json.get('site_language')
        or request.json.get('page_language')
        or request.json.get('language')
        or request.json.get('idioma')
    )
    if site_language and site_language.lower().strip()[:2] == 'eu':
        detected_lang = 'eu'
    elif site_language and site_language.lower().strip()[:2] == 'es':
        detected_lang = 'es'
    else:
        detected_lang = detect_language(question)
    print(f"[LANG] Idioma resuelto: {detected_lang} (site_language={site_language!r})")

    # ── Búsqueda rápida en FAQ (sin embeddings) ──────────────────────────────
    # Si la pregunta viene en euskera, buscar en el FAQ con la pregunta tal cual
    # (el LLM se encarga de responder en euskera usando el contexto en castellano)
    quick_faq_text = _quick_faq_lookup(question)
    # Si no hay match y la pregunta es en euskera, intentar traducirla al castellano
    # para buscar en el FAQ (que está en castellano)
    if not quick_faq_text and detected_lang == 'eu' and azure_client_text:
        try:
            r = azure_client_text.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME_TEXT,
                messages=[
                    {"role": "system", "content": "Traduce al castellano la siguiente pregunta. Solo la pregunta traducida, sin explicaciones."},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_tokens=100,
            )
            question_es = (r.choices[0].message.content or "").strip()
            if question_es:
                print(f"[FAQ] Traduccion EU→ES: {question_es!r}")
                quick_faq_text = _quick_faq_lookup(question_es)
                print(f"[FAQ] Lookup resultado: {'HIT → ' + quick_faq_text[:60] if quick_faq_text else 'MISS'}")
        except Exception as e:
            print(f"[WARN] Error traduciendo pregunta para FAQ: {e}")
    if quick_faq_text:
        print(f"[FAQ] Match rápido encontrado, pasando por LLM (lang={detected_lang})")
        # Pasar siempre por el LLM: reformula la respuesta de forma natural
        # y responde en el idioma de la pregunta
        final_answer = quick_faq_text  # fallback si el LLM falla
        if azure_client_text:
            try:
                if detected_lang == 'eu':
                    # Euskera: traducir el texto exacto del FAQ
                    system_prompt = """Traduce el siguiente texto al euskera. Reproduce todos los datos exactos (números, fechas, importes). Solo el texto traducido, sin explicaciones."""
                    r = azure_client_text.chat.completions.create(
                        model=AZURE_DEPLOYMENT_NAME_TEXT,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": quick_faq_text},
                        ],
                        temperature=0.0,
                        max_tokens=500,
                    )
                    llm_answer = (r.choices[0].message.content or "").strip()
                    if llm_answer:
                        final_answer = _sanitize_answer_output(llm_answer)
                # Castellano: FAQ ya en castellano, devolver directamente
            except Exception as e:
                print(f"[WARN] Error pasando FAQ por LLM: {e}")

        return jsonify({
            'question': question,
            'answer': final_answer,
            'answer_generated': True,
            'answer_extractive': False,
            'display_mode': 'answer',
            'confidence': 'alta',
            'answered_by_faq': True,
            'sources': [],
            'results': [],
            'count': 0,
        }), 200

    if not azure_client:
        return jsonify({'error': 'Azure OpenAI no configurado'}), 500

    top_k = request.json.get('top_k', TOP_K)
    generate_answer = request.json.get('generate_answer', DEFAULT_GENERATE_ANSWER)

    # ── Reescritura y búsqueda semántica ─────────────────────────────────────
    query_variants = rewrite_query(question, detected_lang)
    retrieval_top_k = max(top_k * 3, top_k + 6)

    all_similar = {}
    for variant in query_variants:
        qe = calculate_embedding(variant)
        for chunk in _fetch_scored_chunks(qe, variant, retrieval_top_k):
            cid = chunk['id']
            if cid not in all_similar:
                all_similar[cid] = chunk
            else:
                all_similar[cid]['similarity'] = max(all_similar[cid]['similarity'], chunk['similarity'])

    all_chunks_list = list(all_similar.values())
    top_chunk = max(all_chunks_list, key=lambda x: x['similarity']) if all_chunks_list else None

    # ── ¿Responde el FAQ (vía embedding)? ────────────────────────────────────
    answered_by_faq = (
        top_chunk is not None
        and _is_faq_chunk(top_chunk)
        and top_chunk['similarity'] >= FAQ_MIN_SIMILARITY
    )

    if answered_by_faq:
        print(f"[INFO] Respuesta desde FAQ (sim={top_chunk['similarity']:.3f})")
        faq_chunks = sorted(
            [c for c in all_chunks_list if _is_faq_chunk(c)],
            key=lambda x: x['similarity'], reverse=True
        )[:top_k]
        similar_chunks = faq_chunks
        similar_chunks_expanded = faq_chunks
        completeness = check_answer_completeness(faq_chunks, question)
        max_similarity = faq_chunks[0]['similarity'] if faq_chunks else 0
        search_passes = 1
        second_pass_performed = False
    else:
        # ── Búsqueda en documentos normativos ────────────────────────────────
        answered_by_faq = False
        similar_chunks = sorted(all_chunks_list, key=lambda x: x['similarity'], reverse=True)[:retrieval_top_k]
        similar_chunks = rerank_chunks(similar_chunks, question)
        max_similarity = similar_chunks[0]['similarity'] if similar_chunks else 0
        completeness = check_answer_completeness(similar_chunks, question)
        search_passes = 1
        second_pass_performed = False

        # Segundo pase si faltan keywords normativas
        if max_similarity >= MIN_SIMILARITY_SECOND_PASS and completeness['missing_keywords']:
            print(f"[INFO] Segundo pase activado: sim={max_similarity:.2f}, faltan keywords")
            q_lower = question.lower()
            forced_kws = [kw for kw in ['mayoría', 'quórum', 'aprobación', 'modificación', 'reglamento', 'artículo'] if kw in q_lower]
            if not forced_kws:
                forced_kws = ['reglamento', 'artículo', 'procedimiento']
            try:
                enhanced_q = f"{query_variants[0]} {' '.join(forced_kws)}"
                qe2 = calculate_embedding(enhanced_q)
                second_chunks = find_similar_chunks_with_keywords(qe2, forced_kws, top_k=top_k)
                for chunk in second_chunks:
                    cid = chunk['id']
                    if cid not in all_similar or chunk['similarity'] > all_similar[cid]['similarity'] * 1.1:
                        all_similar[cid] = chunk
                similar_chunks = sorted(all_similar.values(), key=lambda x: x['similarity'], reverse=True)[:retrieval_top_k]
                similar_chunks = rerank_chunks(similar_chunks, question)
                completeness = check_answer_completeness(similar_chunks, question)
                search_passes = 2
                second_pass_performed = True
            except Exception as e:
                print(f"[WARN] Error en segundo pase: {e}")

        max_similarity = similar_chunks[0]['similarity'] if similar_chunks else 0
        similar_chunks_expanded = expand_context_with_neighbors(similar_chunks) if ENABLE_NEIGHBOR_EXPANSION else similar_chunks

    # ── Confianza ─────────────────────────────────────────────────────────────
    if max_similarity >= MIN_SIMILARITY:
        final_confidence = 'alta' if completeness['answer_complete'] else 'media'
        zone, ui_color = 'high', 'green'
        ui_message = 'Información encontrada con alta confianza'
    elif max_similarity >= MIN_SIMILARITY_WARNING:
        final_confidence = 'media'
        zone, ui_color = 'gray', 'yellow'
        ui_message = 'Información encontrada con confianza media — verificar con fuente oficial'
    elif max_similarity >= MIN_SIMILARITY_ABSOLUTE:
        final_confidence = 'baja'
        zone, ui_color = 'low', 'orange'
        ui_message = 'Información de baja confianza — se recomienda consultar directamente'
    else:
        final_confidence = 'muy-baja'
        zone, ui_color = 'very-low', 'red'
        ui_message = 'No se ha encontrado información relevante para esta consulta'

    # ── Preparar resultados visibles ──────────────────────────────────────────
    visible_chunks = [c for c in similar_chunks if not _is_faq_chunk(c)]
    response_results = []
    for item in visible_chunks:
        ci = dict(item)
        ci['text'] = _strip_chunk_metadata(item.get('text', ''))
        response_results.append(ci)

    primary_source = None
    if visible_chunks:
        top_item = visible_chunks[0]
        primary_source = {
            'file': top_item.get('file_name', 'Documento desconocido'),
            'similarity': round(float(top_item.get('similarity', 0)) * 100, 1),
        }

    response_data = {
        'question': question,
        'detected_language': detected_lang,
        'query_variants_used': len(query_variants),
        'max_similarity': float(max_similarity),
        'min_similarity_threshold': MIN_SIMILARITY,
        'min_similarity_warning': MIN_SIMILARITY_WARNING,
        'min_similarity_absolute': MIN_SIMILARITY_ABSOLUTE,
        'zone': zone,
        'ui_confidence_color': ui_color,
        'ui_message': ui_message,
        'answer_complete': completeness['answer_complete'],
        'completeness_score': completeness['completeness_score'],
        'missing_keywords': completeness['missing_keywords'],
        'search_passes': search_passes,
        'second_pass_performed': second_pass_performed,
        'answered_by_faq': answered_by_faq,
        'confidence': final_confidence,
        'sources': [primary_source] if primary_source else [],
        'results': response_results,
        'count': len(response_results),
    }

    # ── Generación de respuesta con LLM ──────────────────────────────────────
    if generate_answer:
        if ENABLE_NEIGHBOR_EXPANSION:
            # Originales primero (TOP_K garantizados), luego vecinos hasta CONTEXT_MAX_SOURCES
            original_ids = {c['id'] for c in similar_chunks}
            neighbors = [c for c in similar_chunks_expanded if c['id'] not in original_ids]
            neighbors_sorted = sorted(neighbors, key=lambda x: x['similarity'], reverse=True)
            context_chunks = similar_chunks + neighbors_sorted
        else:
            context_chunks = similar_chunks

        if answered_by_faq:
            answer, is_llm = generate_faq_answer(question, similar_chunks, detected_lang)
            if answer:
                response_data['answer'] = answer
                response_data['answer_generated'] = is_llm
                response_data['answer_extractive'] = not is_llm
                response_data['display_mode'] = 'answer'
                return jsonify(response_data), 200
            # Fallback: extracción directa
            raw = _extract_faq_answer(similar_chunks[0].get('text', '')) if similar_chunks else None
            if raw:
                response_data['answer'] = raw
                response_data['answer_generated'] = False
                response_data['answer_extractive'] = True
                response_data['display_mode'] = 'answer'
                return jsonify(response_data), 200
        else:
            answer = generate_answer_from_chunks(question, context_chunks, detected_lang)
            response_data['answer'] = answer
            response_data['answer_generated'] = bool(answer)
            response_data['answer_extractive'] = False
            response_data['answer_verified'] = verify_answer_against_chunks(answer or '', context_chunks)
            response_data['display_mode'] = 'answer' if answer else 'results'

    else:
        response_data['display_mode'] = 'results'

    return jsonify(response_data), 200


@app.route('/stats', methods=['GET'])
def stats():
    try:
        conn = connect_db()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
            chunks_with_embeddings = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT file_path) FROM chunks")
            total_files = cur.fetchone()[0]
        conn.close()
        return jsonify({
            'total_chunks': total_chunks,
            'chunks_with_embeddings': chunks_with_embeddings,
            'total_files': total_files,
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===========================
# Main
# ===========================
if __name__ == '__main__':
    port = int(os.getenv('API_PORT', '5000'))
    debug = os.getenv('API_DEBUG', 'False').lower() == 'true'

    print(f"[INFO] Iniciando API en puerto {port}")
    print(f"[INFO] Embeddings:  {'✅' if azure_client else '❌'}")
    print(f"[INFO] Texto (LLM): {'✅ ' + AZURE_DEPLOYMENT_NAME_TEXT if azure_client_text else '❌'}")
    print(f"[INFO] Umbrales — verde: >={MIN_SIMILARITY} | amarillo: >={MIN_SIMILARITY_WARNING} | naranja: >={MIN_SIMILARITY_ABSOLUTE}")
    print(f"[INFO] FAQ: {FAQ_FILE}")
    print(f"[INFO] DB: {DB_HOST}:{DB_PORT}/{DB_NAME}")

    app.run(host='0.0.0.0', port=port, debug=debug)