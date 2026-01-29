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
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.65"))  # Umbral mínimo de similitud
MIN_SIMILARITY_WARNING = float(os.getenv("MIN_SIMILARITY_WARNING", "0.55"))  # Umbral zona gris
REWRITE_QUERY = os.getenv("REWRITE_QUERY", "True").lower() == "true"  # Reescribir preguntas
CONTEXT_SUMMARY_MODE = os.getenv("CONTEXT_SUMMARY_MODE", "heuristic").lower()  # model | heuristic
CONTEXT_MAX_SOURCES = int(os.getenv("CONTEXT_MAX_SOURCES", str(TOP_K)))
CONTEXT_MAX_BULLETS = int(os.getenv("CONTEXT_MAX_BULLETS", "2"))

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

def _normalize_text(text):
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

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
    bullets = [s.strip() for s in sentences if s.strip()]
    return bullets[:max_bullets]

def build_clean_context(question, chunks):
    sources = []
    for chunk in chunks[:CONTEXT_MAX_SOURCES]:
        cleaned_text = _normalize_text(chunk.get('text', ''))
        article = _extract_article_title(cleaned_text)
        sources.append({
            'file_name': chunk.get('file_name', 'Documento'),
            'article': article,
            'text': cleaned_text,
            'similarity': chunk.get('similarity', 0.0)
        })

    if not sources:
        return ""

    if CONTEXT_SUMMARY_MODE == "model" and azure_client_text:
        try:
            source_blocks = []
            for i, s in enumerate(sources, start=1):
                header = f"Fuente {i}: {s['file_name']}"
                if s['article']:
                    header += f" ({s['article']})"
                source_blocks.append(f"{header}\nTexto: {s['text']}")

            prompt = f"""Eres un analista documental. Resume de forma breve y fiel cada fuente para responder la pregunta del usuario.

Pregunta: {question}

Instrucciones:
- Devuelve SOLO resúmenes por fuente en el siguiente formato:
  Fuente N: <Documento> (<Artículo si existe>)
  Resumen del fragmento:
  - ...
  - ...
- No inventes datos ni porcentajes.
- Si el fragmento no es concluyente, indícalo.

Fuentes:
""" + "\n\n".join(source_blocks)

            response = azure_client_text.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME_TEXT,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=700
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[WARN] Error creando contexto con modelo: {e}")

    # Fallback heurístico
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

def generate_answer_from_chunks(question, chunks, max_similarity):
    """
    Genera una respuesta usando el modelo de texto de Azure OpenAI
    basándose en los chunks relevantes encontrados.
    
    Tres niveles de confianza:
    - max_similarity >= MIN_SIMILARITY: respuesta normal
    - MIN_SIMILARITY_WARNING <= max_similarity < MIN_SIMILARITY: respuesta con aviso (zona gris)
    - max_similarity < MIN_SIMILARITY_WARNING: no hay información suficiente
    """
    if not azure_client_text:
        return None
    
    # Nivel 1: Sin información suficiente
    if max_similarity < MIN_SIMILARITY_WARNING:
        return f"No he encontrado en la documentación disponible información relevante sobre tu pregunta (confianza: {max_similarity:.0%}). Te recomiendo consultar con RRHH o revisar el reglamento completo en el portal."
    
    # Nivel 2: Zona gris - respuesta con aviso (0.55-0.65)
    is_gray_zone = max_similarity < MIN_SIMILARITY
    
    try:
        # Construir el contexto limpio a partir de los chunks
        clean_context = build_clean_context(question, chunks)

        # Crear el prompt para el modelo
        if is_gray_zone:
            # Prompt para zona gris: más cauteloso
            system_prompt = """Eres el asistente oficial de la intranet corporativa.
Utiliza únicamente la información proporcionada en las fuentes.
Redacta una respuesta clara, completa y prudente.
Si la información no es concluyente, indícalo explícitamente.
No muestres fragmentos de texto sin explicar."""

            user_prompt = f"""Pregunta:
{question}

Fuentes (resumidas):
{clean_context}

Instrucciones:
- Sintetiza y redacta una respuesta clara y profesional.
- Indica que la información puede no ser exactamente la requerida.
- Sugiere verificar el artículo o documento específico si hay dudas.
- No inventes datos, porcentajes ni requisitos que no estén en las fuentes."""
        else:
            # Prompt normal: alta confianza
            system_prompt = """Eres el asistente oficial de la intranet corporativa.
Utiliza únicamente la información proporcionada en las fuentes.
Redacta una respuesta clara y completa.
Si la información no es totalmente concluyente, indícalo explícitamente.
No muestres fragmentos de texto sin explicar."""

            user_prompt = f"""Pregunta:
{question}

Fuentes (resumidas):
{clean_context}

Instrucciones:
- Sintetiza y redacta la respuesta en lenguaje claro.
- No copies textualmente las fuentes.
- Si falta información específica, indícalo de forma transparente."""
        
        # Llamar al modelo de texto
        response = azure_client_text.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME_TEXT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,  # Temperatura balanceada para respuestas naturales pero precisas
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"[ERROR] Error generando respuesta: {e}")
        return None

def find_similar_chunks(query_embedding, top_k=TOP_K):
    """
    Encuentra los chunks más similares usando similitud del coseno
    """
    conn = connect_db()
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Obtener todos los chunks con embeddings
            cur.execute("""
                SELECT id, file_name, file_path, folder_name, chunk_index, text, embedding
                FROM chunks
                WHERE embedding IS NOT NULL
            """)
            chunks = cur.fetchall()
        
        # Calcular similitud del coseno para cada chunk
        similarities = []
        for chunk in chunks:
            embedding = chunk['embedding']
            if embedding:
                # Convertir embedding de string a lista si es necesario
                if isinstance(embedding, str):
                    # El formato es "[0.1, 0.2, ...]"
                    embedding = [float(x) for x in embedding.strip('[]').split(',')]
                
                similarity = cosine_similarity(query_embedding, embedding)
                similarities.append({
                    'id': chunk['id'],
                    'file_name': chunk['file_name'],
                    'file_path': chunk['file_path'],
                    'folder_name': chunk['folder_name'],
                    'chunk_index': chunk['chunk_index'],
                    'text': chunk['text'],
                    'similarity': float(similarity)
                })
        
        # Ordenar por similitud descendente y retornar los top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
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
        generate_answer = request.json.get('generate_answer', False)
        
        # Validar Azure OpenAI
        if not azure_client:
            return jsonify({
                'error': 'Azure OpenAI no configurado. Verifica las variables de entorno.'
            }), 500
        
        # Reescribir pregunta para mejor matching semántico
        query_variants = rewrite_query(question)
        
        # Buscar con todas las variantes y combinar resultados
        all_similar = {}
        for query_variant in query_variants:
            query_embedding = calculate_embedding(query_variant)
            similar_chunks = find_similar_chunks(query_embedding, top_k)
            
            for chunk in similar_chunks:
                chunk_id = chunk['id']
                if chunk_id not in all_similar:
                    all_similar[chunk_id] = chunk
                else:
                    # Si el mismo chunk aparece en múltiples búsquedas, aumentar su similitud
                    all_similar[chunk_id]['similarity'] = max(all_similar[chunk_id]['similarity'], chunk['similarity'])
        
        # Ordenar por similitud
        similar_chunks = sorted(all_similar.values(), key=lambda x: x['similarity'], reverse=True)[:top_k]
        max_similarity = similar_chunks[0]['similarity'] if similar_chunks else 0
        
        # Preparar respuesta
        response_data = {
            'question': question,
            'query_variants_used': len(query_variants),
            'max_similarity': float(max_similarity),
            'min_similarity_threshold': MIN_SIMILARITY,
            'min_similarity_warning': MIN_SIMILARITY_WARNING,
            'zone': 'high' if max_similarity >= MIN_SIMILARITY else 'gray' if max_similarity >= MIN_SIMILARITY_WARNING else 'low',
            'results': similar_chunks,
            'count': len(similar_chunks)
        }
        
        # Generar respuesta con el modelo de texto si se solicita
        if generate_answer:
            if azure_client_text:
                answer = generate_answer_from_chunks(question, similar_chunks, max_similarity)
                response_data['answer'] = answer
                response_data['answer_generated'] = True
                # Confianza: alta (>=0.65), media-con-aviso (0.55-0.65), baja (<0.55)
                if max_similarity >= MIN_SIMILARITY:
                    response_data['confidence'] = 'alta'
                elif max_similarity >= MIN_SIMILARITY_WARNING:
                    response_data['confidence'] = 'media-con-aviso'
                else:
                    response_data['confidence'] = 'baja'
            else:
                response_data['answer'] = None
                response_data['answer_generated'] = False
                response_data['answer_error'] = 'Modelo de texto no configurado. Configure AZURE_DEPLOYMENT_NAME_TEXT.'
        
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
    print(f"[INFO] Umbrales de similitud:")
    print(f"       - Alta confianza: >= {MIN_SIMILARITY}")
    print(f"       - Zona gris (con aviso): {MIN_SIMILARITY_WARNING} - {MIN_SIMILARITY}")
    print(f"       - Sin información: < {MIN_SIMILARITY_WARNING}")
    print(f"[INFO] Reescritura de preguntas activa: {'Sí' if REWRITE_QUERY else 'No'}")
    print(f"[INFO] Base de datos: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
