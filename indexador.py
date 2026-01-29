#!/usr/bin/env python3
import os
import hashlib
import re
import psycopg2
from psycopg2.extras import execute_values
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential

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
# Directorio base de PDFs
# ===========================
BASE_DIR = os.path.expanduser(os.getenv("BASE_DIR", "~/normativas"))
IGNORE_DIRS = [d.strip() for d in os.getenv("IGNORE_DIRS", "NormativasAPP").split(",") if d.strip()]

# ===========================
# Parámetros de chunking
# ===========================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
FORCE_REINDEX = os.getenv("FORCE_REINDEX", "False").lower() == "true"

# ===========================
# Azure OpenAI Configuration
# ===========================
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "text-embedding-3-small")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

# Cliente de Azure OpenAI
azure_client = None
if AZURE_ENDPOINT and AZURE_API_KEY:
    azure_client = AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY
    )

# ===========================
# Funciones
# ===========================

def hash_file(path):
    """Devuelve SHA256 del archivo"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_text_from_pdf(pdf_path):
    """Extrae texto de PDF"""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"[ERROR] No se pudo leer {pdf_path}: {e}")
    return text

def normalize_pdf_text(text):
    """Limpia saltos de línea y cortes de palabra típicos de PDF."""
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def extract_article_title(text):
    if not text:
        return None
    match = re.search(
        r"(?im)^\s*(ART[IÍ]CULO\s+\d+[A-Za-zºª\-]*[^\n]*)",
        text,
    )
    if match:
        return match.group(1).strip()
    return None

def split_by_structure(text):
    if not text:
        return []
    pattern = (
        r"(?im)(?=^\s*(?:"
        r"ART[IÍ]CULO\s+\d+[A-Za-zºª\-]*|"
        r"DISPOSICION(?:ES)?\s+\w+|"
        r"T[ÍI]TULO\s+\w+|"
        r"CAP[IÍ]TULO\s+\w+|"
        r"SECCI[ÓO]N\s+\w+|"
        r"Uno\.|Dos\.|Tres\.|Cuatro\.|Cinco\.|Seis\.|Siete\.|Ocho\.|Nueve\.|Diez\."
        r"))"
    )
    parts = [p.strip() for p in re.split(pattern, text) if p.strip()]
    return parts

def split_by_sentences(text, max_len=CHUNK_SIZE):
    if not text:
        return []
    sentences = re.split(r"(?<=[\.!\?])\s+", text)
    chunks = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not current:
            current = s
            continue
        if len(current) + len(s) + 1 <= max_len:
            current = f"{current} {s}"
        else:
            chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunking por estructura legal con fallback a frases/tamaño fijo."""
    if not text:
        return []

    structured = split_by_structure(text)
    chunks = []

    if structured:
        for part in structured:
            if len(part) <= chunk_size:
                chunks.append(part)
            else:
                chunks.extend(split_by_sentences(part, max_len=chunk_size))
        return chunks

    # Fallback por párrafos si no hay estructura detectada
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current = ""
    for p in paragraphs:
        if not current:
            if len(p) <= chunk_size:
                current = p
                continue
        if len(current) + len(p) + 2 <= chunk_size:
            current = f"{current}\n\n{p}" if current else p
        else:
            if current:
                chunks.append(current)
            if len(p) > chunk_size:
                start = 0
                while start < len(p):
                    end = start + chunk_size
                    chunks.append(p[start:end])
                    start += chunk_size - overlap
                current = ""
            else:
                current = p
    if current:
        chunks.append(current)

    return chunks

def enrich_chunk_with_metadata(file_name, chunk_text_value):
    article = extract_article_title(chunk_text_value)
    header_lines = [f"Documento: {file_name}"]
    if article:
        header_lines.append(f"Artículo: {article}")
    header = "\n".join(header_lines)
    return f"{header}\nTexto:\n{chunk_text_value}".strip()

def connect_db():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )

def create_table():
    """Tabla de chunks + hash para versionado y esquemas nuevos."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS chunks (
        id SERIAL PRIMARY KEY,
        file_name TEXT,
        file_path TEXT,
        folder_name TEXT,
        chunk_index INT,
        text TEXT,
        embedding vector(1536),
        file_hash TEXT
    );
    """

    migrate_sql = """
    DO $$
    BEGIN
        -- Asegura la columna file_path si la tabla existía sin ella
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'chunks' AND column_name = 'file_path'
        ) THEN
            ALTER TABLE chunks ADD COLUMN file_path TEXT;
        END IF;

        -- Índice para búsquedas por ruta
        IF NOT EXISTS (
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = 'idx_chunks_file_path' AND n.nspname = 'public'
        ) THEN
            CREATE INDEX idx_chunks_file_path ON chunks(file_path);
        END IF;

        -- Índice único para ruta + chunk_index
        IF NOT EXISTS (
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = 'ux_chunks_file_path_chunk_index' AND n.nspname = 'public'
        ) THEN
            CREATE UNIQUE INDEX ux_chunks_file_path_chunk_index ON chunks(file_path, chunk_index);
        END IF;
    END
    $$;
    """

    conn = connect_db()
    with conn:
        with conn.cursor() as cur:
            cur.execute(create_sql)
            cur.execute(migrate_sql)
    conn.close()

def calculate_embeddings(chunks):
    """Calcula embeddings para una lista de chunks usando Azure OpenAI."""
    if not azure_client:
        print("[WARN] Azure OpenAI no configurado, embeddings serán NULL")
        return [None] * len(chunks)
    
    try:
        # Azure OpenAI permite hasta 2048 inputs por request, procesamos en lotes
        batch_size = 2048
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            response = azure_client.embeddings.create(
                input=batch,
                model=AZURE_DEPLOYMENT_NAME
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            print(f"[INFO] Embeddings calculados: {i + len(batch)}/{len(chunks)}")
        
        return all_embeddings
    except Exception as e:
        print(f"[ERROR] Error calculando embeddings: {e}")
        return [None] * len(chunks)

def insert_chunks(file_name, file_path, folder_name, chunks, file_hash):
    """Inserta chunks con sus embeddings en la base de datos."""
    # Añadir metadatos al texto del chunk
    chunks = [enrich_chunk_with_metadata(file_name, c) for c in chunks]
    # Calcular embeddings para todos los chunks
    embeddings = calculate_embeddings(chunks)
    
    conn = connect_db()
    data = [
        (file_name, file_path, folder_name, i, chunk, embeddings[i], file_hash)
        for i, chunk in enumerate(chunks)
    ]
    sql = """
    INSERT INTO chunks (
        file_name,
        file_path,
        folder_name,
        chunk_index,
        text,
        embedding,
        file_hash
    ) VALUES %s
    """
    with conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, data)
    conn.close()

def update_missing_embeddings(file_path):
    """Calcula y actualiza embeddings NULL para un PDF."""
    conn = connect_db()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, text FROM chunks WHERE file_path = %s AND embedding IS NULL ORDER BY chunk_index",
            (file_path,)
        )
        rows = cur.fetchall()
    conn.close()
    
    if not rows:
        return 0
    
    chunk_ids = [row[0] for row in rows]
    chunk_texts = [row[1] for row in rows]
    
    print(f"[INFO] Calculando {len(chunk_texts)} embeddings faltantes para {file_path}")
    embeddings = calculate_embeddings(chunk_texts)
    
    # Actualizar embeddings en la base de datos
    conn = connect_db()
    with conn:
        with conn.cursor() as cur:
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                if embedding is not None:
                    cur.execute(
                        "UPDATE chunks SET embedding = %s WHERE id = %s",
                        (embedding, chunk_id)
                    )
    conn.close()
    
    return len(chunk_ids)

def delete_chunks(file_path):
    """Borra todos los chunks de un PDF (ruta relativa)."""
    conn = connect_db()
    with conn:
        with conn.cursor() as cur:
            if file_path is None:
                cur.execute("DELETE FROM chunks WHERE file_path IS NULL")
            else:
                cur.execute("DELETE FROM chunks WHERE file_path = %s", (file_path,))
    conn.close()

def delete_all_chunks():
    """Borra todos los chunks de la base de datos."""
    conn = connect_db()
    with conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks")
    conn.close()

def process_pdfs():
    """Procesa todos los PDFs, detectando cambios"""
    if FORCE_REINDEX:
        print("[INFO] FORCE_REINDEX activo: eliminando todos los chunks antes de reindexar")
        delete_all_chunks()
    # Lista de PDFs actuales usando ruta relativa para evitar colisiones de nombres
    current_files = {}
    for root, dirs, files in os.walk(BASE_DIR):
        for ignore in IGNORE_DIRS:
            if ignore in dirs:
                dirs.remove(ignore)
        folder_name = os.path.basename(root)
        for f in files:
            if f.lower().endswith(".pdf"):
                full_path = os.path.join(root, f)
                relative_path = os.path.relpath(full_path, BASE_DIR)
                file_hash = hash_file(full_path)
                current_files[relative_path] = file_hash
                # Revisar si está en DB con mismo hash
                conn = connect_db()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT file_hash FROM chunks WHERE file_path = %s LIMIT 1",
                        (relative_path,),
                    )
                    row = cur.fetchone()
                conn.close()
                if row is None or row[0] != file_hash:
                    if row is not None:
                        print(
                            f"[INFO] PDF modificado: {relative_path}, eliminando chunks antiguos"
                        )
                        delete_chunks(relative_path)
                    print(f"[INFO] Procesando PDF nuevo o modificado: {relative_path}")
                    text = extract_text_from_pdf(full_path)
                    text = normalize_pdf_text(text)
                    if text.strip() == "":
                        print(f"[WARN] {full_path} está vacío")
                        continue
                    chunks = chunk_text(text)
                    insert_chunks(f, relative_path, folder_name, chunks, file_hash)
                    print(f"[INFO] {len(chunks)} chunks insertados para {relative_path}")
                else:
                    # PDF no modificado, pero verificar si faltan embeddings
                    missing_count = update_missing_embeddings(relative_path)
                    if missing_count > 0:
                        print(f"[INFO] {missing_count} embeddings actualizados para {relative_path}")

    # Borrar de DB PDFs que ya no existen
    conn = connect_db()
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT file_path FROM chunks")
        all_files_in_db = [r[0] for r in cur.fetchall()]
    conn.close()
    for file_path in all_files_in_db:
        if file_path not in current_files:
            print(f"[INFO] PDF eliminado del disco: {file_path}, borrando chunks")
            delete_chunks(file_path)

# ===========================
# Main
# ===========================
if __name__ == "__main__":
    create_table()
    process_pdfs()
    print("[INFO] Indexación completada con verificación de cambios.")
