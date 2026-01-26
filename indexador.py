#!/usr/bin/env python3
import os
import hashlib
import psycopg2
from psycopg2.extras import execute_values
from pypdf import PdfReader
from dotenv import load_dotenv

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
IGNORE_DIRS = os.getenv("IGNORE_DIRS", "NormativasAPP").split(",")

# ===========================
# Parámetros de chunking
# ===========================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

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

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def connect_db():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )

def create_table():
    """Tabla de chunks + hash para versionado"""
    sql = """
    CREATE TABLE IF NOT EXISTS chunks (
        id SERIAL PRIMARY KEY,
        file_name TEXT,
        folder_name TEXT,
        chunk_index INT,
        text TEXT,
        embedding vector(1536),
        file_hash TEXT
    );
    """
    conn = connect_db()
    with conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    conn.close()

def insert_chunks(file_name, folder_name, chunks, file_hash):
    conn = connect_db()
    data = [(file_name, folder_name, i, chunk, None, file_hash) for i, chunk in enumerate(chunks)]
    sql = "INSERT INTO chunks (file_name, folder_name, chunk_index, text, embedding, file_hash) VALUES %s"
    with conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, data)
    conn.close()

def delete_chunks(file_name):
    """Borra todos los chunks de un PDF"""
    conn = connect_db()
    sql = "DELETE FROM chunks WHERE file_name = %s"
    with conn:
        with conn.cursor() as cur:
            cur.execute(sql, (file_name,))
    conn.close()

def process_pdfs():
    """Procesa todos los PDFs, detectando cambios"""
    # Lista de PDFs actuales
    current_files = {}
    for root, dirs, files in os.walk(BASE_DIR):
        for ignore in IGNORE_DIRS:
            if ignore in dirs:
                dirs.remove(ignore)
        folder_name = os.path.basename(root)
        for f in files:
            if f.lower().endswith(".pdf"):
                full_path = os.path.join(root, f)
                file_hash = hash_file(full_path)
                current_files[f] = file_hash
                # Revisar si está en DB con mismo hash
                conn = connect_db()
                with conn.cursor() as cur:
                    cur.execute("SELECT file_hash FROM chunks WHERE file_name = %s LIMIT 1", (f,))
                    row = cur.fetchone()
                conn.close()
                if row is None or row[0] != file_hash:
                    if row is not None:
                        print(f"[INFO] PDF modificado: {f}, eliminando chunks antiguos")
                        delete_chunks(f)
                    print(f"[INFO] Procesando PDF nuevo o modificado: {full_path}")
                    text = extract_text_from_pdf(full_path)
                    if text.strip() == "":
                        print(f"[WARN] {full_path} está vacío")
                        continue
                    chunks = chunk_text(text)
                    insert_chunks(f, folder_name, chunks, file_hash)
                    print(f"[INFO] {len(chunks)} chunks insertados para {f}")

    # Borrar de DB PDFs que ya no existen
    conn = connect_db()
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT file_name FROM chunks")
        all_files_in_db = [r[0] for r in cur.fetchall()]
    conn.close()
    for f in all_files_in_db:
        if f not in current_files:
            print(f"[INFO] PDF eliminado del disco: {f}, borrando chunks")
            delete_chunks(f)

# ===========================
# Main
# ===========================
if __name__ == "__main__":
    create_table()
    process_pdfs()
    print("[INFO] Indexación completada con verificación de cambios.")
