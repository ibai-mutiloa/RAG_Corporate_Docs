#!/usr/bin/env python3
import os
import hashlib
import re
import importlib
import psycopg2
from psycopg2.extras import execute_values
from pypdf import PdfReader
import pdfplumber
from dotenv import load_dotenv
from openai import AzureOpenAI
import tiktoken

try:
    camelot = importlib.import_module("camelot")
except Exception:
    camelot = None

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
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
FORCE_REINDEX = os.getenv("FORCE_REINDEX", "False").lower() == "true"
MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "2000"))  # Máximo de tokens por chunk
TOC_DOTTED_ALWAYS_SKIP = os.getenv("TOC_DOTTED_ALWAYS_SKIP", "True").lower() == "true"
TABLE_CRITICAL_KEYWORDS = [
    kw.strip().lower()
    for kw in os.getenv(
        "TABLE_CRITICAL_KEYWORDS",
        "irpf,retencion,retención,km,kilometraje,calendario,laborable,festivo"
    ).split(",")
    if kw.strip()
]

# Inicializar tokenizer de OpenAI
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    print(f"[WARN] No se pudo inicializar tiktoken: {e}, se usará estimación por caracteres")
    tokenizer = None

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
    # Preservar saltos antes de encabezados legales para mejorar el chunking estructural
    text = re.sub(
        r"\n(?=\s*(?:ART[IÍ]CULO\s+\d+[A-Za-zºª\-]*|DISPOSICION(?:ES)?\s+\w+|T[ÍI]TULO\s+\w+|CAP[IÍ]TULO\s+\w+|SECCI[ÓO]N\s+\w+|ANEXO\s+\w+))",
        "§§HEADER_BREAK§§",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.replace("§§HEADER_BREAK§§", "\n\n")
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

def extract_section_label(text):
    """Extrae una etiqueta de sección para enriquecer metadatos."""
    if not text:
        return "general"

    patterns = [
        r"(?im)^\s*(\d{1,2}\.\s+[A-ZÁÉÍÓÚÜÑ][^\n]{0,120})",
        r"(?im)^\s*(ART[IÍ]CULO\s+\d+[A-Za-zºª\-]*)",
        r"(?im)^\s*(DISPOSICION(?:ES)?\s+\w+)",
        r"(?im)^\s*(T[ÍI]TULO\s+\w+)",
        r"(?im)^\s*(CAP[IÍ]TULO\s+\w+)",
        r"(?im)^\s*(SECCI[ÓO]N\s+\w+)",
        r"(?im)^\s*(ANEXO\s+\w+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return re.sub(r"\s+", " ", match.group(1).strip())
    return "general"

def normalize_cell(value):
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text

def clean_table_rows(raw_rows):
    if not raw_rows:
        return []
    cleaned = []
    for row in raw_rows:
        if not row:
            continue
        normalized_row = [normalize_cell(cell) for cell in row]
        if any(cell for cell in normalized_row):
            cleaned.append(normalized_row)
    return cleaned

def table_contains_critical_keywords(rows):
    if not rows:
        return False
    flat_text = " ".join(" ".join(r) for r in rows).lower()
    return any(keyword in flat_text for keyword in TABLE_CRITICAL_KEYWORDS)

def table_to_row_chunks(file_name, table_id, page_number, section, rows):
    """Genera un chunk por fila de tabla con metadatos."""
    if not rows:
        return []

    headers = rows[0]
    body_rows = rows[1:] if len(rows) > 1 else rows
    row_chunks = []

    for row_index, row in enumerate(body_rows):
        max_cols = max(len(headers), len(row))
        pairs = []
        for col_idx in range(max_cols):
            header = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx + 1}"
            value = row[col_idx] if col_idx < len(row) else ""
            header = normalize_cell(header) or f"col_{col_idx + 1}"
            value = normalize_cell(value)
            if value:
                pairs.append(f"{header}: {value}")

        if not pairs:
            continue

        row_text = " | ".join(pairs)
        chunk_text = (
            f"Tabla crítica detectada\n"
            f"Documento: {file_name}\n"
            f"Sección: {section}\n"
            f"Página: {page_number}\n"
            f"Tabla: {table_id}\n"
            f"Fila: {row_index}\n"
            f"Datos: {row_text}"
        )

        row_chunks.append({
            'text': chunk_text,
            'is_table': True,
            'table_id': table_id,
            'row_index': row_index,
            'page_number': page_number,
            'section': section,
        })

    return row_chunks

def extract_tables_with_pdfplumber(page):
    try:
        return page.extract_tables() or []
    except Exception:
        return []

def extract_tables_with_camelot(pdf_path, page_number):
    if camelot is None:
        return []
    try:
        parsed = camelot.read_pdf(pdf_path, pages=str(page_number), flavor='stream')
        rows = []
        for table in parsed:
            rows.append(table.df.values.tolist())
        return rows
    except Exception:
        return []

def extract_critical_table_row_chunks(pdf_path, file_name):
    """Extrae filas de tablas críticas con pdfplumber y fallback a camelot."""
    all_table_chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                page_text = normalize_pdf_text(page.extract_text() or "")
                section = extract_section_label(page_text)

                raw_tables = extract_tables_with_pdfplumber(page)
                source = "pdfplumber"
                if not raw_tables:
                    raw_tables = extract_tables_with_camelot(pdf_path, page_idx)
                    source = "camelot"

                for table_idx, raw_table in enumerate(raw_tables, start=1):
                    rows = clean_table_rows(raw_table)
                    if not rows or not table_contains_critical_keywords(rows):
                        continue

                    table_id = f"p{page_idx}_t{table_idx}_{source}"
                    all_table_chunks.extend(
                        table_to_row_chunks(
                            file_name=file_name,
                            table_id=table_id,
                            page_number=page_idx,
                            section=section,
                            rows=rows,
                        )
                    )
    except Exception as e:
        print(f"[WARN] Error extrayendo tablas en {pdf_path}: {e}")

    return all_table_chunks

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
        r"ANEXO\s+\w+|"
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
        if len(s) > max_len:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            step = max(max_len - CHUNK_OVERLAP, 1)
            while start < len(s):
                end = start + max_len
                chunks.append(s[start:end].strip())
                start += step
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
    header = f"Documento: {file_name}"
    return f"{header}\nTexto:\n{chunk_text_value}".strip()

def connect_db():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )

def count_tokens(text):
    """Cuenta el número de tokens en un texto usando tiktoken"""
    if not text:
        return 0
    
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            print(f"[WARN] Error contando tokens: {e}")
            # Fallback: estimación aproximada (1 token ≈ 4 caracteres)
            return len(text) // 4
    else:
        # Estimación aproximada si no hay tokenizer
        return len(text) // 4

def split_large_chunk(text, max_tokens=MAX_TOKENS_PER_CHUNK):
    """Divide un chunk grande en sub-chunks más pequeños"""
    if not text:
        return []
    
    # Intentar dividir por párrafos primero
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sub_chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        test_chunk = f"{current_chunk}\n\n{paragraph}".strip() if current_chunk else paragraph
        
        if count_tokens(test_chunk) <= max_tokens:
            current_chunk = test_chunk
        else:
            # Si el chunk actual no está vacío, guardarlo
            if current_chunk:
                sub_chunks.append(current_chunk)
                current_chunk = ""
            
            # Si el párrafo solo es demasiado grande, dividirlo por frases
            if count_tokens(paragraph) > max_tokens:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                temp_chunk = ""
                
                for sentence in sentences:
                    test_sentence = f"{temp_chunk} {sentence}".strip() if temp_chunk else sentence
                    
                    if count_tokens(test_sentence) <= max_tokens:
                        temp_chunk = test_sentence
                    else:
                        if temp_chunk:
                            sub_chunks.append(temp_chunk)
                        
                        # Si una sola frase es muy grande, dividir por caracteres
                        if count_tokens(sentence) > max_tokens:
                            # Dividir en trozos de aproximadamente max_tokens/4 caracteres
                            chunk_size = max_tokens * 3  # ~750 caracteres por cada 250 tokens
                            for i in range(0, len(sentence), chunk_size):
                                sub_chunks.append(sentence[i:i + chunk_size])
                            temp_chunk = ""
                        else:
                            temp_chunk = sentence
                
                if temp_chunk:
                    sub_chunks.append(temp_chunk)
            else:
                current_chunk = paragraph
    
    if current_chunk:
        sub_chunks.append(current_chunk)
    
    return sub_chunks if sub_chunks else [text]

def is_page_index_chunk(text):
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) <= 6 and stripped.isdigit():
        return True
    if re.fullmatch(r"\d+\s*/\s*\d+", stripped):
        return True
    if re.fullmatch(r"(?:p(?:a|á)g(?:ina)?\.?\s*)?\d+(?:\s*(?:de|/)\s*\d+)?", stripped, re.IGNORECASE):
        return True
    if re.fullmatch(r"[ivxlcdm]+", stripped, re.IGNORECASE):
        return True
    return False

def is_title_like(text):
    stripped = re.sub(r"\s+", " ", text.strip())
    if not stripped:
        return True
    words = stripped.split()
    if len(words) > 12:
        return False
    if re.search(r"[.!?;:]", stripped):
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio >= 0.6

def is_non_relevant_title_chunk(text):
    stripped = re.sub(r"\s+", " ", text.strip())
    if not stripped:
        return True
    if not is_title_like(stripped):
        return False

    keyword_pattern = r"\b(INDICE|[ÍI]NDICE|TABLA DE CONTENIDOS?|CONTENIDOS?|SUMARIO|LISTA DE FIGURAS|LISTA DE TABLAS)\b"
    if re.search(keyword_pattern, stripped, re.IGNORECASE):
        return True

    heading_pattern = r"^(T[ÍI]TULO|CAP[ÍI]TULO|SECCI[ÓO]N|ANEXO|AP[ÉE]NDICE)\s+[\wIVXLCDM]+\.?$"
    return re.fullmatch(heading_pattern, stripped, re.IGNORECASE) is not None

def is_toc_like_chunk(text):
    stripped = text.strip()
    if not stripped:
        return True

    # Patrones típicos de tabla de contenidos/índice
    has_toc_word = re.search(r"\b([ÍI]NDICE|SUMARIO|TABLA DE CONTENIDOS?|CONTENIDOS?)\b", stripped, re.IGNORECASE) is not None
    dotted_leaders = len(re.findall(r"\.{2,}\s*\d+\b", stripped))
    dotted_leaders_generic = len(re.findall(r"\.{4,}", stripped))
    heading_refs = len(re.findall(r"\b(art[ií]culo|art\.?|cap[ií]tulo|secci[óo]n|t[íi]tulo|anexo)\b", stripped, re.IGNORECASE))
    page_numbers = len(re.findall(r"\b\d{1,4}\b", stripped))

    # Modo estricto: si detectamos líderes de puntos tipo "....", tratarlo siempre como índice
    if TOC_DOTTED_ALWAYS_SKIP and (dotted_leaders >= 1 or dotted_leaders_generic >= 1):
        return True

    if has_toc_word and (dotted_leaders >= 1 or heading_refs >= 3):
        return True
    if dotted_leaders >= 2 and page_numbers >= 4:
        return True
    if heading_refs >= 6 and page_numbers >= 6 and len(stripped) < 1800:
        return True

    return False

def is_noise_chunk(text):
    if is_page_index_chunk(text):
        return True
    if is_non_relevant_title_chunk(text):
        return True
    if is_toc_like_chunk(text):
        return True
    return False

def create_table():
    """Tabla de chunks + hash para versionado y esquemas nuevos."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS chunks (
        id SERIAL PRIMARY KEY,
        file_name TEXT,
        file_path TEXT,
        folder_name TEXT,
        chunk_index INT,
        is_table BOOLEAN DEFAULT FALSE,
        table_id TEXT,
        row_index INT,
        page_number INT,
        section TEXT,
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

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'chunks' AND column_name = 'is_table'
        ) THEN
            ALTER TABLE chunks ADD COLUMN is_table BOOLEAN DEFAULT FALSE;
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'chunks' AND column_name = 'table_id'
        ) THEN
            ALTER TABLE chunks ADD COLUMN table_id TEXT;
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'chunks' AND column_name = 'row_index'
        ) THEN
            ALTER TABLE chunks ADD COLUMN row_index INT;
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'chunks' AND column_name = 'page_number'
        ) THEN
            ALTER TABLE chunks ADD COLUMN page_number INT;
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'chunks' AND column_name = 'section'
        ) THEN
            ALTER TABLE chunks ADD COLUMN section TEXT;
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
        all_embeddings = [None] * len(chunks)  # Inicializar con None para todos los índices
        valid_chunks_count = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_end = i + len(batch)
            
            # Validar tamaño de tokens para cada chunk en el batch
            filtered_batch = []
            filtered_indices = []
            
            for idx, chunk in enumerate(batch):
                actual_idx = i + idx  # Índice real en el array chunks
                char_count = len(chunk)
                token_count = count_tokens(chunk)
                
                if token_count > MAX_TOKENS_PER_CHUNK:
                    print(f"[ERROR] Chunk {actual_idx} demasiado grande:")
                    print(f"        • {char_count} caracteres")
                    print(f"        • {token_count} tokens (límite: {MAX_TOKENS_PER_CHUNK})")
                    print(f"        • Ratio: {token_count/char_count:.3f} tokens/char")
                    
                    # Intentar dividir el chunk
                    print(f"[INFO] Intentando dividir chunk {actual_idx}...")
                    sub_chunks = split_large_chunk(chunk, MAX_TOKENS_PER_CHUNK)
                    
                    if len(sub_chunks) > 1:
                        print(f"[INFO] Chunk dividido en {len(sub_chunks)} sub-chunks")
                        # Procesar cada sub-chunk y tomar el primero válido
                        best_embedding = None
                        best_token_count = 0
                        
                        for sub_idx, sub_chunk in enumerate(sub_chunks):
                            sub_token_count = count_tokens(sub_chunk)
                            if sub_token_count <= MAX_TOKENS_PER_CHUNK:
                                print(f"[DEBUG] Sub-chunk {actual_idx}.{sub_idx} → {sub_token_count} tokens")
                                # Tomar el sub-chunk más largo como representante
                                if sub_token_count > best_token_count:
                                    best_embedding = sub_chunk
                                    best_token_count = sub_token_count
                            else:
                                print(f"[WARN] Sub-chunk {actual_idx}.{sub_idx} aún muy grande ({sub_token_count} tokens) → omitido")
                        
                        if best_embedding:
                            # Usar el mejor sub-chunk encontrado
                            filtered_batch.append(best_embedding)
                            filtered_indices.append(actual_idx)
                        else:
                            print(f"[ERROR] Ningún sub-chunk válido para chunk {actual_idx} → SIN EMBEDDING")
                            all_embeddings[actual_idx] = None
                    else:
                        print(f"[ERROR] No se pudo dividir el chunk {actual_idx} → SIN EMBEDDING")
                        all_embeddings[actual_idx] = None
                else:
                    print(f"[DEBUG] Embedding chunk {actual_idx} → {token_count} tokens ({char_count} chars)")
                    filtered_batch.append(chunk)
                    filtered_indices.append(actual_idx)
            
            # Si hay chunks válidos en el batch, procesar
            if filtered_batch:
                response = azure_client.embeddings.create(
                    input=filtered_batch,
                    model=AZURE_DEPLOYMENT_NAME
                )
                batch_embeddings = [item.embedding for item in response.data]
                
                # Asignar embeddings a los índices correctos
                for local_idx, embedding in zip(filtered_indices, batch_embeddings):
                    all_embeddings[local_idx] = embedding
                    valid_chunks_count += 1
                
                print(f"[INFO] Embeddings procesados: {batch_end}/{len(chunks)} (válidos: {valid_chunks_count})")
        
        return all_embeddings
    except Exception as e:
        print(f"[ERROR] Error calculando embeddings: {e}")
        return [None] * len(chunks)

def insert_chunks(file_name, file_path, folder_name, chunks, file_hash):
    """Inserta chunks con sus embeddings en la base de datos."""
    # Filtrar chunks poco informativos (indices de pagina, titulos sin contenido)
    filtered_chunks = []
    skipped_noise = 0
    for c in chunks:
        if is_noise_chunk(c):
            skipped_noise += 1
        else:
            filtered_chunks.append(c)

    if not filtered_chunks:
        print(f"[WARN] Todos los chunks filtrados por ruido para {file_path}")
        return

    # Añadir metadatos al texto del chunk
    chunks = [enrich_chunk_with_metadata(file_name, c) for c in filtered_chunks]
    # Calcular embeddings para todos los chunks
    embeddings = calculate_embeddings(chunks)
    
    # Filtrar: solo insertar chunks con embeddings válidos (no None)
    valid_data = []
    skipped_count = 0
    
    for i, chunk in enumerate(chunks):
        if embeddings[i] is None:
            print(f"[WARN] Chunk {i} sin embedding → NO se insertará")
            skipped_count += 1
        else:
            valid_data.append((
                file_name,
                file_path,
                folder_name,
                i,
                False,
                None,
                None,
                None,
                None,
                chunk,
                embeddings[i],
                file_hash,
            ))
    
    if not valid_data:
        print(f"[ERROR] Ningún chunk válido con embedding para {file_path}")
        return
    
    conn = connect_db()
    sql = """
    INSERT INTO chunks (
        file_name,
        file_path,
        folder_name,
        chunk_index,
        is_table,
        table_id,
        row_index,
        page_number,
        section,
        text,
        embedding,
        file_hash
    ) VALUES %s
    """
    with conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, valid_data)
    conn.close()
    
    print(f"[INFO] {len(valid_data)} chunks insertados, {skipped_count} omitidos (sin embedding), {skipped_noise} filtrados por ruido")

def delete_table_chunks(file_path):
    """Borra solo chunks de tablas para un PDF, preservando chunks narrativos."""
    conn = connect_db()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE file_path = %s AND COALESCE(is_table, FALSE) = TRUE",
                (file_path,),
            )
    conn.close()

def insert_table_chunks(file_name, file_path, folder_name, table_chunks, file_hash):
    """Inserta chunks tabulares (un chunk por fila) con metadatos."""
    if not table_chunks:
        return 0

    table_texts = [chunk['text'] for chunk in table_chunks]
    embeddings = calculate_embeddings(table_texts)

    if not embeddings:
        return 0

    valid_data = []
    skipped_count = 0
    base_chunk_index = 1_000_000

    for idx, item in enumerate(table_chunks):
        embedding = embeddings[idx]
        if embedding is None:
            skipped_count += 1
            continue

        page_number = item.get('page_number')
        row_index = item.get('row_index')
        derived_index = base_chunk_index + idx

        valid_data.append((
            file_name,
            file_path,
            folder_name,
            derived_index,
            True,
            item.get('table_id'),
            row_index,
            page_number,
            item.get('section', 'general'),
            enrich_chunk_with_metadata(file_name, item['text']),
            embedding,
            file_hash,
        ))

    if not valid_data:
        print(f"[WARN] No hay chunks tabulares válidos para {file_path}")
        return 0

    conn = connect_db()
    sql = """
    INSERT INTO chunks (
        file_name,
        file_path,
        folder_name,
        chunk_index,
        is_table,
        table_id,
        row_index,
        page_number,
        section,
        text,
        embedding,
        file_hash
    ) VALUES %s
    ON CONFLICT (file_path, chunk_index) DO UPDATE SET
        is_table = EXCLUDED.is_table,
        table_id = EXCLUDED.table_id,
        row_index = EXCLUDED.row_index,
        page_number = EXCLUDED.page_number,
        section = EXCLUDED.section,
        text = EXCLUDED.text,
        embedding = EXCLUDED.embedding,
        file_hash = EXCLUDED.file_hash
    """
    with conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, valid_data)
    conn.close()

    print(f"[INFO] {len(valid_data)} chunks tabulares insertados, {skipped_count} omitidos (sin embedding)")
    return len(valid_data)

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
    
    # Actualizar embeddings en la base de datos (solo los que no son None)
    conn = connect_db()
    updated_count = 0
    with conn:
        with conn.cursor() as cur:
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                if embedding is not None:
                    cur.execute(
                        "UPDATE chunks SET embedding = %s WHERE id = %s",
                        (embedding, chunk_id)
                    )
                    updated_count += 1
                else:
                    print(f"[WARN] Chunk {chunk_id} sin embedding → NO se actualizará")
    conn.close()
    
    return updated_count

def update_all_missing_embeddings():
    """Calcula y actualiza TODOS los embeddings NULL en la base de datos."""
    conn = connect_db()
    with conn.cursor() as cur:
        # Obtener lista de archivos con chunks sin embedding
        cur.execute("""
            SELECT DISTINCT file_path 
            FROM chunks 
            WHERE embedding IS NULL
            ORDER BY file_path
        """)
        files_with_missing = [row[0] for row in cur.fetchall()]
    conn.close()
    
    if not files_with_missing:
        print("[INFO] ✅ No hay chunks sin embedding")
        return 0
    
    print(f"\n[INFO] 🔄 Actualizando embeddings para {len(files_with_missing)} archivos...")
    print("="*70)
    
    total_updated = 0
    for file_path in files_with_missing:
        updated = update_missing_embeddings(file_path)
        total_updated += updated
    
    print("="*70)
    print(f"[INFO] ✅ Total chunks actualizados: {total_updated}\n")
    
    return total_updated

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

def check_null_embeddings():
    """Valida post-indexación: verifica si hay chunks con embedding NULL"""
    conn = connect_db()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT file_name, COUNT(*) as count
            FROM chunks
            WHERE embedding IS NULL
            GROUP BY file_name
            ORDER BY count DESC
        """)
        rows = cur.fetchall()
    conn.close()
    
    if rows:
        print("\n" + "="*70)
        print("🚨 [ALARMA] Chunks sin embedding detectados:")
        print("="*70)
        for file_name, count in rows:
            print(f"  • {file_name}: {count} chunks sin embedding")
        print("="*70 + "\n")
        return False
    else:
        print("\n" + "="*70)
        print("✅ [VALIDACIÓN] Todos los chunks tienen embeddings")
        print("="*70 + "\n")
        return True

def diagnose_problematic_chunks():
    """Muestra información detallada sobre chunks sin embedding"""
    conn = connect_db()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, file_path, chunk_index, 
                   LENGTH(text) as char_count,
                   LEFT(text, 200) as preview
            FROM chunks
            WHERE embedding IS NULL
            ORDER BY LENGTH(text) DESC
            LIMIT 5
        """)
        rows = cur.fetchall()
    conn.close()
    
    if rows:
        print("\n" + "="*70)
        print("🔍 [DIAGNÓSTICO] Top 5 chunks problemáticos:")
        print("="*70)
        for chunk_id, file_path, chunk_idx, char_count, preview in rows:
            token_estimate = count_tokens(preview) * (char_count / len(preview)) if preview else 0
            print(f"\nChunk ID: {chunk_id}")
            print(f"  Archivo: {file_path}")
            print(f"  Índice: {chunk_idx}")
            print(f"  Tamaño: {char_count} caracteres (~{int(token_estimate)} tokens estimados)")
            print(f"  Preview: {preview[:100]}...")
        print("="*70 + "\n")

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

                # Reindexación selectiva de tablas críticas (sin tocar chunks narrativos)
                table_chunks = extract_critical_table_row_chunks(full_path, f)
                if table_chunks:
                    delete_table_chunks(relative_path)
                    inserted_table_chunks = insert_table_chunks(
                        file_name=f,
                        file_path=relative_path,
                        folder_name=folder_name,
                        table_chunks=table_chunks,
                        file_hash=file_hash,
                    )
                    print(
                        f"[INFO] Reindexación selectiva de tablas críticas en {relative_path}: "
                        f"{inserted_table_chunks} filas indexadas"
                    )

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
    
    # Validación post-indexación
    print("\n[INFO] Ejecutando validación post-indexación...")
    all_valid = check_null_embeddings()
    
    # Si hay chunks sin embedding, mostrar diagnóstico e intentar reparar
    if not all_valid:
        diagnose_problematic_chunks()
        print("[INFO] 🔧 Intentando corregir embeddings faltantes...")
        update_all_missing_embeddings()
        # Re-validar después de actualizar
        print("[INFO] Re-validando después de actualización...")
        still_invalid = not check_null_embeddings()
        
        if still_invalid:
            print("[WARN] ⚠️  Algunos chunks siguen sin embedding después del intento de reparación")
            diagnose_problematic_chunks()
    
    # Mostrar estadísticas finales
    conn = connect_db()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
        chunks_with_embedding = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT file_path) FROM chunks")
        total_files = cur.fetchone()[0]
    conn.close()
    
    print(f"\n📊 Estadísticas finales:")
    print(f"  • Total chunks: {total_chunks}")
    print(f"  • Chunks con embedding: {chunks_with_embedding}")
    print(f"  • Archivos indexados: {total_files}")
    if total_chunks > 0:
        percentage = (chunks_with_embedding / total_chunks) * 100
        print(f"  • Cobertura de embeddings: {percentage:.1f}%")
