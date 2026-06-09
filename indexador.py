#!/usr/bin/env python3
"""
Indexador simplificado — solo carpeta es/, un único FAQ en castellano.
Usa markitdown como extractor primario de PDFs (fallback: pdfplumber → pypdf).
El LLM se encarga de responder en el idioma de la pregunta.
"""
import os
import hashlib
import re
import importlib
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from openai import AzureOpenAI
import tiktoken

try:
    from markitdown import MarkItDown
    _markitdown = MarkItDown()
except Exception:
    _markitdown = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

from pypdf import PdfReader

load_dotenv(override=True)

# ===========================
# Configuración de DB
# ===========================
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "normativas_db")
DB_USER = os.getenv("DB_USER", "normativas_user")
DB_PASS = os.getenv("DB_PASS", "")

# ===========================
# Directorio base — SOLO es/
# ===========================
BASE_DIR = os.path.expanduser(os.getenv("BASE_DIR", "~/normativas"))
# La carpeta de documentos a indexar (siempre el subdirectorio es/)
DOCS_DIR = os.path.expanduser(os.getenv("DOCS_DIR", "") or BASE_DIR)
IGNORE_DIRS = [d.strip() for d in os.getenv("IGNORE_DIRS", "NormativasAPP").split(",") if d.strip()]

# ===========================
# FAQ — único fichero en castellano
# ===========================
# Soporta ruta absoluta, ruta con ~, o ruta relativa al repo
_faq_raw = os.getenv("FAQ_FILE", "es/faq_castellano.md")
_faq_expanded = os.path.expanduser(_faq_raw)
if os.path.isabs(_faq_expanded):
    # Ruta absoluta o con ~: usarla directamente
    FAQ_FILE_ABS = _faq_expanded
    FAQ_FILE = _faq_expanded  # se usa como clave en BD también
else:
    # Ruta relativa: resolver desde el directorio del script
    _repo_root = os.path.dirname(os.path.abspath(__file__))
    FAQ_FILE_ABS = os.path.join(_repo_root, _faq_expanded)
    FAQ_FILE = _faq_expanded

# ===========================
# Parámetros de chunking
# ===========================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
FORCE_REINDEX = os.getenv("FORCE_REINDEX", "False").lower() == "true"
MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "2000"))
TOC_DOTTED_ALWAYS_SKIP = os.getenv("TOC_DOTTED_ALWAYS_SKIP", "True").lower() == "true"
MAX_INDEXING_MODE = os.getenv("MAX_INDEXING_MODE", "False").lower() == "true"

# ===========================
# Azure OpenAI
# ===========================
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "text-embedding-3-small")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

azure_client = None
if AZURE_ENDPOINT and AZURE_API_KEY:
    azure_client = AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    )

try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    tokenizer = None


# ===========================
# Helpers
# ===========================

def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def count_tokens(text):
    if not text:
        return 0
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return len(text) // 4


# ===========================
# Extracción de texto
# ===========================

def _score_pdf_text(text):
    """
    Puntúa la calidad del texto extraído de un PDF.
    Penaliza texto con muchas líneas de índice/tabla (....., ---) y
    premia texto con frases completas.
    """
    if not text or not text.strip():
        return 0
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0
    total = len(lines)
    noise = sum(1 for l in lines if (
        l.count(".") > len(l) * 0.4 or
        l.startswith("| ---") or
        l == "---" or
        (len(l) < 8 and not l[0].isalpha())
    ))
    useful_chars = sum(len(l) for l in lines if len(l) > 30)
    noise_ratio = noise / total if total else 1
    return useful_chars * (1 - noise_ratio)


def extract_text_from_pdf(pdf_path):
    """
    Extrae texto de un PDF.
    Prueba markitdown y pdfplumber, devuelve el de mejor calidad.
    Fallback: pypdf.
    """
    markitdown_text = ""
    pdfplumber_text = ""

    # 1) markitdown
    if _markitdown:
        try:
            result = _markitdown.convert(pdf_path)
            markitdown_text = result.text_content or ""
        except Exception as e:
            print(f"[WARN] markitdown falló en {pdf_path}: {e}")

    # 2) pdfplumber
    if pdfplumber:
        try:
            page_texts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    parts = []
                    text = page.extract_text() or ""
                    if text.strip():
                        parts.append(text)
                    try:
                        tables = page.extract_tables()
                    except Exception:
                        tables = []
                    for table in tables or []:
                        if not table:
                            continue
                        header = table[0]
                        if not any(cell for cell in header if cell and str(cell).strip()):
                            cols = len(header)
                            header = [f"col{i+1}" for i in range(cols)]
                            body_rows = table
                        else:
                            body_rows = table[1:]
                        md = "| " + " | ".join([str(h).strip() if h is not None else "" for h in header]) + " |\n"
                        md += "| " + " | ".join(["---"] * len(header)) + " |\n"
                        for row in body_rows:
                            row_cells = [str(c).strip() if c is not None else "" for c in row]
                            if len(row_cells) < len(header):
                                row_cells += [""] * (len(header) - len(row_cells))
                            md += "| " + " | ".join(row_cells) + " |\n"
                        parts.append(md)
                    page_texts.append("\n\n".join(parts).strip())
            combined = "\n".join(page_texts)
            if not MAX_INDEXING_MODE:
                page_texts = strip_front_matter_pages(page_texts)
                combined = "\n".join(page_texts)
            pdfplumber_text = combined
        except Exception as e:
            print(f"[WARN] pdfplumber falló en {pdf_path}: {e}")

    # Elegir el mejor resultado
    score_md = _score_pdf_text(markitdown_text)
    score_pl = _score_pdf_text(pdfplumber_text)

    if score_md > 0 or score_pl > 0:
        if score_pl > score_md * 1.2:
            print(f"[PDF] pdfplumber mejor ({int(score_pl)} vs {int(score_md)}): {os.path.basename(pdf_path)}")
            return pdfplumber_text
        elif markitdown_text.strip():
            print(f"[PDF] markitdown mejor ({int(score_md)} vs {int(score_pl)}): {os.path.basename(pdf_path)}")
            return markitdown_text
        else:
            return pdfplumber_text

    # 3) pypdf fallback
    try:
        reader = PdfReader(pdf_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        if not MAX_INDEXING_MODE:
            pages = strip_front_matter_pages(pages)
        return "\n".join(pages)
    except Exception as e:
        print(f"[ERROR] No se pudo leer {pdf_path}: {e}")
        return ""


def extract_text_from_markdown(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ===========================
# Limpieza de texto PDF
# ===========================

def normalize_pdf_text(text):
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    short_fragment_stopwords = {
        "a", "al", "de", "del", "el", "en", "la", "las", "lo", "los",
        "o", "por", "que", "se", "su", "sus", "un", "una", "y",
    }
    merged_lines = []
    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]{2,}", " ", raw_line).strip()
        if not line:
            if merged_lines and merged_lines[-1] != "":
                merged_lines.append("")
            continue
        if merged_lines:
            previous_line = merged_lines[-1].rstrip()
            previous_word_match = re.search(r"(\S+)$", previous_line)
            previous_word = previous_word_match.group(1) if previous_word_match else ""
            previous_fragment = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", "", previous_word)
            if (
                1 <= len(previous_fragment) <= 2
                and previous_fragment.lower() not in short_fragment_stopwords
                and line[:1].islower()
            ):
                merged_lines[-1] = previous_line + line.lstrip()
                continue
        merged_lines.append(line)

    text = "\n".join(merged_lines)
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


# ===========================
# Detección portada/índice
# ===========================

def _page_has_body_marker(text):
    if not text:
        return False
    normalized = re.sub(r"\s+", " ", text).strip()
    if re.search(r"\b(?:ÍNDICE|INDICE|SUMARIO|TABLA DE CONTENIDOS|CONTENIDOS)\b", normalized, re.IGNORECASE):
        return False
    if re.search(r"\.{2,}\s*\d+\b", normalized):
        return False
    body_patterns = [
        r"\bART[IÍ]CULO\s+1\b",
        r"\bCAP[IÍ]TULO\s+I\b",
        r"\bT[ÍI]TULO\s+I\b",
        r"\bSECCI[ÓO]N\s+1\b",
    ]
    return any(re.search(p, normalized, re.IGNORECASE) for p in body_patterns)


def _looks_like_front_matter_page(text):
    if not text:
        return True
    stripped = re.sub(r"\s+", " ", text).strip()
    if not stripped:
        return True
    if len(stripped) <= 220 and not _page_has_body_marker(stripped):
        return True
    if is_toc_like_chunk(stripped):
        return True
    if len(stripped) <= 220 and is_title_like(stripped):
        return True
    return bool(re.search(r"\b(?:ÍNDICE|INDICE|SUMARIO|TABLA DE CONTENIDOS|CONTENIDOS)\b", stripped, re.IGNORECASE))


def strip_front_matter_pages(page_texts):
    if not page_texts:
        return page_texts
    body_start = None
    for idx, page_text in enumerate(page_texts):
        if _page_has_body_marker(page_text):
            body_start = idx
            break
    if body_start is None or body_start == 0:
        return page_texts
    leading = page_texts[:body_start]
    if leading and all(_looks_like_front_matter_page(p) for p in leading):
        print(f"[INFO] Recortando {body_start} páginas de portada/índice")
        return page_texts[body_start:]
    return page_texts


# ===========================
# Chunking
# ===========================

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
    return [p.strip() for p in re.split(pattern, text) if p.strip()]


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
                chunks.append(s[start:start + max_len].strip())
                start += step
            continue
        if not current:
            current = s
        elif len(current) + len(s) + 1 <= max_len:
            current = f"{current} {s}"
        else:
            chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    structured = split_by_structure(text)
    if structured:
        chunks = []
        for part in structured:
            if len(part) <= chunk_size:
                chunks.append(part)
            else:
                chunks.extend(split_by_sentences(part, max_len=chunk_size))
        return chunks

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
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
                    chunks.append(p[start:start + chunk_size])
                    start += chunk_size - overlap
                current = ""
            else:
                current = p
    if current:
        chunks.append(current)
    return chunks


def chunk_faq_markdown(text):
    """Un chunk por par pregunta+respuesta (## heading)."""
    chunks = []
    current_q = None
    current_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('###') and not stripped.startswith('####'):
            continue
        if stripped.startswith('##'):
            if current_q is not None:
                answer = ' '.join(current_lines).strip()
                chunk = f"{current_q}\n{answer}" if answer else current_q
                if chunk:
                    chunks.append(chunk)
            current_q = stripped
            current_lines = []
        else:
            if current_q is not None and stripped:
                current_lines.append(stripped)
    if current_q is not None:
        answer = ' '.join(current_lines).strip()
        chunk = f"{current_q}\n{answer}" if answer else current_q
        if chunk:
            chunks.append(chunk)
    return chunks


# ===========================
# Filtros de ruido
# ===========================

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
    heading_pattern = r"^(T[ÍI]TULO|CAP[IÍ]TULO|SECCI[ÓO]N|ANEXO|AP[ÉE]NDICE)\s+[\wIVXLCDM]+\.?$"
    return re.fullmatch(heading_pattern, stripped, re.IGNORECASE) is not None


def is_toc_like_chunk(text):
    stripped = text.strip()
    if not stripped:
        return True
    has_toc_word = re.search(r"\b(?:[ÍI]NDICE|SUMARIO|TABLA DE CONTENIDOS|CONTENIDOS)\b", stripped, re.IGNORECASE) is not None
    dotted_leaders = len(re.findall(r"\.{2,}\s*\d+\b", stripped))
    dotted_leaders_generic = len(re.findall(r"\.{4,}", stripped))
    heading_refs = len(re.findall(r"\b(art[ií]culo|art\.?|cap[ií]tulo|secci[óo]n|t[íi]tulo|anexo)\b", stripped, re.IGNORECASE))
    page_numbers = len(re.findall(r"\b\d{1,4}\b", stripped))
    if TOC_DOTTED_ALWAYS_SKIP and (dotted_leaders >= 2 or dotted_leaders_generic >= 2):
        if has_toc_word or heading_refs >= 3 or page_numbers >= 4 or len(stripped) < 1800:
            return True
    if TOC_DOTTED_ALWAYS_SKIP and has_toc_word and (dotted_leaders >= 1 or dotted_leaders_generic >= 1):
        return True
    if has_toc_word and (dotted_leaders >= 1 or heading_refs >= 3):
        return True
    if dotted_leaders >= 2 and page_numbers >= 4:
        return True
    if heading_refs >= 6 and page_numbers >= 6 and len(stripped) < 1800:
        return True
    return False


def is_noise_chunk(text):
    return is_page_index_chunk(text) or is_non_relevant_title_chunk(text) or is_toc_like_chunk(text)


# ===========================
# DB
# ===========================

def connect_db():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )


def create_table():
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
        search_tsv tsvector,
        is_front_matter BOOLEAN DEFAULT FALSE,
        file_hash TEXT
    );
    """
    migrate_sql = """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'chunks' AND column_name = 'file_path'
        ) THEN ALTER TABLE chunks ADD COLUMN file_path TEXT; END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'chunks' AND column_name = 'search_tsv'
        ) THEN ALTER TABLE chunks ADD COLUMN search_tsv tsvector; END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'chunks' AND column_name = 'is_front_matter'
        ) THEN ALTER TABLE chunks ADD COLUMN is_front_matter BOOLEAN DEFAULT FALSE; END IF;

        IF NOT EXISTS (
            SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = 'idx_chunks_file_path' AND n.nspname = 'public'
        ) THEN CREATE INDEX idx_chunks_file_path ON chunks(file_path); END IF;

        IF NOT EXISTS (
            SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = 'ux_chunks_file_path_chunk_index' AND n.nspname = 'public'
        ) THEN CREATE UNIQUE INDEX ux_chunks_file_path_chunk_index ON chunks(file_path, chunk_index); END IF;

        IF NOT EXISTS (
            SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = 'idx_chunks_search_tsv' AND n.nspname = 'public'
        ) THEN CREATE INDEX idx_chunks_search_tsv ON chunks USING GIN (search_tsv); END IF;

        IF NOT EXISTS (
            SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = 'idx_chunks_embedding_ivfflat' AND n.nspname = 'public'
        ) THEN
            CREATE INDEX idx_chunks_embedding_ivfflat
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        END IF;

        UPDATE chunks SET search_tsv = to_tsvector('simple', COALESCE(text, ''))
        WHERE search_tsv IS NULL;
    END
    $$;
    """
    conn = connect_db()
    with conn:
        with conn.cursor() as cur:
            cur.execute(create_sql)
            cur.execute(migrate_sql)
    conn.close()
    print("[INFO] Tabla 'chunks' lista.")


def delete_chunks(file_path):
    conn = connect_db()
    with conn:
        with conn.cursor() as cur:
            if file_path is None:
                cur.execute("DELETE FROM chunks WHERE file_path IS NULL")
            else:
                cur.execute("DELETE FROM chunks WHERE file_path = %s", (file_path,))
    conn.close()


def delete_all_chunks():
    conn = connect_db()
    with conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks")
    conn.close()


# ===========================
# Embeddings
# ===========================

def calculate_embeddings(texts, batch_size=20):
    if not azure_client:
        raise Exception("Azure OpenAI no configurado")
    results = [None] * len(texts)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        valid_indices = [j for j, t in enumerate(batch) if t and t.strip()]
        if not valid_indices:
            continue
        valid_texts = [batch[j] for j in valid_indices]
        try:
            response = azure_client.embeddings.create(input=valid_texts, model=AZURE_DEPLOYMENT_NAME)
            for k, j in enumerate(valid_indices):
                results[i + j] = response.data[k].embedding
        except Exception as e:
            print(f"[ERROR] Batch {i//batch_size + 1} falló: {e}")
    return results


def enrich_chunk_with_metadata(file_name, chunk_text_value):
    return f"Documento: {file_name}\nTexto:\n{chunk_text_value}".strip()


def insert_chunks(file_name, file_path, folder_name, chunks, file_hash):
    if MAX_INDEXING_MODE:
        filtered_chunks = list(chunks)
        skipped_noise = 0
    else:
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

    enriched_chunks = [enrich_chunk_with_metadata(file_name, c) for c in filtered_chunks]
    embeddings = calculate_embeddings(filtered_chunks)

    valid_data = []
    for i, clean_chunk in enumerate(filtered_chunks):
        if embeddings[i] is None:
            print(f"[WARN] Chunk {i} sin embedding → se omite")
            continue
        enriched = enriched_chunks[i]
        valid_data.append((file_name, file_path, folder_name, i, enriched, embeddings[i], clean_chunk, False, file_hash))

    if not valid_data:
        print(f"[ERROR] Ningún chunk válido para {file_path}")
        return

    conn = connect_db()
    sql = """
    INSERT INTO chunks (file_name, file_path, folder_name, chunk_index, text, embedding,
                        search_tsv, is_front_matter, file_hash)
    VALUES %s
    """
    with conn:
        with conn.cursor() as cur:
            execute_values(
                cur, sql, valid_data,
                template="(%s, %s, %s, %s, %s, %s, to_tsvector('simple', %s), %s, %s)"
            )
    conn.close()
    print(f"[INFO] {len(valid_data)} chunks insertados ({skipped_noise} ruido filtrado) → {file_path}")


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
    chunk_ids = [r[0] for r in rows]
    chunk_texts = [r[1] for r in rows]
    print(f"[INFO] Calculando {len(chunk_texts)} embeddings faltantes para {file_path}")
    embeddings = calculate_embeddings(chunk_texts)
    conn = connect_db()
    updated = 0
    with conn:
        with conn.cursor() as cur:
            for chunk_id, emb in zip(chunk_ids, embeddings):
                if emb is not None:
                    cur.execute("UPDATE chunks SET embedding = %s WHERE id = %s", (emb, chunk_id))
                    updated += 1
    conn.close()
    return updated


def update_all_missing_embeddings():
    conn = connect_db()
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT file_path FROM chunks WHERE embedding IS NULL ORDER BY file_path")
        files = [r[0] for r in cur.fetchall()]
    conn.close()
    if not files:
        print("[INFO] ✅ No hay chunks sin embedding")
        return 0
    total = 0
    for fp in files:
        total += update_missing_embeddings(fp)
    print(f"[INFO] Total embeddings actualizados: {total}")
    return total


def check_null_embeddings():
    conn = connect_db()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT file_name, COUNT(*) FROM chunks WHERE embedding IS NULL
            GROUP BY file_name ORDER BY COUNT(*) DESC
        """)
        rows = cur.fetchall()
    conn.close()
    if rows:
        print("🚨 Chunks sin embedding:")
        for file_name, count in rows:
            print(f"  • {file_name}: {count}")
        return False
    print("✅ Todos los chunks tienen embeddings")
    return True


def diagnose_problematic_chunks():
    conn = connect_db()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, file_path, chunk_index, LENGTH(text), LEFT(text, 200)
            FROM chunks WHERE embedding IS NULL
            ORDER BY LENGTH(text) DESC LIMIT 5
        """)
        rows = cur.fetchall()
    conn.close()
    for chunk_id, file_path, chunk_idx, char_count, preview in rows:
        print(f"  Chunk {chunk_id} | {file_path} | idx={chunk_idx} | {char_count} chars")
        print(f"    {preview[:80]}...")


# ===========================
# Proceso principal
# ===========================

def process_documents():
    """Indexa el FAQ y todos los PDFs de DOCS_DIR (es/)."""
    if FORCE_REINDEX:
        print("[INFO] FORCE_REINDEX: eliminando todos los chunks")
        delete_all_chunks()

    common_root = DOCS_DIR  # Solo indexamos es/, la raíz es esa misma carpeta

    # ── FAQ ──────────────────────────────────────────────────────────────────
    faq_abs = FAQ_FILE_ABS
    faq_relative = FAQ_FILE  # clave guardada en BD
    indexed_faq_paths = set()

    if os.path.exists(faq_abs):
        file_hash = hash_file(faq_abs)
        conn = connect_db()
        with conn.cursor() as cur:
            cur.execute("SELECT file_hash FROM chunks WHERE file_path = %s LIMIT 1", (faq_relative,))
            row = cur.fetchone()
        conn.close()

        if row is None or row[0] != file_hash:
            if row is not None:
                print(f"[INFO] FAQ modificado, eliminando chunks antiguos")
                delete_chunks(faq_relative)
            print(f"[INFO] Indexando FAQ: {faq_abs}")
            text = extract_text_from_markdown(faq_abs)
            if text.strip():
                chunks = chunk_faq_markdown(text)
                print(f"[INFO] FAQ: {len(chunks)} entradas Q&A")
                insert_chunks(os.path.basename(faq_abs), faq_relative, "faq", chunks, file_hash)
        else:
            missing = update_missing_embeddings(faq_relative)
            if missing:
                print(f"[INFO] {missing} embeddings del FAQ actualizados")
        indexed_faq_paths.add(faq_relative)
    else:
        print(f"[WARN] FAQ no encontrado: {faq_abs}")

    # ── PDFs en es/ ──────────────────────────────────────────────────────────
    if not os.path.isdir(DOCS_DIR):
        print(f"[ERROR] Directorio de documentos no encontrado: {DOCS_DIR}")
        return

    current_files = {rp: None for rp in indexed_faq_paths}

    for root, dirs, files in os.walk(DOCS_DIR):
        for ignore in IGNORE_DIRS:
            if ignore in dirs:
                dirs.remove(ignore)
        folder_name = os.path.basename(root)
        for f in files:
            if not f.lower().endswith(".pdf"):
                continue
            full_path = os.path.join(root, f)
            relative_path = os.path.relpath(full_path, common_root)
            file_hash = hash_file(full_path)
            current_files[relative_path] = file_hash

            conn = connect_db()
            with conn.cursor() as cur:
                cur.execute("SELECT file_hash FROM chunks WHERE file_path = %s LIMIT 1", (relative_path,))
                row = cur.fetchone()
            conn.close()

            if row is None or row[0] != file_hash:
                if row is not None:
                    print(f"[INFO] PDF modificado: {relative_path}, eliminando chunks antiguos")
                    delete_chunks(relative_path)
                print(f"[INFO] Procesando: {relative_path}")
                text = extract_text_from_pdf(full_path)
                text = normalize_pdf_text(text)
                if not text.strip():
                    print(f"[WARN] {full_path} vacío tras extracción")
                    continue
                chunks = chunk_text(text)
                insert_chunks(f, relative_path, folder_name, chunks, file_hash)
            else:
                missing = update_missing_embeddings(relative_path)
                if missing:
                    print(f"[INFO] {missing} embeddings actualizados para {relative_path}")

    # ── Limpiar documentos eliminados ────────────────────────────────────────
    conn = connect_db()
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT file_path FROM chunks")
        all_in_db = [r[0] for r in cur.fetchall()]
    conn.close()
    for fp in all_in_db:
        if fp not in current_files:
            print(f"[INFO] Documento eliminado del disco: {fp}, borrando chunks")
            delete_chunks(fp)


# ===========================
# Main
# ===========================
if __name__ == "__main__":
    if not _markitdown:
        print("[WARN] markitdown no disponible. Instala con: pip install markitdown")
    create_table()
    process_documents()
    print("[INFO] Indexación completada.")

    print("\n[INFO] Validación post-indexación...")
    all_valid = check_null_embeddings()
    if not all_valid:
        diagnose_problematic_chunks()
        print("[INFO] Intentando reparar embeddings faltantes...")
        update_all_missing_embeddings()
        check_null_embeddings()

    conn = connect_db()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
        chunks_with_emb = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT file_path) FROM chunks")
        total_files = cur.fetchone()[0]
    conn.close()

    print(f"\n📊 Estadísticas finales:")
    print(f"  • Total chunks:          {total_chunks}")
    print(f"  • Chunks con embedding:  {chunks_with_emb}")
    print(f"  • Archivos indexados:    {total_files}")
    if total_chunks > 0:
        print(f"  • Cobertura embeddings:  {chunks_with_emb/total_chunks*100:.1f}%")