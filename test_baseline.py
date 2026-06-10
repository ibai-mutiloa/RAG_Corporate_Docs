#!/usr/bin/env python3
"""
Script de baseline para el RAG de normativas MGEP.
Lanza preguntas conocidas contra la API y guarda resultados en JSON con timestamp.

Uso:
    python test_baseline.py [--url http://localhost:5000] [--output dir/] [--no-answer]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import requests

# ===========================
# Preguntas de prueba
# ===========================

QUESTIONS = [
    # --- Validadas (respuesta correcta confirmada) ---
    {
        "id": 1,
        "question": "quién conforma el comité de cumplimiento",
        "expected_file": "Manual Prevención Riesgos Penales.pdf",
        "expected_chunk_index": 32,
        "partial_expected": False,
        "notes": "chunk_index=32 específico — caso crítico para neighbor expansion",
    },
    {
        "id": 2,
        "question": "qué hago si conozco un delito cometido en Eskola",
        "expected_file": "Manual Prevención Riesgos Penales.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "keyword 'Eskola' puede desviar retrieval",
    },
    {
        "id": 3,
        "question": "cómo puedo hacer una denuncia siendo un trabajador",
        "expected_file": "Política de conflicto acoso laboral y acoso sexual.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "Canal de denuncia está en chunk 238 de Política acoso (web MGEP, Compliance Officer). Antes se esperaba Reglamento — corregido tras validación manual.",
    },
    {
        "id": 4,
        "question": "que canal puedo utilizar para hacer una denuncia",
        "expected_file": "Manual Prevención Riesgos Penales.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "keyword 'denuncia' + 'canal' — riesgo de expansión ruidosa",
    },
    {
        "id": 5,
        "question": "que gastos de viaje se pagan en MGEP",
        "expected_file": "3 Compensación gastos desplazamiento y manutención.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 6,
        "question": "cuando se calcula el IRPF",
        "expected_file": "1 Anticipos y retribución.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 7,
        "question": "que es el nivel de anticipos",
        "expected_file": "1 Anticipos y retribución.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 8,
        "question": "vamos a cobrar retribucion variable",
        "expected_file": "1 Anticipos y retribución.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 9,
        "question": "le he dado un golpe al coche, que tengo que hacer",
        "expected_file": "5 Compensación de daños en vehículos.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 10,
        "question": "tengo que pagar la franquicia del vehiculo",
        "expected_file": "5 Compensación de daños en vehículos.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 11,
        "question": "a quien tengo que avisar si he tenido un golpe con mi coche",
        "expected_file": "5 Compensación de daños en vehículos.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 12,
        "question": "que tengo que hacer si la huelga es vinculante",
        "expected_file": "Huelga trabajadores.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 13,
        "question": "cuando se vota cuando hay convocatoria de huelga",
        "expected_file": "Huelga trabajadores.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 14,
        "question": "puedo ir a trabajar un dia de huelga",
        "expected_file": "Huelga trabajadores.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 15,
        "question": "cuales son los pasos si fallece un empleado de Eskola",
        "expected_file": "Protocolo en caso de fallecimiento de trabajador o familiar de trabajador.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "keyword 'Eskola' puede desviar retrieval",
    },
    {
        "id": 16,
        "question": "cuantas horas tengo que trabajar durante el curso 2024-2025",
        "expected_file": "Normativa laboral.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "Normativa laboral.pdf contiene la jornada actualizada (1.722h curso 2025-2026). Antes se esperaba 2 Horario de trabajo.pdf — corregido tras validación manual.",
    },
    {
        "id": 17,
        "question": "cual es la jornada diaria de trabajo",
        "expected_file": "Restricciones de horarios.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 18,
        "question": "cual es el horario en julio",
        "expected_file": "2 Horario de trabajo.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 19,
        "question": "que requisito debo cumplir para solicitar un año sabatico",
        "expected_file": "Año sabático.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 20,
        "question": "estoy de año sabatico y quiero renunciar, como debo proceder",
        "expected_file": "Año sabático.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 21,
        "question": "que condiciones debo cumplir para acogerme a la ayuda por cese de la actividad",
        "expected_file": "7 Ayudas por cese de la actividad laboral.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 22,
        "question": "cuando debo comunicar que quiero acogerme a la ayuda por cese de la actividad",
        "expected_file": "7 Ayudas por cese de la actividad laboral.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 23,
        "question": "soy una persona socia que he recibido formación con cargo a la cooperativa, cuanto tiempo de permanencia me pueden exigir",
        "expected_file": "Reglamento de régimen interno.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 24,
        "question": "cuales son las cuatro clases de personas socias en la Cooperativa",
        "expected_file": "Reglamento de régimen interno.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 25,
        "question": "en caso de empate en votación para el Consejo Rector, qué criterios de desempate hay",
        "expected_file": "Estatutos.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 26,
        "question": "que diferencia hay entre el procedimiento sancionador de falta leve y falta grave",
        "expected_file": "Reglamento de régimen interno.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 27,
        "question": "puedo solicitar un cambio de campus",
        "expected_file": "4 Compensación por cambio de campus base.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "faq_acceptable": True,
        "notes": "FAQ entry válida — '¿Cómo funciona el cambio de campus base para trabajadores?'",
    },
    {
        "id": 28,
        "question": "como se compensa si MGEP me propone un cambio de campus",
        "expected_file": "4 Compensación por cambio de campus base.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 29,
        "question": "he empezado hoy a trabajar, tengo que realizar el registro de jornada",
        "expected_file": "2 Horario de trabajo.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    {
        "id": 30,
        "question": "al cabo de cuanto tiempo me pueden consolidar en MGEP",
        "expected_file": "Normativa de contratación y consolidación.pdf",
        "expected_chunk_index": None,
        "partial_expected": False,
        "notes": "",
    },
    # --- Erdizka / respuesta parcial ---
    {
        "id": 31,
        "question": "a cuanto se paga el kilometraje",
        "expected_file": "3 Compensación gastos desplazamiento y manutención.pdf",
        "expected_chunk_index": None,
        "partial_expected": True,
        "notes": "Esperado: 0,32 euros/km. Tabla no indexada correctamente en versión anterior.",
    },
    {
        "id": 32,
        "question": "el canon de educacion tiene retencion",
        "expected_file": "Reglamento de régimen interno.pdf",
        "expected_chunk_index": None,
        "partial_expected": True,
        "notes": "Esperado: respuesta completa sobre retención del canon.",
    },
    {
        "id": 33,
        "question": "cuanta aportacion paga un socio indefinido",
        "expected_file": "Estatutos.pdf",
        "expected_chunk_index": None,
        "partial_expected": True,
        "notes": "Esperado: desglose completo de aportación inicial.",
    },
    {
        "id": 34,
        "question": "quien puede acogerse a la ayuda por cese de la actividad",
        "expected_file": "7 Ayudas por cese de la actividad laboral.pdf",
        "expected_chunk_index": None,
        "partial_expected": True,
        "notes": "Esperado: persona socia que cese y cause baja como socia de trabajo.",
    },
    {
        "id": 35,
        "question": "cual va a ser el importe de la ayuda que voy a recibir por el cese",
        "expected_file": "7 Ayudas por cese de la actividad laboral.pdf",
        "expected_chunk_index": None,
        "partial_expected": True,
        "notes": "Esperado: tabla según índice laboral 2,3 a 4,0.",
    },
]


# ===========================
# Helpers
# ===========================

def _file_matches(result_file: str, expected_file: str) -> bool:
    """Comparación tolerante: normaliza mayúsculas y espacios."""
    if not result_file or not expected_file:
        return False
    return result_file.strip().lower() == expected_file.strip().lower()


def _check_hit(results: list, expected_file: str, expected_chunk_index) -> dict:
    """
    Devuelve información sobre si la fuente esperada aparece en los top 3 resultados.
    - file_hit: el expected_file aparece en top 3
    - chunk_hit: el expected_chunk_index también coincide (solo si se especificó)
    - hit_position: posición (1-3) donde apareció, o None
    """
    for pos, r in enumerate(results[:3], start=1):
        if _file_matches(r.get("file_name", ""), expected_file):
            chunk_ok = (
                expected_chunk_index is None
                or r.get("chunk_index") == expected_chunk_index
            )
            return {
                "file_hit": True,
                "chunk_hit": chunk_ok,
                "hit_position": pos,
                "hit_chunk_index": r.get("chunk_index"),
            }
    return {
        "file_hit": False,
        "chunk_hit": False,
        "hit_position": None,
        "hit_chunk_index": None,
    }


def run_question(api_url: str, q: dict, generate_answer: bool, delay: float) -> dict:
    payload = {
        "question": q["question"],
        "generate_answer": generate_answer,
        "top_k": 8,
    }
    try:
        resp = requests.post(f"{api_url}/search", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return {"error": "timeout", "raw": None}
    except Exception as e:
        return {"error": str(e), "raw": None}

    results_raw = data.get("results", [])
    top3 = [
        {
            "file_name": r.get("file_name", ""),
            "chunk_index": r.get("chunk_index"),
            "similarity": round(float(r.get("similarity", 0)), 4),
        }
        for r in results_raw[:3]
    ]

    hit = _check_hit(results_raw, q["expected_file"], q.get("expected_chunk_index"))

    result = {
        "id": q["id"],
        "question": q["question"],
        "expected_file": q["expected_file"],
        "expected_chunk_index": q.get("expected_chunk_index"),
        "partial_expected": q.get("partial_expected", False),
        "faq_acceptable": q.get("faq_acceptable", False),
        "notes": q.get("notes", ""),
        # Métricas de retrieval
        "max_similarity": round(float(data.get("max_similarity", 0)), 4),
        "zone": data.get("zone", ""),
        "confidence": data.get("confidence", ""),
        "answered_by_faq": data.get("answered_by_faq", False),
        "second_pass_performed": data.get("second_pass_performed", False),
        "query_variants_used": data.get("query_variants_used", 0),
        # Top 3 fuentes recuperadas
        "top3_results": top3,
        # Resultado del hit check
        # faq_acceptable=True + answered_by_faq=True cuenta como hit válido
        "file_hit": hit["file_hit"] or (q.get("faq_acceptable", False) and data.get("answered_by_faq", False)),
        "chunk_hit": hit["chunk_hit"],
        "hit_position": hit["hit_position"],
        "hit_chunk_index": hit["hit_chunk_index"],
        # Respuesta generada
        "answer": data.get("answer", None) if generate_answer else None,
        "answer_generated": data.get("answer_generated", False),
    }

    if delay > 0:
        time.sleep(delay)

    return result


# ===========================
# Formateo de consola
# ===========================

RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"


def _col(text, color):
    return f"{color}{text}{RESET}"


def print_question_result(r: dict, idx: int, total: int):
    prefix = f"[{idx:02d}/{total}]"
    qid = f"#{r['id']:02d}"
    partial_tag = _col(" [ERDIZKA]", YELLOW) if r.get("partial_expected") else ""

    if r.get("error"):
        print(f"{prefix} {qid}{partial_tag} {_col('ERROR', RED)}: {r['error']}")
        print(f"       Q: {r['question'][:80]}")
        return

    hit = r["file_hit"]
    icon = _col("✅", GREEN) if hit else _col("❌", RED)
    chunk_note = ""
    if r.get("expected_chunk_index") is not None:
        chunk_hit = r.get("chunk_hit", False)
        chunk_icon = _col("✅", GREEN) if chunk_hit else _col("❌", RED)
        chunk_note = f" chunk_idx={r['hit_chunk_index']} {chunk_icon}"

    sim = r.get("max_similarity", 0)
    zone = r.get("zone", "?")
    faq = _col(" [FAQ]", YELLOW) if r.get("answered_by_faq") else ""
    sp = _col(" [2nd-pass]", YELLOW) if r.get("second_pass_performed") else ""

    print(f"{prefix} {qid}{partial_tag} {icon}  sim={sim:.3f} zone={zone}{faq}{sp}{chunk_note}")
    print(f"       Q: {r['question'][:90]}")
    print(f"       Esperado : {r['expected_file']}")

    for i, res in enumerate(r.get("top3_results", []), start=1):
        match_mark = _col("◀", GREEN) if _file_matches(res["file_name"], r["expected_file"]) else " "
        print(f"       Top{i} {match_mark}: {res['file_name']}  chunk={res['chunk_index']}  sim={res['similarity']:.4f}")

    if r.get("answer"):
        answer_preview = r["answer"][:120].replace("\n", " ")
        print(f"       Answer : {_col(answer_preview + '…', DIM)}")

    print()


def print_summary(results: list):
    total = len(results)
    validated = [r for r in results if not r.get("partial_expected") and not r.get("error")]
    partial = [r for r in results if r.get("partial_expected") and not r.get("error")]
    errors = [r for r in results if r.get("error")]

    val_hits = sum(1 for r in validated if r["file_hit"])
    par_hits = sum(1 for r in partial if r["file_hit"])

    # Chunk-level hit (solo pregunta #1 tiene expected_chunk_index=32)
    chunk_hits = [r for r in results if r.get("expected_chunk_index") is not None]
    chunk_ok = sum(1 for r in chunk_hits if r.get("chunk_hit"))

    print("=" * 60)
    print(f"{BOLD}RESUMEN BASELINE{RESET}")
    print("=" * 60)
    print(f"  Preguntas validadas  : {val_hits}/{len(validated)}")
    print(f"  Preguntas Erdizka    : {par_hits}/{len(partial)}")
    print(f"  TOTAL file_hit       : {val_hits + par_hits}/{total - len(errors)}")
    if chunk_hits:
        print(f"  Chunk exacto (#1)    : {chunk_ok}/{len(chunk_hits)}")
    if errors:
        print(f"  {_col('Errores API', RED)}          : {len(errors)}")

    # Preguntas con FAQ bypass (pueden enmascarar fallos de retrieval)
    faq_bypassed = [r for r in results if r.get("answered_by_faq")]
    if faq_bypassed:
        print(f"\n  {_col('FAQ bypass', YELLOW)} ({len(faq_bypassed)} preguntas — no pasan por retrieval vectorial):")
        for r in faq_bypassed:
            print(f"    #{r['id']:02d}: {r['question'][:70]}")

    # Segundo pase activado
    second_pass = [r for r in results if r.get("second_pass_performed")]
    if second_pass:
        print(f"\n  2nd-pass activado ({len(second_pass)} preguntas):")
        for r in second_pass:
            icon = _col("✅", GREEN) if r["file_hit"] else _col("❌", RED)
            print(f"    #{r['id']:02d} {icon}: {r['question'][:70]}")

    # Fallos
    failures = [r for r in results if not r.get("error") and not r["file_hit"]]
    if failures:
        print(f"\n  {_col('Fallos de retrieval', RED)} ({len(failures)}):")
        for r in failures:
            partial_tag = " [ERDIZKA]" if r.get("partial_expected") else ""
            top1 = r["top3_results"][0]["file_name"] if r.get("top3_results") else "—"
            print(f"    #{r['id']:02d}{partial_tag}: {r['question'][:60]}")
            print(f"          → Top1 real: {top1}")
            print(f"          → Esperado : {r['expected_file']}")

    print("=" * 60)


# ===========================
# Main
# ===========================

def main():
    parser = argparse.ArgumentParser(description="Baseline test para RAG normativas MGEP")
    parser.add_argument("--url", default="http://localhost:5000", help="URL base de la API")
    parser.add_argument("--output", default=".", help="Directorio donde guardar el JSON")
    parser.add_argument("--no-answer", action="store_true", help="No generar respuesta LLM (más rápido)")
    parser.add_argument("--delay", type=float, default=0.5, help="Segundos entre preguntas (default: 0.5)")
    parser.add_argument("--ids", help="Correr solo estos IDs, separados por coma. Ej: 1,2,31")
    args = parser.parse_args()

    generate_answer = not args.no_answer
    api_url = args.url.rstrip("/")

    # Filtro opcional por ID
    questions = QUESTIONS
    if args.ids:
        ids_filter = {int(x.strip()) for x in args.ids.split(",")}
        questions = [q for q in QUESTIONS if q["id"] in ids_filter]
        if not questions:
            print(f"ERROR: ningún ID válido en --ids={args.ids}")
            sys.exit(1)

    # Verificar que la API responde
    try:
        health = requests.get(f"{api_url}/health", timeout=10)
        health.raise_for_status()
        print(f"{_col('API OK', GREEN)}: {api_url}")
    except Exception as e:
        print(f"{_col('ERROR', RED)}: No se puede conectar a la API en {api_url}: {e}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output, f"baseline_{timestamp}.json")

    print(f"\nEjecutando {len(questions)} preguntas (generate_answer={generate_answer})\n")
    print("-" * 60)

    results = []
    for idx, q in enumerate(questions, start=1):
        r = run_question(api_url, q, generate_answer, args.delay)
        if "error" in r and "id" not in r:
            r["id"] = q["id"]
            r["question"] = q["question"]
            r["expected_file"] = q["expected_file"]
            r["partial_expected"] = q.get("partial_expected", False)
            r["file_hit"] = False
            r["chunk_hit"] = False
        results.append(r)
        print_question_result(r, idx, len(questions))

    # Guardar JSON
    output = {
        "metadata": {
            "timestamp": timestamp,
            "api_url": api_url,
            "generate_answer": generate_answer,
            "total_questions": len(questions),
            "file_hits": sum(1 for r in results if r.get("file_hit")),
        },
        "results": results,
    }
    os.makedirs(args.output, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print_summary(results)
    print(f"\nResultados guardados en: {_col(output_file, GREEN)}\n")


if __name__ == "__main__":
    main()
