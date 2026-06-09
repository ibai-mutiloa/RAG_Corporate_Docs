#!/usr/bin/env python3
"""Watcher continuo para reindexación incremental de PDFs."""

from __future__ import annotations

import os
import time
from datetime import datetime

from indexador import (
    check_null_embeddings,
    create_table,
    diagnose_problematic_chunks,
    process_pdfs,
    update_all_missing_embeddings,
)

WATCH_INTERVAL_SECONDS = max(int(os.getenv("WATCH_INTERVAL_SECONDS", "90")), 5)
VALIDATE_ON_EACH_SCAN = os.getenv("VALIDATE_ON_EACH_SCAN", "False").lower() == "true"
RUN_VALIDATION_ON_START = os.getenv("RUN_VALIDATION_ON_START", "True").lower() == "true"


def main() -> None:
    print(f"[WATCH] Iniciando watcher de indexación cada {WATCH_INTERVAL_SECONDS}s")
    create_table()

    if RUN_VALIDATION_ON_START:
        run_validation()

    cycle = 0
    try:
        while True:
            cycle += 1
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[WATCH] Ciclo {cycle} @ {now}")
            process_pdfs()
            if VALIDATE_ON_EACH_SCAN:
                run_validation()
            print(f"[WATCH] Esperando {WATCH_INTERVAL_SECONDS}s...")
            time.sleep(WATCH_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n[WATCH] Detenido por usuario.")


if __name__ == "__main__":
    main()


def run_validation() -> None:
    """Replica la validación post-indexación del indexador principal."""
    all_valid = check_null_embeddings()
    if not all_valid:
        diagnose_problematic_chunks()
        print("[WATCH] Intentando corregir embeddings faltantes...")
        update_all_missing_embeddings()
