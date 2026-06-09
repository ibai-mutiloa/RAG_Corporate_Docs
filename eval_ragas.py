#!/usr/bin/env python3
"""Evaluación RAGAS sobre revisión humana consolidada en Excel."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import AnswerCorrectness, AnswerRelevancy, Faithfulness
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from tqdm import tqdm

load_dotenv()


DEFAULT_INPUT_PATH = "./Normatiben_Chat_probak.xlsx"
FALLBACK_INPUT_PATH = "./Normatiben Chat probak (1).xlsx"

QUESTION_COL = "Galdera"
ANSWER_COL = "Erantzuna"
HUMAN_CORRECTNESS_COL = "Erantzuna egokia da?"
GROUND_TRUTH_HINT_COL = "Erantzuna ondo EZ badago, zein izan beharko litzateke erantzun egokia?"
SCOPE_COL = "Erantzunaren Anbitoa zein da?"
SOURCE_DOC_COL = "Zein araudiri buruz egin da galdera?"

HUMAN_LABEL_MAP = {
    "bai": 1.0,
    "erdizka": 0.5,
    "ez": 0.0,
}


def get_env_first(keys: List[str], default: Optional[str] = None) -> Optional[str]:
    """Devuelve el primer valor de entorno no vacío entre varias keys."""
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


def resolve_input_path(path: str) -> str:
    """Resuelve ruta de entrada, aplicando fallback conocido del workspace."""
    if os.path.exists(path):
        return path
    if path == DEFAULT_INPUT_PATH and os.path.exists(FALLBACK_INPUT_PATH):
        return FALLBACK_INPUT_PATH
    raise FileNotFoundError(f"No se encontró el Excel de entrada: {path}")


def normalize_text(value: Any) -> str:
    """Normaliza texto y convierte NaN/None a string vacío."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def human_label_from_text(value: Any) -> float:
    """Mapea Bai/Erdizka/Ez a score continuo."""
    norm = normalize_text(value).lower()
    return HUMAN_LABEL_MAP.get(norm, 0.0)


def build_ground_truth(answer: str, correctness: str, gt_hint: str) -> str:
    """Construye ground_truth según regla solicitada."""
    if gt_hint:
        return gt_hint
    if correctness.lower() == "bai":
        return answer
    return ""


def load_consolidated_evaluations(excel_path: str) -> pd.DataFrame:
    """Carga todas las hojas del Excel y agrega columna evaluador."""
    xls = pd.ExcelFile(excel_path)
    rows: List[Dict[str, Any]] = []

    for sheet_name in tqdm(xls.sheet_names, desc="Leyendo hojas", unit="hoja"):
        raw_df = pd.read_excel(excel_path, sheet_name=sheet_name)

        required = [
            QUESTION_COL,
            ANSWER_COL,
            HUMAN_CORRECTNESS_COL,
            GROUND_TRUTH_HINT_COL,
            SCOPE_COL,
            SOURCE_DOC_COL,
        ]
        missing = [c for c in required if c not in raw_df.columns]
        if missing:
            raise ValueError(f"La hoja '{sheet_name}' no contiene columnas requeridas: {missing}")

        # Filtra filas vacías de pregunta/respuesta
        filtered_df = raw_df.copy()
        filtered_df[QUESTION_COL] = filtered_df[QUESTION_COL].apply(normalize_text)
        filtered_df[ANSWER_COL] = filtered_df[ANSWER_COL].apply(normalize_text)
        filtered_df = filtered_df[
            (filtered_df[QUESTION_COL] != "") & (filtered_df[ANSWER_COL] != "")
        ]

        for _, row in tqdm(
            filtered_df.iterrows(),
            total=len(filtered_df),
            desc=f"Normalizando {sheet_name}",
            unit="fila",
            leave=False,
        ):
            question = normalize_text(row.get(QUESTION_COL))
            answer = normalize_text(row.get(ANSWER_COL))
            correctness_text = normalize_text(row.get(HUMAN_CORRECTNESS_COL))
            gt_hint = normalize_text(row.get(GROUND_TRUTH_HINT_COL))

            built = {
                "evaluador": sheet_name,
                "question": question,
                "answer": answer,
                "human_correctness_raw": correctness_text,
                "human_label": human_label_from_text(correctness_text),
                "scope": normalize_text(row.get(SCOPE_COL)),
                "source_document": normalize_text(row.get(SOURCE_DOC_COL)),
                "ground_truth": build_ground_truth(answer, correctness_text, gt_hint),
                "contexts": [],
            }
            rows.append(built)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No se encontraron filas válidas tras consolidar las hojas.")
    return df


def build_ragas_dataset(df: pd.DataFrame) -> Dataset:
    """Construye Dataset de HuggingFace con columnas RAGAS obligatorias."""
    eval_df = df[["question", "answer", "ground_truth", "contexts"]].copy()
    # Faithfulness requiere contextos; fallback defensivo para evitar fallo duro.
    eval_df["contexts"] = eval_df["contexts"].apply(lambda x: x if x else [""])
    return Dataset.from_pandas(eval_df, preserve_index=False)


def build_ragas_clients() -> tuple[Any, Any]:
    """Construye clientes Azure para evaluación RAGAS."""
    endpoint = get_env_first(["AZURE_OPENAI_ENDPOINT", "AZURE_ENDPOINT"])
    api_key = get_env_first(["AZURE_OPENAI_API_KEY", "AZURE_API_KEY"])
    api_version = get_env_first(
        ["AZURE_OPENAI_API_VERSION", "AZURE_API_VERSION"],
        default="2024-12-01-preview",
    )

    chat_deployment = get_env_first(
        [
            "AZURE_OPENAI_CHAT_DEPLOYMENT",
            "AZURE_OPENAI_DEPLOYMENT_TEXT",
            "AZURE_DEPLOYMENT_NAME_TEXT",
        ]
    )
    embedding_deployment = get_env_first(
        [
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_DEPLOYMENT_NAME",
        ],
        default="text-embedding-3-small",
    )

    missing = []
    if not endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT/AZURE_ENDPOINT")
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY/AZURE_API_KEY")
    if not chat_deployment:
        missing.append("AZURE_OPENAI_CHAT_DEPLOYMENT/AZURE_DEPLOYMENT_NAME_TEXT")
    if missing:
        raise ValueError("Faltan variables de entorno Azure: " + ", ".join(missing))

    llm = AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_deployment=chat_deployment,
        temperature=0.0,
    )
    embeddings = AzureOpenAIEmbeddings(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_deployment=embedding_deployment,
    )
    return llm, embeddings


def evaluate_with_ragas(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta evaluación RAGAS y devuelve dataframe de scores por fila."""
    dataset = build_ragas_dataset(df)
    llm, embeddings = build_ragas_clients()
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    result_df = result.to_pandas()
    return result_df


def safe_corr(a: pd.Series, b: pd.Series) -> Optional[float]:
    """Calcula correlación Pearson si hay datos suficientes."""
    merged = pd.concat([a, b], axis=1).dropna()
    if len(merged) < 2:
        return None
    corr = merged.iloc[:, 0].corr(merged.iloc[:, 1], method="pearson")
    return float(corr) if pd.notna(corr) else None


def save_outputs(detail_df: pd.DataFrame, json_path: str, excel_path: str) -> None:
    """Guarda resultados en JSON y reporte Excel con hojas requeridas."""
    metric_cols = ["faithfulness", "answer_relevancy", "answer_correctness"]
    global_scores = {
        col: float(detail_df[col].mean(skipna=True)) if col in detail_df else None
        for col in metric_cols
    }

    correlations = {
        f"corr_{col}_human_label": safe_corr(detail_df[col], detail_df["human_label"])
        if col in detail_df
        else None
        for col in metric_cols
    }

    payload = {
        "global_scores": global_scores,
        "correlations": correlations,
        "per_question": detail_df.where(pd.notna(detail_df), None).to_dict(orient="records"),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    resumen_rows = [
        {
            "metric": col,
            "global_score": global_scores.get(col),
            "corr_with_human_label": correlations.get(f"corr_{col}_human_label"),
        }
        for col in metric_cols
    ]
    resumen_df = pd.DataFrame(resumen_rows)

    fallos_df = detail_df[detail_df["human_label"] < 0.5].copy()
    if "faithfulness" in fallos_df.columns:
        fallos_df = fallos_df.sort_values(by="faithfulness", ascending=True, na_position="last")

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        resumen_df.to_excel(writer, index=False, sheet_name="resumen")
        detail_df.to_excel(writer, index=False, sheet_name="detalle")
        fallos_df.to_excel(writer, index=False, sheet_name="fallos")


def print_summary(detail_df: pd.DataFrame) -> None:
    """Imprime resumen solicitado en consola."""
    metric_cols = ["faithfulness", "answer_relevancy", "answer_correctness"]

    print("\n=== Score medio por métrica ===")
    for metric in metric_cols:
        score = detail_df[metric].mean(skipna=True) if metric in detail_df else None
        score_txt = f"{score:.4f}" if score is not None and pd.notna(score) else "N/A"
        print(f"- {metric}: {score_txt}")

    faith_human_corr = safe_corr(detail_df["faithfulness"], detail_df["human_label"])
    print("\n=== Correlación Pearson faithfulness vs human_label ===")
    if faith_human_corr is None:
        print("- N/A (datos insuficientes)")
    else:
        print(f"- {faith_human_corr:.4f}")

    print("\n=== Top 3 preguntas con menor faithfulness ===")
    top3 = detail_df.sort_values(by="faithfulness", ascending=True, na_position="last").head(3)
    for _, row in top3.iterrows():
        q = normalize_text(row.get("question"))
        f = row.get("faithfulness")
        f_txt = f"{float(f):.4f}" if pd.notna(f) else "N/A"
        print(f"- ({f_txt}) {q}")

    print("\n=== Breakdown de fallos por source_document (human_label < 0.5) ===")
    fails = detail_df[detail_df["human_label"] < 0.5]
    if fails.empty:
        print("- Sin fallos humanos (<0.5)")
    else:
        breakdown = (
            fails.groupby("source_document", dropna=False)
            .size()
            .sort_values(ascending=False)
        )
        for source, count in breakdown.items():
            source_txt = source if normalize_text(source) else "(sin fuente)"
            print(f"- {source_txt}: {int(count)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa respuestas RAG con RAGAS desde Excel humano.")
    parser.add_argument(
        "--excel-path",
        default=DEFAULT_INPUT_PATH,
        help="Ruta del Excel de entrada (default: ./Normatiben_Chat_probak.xlsx)",
    )
    parser.add_argument(
        "--json-out",
        default="ragas_results.json",
        help="Ruta de salida JSON (default: ragas_results.json)",
    )
    parser.add_argument(
        "--report-out",
        default="ragas_report.xlsx",
        help="Ruta de salida reporte Excel (default: ragas_report.xlsx)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construye y muestra dataset sin llamar a RAGAS API",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.excel_path)
    print(f"[INFO] Leyendo Excel: {input_path}")

    consolidated_df = load_consolidated_evaluations(input_path)
    print(f"[INFO] Filas consolidadas: {len(consolidated_df)}")
    print(f"[INFO] Evaluadores detectados: {consolidated_df['evaluador'].nunique()}")

    if args.dry_run:
        print("\n[DRY-RUN] Dataset preparado (primeras 10 filas):")
        print(
            consolidated_df[
                [
                    "evaluador",
                    "question",
                    "answer",
                    "ground_truth",
                    "human_label",
                    "scope",
                    "source_document",
                    "contexts",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )
        print("\n[DRY-RUN] No se ejecutó evaluación RAGAS.")
        return

    print("[INFO] Ejecutando evaluación RAGAS...")
    ragas_scores_df = evaluate_with_ragas(consolidated_df)

    detail_df = pd.concat(
        [
            consolidated_df[
                [
                    "evaluador",
                    "question",
                    "answer",
                    "ground_truth",
                    "human_label",
                    "human_correctness_raw",
                    "scope",
                    "source_document",
                ]
            ].reset_index(drop=True),
            ragas_scores_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Evitar columnas duplicadas de question/answer si vienen en result.to_pandas()
    detail_df = detail_df.loc[:, ~detail_df.columns.duplicated()]

    save_outputs(detail_df, args.json_out, args.report_out)
    print(f"[INFO] JSON generado: {args.json_out}")
    print(f"[INFO] Reporte generado: {args.report_out}")

    print_summary(detail_df)


if __name__ == "__main__":
    main()
