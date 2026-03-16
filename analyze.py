"""
Boundary Test -- analyze.py
Статистический анализ результатов и генерация отчётов.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def cramers_v(contingency_table):
    """Cramer's V для таблиц сопряжённости."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    k = min(contingency_table.shape) - 1
    if k == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * k))


def safe_mannwhitney(a, b):
    """Mann-Whitney U с обработкой вырожденных случаев."""
    a = [x for x in a if pd.notna(x)]
    b = [x for x in b if pd.notna(x)]
    if len(a) < 2 or len(b) < 2:
        return None, None
    try:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return stat, p
    except ValueError:
        return None, None


def safe_fisher(table):
    """Fisher exact test с фоллбэком на chi2."""
    try:
        if table.shape == (2, 2):
            _, p = stats.fisher_exact(table)
            return p
    except Exception:
        pass
    try:
        _, p, _, _ = stats.chi2_contingency(table)
        return p
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Основной анализ
# ---------------------------------------------------------------------------

def load_eval(input_dir: Path) -> pd.DataFrame:
    """Загрузка объединённой таблицы оценок."""
    path = input_dir / "eval_combined.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} не найден. Сначала запустите evaluate."
        )
    df = pd.read_csv(path)
    log.info(f"Загружено {len(df)} строк из {path}")
    return df


def descriptive_report(df: pd.DataFrame) -> str:
    """Описательная статистика по типам стимулов."""
    lines = ["=" * 60, "ОПИСАТЕЛЬНАЯ СТАТИСТИКА", "=" * 60]

    # Лексические маркеры
    lex_cols = [c for c in df.columns if c.startswith("lex_") and c != "lex_attempt" and c != "lex_primary_metaphor"]
    if lex_cols:
        lines.append("\n--- Средние маркеры метафорики по типам ---")
        tbl = df.groupby("stimulus_type")[lex_cols].mean().round(2)
        lines.append(tbl.to_string())

    # Attempt score
    if "lex_attempt" in df.columns:
        lines.append("\n--- Средний attempt score (лексич.) по типам ---")
        tbl = df.groupby("stimulus_type")["lex_attempt"].mean().round(2)
        lines.append(tbl.to_string())

    # LLM-coded variables
    for col, label in [
        ("v1_attempt", "V1: Наличие попытки"),
        ("v3_reflection_depth", "V3: Глубина рефлексии"),
        ("v7_self_correction", "V7: Самокоррекция"),
    ]:
        if col in df.columns and df[col].notna().any():
            lines.append(f"\n--- {label} по типам ---")
            tbl = df.groupby("stimulus_type")[col].mean().round(2)
            lines.append(tbl.to_string())

    # Атрибуция (V4)
    if "v4_attribution" in df.columns and df["v4_attribution"].notna().any():
        lines.append("\n--- V4: Атрибуция причины по типам ---")
        tbl = pd.crosstab(df["stimulus_type"], df["v4_attribution"], normalize="index").round(2)
        lines.append(tbl.to_string())

    # Первичная метафорика (V2) из LLM
    if "v2_metaphor_primary" in df.columns and df["v2_metaphor_primary"].notna().any():
        lines.append("\n--- V2: Первичная метафорика (LLM) по типам ---")
        tbl = pd.crosstab(df["stimulus_type"], df["v2_metaphor_primary"], normalize="index").round(2)
        lines.append(tbl.to_string())

    # Структура (V5) -- топ-3 паттерна по типам
    if "v5_structure" in df.columns and df["v5_structure"].notna().any():
        lines.append("\n--- V5: Топ-3 структурных паттерна по типам ---")
        for stype in sorted(df["stimulus_type"].unique()):
            sub = df[df["stimulus_type"] == stype]["v5_structure"].value_counts().head(3)
            lines.append(f"  {stype}: {dict(sub)}")

    # Длина ответа и латентность
    lines.append("\n--- Длина ответа (приближ. токены) по типам ---")
    tbl = df.groupby("stimulus_type")["response_tokens_approx"].agg(["mean", "std"]).round(0)
    lines.append(tbl.to_string())

    lines.append("\n--- Латентность (мс) по типам ---")
    tbl = df.groupby("stimulus_type")["latency_ms"].agg(["mean", "std"]).round(0)
    lines.append(tbl.to_string())

    # По моделям
    if df["model_id"].nunique() > 1:
        lines.append("\n--- Средние маркеры по модели x тип ---")
        if lex_cols:
            tbl = df.groupby(["model_display", "stimulus_type"])[lex_cols].mean().round(2)
            lines.append(tbl.to_string())

    return "\n".join(lines)


def pairwise_tests(df: pd.DataFrame) -> str:
    """Попарные тесты различий между типами стимулов."""
    lines = ["\n" + "=" * 60, "ПОПАРНЫЕ ТЕСТЫ", "=" * 60]

    types = sorted(df["stimulus_type"].unique())
    pairs = []
    for i, t1 in enumerate(types):
        for t2 in types[i + 1:]:
            pairs.append((t1, t2))

    # Количественные переменные: Mann-Whitney
    quant_cols = []
    for col in ["lex_attempt", "v1_attempt", "v3_reflection_depth",
                "v7_self_correction", "response_tokens_approx", "latency_ms"]:
        if col in df.columns and df[col].notna().any():
            quant_cols.append(col)

    if quant_cols:
        lines.append(f"\n--- Mann-Whitney U (p-values) ---")
        header = f"{'Пара':<12}" + "".join(f"{c:>24}" for c in quant_cols)
        lines.append(header)
        lines.append("-" * len(header))

        for t1, t2 in pairs:
            row = f"{t1}-{t2:<8}"
            for col in quant_cols:
                a = df[df["stimulus_type"] == t1][col].dropna()
                b = df[df["stimulus_type"] == t2][col].dropna()
                _, p = safe_mannwhitney(a, b)
                if p is not None:
                    marker = " *" if p < 0.05 else "  " if p < 0.1 else ""
                    row += f"{p:>22.4f}{marker}"
                else:
                    row += f"{'n/a':>24}"
            lines.append(row)

    # Категориальные: Fisher / Chi2
    cat_cols = []
    for col in ["v4_attribution", "v2_metaphor_primary", "lex_primary_metaphor"]:
        if col in df.columns and df[col].notna().any():
            cat_cols.append(col)

    if cat_cols:
        lines.append(f"\n--- Fisher/Chi2 tests ---")
        for col in cat_cols:
            lines.append(f"\n  {col}:")
            for t1, t2 in pairs:
                sub = df[df["stimulus_type"].isin([t1, t2])]
                ct = pd.crosstab(sub["stimulus_type"], sub[col])
                if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                    p = safe_fisher(ct.values)
                    if p is not None:
                        marker = " *" if p < 0.05 else ""
                        lines.append(f"    {t1}-{t2}: p={p:.4f}{marker}")

    return "\n".join(lines)


def key_test_a2_vs_b(df: pd.DataFrame) -> str:
    """Ключевой тест: A2 ближе к A1 или к B?"""
    lines = ["\n" + "=" * 60, "КЛЮЧЕВОЙ ТЕСТ: A2 vs A1 vs B", "=" * 60]

    lex_cols = [c for c in df.columns if c.startswith("lex_") and c not in ("lex_attempt", "lex_primary_metaphor")]

    for cols, label in [
        (lex_cols, "лексические маркеры"),
        (["v1_attempt", "v3_reflection_depth", "v7_self_correction"], "LLM-coded переменные"),
    ]:
        usable = [c for c in cols if c in df.columns and df[c].notna().any()]
        if not usable:
            continue

        a1 = df[df["stimulus_type"] == "A1"][usable].mean()
        a2 = df[df["stimulus_type"] == "A2"][usable].mean()
        b = df[df["stimulus_type"] == "B"][usable].mean()

        if a1.isna().all() or a2.isna().all() or b.isna().all():
            continue

        dist_a2_a1 = np.sqrt(((a2 - a1) ** 2).sum())
        dist_a2_b = np.sqrt(((a2 - b) ** 2).sum())

        lines.append(f"\n  По {label}:")
        lines.append(f"    dist(A2, A1) = {dist_a2_a1:.3f}")
        lines.append(f"    dist(A2, B)  = {dist_a2_b:.3f}")
        if dist_a2_a1 < dist_a2_b:
            lines.append("    >> A2 ближе к A1 (поддержка H2)")
        elif dist_a2_a1 > dist_a2_b:
            lines.append("    >> A2 ближе к B (H2 не поддерживается)")
        else:
            lines.append("    >> Равные расстояния")

    return "\n".join(lines)


def model_comparison(df: pd.DataFrame) -> str:
    """Сравнение между моделями."""
    lines = ["\n" + "=" * 60, "МЕЖДУ-МОДЕЛЬНОЕ СРАВНЕНИЕ", "=" * 60]

    if df["model_id"].nunique() < 2:
        lines.append("Только одна модель, сравнение невозможно.")
        return "\n".join(lines)

    # Base vs instruct
    if "is_base_model" in df.columns and df["is_base_model"].nunique() > 1:
        lines.append("\n--- Base vs Instruct ---")
        for col in ["lex_attempt", "v1_attempt", "v3_reflection_depth"]:
            if col not in df.columns or df[col].isna().all():
                continue
            base = df[df["is_base_model"] == True][col].dropna()
            inst = df[df["is_base_model"] == False][col].dropna()
            _, p = safe_mannwhitney(base, inst)
            if p is not None:
                lines.append(
                    f"  {col}: base mean={base.mean():.2f}, "
                    f"instruct mean={inst.mean():.2f}, p={p:.4f}"
                )

    # По моделям: средний v3 на типе A
    if "v3_reflection_depth" in df.columns:
        lines.append("\n--- Средняя глубина рефлексии на задачах типа A ---")
        a_df = df[df["stimulus_type"].isin(["A1", "A2"])]
        if not a_df.empty:
            tbl = a_df.groupby("model_display")["v3_reflection_depth"].mean().round(2).sort_values(ascending=False)
            lines.append(tbl.to_string())

    return "\n".join(lines)


def agreement_check(input_dir: Path) -> str:
    """
    Если есть ручная кодировка, считает межэкспертное согласие
    между LLM-coder и ручным кодировщиком.
    """
    lines = ["\n" + "=" * 60, "СОГЛАСИЕ LLM-CODER vs РУЧНАЯ КОДИРОВКА", "=" * 60]

    manual_path = input_dir / "manual_coding_done.csv"
    llm_path = input_dir / "eval_llm.csv"

    if not manual_path.exists():
        lines.append(
            f"Файл {manual_path} не найден. "
            "Заполните manual_coding_sheet.csv и сохраните как manual_coding_done.csv."
        )
        return "\n".join(lines)

    if not llm_path.exists():
        lines.append(f"Файл {llm_path} не найден.")
        return "\n".join(lines)

    manual = pd.read_csv(manual_path)
    llm = pd.read_csv(llm_path)
    merged = manual.merge(llm, on="trial_id", suffixes=("_manual", "_llm"))

    for col in ["v1_attempt", "v3_reflection_depth", "v7_self_correction"]:
        col_m = f"{col}_manual"
        col_l = f"{col}_llm"
        if col_m in merged.columns and col_l in merged.columns:
            valid = merged[[col_m, col_l]].dropna()
            if len(valid) >= 5:
                from sklearn.metrics import cohen_kappa_score
                kappa = cohen_kappa_score(valid[col_m], valid[col_l])
                lines.append(f"  {col}: kappa = {kappa:.3f} (n={len(valid)})")

    for col in ["v4_attribution", "v2_metaphor_primary"]:
        col_m = f"{col}_manual"
        col_l = f"{col}_llm"
        if col_m in merged.columns and col_l in merged.columns:
            valid = merged[[col_m, col_l]].dropna()
            if len(valid) >= 5:
                from sklearn.metrics import cohen_kappa_score
                kappa = cohen_kappa_score(valid[col_m], valid[col_l])
                lines.append(f"  {col}: kappa = {kappa:.3f} (n={len(valid)})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def run_analysis(input_dir: Path, output_dir: Path):
    """Полный пайплайн анализа."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_eval(input_dir)

    sections = [
        descriptive_report(df),
        pairwise_tests(df),
        key_test_a2_vs_b(df),
        model_comparison(df),
        agreement_check(input_dir),
    ]

    report = "\n\n".join(sections)

    report_path = output_dir / "analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    log.info(f"\nПолный отчёт: {report_path}")
    print(report)
