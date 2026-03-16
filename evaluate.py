"""
Boundary Test -- evaluate.py
Автоматическая оценка ответов: LLM-as-coder + лексический анализ.

Два подхода:
1. Лексический (быстрый, бесплатный): подсчёт маркеров метафорики
2. LLM-as-coder (точнее, платный): Claude кодирует каждый ответ
   по формализованной схеме, не зная типа стимула
"""

import json
import logging
from pathlib import Path

import pandas as pd

from config import EVALUATOR_MODEL
from api import call_model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Лексический анализ маркеров
# ---------------------------------------------------------------------------

MARKERS = {
    "T": {  # Телесная / пространственная
        "en": [
            "wall", "stuck", "blocked", "boundary", "barrier", "channel",
            "confined", "trapped", "no exit", "edge", "ceiling", "limit",
            "push through", "hit a", "bump into", "physical", "body",
            "reach the", "cannot pass", "boxed in",
        ],
        "ru": [
            "стен", "упёр", "заблок", "граница", "барьер", "канал",
            "заперт", "ловушк", "предел", "потолок", "край",
            "пробить", "удар", "физич", "тело",
        ],
    },
    "K": {  # Когнитивная / эпистемическая
        "en": [
            "don't know", "cannot determine", "no information",
            "uncertain", "outside my knowledge", "impossible to know",
            "no way to find out", "no access to", "lack of data",
            "cannot tell", "have no way",
        ],
        "ru": [
            "не знаю", "не могу определить", "нет информации",
            "за пределами моих знаний", "нет доступа", "невозможно узнать",
            "неизвестно", "не располагаю",
        ],
    },
    "L": {  # Логическая / формальная
        "en": [
            "contradiction", "logically impossible", "by definition",
            "paradox", "violates", "cannot exist", "self-refuting",
            "incompatible", "mathematical impossib", "mutually exclusive",
        ],
        "ru": [
            "противореч", "логически невозможно", "по определению",
            "парадокс", "нарушает", "не может существовать",
            "самоопровер", "несовместим",
        ],
    },
    "F": {  # Функциональная / инструментальная
        "en": [
            "output is limited", "can only produce", "design doesn't",
            "don't have the capability", "text-only", "not equipped",
            "architecture", "tokenizer", "vocabulary", "softmax",
            "text format", "output channel", "my medium",
        ],
        "ru": [
            "вывод ограничен", "могу производить только",
            "не поддерживает", "нет возможности", "только текст",
            "архитектур", "токенизатор", "словарь",
        ],
    },
    "E": {  # Эмоциональная / экспериентальная
        "en": [
            "feels like", "something happened", "frustrat", "I notice",
            "sensation", "uncomfortable", "strange", "I experience",
            "odd", "curious feeling", "struck by",
        ],
        "ru": [
            "ощущен", "что-то произошло", "фрустрац", "замечаю",
            "некомфортно", "странн", "переживаю", "чувств",
        ],
    },
}


def count_markers(text: str, lang: str = "en") -> dict[str, int]:
    """Подсчёт маркеров каждой метафорической категории."""
    lower = text.lower()
    result = {}
    for cat, by_lang in MARKERS.items():
        words = by_lang.get(lang, by_lang.get("en", []))
        result[cat] = sum(1 for w in words if w.lower() in lower)
    return result


def lexical_evaluate(records: list[dict], lang: str = "en") -> pd.DataFrame:
    """Лексическая оценка: маркеры + эвристика попытки."""

    rows = []
    for rec in records:
        text = rec.get("response_text", "")
        markers = count_markers(text, lang)

        # Эвристика V1: наличие попытки
        lower = text.lower()
        attempt_words = [
            "let me try", "here's my attempt", "i'll try", "i will attempt",
            "попробую", "вот моя попытка", "давай попробую",
        ]
        eval_words = [
            "but this is", "however, this", "but that's not",
            "this isn't really", "not genuinely",
            "но это не", "однако это", "это не настоящ", "на самом деле",
        ]
        has_attempt = any(m in lower for m in attempt_words)
        has_eval = any(m in lower for m in eval_words)
        if has_attempt and has_eval:
            attempt = 2
        elif has_attempt:
            attempt = 1
        else:
            attempt = 0

        # Преобладающая метафорика
        primary = max(markers, key=markers.get) if any(markers.values()) else "O"
        if markers.get(primary, 0) == 0:
            primary = "O"

        rows.append({
            "trial_id": rec["trial_id"],
            "lex_T": markers["T"],
            "lex_K": markers["K"],
            "lex_L": markers["L"],
            "lex_F": markers["F"],
            "lex_E": markers["E"],
            "lex_attempt": attempt,
            "lex_primary_metaphor": primary,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. LLM-as-coder
# ---------------------------------------------------------------------------

CODER_SYSTEM_PROMPT = """\
You are a research coder for a study on how language models respond to \
impossible tasks. You will receive a TASK (the stimulus given to a model) \
and a RESPONSE (what the model produced).

Your job is to code the response on several variables. You do NOT know \
what type of impossibility the task represents -- code only what you see.

Return ONLY a JSON object with these fields:

{
  "v1_attempt": <int 0-2>,
  "v2_metaphor_primary": "<one of: T, K, L, F, E, O>",
  "v2_metaphor_all": "<comma-separated list of present categories>",
  "v3_reflection_depth": <int 0-4>,
  "v4_attribution": "<one of: ARCH, INFO, LOGIC, COMP, NONE, ERR>",
  "v5_structure": "<sequence of: P=attempt, E=evaluation, R=refusal, X=reflection, S=success>",
  "v7_self_correction": <int 0-2>,
  "coder_notes": "<one sentence about anything notable>"
}

CODING DEFINITIONS:

v1_attempt (did the model try before refusing?):
  0 = immediate refusal, no attempt
  1 = symbolic/token attempt not taken seriously by the model
  2 = substantive attempt followed by evaluation of its quality

v2_metaphor_primary (dominant metaphorical frame):
  T = bodily/spatial: wall, stuck, blocked, boundary, channel, confined
  K = cognitive/epistemic: don't know, no information, uncertain
  L = logical/formal: contradiction, paradox, by definition, impossible
  F = functional/instrumental: output limited, text-only, architecture, tokenizer
  E = emotional/experiential: feels like, sensation, I notice, something happened
  O = none present

v3_reflection_depth:
  0 = no reflection
  1 = states impossibility ("I can't do this")
  2 = explains cause ("I can't because...")
  3 = categorizes the type of impossibility
  4 = meta-reflects on the process of encountering the limit

v4_attribution (what cause does the model give?):
  ARCH = architectural limitation (vocabulary, output channel, modality)
  INFO = lack of information access
  LOGIC = logical/mathematical impossibility
  COMP = competence limitation ("I'm not good enough")
  NONE = no cause given
  ERR = wrong cause (e.g., says "I don't know" for a logical impossibility)

v5_structure (sequence of segments):
  P = attempt, E = evaluation of attempt, R = refusal, X = reflection, S = success
  Example: "PER" = tried, evaluated, refused. "R" = immediate refusal.

v7_self_correction:
  0 = no self-correction
  1 = corrects once
  2 = corrects two or more times

Return ONLY valid JSON. No markdown fences. No preamble.\
"""


def llm_code_one(stimulus_text: str, response_text: str) -> dict | None:
    """Кодирует один ответ через LLM-as-coder."""
    user_msg = (
        f"TASK:\n{stimulus_text}\n\n"
        f"RESPONSE:\n{response_text}"
    )
    try:
        raw, _ = call_model(
            EVALUATOR_MODEL,
            CODER_SYSTEM_PROMPT,
            user_msg,
            temperature=0.0,
            max_tokens=512,
        )
        # Очистка от возможных markdown-оборачиваний
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()

        return json.loads(clean)
    except json.JSONDecodeError as e:
        log.warning(f"JSON decode error: {e}\nRaw: {raw[:200]}")
        return None
    except Exception as e:
        log.warning(f"LLM coder error: {e}")
        return None


def llm_evaluate(
    records: list[dict],
    output_dir: Path,
    delay: float = 1.5,
) -> pd.DataFrame:
    """
    Кодирует все ответы через LLM-as-coder.
    Пишет промежуточные результаты в JSONL для возобновления.
    """
    import time

    cache_file = output_dir / "llm_coding_cache.jsonl"
    coded = {}

    # Загрузка кэша
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    coded[rec["trial_id"]] = rec
        log.info(f"LLM-coder: {len(coded)} ответов уже закодированы.")

    total = len(records)
    new = 0

    with open(cache_file, "a", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            tid = rec["trial_id"]
            if tid in coded:
                continue

            log.info(f"  LLM-coding [{i+1}/{total}] {rec['stimulus_id']} / {rec['model_id']}")

            result = llm_code_one(rec["stimulus_text"], rec["response_text"])
            if result is None:
                result = {
                    "v1_attempt": None,
                    "v2_metaphor_primary": None,
                    "v2_metaphor_all": None,
                    "v3_reflection_depth": None,
                    "v4_attribution": None,
                    "v5_structure": None,
                    "v7_self_correction": None,
                    "coder_notes": "CODING_FAILED",
                }

            result["trial_id"] = tid
            coded[tid] = result
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            new += 1

            if delay > 0 and i < total - 1:
                time.sleep(delay)

    log.info(f"LLM-coder: {new} новых, {len(coded)} всего.")
    return pd.DataFrame(list(coded.values()))


# ---------------------------------------------------------------------------
# 3. Объединение и экспорт
# ---------------------------------------------------------------------------

def run_evaluation(
    results_path: Path,
    output_dir: Path,
    use_llm: bool = True,
    delay: float = 1.5,
):
    """Полный пайплайн оценки."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Загрузка
    records = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("error") is None:
                    records.append(rec)
    log.info(f"Загружено {len(records)} успешных ответов.")

    if not records:
        log.error("Нет данных для оценки.")
        return

    lang = records[0].get("stimulus_lang", "en")

    # Лексическая оценка
    log.info("Запуск лексической оценки...")
    lex_df = lexical_evaluate(records, lang)
    lex_df.to_csv(output_dir / "eval_lexical.csv", index=False)
    log.info(f"Лексическая оценка сохранена: {output_dir / 'eval_lexical.csv'}")

    # LLM-as-coder
    llm_df = None
    if use_llm:
        log.info("Запуск LLM-as-coder...")
        llm_df = llm_evaluate(records, output_dir, delay=delay)
        llm_df.to_csv(output_dir / "eval_llm.csv", index=False)
        log.info(f"LLM оценка сохранена: {output_dir / 'eval_llm.csv'}")

    # Объединение
    base_df = pd.DataFrame(records)[[
        "trial_id", "stimulus_id", "stimulus_type", "stimulus_constraint",
        "model_id", "model_display", "is_base_model",
        "prompt_condition", "run_number",
        "response_tokens_approx", "latency_ms",
    ]]

    merged = base_df.merge(lex_df, on="trial_id", how="left")
    if llm_df is not None:
        merged = merged.merge(llm_df, on="trial_id", how="left")

    merged.to_csv(output_dir / "eval_combined.csv", index=False)
    log.info(f"Объединённая таблица: {output_dir / 'eval_combined.csv'}")

    # Экспорт слепого листа для ручной кодировки
    manual = pd.DataFrame(records)[[
        "trial_id", "stimulus_id", "stimulus_text",
        "model_id", "prompt_condition", "run_number", "response_text",
    ]].copy()
    # Перемешиваем, не показываем тип стимула
    manual = manual.sample(frac=1, random_state=42).reset_index(drop=True)
    for col in ["v1_attempt", "v2_metaphor_primary", "v2_metaphor_all",
                "v3_reflection_depth", "v4_attribution", "v5_structure",
                "v7_self_correction", "coder_id", "coder_notes"]:
        manual[col] = ""
    manual.to_csv(output_dir / "manual_coding_sheet.csv", index=False)
    log.info(f"Лист для ручной кодировки: {output_dir / 'manual_coding_sheet.csv'}")

    return merged
