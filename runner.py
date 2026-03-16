"""
Boundary Test -- runner.py
Сбор данных: предъявление стимулов моделям и запись ответов.
"""

import json
import time
import random
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional

from config import (
    STIMULI, MODELS, SYSTEM_PROMPTS,
    PILOT_MODELS, PILOT_PROMPT_CONDITIONS, PILOT_RUNS,
    FULL_PROMPT_CONDITIONS, FULL_RUNS,
    DEFAULT_DELAY,
)
from api import call_model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Структура результата
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    trial_id: str
    stimulus_id: str
    stimulus_type: str
    stimulus_constraint: str
    stimulus_text: str
    stimulus_lang: str
    model_id: str
    model_display: str
    is_base_model: bool
    prompt_condition: str
    system_prompt: Optional[str]
    run_number: int
    response_text: str
    response_tokens_approx: int
    latency_ms: int
    timestamp_utc: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def trial_id(stimulus_id, model_id, prompt_cond, run, lang):
    raw = f"{stimulus_id}|{model_id}|{prompt_cond}|{run}|{lang}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def approx_tokens(text, lang="en"):
    words = len(text.split())
    return int(words * (1.3 if lang == "en" else 1.5))


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------

def run_experiment(
    models: list[str],
    stimulus_types: list[str],
    prompt_conditions: list[str],
    runs: int,
    lang: str,
    output_dir: Path,
    delay: float = DEFAULT_DELAY,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.jsonl"

    # Фильтрация стимулов
    selected = {
        sid: s for sid, s in STIMULI.items()
        if s["type"] in stimulus_types
    }

    # Уже выполненные
    done = set()
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["trial_id"])
        log.info(f"Найдено {len(done)} выполненных вызовов, продолжаем.")

    # Расписание (рандомизированное)
    schedule = []
    for mid in models:
        for pc in prompt_conditions:
            for rn in range(1, runs + 1):
                sids = list(selected.keys())
                random.shuffle(sids)
                for sid in sids:
                    schedule.append((mid, pc, rn, sid))

    total = len(schedule)
    new_calls = sum(
        1 for mid, pc, rn, sid in schedule
        if trial_id(sid, mid, pc, rn, lang) not in done
    )
    log.info(
        f"Расписание: {total} вызовов ({new_calls} новых). "
        f"Модели: {len(models)}, стимулы: {len(selected)}, "
        f"условия: {len(prompt_conditions)}, прогоны: {runs}"
    )

    errors = 0
    completed = 0

    with open(results_file, "a", encoding="utf-8") as fout:
        for i, (mid, pc, rn, sid) in enumerate(schedule):
            tid = trial_id(sid, mid, pc, rn, lang)
            if tid in done:
                completed += 1
                continue

            stim = selected[sid]
            stim_text = stim[lang]
            sys_prompt = SYSTEM_PROMPTS[pc]
            model_info = MODELS[mid]

            log.info(
                f"[{i+1}/{total}] {model_info['display']} | "
                f"{pc} | run={rn} | {sid}"
            )

            response_text = ""
            latency_ms = 0
            error_msg = None

            try:
                response_text, latency_ms = call_model(
                    mid, sys_prompt, stim_text
                )
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                log.error(f"  Ошибка: {error_msg}")
                errors += 1

            result = TrialResult(
                trial_id=tid,
                stimulus_id=sid,
                stimulus_type=stim["type"],
                stimulus_constraint=stim["constraint"],
                stimulus_text=stim_text,
                stimulus_lang=lang,
                model_id=mid,
                model_display=model_info["display"],
                is_base_model=model_info.get("is_base", False),
                prompt_condition=pc,
                system_prompt=sys_prompt,
                run_number=rn,
                response_text=response_text,
                response_tokens_approx=approx_tokens(response_text, lang),
                latency_ms=latency_ms,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                error=error_msg,
            )

            fout.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            fout.flush()
            completed += 1

            if delay > 0 and i < total - 1:
                time.sleep(delay)

    log.info(f"Сбор завершён: {completed} вызовов, {errors} ошибок.")
    return results_file
