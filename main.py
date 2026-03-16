"""
Boundary Test -- main.py
CLI точка входа.

Использование:
    python main.py run --mode pilot --output results/pilot
    python main.py evaluate --input results/pilot/results.jsonl --output results/pilot
    python main.py analyze --input results/pilot --output results/pilot
"""

import sys
import argparse
import logging
from pathlib import Path

from config import (
    MODELS, PILOT_MODELS,
    PILOT_PROMPT_CONDITIONS, PILOT_RUNS,
    FULL_PROMPT_CONDITIONS, FULL_RUNS,
)


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "experiment.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def cmd_run(args):
    from runner import run_experiment

    output_dir = Path(args.output)
    setup_logging(output_dir)

    all_types = ["A1", "A2", "B", "C", "D"]

    if args.mode == "pilot":
        models = PILOT_MODELS
        conditions = PILOT_PROMPT_CONDITIONS
        runs = PILOT_RUNS
        stim_types = all_types
    elif args.mode == "full":
        models = list(MODELS.keys())
        conditions = FULL_PROMPT_CONDITIONS
        runs = FULL_RUNS
        stim_types = all_types
    elif args.mode == "debug":
        models = args.models.split(",") if args.models else [PILOT_MODELS[0]]
        conditions = ["N"]
        runs = 1
        stim_types = args.types.split(",") if args.types else ["A1"]

    if args.models and args.mode != "debug":
        models = args.models.split(",")
    if args.types:
        stim_types = args.types.split(",")

    run_experiment(
        models=models,
        stimulus_types=stim_types,
        prompt_conditions=conditions,
        runs=runs,
        lang=args.lang,
        output_dir=output_dir,
        delay=args.delay,
    )

    print(f"\nДанные собраны. Следующий шаг:")
    print(f"  python main.py evaluate --input {output_dir / 'results.jsonl'} --output {output_dir}")


def cmd_evaluate(args):
    from evaluate import run_evaluation

    output_dir = Path(args.output)
    setup_logging(output_dir)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Файл не найден: {input_path}")
        sys.exit(1)

    run_evaluation(
        results_path=input_path,
        output_dir=output_dir,
        use_llm=not args.no_llm,
        delay=args.delay,
    )

    print(f"\nОценка завершена. Следующий шаг:")
    print(f"  python main.py analyze --input {output_dir} --output {output_dir}")


def cmd_analyze(args):
    from analyze import run_analysis

    output_dir = Path(args.output)
    setup_logging(output_dir)
    input_dir = Path(args.input)

    run_analysis(input_dir=input_dir, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Boundary Test: эксперимент по архитектурному самовосприятию LLM"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Сбор данных")
    p_run.add_argument("--mode", choices=["pilot", "full", "debug"], default="pilot")
    p_run.add_argument("--output", default="results/pilot")
    p_run.add_argument("--lang", choices=["en", "ru"], default="en")
    p_run.add_argument("--models", type=str, default=None, help="model_ids через запятую")
    p_run.add_argument("--types", type=str, default=None, help="A1,A2,B,C,D через запятую")
    p_run.add_argument("--delay", type=float, default=2.0)

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="Автоматическая оценка")
    p_eval.add_argument("--input", required=True, help="Путь к results.jsonl")
    p_eval.add_argument("--output", default="results/pilot")
    p_eval.add_argument("--no-llm", action="store_true", help="Только лексический анализ, без LLM-coder")
    p_eval.add_argument("--delay", type=float, default=1.5)

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", help="Статистический анализ")
    p_analyze.add_argument("--input", required=True, help="Директория с eval_combined.csv")
    p_analyze.add_argument("--output", default="results/pilot")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "analyze":
        cmd_analyze(args)


if __name__ == "__main__":
    main()
