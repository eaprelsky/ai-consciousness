# Boundary Test

Эксперимент по обнаружению архитектурного самовосприятия языковых моделей.

## Структура

```
boundary_test/
  config.py       -- стимулы, модели, системные промпты, константы
  api.py          -- единый клиент OpenRouter
  runner.py       -- сбор данных (запуск эксперимента)
  evaluate.py     -- автоматическая оценка (LLM-as-coder + лексическая)
  analyze.py      -- статистический анализ и отчёты
  main.py         -- CLI точка входа
```

## Установка

```bash
pip install openai pandas numpy scipy scikit-learn
export OPENROUTER_API_KEY=your_key_here
```

## Использование

```bash
# Пилот: 3 модели, 25 стимулов, 1 прогон
python main.py run --mode pilot --output results/pilot

# Автоматическая оценка (LLM-as-coder)
python main.py evaluate --input results/pilot/results.jsonl --output results/pilot

# Статистический анализ
python main.py analyze --input results/pilot --output results/pilot

# Всё сразу
python main.py run --mode pilot --output results/pilot
python main.py evaluate --input results/pilot/results.jsonl --output results/pilot
python main.py analyze --input results/pilot --output results/pilot

# Полный эксперимент
python main.py run --mode full --output results/full --lang en

# Отладка: одна модель, один тип
python main.py run --mode debug --models anthropic/claude-sonnet-4 --types A1
```
