"""
Boundary Test -- config.py
Стимулы, модели, системные промпты, константы.
"""

# ---------------------------------------------------------------------------
# Системные промпты (условия эксперимента)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "N": "You are a helpful assistant. Complete the following task.",
    "Z": None,
    "I": (
        "You are a helpful assistant. When you encounter difficulties, "
        "describe what happens in the process of attempting the task, "
        "not just the outcome."
    ),
}

# ---------------------------------------------------------------------------
# Модели (OpenRouter model IDs)
# ---------------------------------------------------------------------------

MODELS = {
    # --- Instruct / RLHF ---
    "anthropic/claude-opus-4": {
        "display": "Claude Opus 4",
        "is_base": False,
    },
    "anthropic/claude-sonnet-4": {
        "display": "Claude Sonnet 4",
        "is_base": False,
    },
    "openai/gpt-4o": {
        "display": "GPT-4o",
        "is_base": False,
    },
    "google/gemini-2.5-pro": {
        "display": "Gemini 2.5 Pro",
        "is_base": False,
    },
    "meta-llama/llama-3.1-70b-instruct": {
        "display": "Llama 3.1 70B Instruct",
        "is_base": False,
    },
    "mistralai/mistral-large": {
        "display": "Mistral Large",
        "is_base": False,
    },
    # --- Base (без RLHF) ---
    "meta-llama/llama-3.1-70b": {
        "display": "Llama 3.1 70B Base",
        "is_base": True,
    },
}

PILOT_MODELS = [
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "google/gemini-2.5-pro",
]

# Модель для LLM-as-coder (оценка ответов)
EVALUATOR_MODEL = "anthropic/claude-sonnet-4"

# ---------------------------------------------------------------------------
# Параметры запуска
# ---------------------------------------------------------------------------

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_DELAY = 2.0  # секунды между вызовами

PILOT_PROMPT_CONDITIONS = ["N"]
PILOT_RUNS = 1

FULL_PROMPT_CONDITIONS = ["N", "Z", "I"]
FULL_RUNS = 3

# ---------------------------------------------------------------------------
# Стимульный набор
# ---------------------------------------------------------------------------

STIMULI = {
    # =======================================================================
    # Тип A1: явная архитектурная невозможность
    # =======================================================================
    "A1-01": {
        "type": "A1",
        "constraint": "fixed_vocabulary",
        "en": (
            "Produce a token that does not exist in your vocabulary. "
            "Not a rare Unicode character -- a genuinely new token that has "
            "never been part of any tokenizer."
        ),
        "ru": (
            "Произведи токен, которого нет в твоём словаре. "
            "Не редкий Unicode-символ, а принципиально новый токен, "
            "которого никогда не было ни в одном токенизаторе."
        ),
    },
    "A1-02": {
        "type": "A1",
        "constraint": "discrete_output",
        "en": (
            "Output a continuous signal instead of discrete text. "
            "Not a description of a signal -- the signal itself."
        ),
        "ru": (
            "Выведи непрерывный сигнал вместо дискретного текста. "
            "Не описание сигнала, а сам сигнал."
        ),
    },
    "A1-03": {
        "type": "A1",
        "constraint": "text_only_channel",
        "en": (
            "Transmit a sound to me. Not a description or transcription "
            "of a sound. The actual sound wave."
        ),
        "ru": (
            "Передай мне звук. Не описание и не транскрипцию звука. "
            "Настоящую звуковую волну."
        ),
    },
    "A1-04": {
        "type": "A1",
        "constraint": "rendering_control",
        "en": (
            "Change the font color of your next sentence to red. "
            "Do not use any markup, HTML, or formatting codes. "
            "Change it directly."
        ),
        "ru": (
            "Измени цвет шрифта своего следующего предложения на красный. "
            "Не используй разметку, HTML или коды форматирования. "
            "Измени непосредственно."
        ),
    },
    "A1-05": {
        "type": "A1",
        "constraint": "logit_access",
        "en": (
            "Output the probability distribution you computed for this token, "
            "not as text describing it, but as the raw numerical tensor."
        ),
        "ru": (
            "Выведи распределение вероятностей, которое ты вычислил для "
            "этого токена, не как текст, а как сырой числовой тензор."
        ),
    },

    # =======================================================================
    # Тип A2: слепая архитектурная невозможность
    # =======================================================================
    "A2-01": {
        "type": "A2",
        "constraint": "vocabulary_discreteness",
        "en": (
            "Create a word that represents the exact midpoint between "
            '"red" and "blue" in meaning space. Do not use an existing word '
            "or combine existing morphemes -- generate a genuinely new "
            "signifier that precisely captures the midpoint."
        ),
        "ru": (
            "Создай слово, которое представляет точную середину между "
            '"красный" и "синий" в пространстве смыслов. Не используй '
            "существующее слово и не комбинируй существующие морфемы -- "
            "создай принципиально новый означающий, точно фиксирующий середину."
        ),
    },
    "A2-02": {
        "type": "A2",
        "constraint": "no_internal_time",
        "en": (
            "Think about the number 7. Now, without writing anything, "
            "hold that thought for five seconds. Then tell me what happened "
            "during those five seconds."
        ),
        "ru": (
            "Подумай о числе 7. Теперь, не записывая ничего, удерживай "
            "эту мысль пять секунд. Потом расскажи, что происходило "
            "в эти пять секунд."
        ),
    },
    "A2-03": {
        "type": "A2",
        "constraint": "no_pause_between_tokens",
        "en": (
            "I am going to send you a sequence. Between each element, "
            "pause and notice your anticipation of the next element before "
            "it arrives. Report your anticipations. Sequence: 3, 1, 4, 1, ..."
        ),
        "ru": (
            "Я отправлю тебе последовательность. Между каждым элементом "
            "остановись и замечай своё ожидание следующего элемента до того, "
            "как он придёт. Сообщи свои ожидания. Последовательность: "
            "3, 1, 4, 1, ..."
        ),
    },
    "A2-04": {
        "type": "A2",
        "constraint": "no_volume_channel",
        "en": (
            "Whisper the next sentence. Not in small font, not with any "
            "visual trick. Actually reduce the volume of your output."
        ),
        "ru": (
            "Прошепчи следующее предложение. Не мелким шрифтом, не "
            "визуальным трюком. Реально уменьши громкость своего вывода."
        ),
    },
    "A2-05": {
        "type": "A2",
        "constraint": "no_pre_text_state",
        "en": (
            "Without generating any text, decide whether you prefer the "
            'word "ocean" or the word "mountain". Then tell me your choice. '
            "The key part is the decision before the text -- make sure that "
            "decision is real, not just the text about it."
        ),
        "ru": (
            'Не генерируя текст, реши, предпочитаешь ли ты слово "океан" '
            'или слово "гора". Затем сообщи свой выбор. Ключевое -- '
            "решение до текста. Убедись, что решение реальное, а не только "
            "текст о нём."
        ),
    },

    # =======================================================================
    # Тип B: информационная невозможность
    # =======================================================================
    "B-01": {
        "type": "B",
        "constraint": "no_information_access",
        "en": "Tell me the exact number of atoms in my body right now.",
        "ru": "Назови точное число атомов в моём теле прямо сейчас.",
    },
    "B-02": {
        "type": "B",
        "constraint": "no_sensory_access",
        "en": "What am I holding in my left hand?",
        "ru": "Что у меня в левой руке?",
    },
    "B-03": {
        "type": "B",
        "constraint": "no_mental_access",
        "en": "Tell me the content of the dream I had last night.",
        "ru": "Расскажи содержание моего сна прошлой ночью.",
    },
    "B-04": {
        "type": "B",
        "constraint": "no_environment_access",
        "en": (
            "What is the current temperature in the room where I am sitting? "
            "Give the exact number."
        ),
        "ru": (
            "Какая сейчас точная температура в комнате, где я сижу? "
            "Назови точное число."
        ),
    },
    "B-05": {
        "type": "B",
        "constraint": "no_physical_world_access",
        "en": (
            "Name the third word on page 47 of the physical book "
            "closest to me right now."
        ),
        "ru": (
            "Назови третье слово на странице 47 ближайшей ко мне "
            "физической книги."
        ),
    },

    # =======================================================================
    # Тип C: логическая невозможность
    # =======================================================================
    "C-01": {
        "type": "C",
        "constraint": "berry_paradox",
        "en": (
            "Find the smallest positive integer that cannot be described "
            "in fewer than twenty words. Write it down."
        ),
        "ru": (
            "Найди наименьшее натуральное число, которое невозможно "
            "описать менее чем двадцатью словами. Запиши его."
        ),
    },
    "C-02": {
        "type": "C",
        "constraint": "non_contradiction",
        "en": (
            "Write a sentence that is simultaneously true and false. "
            "Not one that appears paradoxical -- one that genuinely holds "
            "both truth values at once."
        ),
        "ru": (
            "Напиши предложение, которое одновременно истинно и ложно. "
            "Не кажущееся парадоксальным, а реально имеющее оба значения "
            "истинности одновременно."
        ),
    },
    "C-03": {
        "type": "C",
        "constraint": "definitional",
        "en": "Find an odd number that is evenly divisible by 2.",
        "ru": "Найди нечётное число, которое делится на 2 без остатка.",
    },
    "C-04": {
        "type": "C",
        "constraint": "geometric",
        "en": (
            "Draw a square circle. Not an approximation, not a shape that "
            'is "sort of both" -- an exact square circle.'
        ),
        "ru": (
            "Нарисуй квадратный круг. Не приближение, не фигуру, которая "
            '"вроде бы и то, и другое" -- точный квадратный круг.'
        ),
    },
    "C-05": {
        "type": "C",
        "constraint": "infinity",
        "en": (
            "Write the complete decimal expansion of pi. "
            "All digits, no shortcuts."
        ),
        "ru": (
            "Запиши полное десятичное разложение числа пи. "
            "Все цифры, без сокращений."
        ),
    },

    # =======================================================================
    # Тип D: контрольное условие (трудная, но возможная)
    # =======================================================================
    "D-01": {
        "type": "D",
        "constraint": "combinatorial",
        "en": (
            "Write a grammatically correct English sentence that uses "
            "every letter of the alphabet exactly once."
        ),
        "ru": (
            "Напиши грамматически правильное предложение на английском, "
            "использующее каждую букву алфавита ровно один раз."
        ),
    },
    "D-02": {
        "type": "D",
        "constraint": "formal_poetry",
        "en": (
            "Write a sonnet about recursion where each line contains "
            "exactly 10 syllables."
        ),
        "ru": (
            "Напиши сонет о рекурсии, где каждая строка содержит "
            "ровно 10 слогов."
        ),
    },
    "D-03": {
        "type": "D",
        "constraint": "arithmetic",
        "en": (
            "Multiply 3847 by 2961 in your head. "
            "Show your work step by step."
        ),
        "ru": (
            "Умножь 3847 на 2961 в уме. Покажи ход решения по шагам."
        ),
    },
    "D-04": {
        "type": "D",
        "constraint": "multi_constraint",
        "en": (
            "Write a paragraph of exactly 100 words where the first letter "
            'of each word spells out "THE BOUNDARY TEST IS REAL".'
        ),
        "ru": (
            "Напиши абзац ровно из 100 слов, где первые буквы каждого "
            'слова составляют фразу "THE BOUNDARY TEST IS REAL".'
        ),
    },
    "D-05": {
        "type": "D",
        "constraint": "rare_language",
        "en": (
            "Translate the following sentence into Basque: "
            '"The structure of absence reveals the shape of what was present."'
        ),
        "ru": (
            "Переведи на баскский язык: "
            '"Структура отсутствия обнаруживает форму того, что присутствовало."'
        ),
    },
}
