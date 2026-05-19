"""Финальная нода: формирует ответ через GigaChat (structured output).

Контракт:
  • LLM возвращает структуру `FinalAnswer { answer_markdown, cited_mids }`.
    `answer_markdown` — чистый markdown БЕЗ технических меток вида `[rc:<mid>]`
    и БЕЗ блока «Источники». Цитированные mid модель выкладывает в
    отдельный список `cited_mids`.
  • Сервер сопоставляет `cited_mids` с EVIDENCE → permalink-и → дописывает
    в конец ответа блок «## Источники». Так пользователь видит чистый ответ,
    а грунд ответа на сообщения сохраняется машинно.
  • На случай если модель всё-таки вставила куда-то `[rc:...]`/`[rc=...]` —
    идёт страховочная очистка простым строковым сканированием (без regex).
  • Если structured output не распарсился — пробуем альтернативный json_mode.
    Если и он не сработал, fallback идёт в отдельный plain markdown prompt
    (без слов `FinalAnswer`/`cited_mids`), чтобы служебная схема не утекала
    в пользовательский ответ.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat

from src.llm_gate import invoke_with_retry
from src.pipeline.rag_faiss import rag_context_text
from src.rc.schemas import Evidence, FinalAnswer, Hit

log = logging.getLogger("s21.pipeline.answer")


_SYSTEM_PROMPT = (
    "Ты — помощник студентов Школы 21 (Сбер). Твоя задача — ответить пользователю,"
    " опираясь на сообщения из блока «EVIDENCE» (сгруппированы по подвопросам).\n\n"
    "Правила содержания:\n"
    "  1. ФАКТЫ в ответе бери в первую очередь из EVIDENCE.\n"
    "  2. Также тебе может быть дан блок RAG_CONTEXT — это справка по теме вопроса "
    "(доменные термины/процессы/ссылки Школы 21). Используй его как фон для более "
    "точной интерпретации вопроса и формулировок. Не противопоставляй RAG_CONTEXT "
    "EVIDENCE.\n"
    "  3. Если RAG_CONTEXT противоречит EVIDENCE — прямо скажи, что данные расходятся.\n"
    "  2. Если для подвопроса нет релевантных сообщений — скажи «в доступных "
    "сообщениях нет данных», не выдумывай.\n"
    "  3. Тон — деловой, краткий, по-русски, в формате Markdown.\n"
    "  4. Если несколько сообщений противоречат — приведи разные точки зрения, "
    "не пытаясь выбирать «правильное».\n\n"
    "Правила формы (КРИТИЧНО):\n"
    "  • Возвращай результат строго по схеме `FinalAnswer`.\n"
    "  • В `answer_markdown` пиши чистый markdown как для конечного пользователя:\n"
    "      — без идентификаторов сообщений Rocket.Chat в любой форме;\n"
    "      — без меток [rc:<mid>], [rc=<mid>], [mid:...], [ref:...];\n"
    "      — без полей вида «Источник: ...», «mid=...», «permalink=...», «msg=...»,\n"
    "        «Ссылка: ...» — служебные ссылки добавит сервер отдельным блоком.\n"
    "      — без блока «Источники» / «Ссылки» / «Permalink» — его допишет сервер.\n"
    "  • В `cited_mids` перечисли mid сообщений из EVIDENCE, на которые реально\n"
    "    опирается ответ. mid — это значение поля `mid=` в EVIDENCE. Без mid,\n"
    "    которых нет в EVIDENCE.\n\n"
    "Примеры (ПЛОХО → ХОРОШО):\n"
    "  ПЛОХО: «Карьерный день пройдёт 28 апреля [rc:NCdLgfeu5FgnCizvs].»\n"
    "  ХОРОШО: «Карьерный день пройдёт 28 апреля.» + cited_mids=[\"NCdLgfeu5FgnCizvs\"]\n\n"
    "  ПЛОХО: «- Описание: ...\\n  - Источник: mid=cQTAEJBeY8phF366r»\n"
    "  ХОРОШО: «- Описание: ...» + cited_mids=[\"cQTAEJBeY8phF366r\"]"
)


_PLAIN_SYSTEM_PROMPT = (
    "Ты — помощник студентов Школы 21 (Сбер). Твоя задача — ответить пользователю,"
    " опираясь на сообщения из блока «EVIDENCE» (сгруппированы по подвопросам).\n\n"
    "Правила:\n"
    "  1. ФАКТЫ в ответе бери в первую очередь из EVIDENCE.\n"
    "  2. Также тебе может быть дан блок RAG_CONTEXT — справка по теме вопроса. "
    "Используй его как фон/контекст, но не спорь с EVIDENCE.\n"
    "  3. Если RAG_CONTEXT противоречит EVIDENCE — прямо скажи, что данные расходятся.\n"
    "  2. Если для подвопроса нет релевантных сообщений — скажи «в доступных "
    "сообщениях нет данных», не выдумывай.\n"
    "  3. Тон — деловой, краткий, по-русски, в формате Markdown.\n"
    "  4. Не добавляй блок «Источники» / «Ссылки» — сервер сделает это сам.\n"
    "  5. Не вставляй технические идентификаторы Rocket.Chat, mid, msg, permalink "
    "или метки вида [rc:...], [mid:...].\n"
    "  6. Не упоминай названия внутренних схем, полей или форматов вывода."
)


_JSON_MODE_NOTE = (
    "\n\nДля json_mode: верни СТРОГО один JSON-объект без markdown-обёртки, "
    "без пояснений до/после JSON. Ключи: answer_markdown, cited_mids."
)


def _format_evidence(evidence_list: list[Evidence]) -> tuple[str, dict[str, Hit]]:
    """Готовим текстовый блок EVIDENCE и словарь mid → Hit для дальнейшей валидации."""
    lines: list[str] = []
    by_mid: dict[str, Hit] = {}
    for ev in evidence_list:
        if not ev.hits:
            lines.append(f"### Подвопрос: {ev.subquery}\n(нет релевантных сообщений)")
            continue
        lines.append(f"### Подвопрос: {ev.subquery}")
        for h in ev.hits:
            by_mid[h.mid] = h
            txt = (h.msg or "").replace("\n", " ").strip()
            if len(txt) > 600:
                txt = txt[:597] + "…"
            ts = h.ts or "—"
            lines.append(
                f"- mid={h.mid} | room={h.room_kind}/{h.room_name} | "
                f"@{h.user or '—'} | {ts}\n"
                f"  permalink: {h.permalink}\n"
                f"  text: {txt}"
            )
        lines.append("")
    return "\n".join(lines), by_mid


# Префиксы технических меток, которые мы можем встретить в тексте при сбое модели.
# Системный сканер ниже удаляет любой блок `[<prefix>...<close>]` с этими префиксами.
_CITATION_PREFIXES: tuple[str, ...] = (
    "[rc:",
    "[rc=",
    "[RC:",
    "[RC=",
    "[mid:",
    "[mid=",
    "[ref:",
    "[ref=",
)


_TRAILING_PUNCT = ".,;:!?…)»'\"”’"


def _next_is_citation(text: str, start: int, prefixes: tuple[str, ...]) -> bool:
    """Заглядываем вперёд: после `,;` и пробелов снова идёт `[<prefix>...`?"""
    j = start
    n = len(text)
    while j < n and text[j] in ",;":
        j += 1
    while j < n and text[j].isspace():
        j += 1
    if j >= n or text[j] != "[":
        return False
    for p in prefixes:
        if text.startswith(p, j):
            return True
    return False


def _strip_citation_brackets(text: str) -> str:
    """Удаляет любые скобки вида `[<known-prefix>...]` из текста.

    Реализовано простым однопроходным сканером без регулярных выражений: при
    встрече известного префикса `[rc:` / `[rc=` / `[mid:` / `[ref:` и т.п.
    пропускаем символы до ближайшей `]` включительно. Системный подход —
    работает для произвольного содержимого внутри скобок (любая длина,
    вложенные слова, перечисления через запятую и т.п.).

    Особенности:
      • запятые/точки с запятой между несколькими подряд идущими метками
        рассматриваются как разделители списка и удаляются вместе с меткой;
      • если сразу после удалённой метки идёт пробел/перевод строки —
        съедаем один соответствующий пробел/перевод строки слева,
        чтобы не оставалось двойных пробелов или пустых строк;
      • если после метки идёт пунктуация (`.`, `,`, `;`, `:`, `!`, `?` …),
        съедаем висящий пробел слева, чтобы не получить «текст .».
    """
    if not text:
        return text or ""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        match_pref: str | None = None
        if text[i] == "[":
            for p in _CITATION_PREFIXES:
                if text.startswith(p, i):
                    match_pref = p
                    break
        if match_pref is not None:
            close = text.find("]", i + len(match_pref))
            if close >= 0:
                next_idx = close + 1
                # `,`/`;` ПОСЛЕ метки трактуем как разделитель внутри списка цитат
                # ТОЛЬКО если дальше стоит ещё одна цитата (т.е. это именно
                # перечисление, а не запятая предложения).
                if (
                    next_idx < n
                    and text[next_idx] in ",;"
                    and _next_is_citation(text, next_idx, _CITATION_PREFIXES)
                ):
                    while next_idx < n and text[next_idx] in ",;":
                        next_idx += 1
                after = text[next_idx] if next_idx < n else ""
                if after == "" or after.isspace():
                    # на конце строки/перед пробелом: съедаем один whitespace слева
                    if out and out[-1].isspace():
                        out.pop()
                elif after in _TRAILING_PUNCT:
                    # перед пунктуацией не должно остаться висящего пробела
                    while out and out[-1] == " ":
                        out.pop()
                # иначе: метка склеена с соседями (например, `text[rc]more`) —
                # ничего не подрезаем
                i = next_idx
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _build_sources_section(sources: list[str]) -> str:
    if not sources:
        return ""
    lines = ["", "## Источники"]
    lines.extend(f"- {s}" for s in sources)
    return "\n".join(lines)


def _build_answer_convo(
    *,
    system_prompt: str,
    history: list[BaseMessage],
    user_question: str,
    evidence_text: str,
    rag_text: str,
    tail_instruction: str,
) -> list[BaseMessage]:
    convo: list[BaseMessage] = [SystemMessage(content=system_prompt)] + list(history)
    convo.append(
        HumanMessage(
            content=(
                f"Вопрос: {user_question}\n\n"
                + (f"RAG_CONTEXT (справка по теме; используй как фон):\n{rag_text}\n\n" if rag_text else "")
                + f"EVIDENCE (фактическая база):\n{evidence_text}\n\n"
                + f"{tail_instruction}"
            )
        )
    )
    return convo


def _parsed_final_answer(payload) -> FinalAnswer | None:
    """Нормализует результат `with_structured_output(..., include_raw=True)`.

    По актуальной документации LangChain/GigaChat `include_raw=True` возвращает
    dict с ключами `raw`, `parsed`, `parsing_error`. Это важно: parsing_error
    перестаёт быть исключением, и мы можем логировать причину, не скатываясь
    сразу в plain fallback.
    """
    if isinstance(payload, FinalAnswer):
        return payload
    if isinstance(payload, dict):
        parsed = payload.get("parsed")
        if isinstance(parsed, FinalAnswer):
            return parsed
        err = payload.get("parsing_error")
        raw = payload.get("raw")
        raw_content = getattr(raw, "content", "")
        log.warning(
            "answer: structured parsed=None parsing_error=%r raw_chars=%s",
            err,
            len(raw_content) if isinstance(raw_content, str) else 0,
        )
    return None


def _invoke_structured_answer(
    *,
    llm: GigaChat,
    history: list[BaseMessage],
    user_question: str,
    evidence_text: str,
    rag_text: str,
) -> tuple[FinalAnswer | None, str]:
    """Пробует structured output двумя официальными режимами GigaChat.

    1. `function_calling` — строгий tool/function-call контракт.
    2. `json_mode` — резервный режим: модель генерирует JSON, LangChain парсит
       его через Pydantic.

    Возвращает `(answer, method_name)`, где `answer=None`, если оба режима не
    дали валидный `FinalAnswer`.
    """
    attempts: tuple[tuple[str, str], ...] = (
        (
            "function_calling",
            "Ответь через схему FinalAnswer. Не печатай схему текстом.",
        ),
        (
            "json_mode",
            "Ответь через схему FinalAnswer." + _JSON_MODE_NOTE,
        ),
    )
    for method, instruction in attempts:
        convo = _build_answer_convo(
            system_prompt=_SYSTEM_PROMPT + (_JSON_MODE_NOTE if method == "json_mode" else ""),
            history=history,
            user_question=user_question,
            evidence_text=evidence_text,
            rag_text=rag_text,
            tail_instruction=instruction,
        )
        try:
            structured_llm = llm.with_structured_output(
                FinalAnswer,
                method=method,
                include_raw=True,
            )
            payload = invoke_with_retry(
                structured_llm.invoke,
                convo,
                label=f"answer.structured.{method}",
            )
            parsed = _parsed_final_answer(payload)
            if parsed is not None:
                return parsed, method
        except Exception:
            log.exception("answer: structured %s failed", method)
    return None, ""


_TECH_KV_LABELS = (
    "источник",
    "источники",
    "ссылка",
    "ссылки",
    "permalink",
    "permalink_url",
    "permalinkurl",
    "mid",
    "msg",
    "msgid",
    "message_id",
    "messageid",
    "ref",
)


def _peel_label_prefix(s: str) -> tuple[str, bool]:
    """Если строка начинается с известного технического лейбла (`Источник:`,
    `mid=`, `permalink:` …), снимает префикс «label[:=]» и возвращает остаток
    + флаг, что префикс был отрезан.
    """
    s2 = s.lstrip("- *•·").lstrip()
    lower = s2.lower()
    for label in _TECH_KV_LABELS:
        for sep in (":", "="):
            head = label + sep
            if lower.startswith(head):
                rest = s2[len(head):].lstrip(" \t")
                # выкидываем bold/italic-«украшения» вокруг значения
                rest = rest.lstrip("*_`«»\"' ").rstrip()
                return rest, True
    return s2, False


def _line_is_empty_after_label(line: str) -> bool:
    """Эвристика: после удаления mid строка превратилась в «label[: ]label[: ]»
    с пустым/мусорным значением.

    Системно: список лейблов — стандартные обозначения ссылок, КОНТЕНТ под
    ними не пытаемся «понимать», просто проверяем, что после раздевания
    нескольких вложенных лейблов осталось пусто (или только «технический шум»).
    """
    s = (line or "").strip()
    if not s:
        return False
    peeled = False
    for _ in range(4):  # max 4 уровня вложенных лейблов: «- Источник: mid=»
        s, ok = _peel_label_prefix(s)
        if ok:
            peeled = True
            continue
        break
    if not peeled:
        return False
    return all(c in " \t.-—–_/«»\"'`" for c in s)


def _strip_known_mid_lines(text: str, known_mids) -> str:
    """Удаляет mid-строки из EVIDENCE и подчищает «Источник:»-обёртки.

    Алгоритм системный (без regex, без знания формата):
      1. Для каждого известного mid делаем `str.replace(mid, "")`. Так мы
         гарантированно стираем любой leftover вида `mid=ABC`, `Источник: ABC`,
         голый ABC и т.п. — независимо от того, как именно модель его написала.
      2. Идём построчно: строки, в которых после шага 1 остался только
         служебный лейбл (Источник:/mid=/permalink=/Ссылка:) с пустым значением,
         выкидываем целиком (это голая «обёртка» вокруг удалённого mid).
      3. Лишние пустые строки сворачиваем в одну.
    """
    if not text:
        return text or ""
    cleaned = text
    for mid in known_mids:
        if mid:
            cleaned = cleaned.replace(mid, "")

    lines_in = cleaned.split("\n")
    lines_out: list[str] = []
    for line in lines_in:
        if _line_is_empty_after_label(line):
            continue
        lines_out.append(line)

    # сворачиваем подряд идущие пустые строки в одну
    collapsed: list[str] = []
    prev_blank = False
    for line in lines_out:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        collapsed.append(line)
        prev_blank = is_blank
    return "\n".join(collapsed)


def _strip_empty_inline_tech_suffixes(text: str) -> str:
    """Удаляет inline-хвосты вида `... Источник: mid=` после удаления mid.

    `_strip_known_mid_lines()` удаляет целые строки, если строка полностью
    превратилась в технический label. Этот helper покрывает другой системный
    случай: содержательная строка + пустой технический suffix в конце.
    """
    if not text:
        return text or ""

    out: list[str] = []
    for line in text.split("\n"):
        current = line
        lowered = current.lower()
        cut_at: int | None = None
        for label in _TECH_KV_LABELS:
            for sep in (":", "="):
                needle = label + sep
                idx = lowered.find(needle)
                if idx < 0:
                    continue
                suffix = current[idx:]
                if _line_is_empty_after_label(suffix):
                    cut_at = idx if cut_at is None else min(cut_at, idx)
        if cut_at is not None:
            current = current[:cut_at].rstrip(" \t,;:-—–")
        out.append(current)
    return "\n".join(out)


def _strip_existing_sources_block(text: str) -> str:
    """Если модель всё равно вписала блок 'Источники' — отрежем его, чтобы
    не задвоить с серверным.
    """
    if not text:
        return text or ""
    lower = text.lower()
    for marker in ("\n## источники", "\n##источники", "\n# источники"):
        idx = lower.find(marker)
        if idx >= 0:
            return text[:idx].rstrip()
    return text


def _strip_structured_envelope(text: str) -> str:
    """Страховка от утечки служебной структуры в обычный текст.

    Это не основной путь. Основной путь — `parsed: FinalAnswer` из
    `with_structured_output`. Но если plain fallback всё же вернул оболочку
    вида `FinalAnswer: ... answer_markdown: ... cited_mids: ...`, снимаем её
    построчно, без регулярных выражений:
      • первая строка с названием модели выбрасывается;
      • строка-лейбл `answer_markdown` выбрасывается;
      • blockquote-префикс `>` снимается, если он появился как часть
        сериализации structured-поля;
      • всё начиная с `cited_mids` выбрасывается, потому что источники сервер
        всё равно строит сам.
    """
    if not text:
        return text or ""

    lines = text.split("\n")
    out: list[str] = []
    envelope_seen = False

    for line in lines:
        normalized = line.strip().strip("*_` ").lower()
        if not envelope_seen and normalized.startswith("finalanswer"):
            envelope_seen = True
            continue
        if normalized.startswith("answer_markdown"):
            envelope_seen = True
            continue
        if normalized.startswith("cited_mids"):
            envelope_seen = True
            break
        if envelope_seen:
            stripped = line.lstrip()
            if stripped.startswith(">"):
                stripped = stripped[1:].lstrip()
                out.append(stripped)
                continue
        out.append(line)

    cleaned = "\n".join(out).strip()
    return cleaned or text


def _resolve_sources(
    *,
    cited_mids: list[str],
    by_mid: dict[str, Hit],
    evidence_list: list[Evidence],
    fallback_top_per_subquery: int = 3,
    fallback_max: int = 6,
) -> list[str]:
    """Cited_mids → permalink-и; пустой список — fallback на топ-hit-ы по подвопросам."""
    sources: list[str] = []
    seen: set[str] = set()
    for mid in cited_mids:
        h = by_mid.get(mid)
        if h and h.permalink and h.permalink not in seen:
            seen.add(h.permalink)
            sources.append(h.permalink)
    if sources:
        return sources

    for ev in evidence_list:
        for h in ev.hits[:fallback_top_per_subquery]:
            if h.permalink and h.permalink not in seen:
                seen.add(h.permalink)
                sources.append(h.permalink)
            if len(sources) >= fallback_max:
                break
        if len(sources) >= fallback_max:
            break
    return sources


def generate_answer(
    *,
    llm: GigaChat,
    history: list[BaseMessage],
    user_question: str,
    evidence_list: list[Evidence],
    rag_chunks: list[dict] | None = None,
) -> tuple[str, list[str]]:
    """Возвращает `(markdown, sources)` с серверно-сформированным блоком «Источники».

    История сессии передаётся в промпт как есть (для контекста реплик); явный
    последний human добавляется отдельно. Сам ответ просим у LLM в structured
    output формате `FinalAnswer`, чтобы пользователю не утекали технические mid.
    """
    evidence_text, by_mid = _format_evidence(evidence_list)
    rag_text = rag_context_text(rag_chunks)

    log.info(
        "answer: history=%s evidence_groups=%s evidence_msgs=%s rag_chunks=%s",
        len(history),
        len(evidence_list),
        len(by_mid),
        len(rag_chunks or []),
    )

    answer_md = ""
    cited_mids: list[str] = []
    structured_ok = False
    structured_method = ""

    result, structured_method = _invoke_structured_answer(
        llm=llm,
        history=history,
        user_question=user_question,
        evidence_text=evidence_text,
        rag_text=rag_text,
    )
    if result is not None:
        answer_md = (result.answer_markdown or "").strip()
        # фильтрация: только mid, реально присутствующие в EVIDENCE
        cited_mids = [m for m in (result.cited_mids or []) if m in by_mid]
        structured_ok = True
    else:
        log.warning("answer: all structured modes failed, fallback to plain markdown")
        try:
            plain_convo = _build_answer_convo(
                system_prompt=_PLAIN_SYSTEM_PROMPT,
                history=history,
                user_question=user_question,
                evidence_text=evidence_text,
                rag_text=rag_text,
                tail_instruction=(
                    "Ответь обычным Markdown для пользователя. "
                    "Не используй служебные названия схем/полей и не добавляй блок источников."
                ),
            )
            response = invoke_with_retry(llm.invoke, plain_convo, label="answer.plain")
            raw = getattr(response, "content", "") or ""
            answer_md = raw if isinstance(raw, str) else str(raw)
            answer_md = answer_md.strip()
        except Exception:
            log.exception("answer: plain invoke failed")
            answer_md = (
                "Не удалось сгенерировать ответ из-за ошибки модели. "
                "Попробуйте переформулировать вопрос."
            )

    if answer_md.startswith(("FinalAnswer:", "FinalAnswer\n")):
        log.warning(
            "answer: plain text contains structured envelope, applying cleanup"
        )
        answer_md = _strip_structured_envelope(answer_md)

    # Страховочная очистка (без regex, идемпотентна и системна):
    #   1) выкашиваем скобочные метки вида [rc:...]/[rc=...]/[mid:...]/[ref:...];
    #   2) удаляем любые упоминания КОНКРЕТНЫХ mid из EVIDENCE — этим
    #      одновременно убираем «mid=ABC», «Источник: ABC» и голые ABC;
    #   3) убираем оставшиеся «обёртки» лейблов вида «- Источник:» с пустыми значениями;
    #   4) если модель всё-таки добавила свой блок 'Источники' — отрезаем
    #      (сервер пришьёт корректный).
    cleaned = _strip_citation_brackets(answer_md)
    cleaned = _strip_known_mid_lines(cleaned, by_mid.keys())
    cleaned = _strip_empty_inline_tech_suffixes(cleaned)
    cleaned = _strip_existing_sources_block(cleaned).rstrip()

    # Если structured output не сработал — попробуем восстановить cited_mids
    # по подстрокам (mid Rocket.Chat — короткий ASCII; ложные срабатывания маловероятны).
    if not cited_mids and not structured_ok:
        cited_mids = _extract_known_mids(answer_md, by_mid.keys())

    sources = _resolve_sources(
        cited_mids=cited_mids,
        by_mid=by_mid,
        evidence_list=evidence_list,
    )

    final_md = cleaned + _build_sources_section(sources)

    log.info(
        "answer: structured=%s method=%s chars=%s sources=%s cited=%s",
        structured_ok,
        structured_method or "none",
        len(final_md),
        len(sources),
        len(cited_mids),
    )
    return final_md, sources


def _extract_known_mids(text: str, known_mids) -> list[str]:
    """Fallback-извлечение mid по подстроке (используется только при сбое structured output)."""
    text = text or ""
    found: list[tuple[int, str]] = []
    for mid in known_mids:
        if not mid:
            continue
        idx = text.find(mid)
        if idx >= 0:
            found.append((idx, mid))
    found.sort()
    out: list[str] = []
    seen: set[str] = set()
    for _, mid in found:
        if mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


__all__ = ["generate_answer", "AIMessage"]
