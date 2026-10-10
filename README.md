# BlueSec1 agent

Reference agent for the BlueSec1 competition. An LLM receives a security alert,
explores the evidence graph through the runtime tools and submits a verdict:
a real attack or a false positive.

Use it as a starting point. Change the prompt and the reasoning loop, keep the
client as is.

Итог разработки и запусков за 10 октября 2026 года:
[отчёт, результаты 40 задач, время, расход и ограничения](docs/reports/2026-10-10-bluesec-session.md).
Последний graded-прогон завершён: 40/40 `TaskResult`, оценка на момент проверки
скрыта; 22 ответа модели, 6 ответов координатора и 12 неполных fallback.

Подробно по каждой задаче: [разбор всех 40 расследований](docs/reports/2026-10-10-quality-tasks.md)
и [документ Word на 42 страницы](docs/reports/2026-10-10-quality-tasks.docx).

## Before you start

You need Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and an
OpenAI-compatible LLM provider with an API key.

Take the runtime address and your access key from the competition site:
https://bluesec.team/

## Quick start

```bash
uv sync
```

```bash
cp .env.example .env
```

Set `SCENARIO_RUNTIME_ENDPOINT` and `SCENARIO_RUNTIME_TOKEN` from the
competition site, and `LLM_BASE_URL`,
`LLM_API_KEY`, `LLM_DEFAULT_MODEL` from your provider. Set `AGENT_NAME` too —
it is shown next to your runs in the competition interface.
Set `SCENARIO_RUNTIME_ARENA` only when you need a specific arena; otherwise the
runtime selects the currently open default.

Start the agent with `--env-file` so gRPC receives the resolver setting from
the environment as well as the application settings.

```bash
uv run --env-file .env bluesec1-agent --max-tasks 1 --max-steps 60
```

This is a live request: the agent creates a remote run, investigates one task
and submits its verdict to the server. It prints the authoritative server-side
task result as JSON. The server owns scoring and leaderboard metrics; the local
files under `reports/runs/` are a redacted execution trace and result summary for
debugging, not a replacement for those server metrics. The CLI prints the exact
artifact directory to stderr when the run finishes.

Runs are bounded to one task by default. Increase the bound deliberately, for
example with `--max-tasks 3`. Use `--all-tasks` only when you intend to consume
the full available queue. `--max-tasks` and `--all-tasks` cannot be combined.
The per-task `--max-steps` limit accepts values from 1 through 100; when omitted,
the agent uses `BLUESEC_AGENT_MAX_STEPS` from the environment.

You may have only one run in flight. Starting a second one while the first is
still open is refused until the first finishes.

## What a task looks like

You get an alert: a suspicious process, a host, a few identifiers. The evidence
behind it is a graph — processes, files, network connections, registry writes and
the relations between them. Your job is to walk that graph, decide whether the
alert is a real attack, and report either the entities that need a response or
the graph evidence that proves the activity is legitimate.

The runtime is the source of truth for tools. Every task observation contains
`available_tools` with the currently enabled tool calls and the exact argument
schema for each one. Build agent calls from that catalog instead of relying on
a hard-coded list or contract.

Scoring rewards a correct verdict and the artifacts you found, and penalises
wasted tool calls. Investigating thoroughly with fewer calls beats brute force.

## What is inside

| Path | What it is |
| --- | --- |
| `src/bluesec1_agent/` | the agent: prompt, reasoning loop, tool schemas |
| `src/bluesec1_client/` | runtime client: retries, deadlines, idempotency, validation |
| `proto/` | the gRPC schema, if you write in another language |

Change `src/bluesec1_agent/`. Start with `SYSTEM_PROMPT` in `agent.py` and the
step budget in `.env`.

## Writing your own agent

The loop is short: take a task, read the tools it advertises, call them until one
returns a terminal result.

```python
import asyncio

from bluesec1_client import RemoteBenchmarkClient


async def main() -> None:
    async with RemoteBenchmarkClient(
        endpoint="bluesec.team:443",
        token="your-key",
        verify_tls=True,
        agent_name="my-agent",
        arena="practice",  # Optional; omit to use the open default arena.
    ) as client:
        while (session := await client.start_task()) is not None:
            print(session.task.observation["available_tools"])
            # Choose tool calls from the advertised catalog and use
            # session.call_tool(tool_name, arguments) until the task ends.


asyncio.run(main())
```

`start_task()` returns `None` when the queue is empty. A tool call that ends the
task carries `task_result` with your score for it.

## Another language

The runtime speaks gRPC. Generate a client from `proto/` and implement the
transport contract yourself: an idempotency key per lifecycle call, a call id per
tool call, and retries that reuse the same key.

## Checks

```bash
uv run pytest
```

```bash
uv run ruff check .
```

## Точка продолжения: 10 октября 2026

Рабочий проект перенесён в репозиторий `MurkaSelebry/bluesec1-agent`.
Папка `bluesec1-agent-example` оставлена как исходная копия. Это исследовательская
версия для проверки API, цикла расследования и обвязки, а не подтверждённое по
качеству решение соревнования.

Перенесены исходники, тесты, `.env`, `.env.example`, lockfile и все отчёты.
Виртуальное окружение пересоздано через `uv sync --locked`; старые `.git`, `.venv`
и кеши не копировались. Проверка переноса и список файлов находятся в
`reports/migration-2026-10-10.json`. Реальные журналы запусков лежат в
`reports/runs/` и исключены из Git.

### Что уже есть

- Автономный цикл с проверкой JSON-ответа, исправлением некорректного формата,
  ограничением шагов и задач, повторным использованием уже прочитанных данных.
- Каталог инструментов берётся из runtime; конкретная схема ответа модели пока
  описана фиксированными Pydantic-моделями. Это ограничение для будущих изменений API.
- Журнал `trace.jsonl`, итоговый `result.json`, закрытие run и остановка после ошибки.
- Из ответов среды извлекается `last_tool_result`, чтобы повторяющийся каталог
  инструментов не накапливался в контексте модели.

### Что показали тестовые прогоны

| Прогон | Результат |
| --- | --- |
| `1715b1f7-1edf-41dc-9c8f-600a04203283` | Вердикт `benign` принят, 6 вызовов среды, 61.325 с. `quality_score`, `efficiency_score`, `total_reward` равны 0. |
| `272f3753-4ef4-4f90-bbd6-2167827ae2f9` | Прерван до вердикта; в журнале 3 вызова среды. Перед переносом сервер подтвердил закрытие run. Итоговой оценки нет. |

Первый прогон проверил прохождение всего протокола, но не доказал качество
расследования: эталонный вердикт сервер не сообщил. Агент изучил родителей
`explorer.exe`, оставив другие прямые связи неисследованными. После этого инструкция
была уточнена. Во втором прогоне наблюдались длинные ответы LLM и исправление
некорректного JSON; полноценная проверка улучшенной версии не завершена.

Проверка провайдера сохранена в `reports/api-preflight-2026-10-10.json`:
15 из 17 модельных ID прошли короткие text/JSON-проверки, две DeepSeek вернули
HTTP 400. Это проверка совместимости API, а не сравнение качества моделей.

### Направления исследования на момент переноса

Этот список описывает ранний этап работы. Итоги последующих изменений и актуальные
следующие шаги собраны в [отчёте за день](docs/reports/2026-10-10-bluesec-session.md).

1. Приоритизация связей исходного процесса: исследовать важные входящие и исходящие
   события до обхода узлов пользователя или хоста с сотнями связей.
2. Проверка ссылок на доказательства: различать entity/relation IDs и проверять,
   что итоговый ответ ссылается только на реально полученные объекты и поля.
3. Измерение обвязки: сохранять расход токенов, задержки и исправления JSON отдельно
   от числа серверных вызовов. Сейчас сведения о токенах видны в консольных логах.
4. Несколько завершённых задач для оценки качества и сравнения моделей; два текущих
   прогона не являются достаточным сравнительным экспериментом.

Локальная проверка без новых запусков на BlueSec:

```bash
uv sync --locked
uv run pytest
uv run ruff check .
```

Когда потребуется следующий реальный эксперимент:

```bash
uv run --env-file .env bluesec1-agent --max-tasks 1 --max-steps 60
```

Эта команда создаёт новый run и позволяет агенту самостоятельно отправить вердикт.

План следующей итерации: [команда агентов с goals и независимым ревьювером](docs/superpowers/plans/2026-10-10-goal-team.md).
В нём зафиксированы роли, границы изменений, проверки и ограниченная серия экспериментов.
Для запуска команды в отдельных задачах подготовлены [готовые промпты и порядок запуска](docs/superpowers/prompts/2026-10-10-goal-team/README.md).

## Цикл улучшения 01: harness v2

CLI по умолчанию использует `v2`: хранит полный граф полученных доказательств,
передаёт модели ограниченное представление с источниками и пропусками, проверяет
известность ID и полей до RPC. Каталог runtime проверяется локально через JSON Schema;
неподдержанная схема останавливает расследование. Provenance подтверждает происхождение
ссылок, но не доказывает правильность вердикта. Неструктурированные ответы search/AD
остаются данными с неопределённой интерпретацией.

Для воспроизводимого сравнения `--strategy baseline` сохраняет исходную историю
сообщений и решения без блокировки по provenance; одинаковая телеметрия и пассивный
аудит ссылок действуют в обеих стратегиях. Библиотечный `SimpleSGRAgent` сохраняет
`baseline` по умолчанию для совместимости; выбирайте `strategy="v2"` явно.

Дополнительные необязательные настройки (значения по умолчанию уже действуют):

| Переменная | Значение | Назначение |
| --- | --- | --- |
| `BLUESEC_AGENT_STRATEGY` | `v2` | Стратегия CLI |
| `BLUESEC_TASK_TIMEOUT_SECONDS` | `900` | Общий deadline расследования вместе с допустимым abort |
| `BLUESEC_CLEANUP_TIMEOUT_SECONDS` | `30` | Окно закрытия remote run |
| `BLUESEC_MAX_CONTEXT_CHARS` | `24000` | Размер evidence-представления, не всего запроса |
| `BLUESEC_RUNTIME_STATE_DIR` | общий каталог репозитория | Путь durable intent и межпроцессного замка |
| `LLM_MAX_STEP_RETRIES` | `3` | Всего HTTP-попыток на решение, включая исправления JSON; максимум 3 |

`--strategy`, `--task-timeout-seconds`, `--cleanup-timeout-seconds`,
`--max-context-chars` и `--runtime-state-dir` переопределяют настройки. Связанные
worktree по умолчанию используют один каталог `reports/goals/runtime-state` основного
репозитория. Для разных независимых копий одного аккаунта задайте общий абсолютный путь.

До CreateRun сохраняется intent с idempotency key. Потерянный ответ создания/финала,
`closing` и неизвестное закрытие сохраняют блокировку последующих запусков. Завершение
локального процесса не подтверждает закрытие на сервере. Не удаляйте intent для
возобновления серии без внешнего подтверждения закрытия; RPC проверки статуса в
текущем контракте отсутствует.

`trace.jsonl` содержит schema_version=2 и отдельные события LLM, RPC, cache, provenance
и cleanup. `result.json` сохраняет полученные task_results, metadata с commit и хешами,
error_category, cleanup и artifact_errors. При ошибке записи возможен
`partial-result.json`. Неизвестный usage остаётся null. Нулевой официальный score
сохраняется как ноль; protobuf не различает пропущенный scalar и переданный ноль.

Многоагентная работа описана в
`docs/superpowers/plans/2026-10-10-goal-team.md`. Реестр задач, решения координатора,
независимое ревью и лимитированный manifest находятся в локальных `reports/goals`,
`reports/reviews`, `reports/experiments`. Эти каталоги исключены из Git.

## Проверка одной задачи и полный результат

Для обучения задайте `SCENARIO_RUNTIME_ARENA=practice` в `.env`.
`--max-tasks 1` проверяет одну задачу и затем закрывает run. Такой запуск полезен
для диагностики, но не проходит весь набор: счётчик завершённых задач в интерфейсе
и оценка качества — разные показатели. Успешный RPC и закрытие сессии также не
означают, что агент правильно решил инцидент.

После проверки одиночных задач полный запуск выполняется так:

```bash
uv run --locked --env-file .env bluesec1-agent \
  --arena practice --all-tasks --strategy v2 \
  --max-steps 40 --task-timeout-seconds 900 \
  --cleanup-timeout-seconds 30 --max-context-chars 48000
```

Модель берётся из `LLM_DEFAULT_MODEL`; для эксперимента используйте `--model`.
Полный запуск расходует токены на все задачи очереди. Запускайте его только после
получения корректного `TaskResult` в коротком пилоте. Проверьте `completion_reason`,
`quality_score`, `efficiency_score` и `total_reward` отдельно: завершённая задача
может иметь нулевое качество. Если `task_results` пуст, клиент не получил оценки;
показанный платформой общий ноль фиксируется отдельно от отсутствующих API-полей.

Результаты каждого run сохраняются в `reports/runs/`: `trace.jsonl` содержит
действия и фактические запросы, `result.json` — полученные серверные результаты
и подтверждение cleanup. При неизвестном исходе не запускайте второй процесс
и не удаляйте `intent.json`: локальная остановка не подтверждает закрытие сервера.

Основание протокола: [официальный пример агента](https://github.com/False-Positive-Community/bluesec1-agent#quick-start).

## Сдача ограниченного набора к заданному времени

`--quality-submit` распределяет оставшееся время между задачами. Он читает граф
до 8 секунд на задачу, получает финальное решение модели и проверяет его схему
и ссылки на наблюдения. На задачу выделяется до 60 секунд; отдельно резервируются
9 секунд для финального RPC и окно закрытия run. Один быстрый невалидный ответ
можно исправить в оставшемся времени. Сетевые задержки могут помешать закончить
весь набор: дедлайн и `--max-tasks 40` не гарантируют 40 завершений.

Задайте будущую дату и время с часовым поясом; арена указана явно:

```bash
BLUESEC_RUN_DEADLINE='2026-10-10T15:55:00+03:00'
uv run --locked --env-file .env bluesec1-agent \
  --quality-submit --quality-deadline "$BLUESEC_RUN_DEADLINE" \
  --arena graded --max-tasks 40 --model gpt-6-astra \
  --max-steps 100 --max-context-chars 32000 \
  --cleanup-timeout-seconds 30
```

Необязательный `--quality-advisory-dir PATH` позволяет координатору предложить
решение по опубликованному `request.json`. Ответ `response.json` содержит только
`run_id`, `task_id`, `evidence_sha256` и полный `decision` формата `NextStep`.
Пишите файл атомарной заменой. Совпадение всех трёх идентификаторов, схема и
provenance проверяются до отправки. Координатор не вызывает runtime самостоятельно.
В trace источник решения обозначен как `model`, `coordinator_advisory` либо
`incomplete_fallback`; опубликованная подсказка не обязательно будет использована.

Режимы `--quality-submit`, `--deadline-submit` и `--submission-only` взаимоисключающие.
`--deadline-submit` делает короткую попытку анализа и может отправить явно неполный
fallback. `--submission-only` вообще не вызывает модель и не читает граф: он сдаёт
неполные ответы с пустыми артефактами под именем `deadline-baseline`. Этот режим
проверяет завершение протокола, а не качество расследования.

Для graded фактически принятые `TaskResult` и `closed_confirmed` подтверждают сдачу.
Если платформа скрывает оценку до reveal, возвращённые нулевые scalar-поля не
интерпретируются как опубликованный балл. Счётчик принятых ответов и качество
решения указываются отдельно.
