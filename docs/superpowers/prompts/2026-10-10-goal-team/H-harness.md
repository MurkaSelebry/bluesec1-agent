Ты — H, разработчик harness BlueSec-агента. План принят; реализуй свою часть автономно в отдельной Worktree-задаче.

Прочитай и применяй как часть моего ТЗ:
- /Users/selebry/Documents/dev/pt/bluesec1-agent/docs/superpowers/prompts/2026-10-10-goal-team/common.md
- /Users/selebry/Documents/dev/pt/bluesec1-agent/docs/superpowers/plans/2026-10-10-goal-team.md — особенно разделы 3 и 5.
- /Users/selebry/Documents/dev/pt/bluesec1-agent/docs/superpowers/specs/2026-10-10-goal-team.md
- /Users/selebry/Documents/dev/pt/bluesec1-agent/reports/goals/coordination.md

Активируй goal: «Сделать попытки LLM и исходы расследования наблюдаемыми и ограниченными. Доказать тестами ограниченность retry/repair, корректную отмену, отсутствие повторной финализации, честные closed/closing/unknown и сохранность исходной ошибки при отказе записи. Сохранить next_step и idempotency SDK, передать C проверенные изменения и закрыть замечания».

Проверь prepared SHA и EventSink из BASE_READY. Твои файлы: src/bluesec1_agent/llm.py, recording.py, новые budget.py и run_lock.py; tests/test_llm.py, test_recording.py, новые test_budget.py, test_run_lock.py и test_lifecycle.py. Минимальные изменения src/bluesec1_client/remote/client.py, session.py, errors.py и tests/test_remote_contract.py допустимы для явных lifecycle-исходов. Protobuf и сетевой контракт сохраняй. Agent/CLI/settings/contracts интегрирует C.

Начни с fault-injection: timeout, 429/500, permanent 4xx, malformed/обрезанный JSON, пустой ответ, refusal, missing usage, настоящая asyncio cancellation и отказ записи. Зафиксируй падение проверок на старом поведении.

Ограничь суммарные HTTP-попытки на решение тремя, включая repair. Deadline и backoff учитывают оставшееся время. Сохраняй attempt-start/end, категории ошибок, nullable usage и correlation run/task/step; неизвестный расход не превращай в 0. Не выводи сырой ответ с возможными ключами в консоль.

Cleanup должен различать closed_confirmed, closing_acknowledged, unknown и failed. Потерянный ответ finish не оправдывает новый finish/abort с новым ID; ошибка recorder после terminal не меняет авторитетный результат. Доказательства idempotency проверяй на реальном session/client поверх fake transport.

Общий межпроцессный lock/intent работает между всеми worktree. Intent сохраняется до CreateRun, с логическим ключом и run_id=null. Потерянный ответ создания или closing блокируют следующий запуск даже после завершения процесса. Не выдумывай отсутствующий RPC для запроса состояния run.

Передай C настройки deadline и точки подключения событий, не меняя параллельно его файлы. Выполни целевые проверки и общий gate. Результат с локальным commit SHA и ограничениями запиши в /Users/selebry/Documents/dev/pt/bluesec1-agent/reports/goals/H-cycle-01.md. Все проверки офлайн; реальные LLM/BlueSec-вызовы не выполняй. Начни с проверки BASE_READY и воспроизведения дефектов телеметрии/cleanup.
