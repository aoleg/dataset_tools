# Правки в файлах агента после перехода

Это список правок для владельца (или для агента под присмотром владельца) в четырёх файлах, которые переезжают в `<папка dataset_tools>\vk_dataset\`: `AGENTS.md`, `opencode.jsonc`, `.opencode\` и `STATUS.md`. Git их не отслеживает, поэтому правки остаются вашими и `git pull` их не затрёт. Заглушки: `<папка dataset_tools>`, `<папка датасетов>` — как в `ПЕРЕХОД.md`; в `opencode.jsonc` пути пишутся прямыми слэшами.

## Папка проекта

- [ ] OpenCode открывается в `<папка dataset_tools>\vk_dataset` — это теперь папка проекта. Старая папка проекта больше не нужна, кроме датасетов: они остаются в `<папка датасетов>`, на неё указывает `base` в `vkds.conf`.
- [ ] Старый `.venv` проекта не нужен: Python — общий, `..\venv\Scripts\python.exe`.

## opencode.jsonc

- [ ] Блок модели, `compaction` (в том числе `buffer: 20000` — не уменьшать), `tool_output` — без изменений.
- [ ] `references`: путь dataset_tools — `..` (родительская папка проекта); описание оставить, убрав из него абсолютный путь.
- [ ] `permissions`, правила для dataset_tools: разрешения `external_directory` и `read` на `<папка dataset_tools>/*` остаются. Запрет `edit` на `<папка dataset_tools>/*` теперь накрывает и сам проект, поэтому после него добавьте разрешения `edit` на `<папка dataset_tools>/vk_dataset/STATUS.md` и `<папка dataset_tools>/vk_dataset/tmp/*`, а уже после них — запреты `edit` на `<папка dataset_tools>/vk_dataset/vkds/*` и `<папка dataset_tools>/vk_dataset/tools/*`. Побеждает последнее совпавшее правило, порядок важен.
- [ ] `permissions`, датасеты: добавьте `external_directory` и `read` с `allow` на `<папка датасетов>/*` — агенту нужно читать сводки (`_albums.json`, начало `_manifest.csv`), сами файлы пишут инструменты.
- [ ] `permissions`, judge: чтение и запуск — как у остальных инструментов (правила `read` на `<папка dataset_tools>/*`); правка запрещена — добавьте `{ "action": "edit", "resource": "<папка dataset_tools>/judge/*", "effect": "deny" }` после разрешений на `STATUS.md` и `tmp`. Прогоны judge запускаете вы, агент только читает `report.txt` (скилл `vk-agent-scenarios`).
- [ ] `permissions`, датасеты: добавьте `{ "action": "edit", "resource": "<папка датасетов>/*", "effect": "deny" }` — файлы датасетов пишут инструменты, не агент.
- [ ] Запрет на токен: правило `{ "action": "read", "resource": "vk_token.txt", "effect": "deny" }` продолжает работать — токен лежит рядом с `vkds.conf` в папке проекта. Правило с абсолютным путём старой папки замените на `<папка dataset_tools>/vk_dataset/vk_token.txt`. Оба правила — последними в списке.
- [ ] `watcher.ignore`: добавьте `_vkds/**`, `_removed/**` и `tests/**`.
- [ ] Встроенный websearch остаётся выключенным; модель — только локальная.

## AGENTS.md

- [ ] §1 «Суть проекта»: из списка скиллов убрать `soviet-vehicle-criteria` и `vehicle-check`, добавить `vk-dataset`; инструмент — `vk_dataset` в dataset_tools, обновления — `git pull` (не архивы).
- [ ] §2 «Рабочее окружение»: рабочая папка — `<папка dataset_tools>\vk_dataset`, датасеты — `<папка датасетов>`; Python — `..\venv\Scripts\python.exe`; убрать исключение про справочник моделей техники (`vehicle_ref.py`, Википедия) — запросов наружу больше нет; убрать ссылки на старые пути и на старые документы из папки `docs`.
- [ ] §3 «Структура папок»: заменить дерево — в папке проекта `vkds.py`, `vkds\`, `vkds.conf`, `vk_token.txt`, `run.bat`, `README.md`, `partner\`, `tools\capture_fixtures.py`, файлы агента; датасеты — в `<папка датасетов>`. Убрать `tests\`, `docs\`, `tools\vehicle_ref.py`, `.venv\`.
- [ ] §4 «Инструмент vkds — команды»: заменить таблицу ссылкой на `README.md` и скилл `vk-dataset`; форма каждой команды — `..\venv\Scripts\python.exe vkds.py <глагол> ...` из папки проекта, никогда не `run.bat`; `--out` — папка темы внутри `base`; вместо `--min-area 307200` — `--min-side 512`; добавить `scan wall` и `wall`; строку про справочник моделей удалить; кэш читают только `scan` и `--dry-run`.
- [ ] §5 «Структура датасета»: имена файлов `photo<владелец>_<id>.jpg` и `.txt`; папка `wall\<ГГГГ-ММ>\` для фото стены; `_wall.json` в папке группы; статусы манифеста `ok`, `exists`, `dup`, `small`, `dead`, `forbidden`, `failed`, `no_url` и колонка `moved`; `.txt` никогда не перезаписывается.
- [ ] §6 «Бюджет»: журнал — `_vkds\ledger.json` рядом с `vkds.conf` (один на копию инструмента), смотреть `budget`; ошибка лимита (код 29) — команду не повторять, ждать сброса, доложить; строку о стратегии тестирования по старому плану проекта убрать.
- [ ] §7 «Стандартный рабочий процесс»: разведка — `scan albums` и `scan wall`; скачивание — `albums` и `wall` с `--min-side 512` и `--budget N`; пост-обработка — шаги 3 и 4 по скиллу `dataset-tools-postprocess`; после шага 4 владелец отправляет папку группы.
- [ ] §8 «dataset_tools»: путь — `..`; запуск — `..\venv\Scripts\python.exe ..\<инструмент>\<скрипт>.py`; из инструментов у вас остаются `deduplicate`, `scripts\unrelated.py` и `judge` (отбор по условию; его запускает владелец, не агент); классификация по эпохе убрана (её делает наш импорт).
- [ ] §1 и §7: два сценария — «найди группы ВК про ...» и «скачай группы, но оставь только фото, где ...» — по скиллу `vk-agent-scenarios`; руководство владельца — `partner\README.md`.
- [ ] §9 «STATUS.md» — без изменений; добавьте в STATUS.md запись о переходе на vk_dataset 0.6.1.
- [ ] §10 «Вопросы владельцу — только инструментом question» — без изменений.
- [ ] §11 «Критерии отбора (тема «советский период»)» — раздел удалить: эпоху решают наш классификатор и локальная модель в `unrelated.py`. Оставить одну строку: плохое качество, дубли и вотермарки — не повод отказываться от альбома при сборе.
- [ ] §12 «Разделение труда»: обновления приходят через `git pull`; после обновления проверять `..\venv\Scripts\python.exe vkds.py --version` и `budget`, а не тесты (папки `tests\` в репозитории нет).
- [ ] Коды выхода (добавить коротко): `0` готово, `1` ошибка, `2` отказ или ошибка лимита, `130` прервано.

## .opencode\

- [ ] `.opencode\skills\vk-dataset\SKILL.md` — новый, текст из `partner\skill_vk-dataset.md`.
- [ ] `.opencode\skills\dataset-tools-postprocess\SKILL.md` — заменить текстом из `partner\skill_dataset-tools-postprocess.md`.
- [ ] `.opencode\skills\vk-agent-scenarios\SKILL.md` — новый, текст из `partner\skill_vk-agent-scenarios.md`.
- [ ] Папки `.opencode\skills\soviet-vehicle-criteria\` и `.opencode\skills\vehicle-check\` — удалить (или убрать из папки скиллов в свой архив, если хотите сохранить).
- [ ] `.opencode\commands\status.md`: путь к STATUS.md — просто `STATUS.md` (файл в папке проекта).

## STATUS.md

- [ ] Ведётся как раньше. Цифры расхода вызовов за месяц — из `budget` (после `budget --seed-ledger --out <пилот>` в журнале есть и вызовы пилота).
