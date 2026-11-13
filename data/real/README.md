# Реальные трассы для шага 6.8

Сырые загруженные архивы хранятся локально и исключены из Git через `.gitignore`.

## KAD availability trace

- Источник: [Repository of Availability Traces, University of Illinois](https://pbg.web.engr.illinois.edu/availability/)
- Архив: `steiner-kad.avt.bz2` (30 361 208 байт), отдельная трасса из этого репозитория.
- Описание источника: 179-дневный crawl более 400 000 KAD-пиров.
- Формат: `.avt`; строка содержит ID пира, число сессий и пары времён начала/конца. В единицах архива — секунды.
- Ограничение: интервал доступности не раскрывает причину недоступности. Завершение наблюдений также может выглядеть как уход/отказ. Это историческая P2P-трасса, не блокчейн.

Первичный потоковый разбор: `scripts/analyze_kad_trace.py`.

## Backblaze Drive Stats

Источник и квартальные архивы: [Backblaze Hard Drive Test Data](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data). Данные содержат ежедневные состояния HDD/SSD, SMART-показатели и индикатор отказа. Они позволяют оценивать компонент аппаратного риска дисков, но не длительность сетевых простоев, пользовательский уход или общие отказы инфраструктуры.

Для Q4 2025: `scripts/analyze_backblaze.py` — квартальная сводка; `scripts/analyze_backblaze_age.py` и `scripts/fit_backblaze_age.py` — исследовательская возрастная оценка с поправкой на модель; `scripts/map_backblaze_to_node.py` — арифметический мост AFR к простою узла; `scripts/run_backblaze_sensitivity.py` — парная изолированная чувствительность S0. Расчётные таблицы находятся в `runs/real_data/backblaze_q4_2025/` и не коммитятся.
