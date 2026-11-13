# I4.4c. Парное сравнение и новая проверка

anchor_i44a/paired_i44b/paired_i44c — одни и те же три мира 24–26. validation_i44c — новые миры 27–29, без нового базиса I4.4b. Дополнительные 18 исполнений простого I4.4a не добавляют новых миров. В каждой строке 135 объектов.

| Серия | Канал | Стратегия | Новых чанков | Полных буферов | Потери | Бюджет | Хвост | Наблюдённое ожидание, сут. |
|---|---|---|---:|---:|---:|---:|---:|---:|
| anchor_i44a | ample | classic | 30 | 30 | 0 | 45.515 | 0 | 6.893 |
| anchor_i44a | ample | graph | 41 | 41 | 0 | 62.248 | 0 | 10.990 |
| anchor_i44a | ample | hybrid | 32 | 32 | 0 | 48.659 | 0 | 8.351 |
| anchor_i44a | constrained | classic | 18 | 18 | 0 | 37.120 | 10 | 266.781 |
| anchor_i44a | constrained | graph | 17 | 19 | 0 | 36.152 | 24 | 387.828 |
| anchor_i44a | constrained | hybrid | 19 | 20 | 0 | 38.440 | 14 | 369.364 |
| paired_i44b | ample | classic | 30 | 30 | 0 | 45.515 | 0 | 6.893 |
| paired_i44b | ample | graph | 41 | 41 | 0 | 62.248 | 0 | 10.990 |
| paired_i44b | ample | hybrid | 32 | 32 | 0 | 48.659 | 0 | 8.351 |
| paired_i44b | constrained | classic | 7 | 10 | 0 | 21.898 | 8 | 92.289 |
| paired_i44b | constrained | graph | 13 | 17 | 0 | 32.032 | 22 | 310.745 |
| paired_i44b | constrained | hybrid | 10 | 14 | 0 | 24.968 | 13 | 204.760 |
| paired_i44c | ample | classic | 30 | 30 | 0 | 45.515 | 0 | 6.893 |
| paired_i44c | ample | graph | 41 | 41 | 0 | 62.248 | 0 | 10.990 |
| paired_i44c | ample | hybrid | 32 | 32 | 0 | 48.659 | 0 | 8.351 |
| paired_i44c | constrained | classic | 10 | 14 | 0 | 28.533 | 8 | 184.485 |
| paired_i44c | constrained | graph | 12 | 17 | 0 | 32.311 | 24 | 329.792 |
| paired_i44c | constrained | hybrid | 10 | 14 | 0 | 24.949 | 13 | 206.081 |
| validation_i44c | ample | classic | 30 | 30 | 0 | 45.480 | 0 | 7.012 |
| validation_i44c | ample | graph | 39 | 39 | 0 | 59.273 | 0 | 8.616 |
| validation_i44c | ample | hybrid | 34 | 34 | 0 | 51.623 | 0 | 7.838 |
| validation_i44c | constrained | classic | 10 | 13 | 0 | 23.758 | 3 | 172.788 |
| validation_i44c | constrained | graph | 12 | 12 | 0 | 24.253 | 16 | 300.043 |
| validation_i44c | constrained | hybrid | 6 | 11 | 0 | 19.381 | 3 | 167.811 |

Ожидание включает возраст ещё не начатых заданий. Хвост суммируется по трём прогонам; бюджет включает неуспешные блоки, проверки возврата и закрытия буфера. Меньший расход не означает завершённость ремонта.

Сохранность редких событий и преимущество одной политики этими тремя проверочными мирами не доказаны.
