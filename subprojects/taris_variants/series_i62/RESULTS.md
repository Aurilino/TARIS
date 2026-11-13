# I6.2. Независимая проверка очереди

Дата: 13 ноября 2026 года.

| Мир | Волны | Правило | Канал | Стратегия | Объектов | Безвозвратные потери | Неполный приём | Дисковый дефицит, сут. | Спасает только буфер, сут. | Бюджет передачи | Хвост заданий |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 201 | close | legacy | ample | classic | 45 | 7 | 0 | 206.500000 | 0.000000 | 19.935 | 0 |
| 201 | close | legacy | ample | graph | 45 | 5 | 0 | 147.500000 | 0.000000 | 25.596 | 0 |
| 201 | close | legacy | ample | hybrid | 45 | 6 | 0 | 177.000000 | 0.000000 | 23.519 | 0 |
| 201 | close | legacy | constrained | classic | 45 | 9 | 0 | 207.173748 | 0.000000 | 1.010 | 8 |
| 201 | close | legacy | constrained | graph | 45 | 10 | 0 | 149.184369 | 0.000000 | 13.180 | 8 |
| 201 | close | legacy | constrained | hybrid | 45 | 12 | 0 | 179.021243 | 0.000000 | 1.010 | 8 |
| 201 | close | combined | ample | classic | 45 | 7 | 0 | 206.500000 | 0.000000 | 19.935 | 0 |
| 201 | close | combined | ample | graph | 45 | 5 | 0 | 147.500000 | 0.000000 | 25.596 | 0 |
| 201 | close | combined | ample | hybrid | 45 | 6 | 0 | 177.000000 | 0.000000 | 23.519 | 0 |
| 201 | close | combined | constrained | classic | 45 | 9 | 0 | 207.173748 | 0.000000 | 1.010 | 8 |
| 201 | close | combined | constrained | graph | 45 | 10 | 0 | 149.184369 | 0.000000 | 13.180 | 8 |
| 201 | close | combined | constrained | hybrid | 45 | 12 | 0 | 179.021243 | 0.000000 | 1.010 | 8 |
| 201 | close | pressure | ample | classic | 45 | 7 | 0 | 206.500000 | 0.000000 | 19.935 | 0 |
| 201 | close | pressure | ample | graph | 45 | 5 | 0 | 147.500000 | 0.000000 | 25.596 | 0 |
| 201 | close | pressure | ample | hybrid | 45 | 6 | 0 | 177.000000 | 0.000000 | 23.519 | 0 |
| 201 | close | pressure | constrained | classic | 45 | 9 | 0 | 207.173748 | 0.000000 | 11.070 | 6 |
| 201 | close | pressure | constrained | graph | 45 | 10 | 0 | 149.184369 | 0.000000 | 13.180 | 8 |
| 201 | close | pressure | constrained | hybrid | 45 | 10 | 0 | 178.347495 | 0.000000 | 10.557 | 7 |
| 201 | far | legacy | ample | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 62.570 | 0 |
| 201 | far | legacy | ample | graph | 45 | 2 | 0 | 46.000000 | 0.000000 | 52.111 | 0 |
| 201 | far | legacy | ample | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 66.112 | 0 |
| 201 | far | legacy | constrained | classic | 45 | 9 | 0 | 161.673748 | 0.000000 | 4.280 | 8 |
| 201 | far | legacy | constrained | graph | 45 | 11 | 0 | 117.021243 | 0.000000 | 12.857 | 8 |
| 201 | far | legacy | constrained | hybrid | 45 | 12 | 0 | 140.021243 | 0.000000 | 1.010 | 8 |
| 201 | far | combined | ample | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 62.565 | 0 |
| 201 | far | combined | ample | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 66.930 | 0 |
| 201 | far | combined | ample | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 66.555 | 0 |
| 201 | far | combined | constrained | classic | 45 | 9 | 0 | 161.673748 | 0.000000 | 4.280 | 8 |
| 201 | far | combined | constrained | graph | 45 | 11 | 0 | 117.021243 | 0.000000 | 12.897 | 8 |
| 201 | far | combined | constrained | hybrid | 45 | 12 | 0 | 140.021243 | 0.000000 | 1.010 | 8 |
| 201 | far | pressure | ample | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 62.565 | 0 |
| 201 | far | pressure | ample | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 66.930 | 0 |
| 201 | far | pressure | ample | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 66.555 | 0 |
| 201 | far | pressure | constrained | classic | 45 | 9 | 0 | 161.673748 | 0.000000 | 11.919 | 7 |
| 201 | far | pressure | constrained | graph | 45 | 11 | 0 | 117.021243 | 0.000000 | 12.897 | 8 |
| 201 | far | pressure | constrained | hybrid | 45 | 10 | 0 | 139.347495 | 0.000000 | 10.876 | 7 |
| 202 | close | legacy | ample | classic | 45 | 3 | 0 | 88.500000 | 0.000000 | 30.277 | 0 |
| 202 | close | legacy | ample | graph | 45 | 5 | 0 | 147.500000 | 0.000000 | 24.630 | 0 |
| 202 | close | legacy | ample | hybrid | 45 | 7 | 0 | 206.500000 | 0.000000 | 23.703 | 0 |
| 202 | close | legacy | constrained | classic | 45 | 3 | 0 | 88.500000 | 0.000000 | 14.554 | 8 |
| 202 | close | legacy | constrained | graph | 45 | 5 | 0 | 147.500000 | 0.000000 | 12.956 | 6 |
| 202 | close | legacy | constrained | hybrid | 45 | 7 | 0 | 206.500000 | 0.000000 | 14.717 | 8 |
| 202 | close | combined | ample | classic | 45 | 3 | 0 | 88.500000 | 0.000000 | 30.340 | 0 |
| 202 | close | combined | ample | graph | 45 | 5 | 0 | 147.500000 | 0.000000 | 24.692 | 0 |
| 202 | close | combined | ample | hybrid | 45 | 7 | 0 | 206.500000 | 0.000000 | 23.703 | 0 |
| 202 | close | combined | constrained | classic | 45 | 3 | 0 | 88.500000 | 0.000000 | 14.554 | 8 |
| 202 | close | combined | constrained | graph | 45 | 5 | 0 | 147.500000 | 0.000000 | 12.956 | 8 |
| 202 | close | combined | constrained | hybrid | 45 | 7 | 0 | 206.500000 | 0.000000 | 14.717 | 5 |
| 202 | close | pressure | ample | classic | 45 | 3 | 0 | 88.500000 | 0.000000 | 30.340 | 0 |
| 202 | close | pressure | ample | graph | 45 | 5 | 0 | 147.500000 | 0.000000 | 24.692 | 0 |
| 202 | close | pressure | ample | hybrid | 45 | 7 | 0 | 206.500000 | 0.000000 | 23.703 | 0 |
| 202 | close | pressure | constrained | classic | 45 | 3 | 0 | 88.500000 | 0.000000 | 14.554 | 8 |
| 202 | close | pressure | constrained | graph | 45 | 5 | 0 | 147.500000 | 0.000000 | 12.956 | 8 |
| 202 | close | pressure | constrained | hybrid | 45 | 7 | 0 | 206.500000 | 0.000000 | 14.717 | 5 |
| 202 | far | legacy | ample | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 39.270 | 0 |
| 202 | far | legacy | ample | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 48.843 | 0 |
| 202 | far | legacy | ample | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 48.280 | 0 |
| 202 | far | legacy | constrained | classic | 45 | 3 | 0 | 69.000000 | 0.000000 | 14.535 | 8 |
| 202 | far | legacy | constrained | graph | 45 | 5 | 0 | 115.000000 | 0.000000 | 12.895 | 8 |
| 202 | far | legacy | constrained | hybrid | 45 | 7 | 0 | 161.000000 | 0.000000 | 14.257 | 5 |
| 202 | far | combined | ample | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 40.879 | 0 |
| 202 | far | combined | ample | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 48.874 | 0 |
| 202 | far | combined | ample | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 49.884 | 0 |
| 202 | far | combined | constrained | classic | 45 | 3 | 0 | 69.000000 | 0.000000 | 14.535 | 8 |
| 202 | far | combined | constrained | graph | 45 | 5 | 0 | 115.000000 | 0.000000 | 12.935 | 7 |
| 202 | far | combined | constrained | hybrid | 45 | 7 | 0 | 161.000000 | 0.000000 | 14.338 | 8 |
| 202 | far | pressure | ample | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 40.879 | 0 |
| 202 | far | pressure | ample | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 48.874 | 0 |
| 202 | far | pressure | ample | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 49.884 | 0 |
| 202 | far | pressure | constrained | classic | 45 | 3 | 0 | 69.000000 | 0.000000 | 14.535 | 8 |
| 202 | far | pressure | constrained | graph | 45 | 5 | 0 | 115.000000 | 0.000000 | 12.935 | 7 |
| 202 | far | pressure | constrained | hybrid | 45 | 7 | 0 | 161.000000 | 0.000000 | 14.343 | 7 |
| 203 | close | legacy | ample | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 47.346 | 0 |
| 203 | close | legacy | ample | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 39.235 | 0 |
| 203 | close | legacy | ample | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 47.326 | 0 |
| 203 | close | legacy | constrained | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 12.775 | 8 |
| 203 | close | legacy | constrained | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 13.175 | 8 |
| 203 | close | legacy | constrained | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 13.055 | 8 |
| 203 | close | combined | ample | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 47.356 | 0 |
| 203 | close | combined | ample | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 37.720 | 0 |
| 203 | close | combined | ample | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 50.356 | 0 |
| 203 | close | combined | constrained | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 12.775 | 8 |
| 203 | close | combined | constrained | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 13.175 | 8 |
| 203 | close | combined | constrained | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 13.055 | 8 |
| 203 | close | pressure | ample | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 47.356 | 0 |
| 203 | close | pressure | ample | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 37.720 | 0 |
| 203 | close | pressure | ample | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 50.356 | 0 |
| 203 | close | pressure | constrained | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 12.775 | 8 |
| 203 | close | pressure | constrained | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 13.175 | 8 |
| 203 | close | pressure | constrained | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 13.055 | 8 |
| 203 | far | legacy | ample | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 73.119 | 0 |
| 203 | far | legacy | ample | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 65.528 | 0 |
| 203 | far | legacy | ample | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 67.059 | 0 |
| 203 | far | legacy | constrained | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 12.594 | 8 |
| 203 | far | legacy | constrained | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 13.194 | 8 |
| 203 | far | legacy | constrained | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 12.854 | 8 |
| 203 | far | combined | ample | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 71.546 | 0 |
| 203 | far | combined | ample | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 65.445 | 0 |
| 203 | far | combined | ample | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 66.996 | 0 |
| 203 | far | combined | constrained | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 12.594 | 8 |
| 203 | far | combined | constrained | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 13.133 | 8 |
| 203 | far | combined | constrained | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 12.854 | 8 |
| 203 | far | pressure | ample | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 71.546 | 0 |
| 203 | far | pressure | ample | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 65.445 | 0 |
| 203 | far | pressure | ample | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 66.996 | 0 |
| 203 | far | pressure | constrained | classic | 45 | 0 | 0 | 0.000000 | 0.000000 | 12.594 | 8 |
| 203 | far | pressure | constrained | graph | 45 | 0 | 0 | 0.000000 | 0.000000 | 13.133 | 8 |
| 203 | far | pressure | constrained | hybrid | 45 | 0 | 0 | 0.000000 | 0.000000 | 12.854 | 8 |

Ветвей: 18; изоляция и общие случайные числа: True.

Бюджет передачи включает неуспешные попытки и прошедшую часть фазы на горизонте. Это не точные доставленные байты. Первичные записи не конкурируют за этот канал. Здоровый буфер освобождается только после оплаченной проверки k разных копий.

Каждая строка — отдельное исполнение одного мира; ветви одного мира не являются независимыми мирами. Полные журналы содержат ожидание начатых заданий и отдельно возраст ещё ожидающих. Редкая сохранность этой серией не подтверждается.
