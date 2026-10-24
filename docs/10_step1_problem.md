# Шаг 1. Проблема и существующие решения

> [[00_tema|Тема]] · [[01_plan|План]] · [[02_student|Студент]] · [[20_step2_math|Шаг 2]] →

### 1.1. Контекст
В классических блокчейнах применяется полная репликация: каждый узел хранит копию всех блоков, а согласованность данных обеспечивает протокол консенсуса. С ростом цепочки это становится всё дороже. Поэтому схемы BFT-Store и PartitionChain хранят блоки с помощью стирающих кодов, чтобы сократить затраты на хранение [4]. Объект режут на $n$ фрагментов, из которых любые $k$ восстанавливают данные, и раздают фрагменты по узлам. [hep](https://academic.hep.com.cn/fcs/EN/10.1007/s11704-023-3209-3)[hep](https://academic.hep.com.cn/fcs/EN/10.1007/s11704-023-3209-3)
### 1.2. Две задачи сети
**Сохранность:** данные не теряются при отказах узлов.

**Конфиденциальность:** узлы не могут прочитать данные. Это не теоретическая угроза. Документация Filecoin прямо предупреждает, что незашифрованный файл провайдер может прочитать и передать копии третьим лицам [12]. Storj считает сильное шифрование обязательным, раз данные лежат на узлах сторонних операторов [14]. [filecoin](https://docs.filecoin.io/builder-cookbook/data-storage/privacy-and-access-control)[storj](https://docs.storj.io/dcs/concepts/encryption-key/how-encryption-is-implemented)
### 1.3. Общий источник риска
Связанные узлы (общий владелец, дата-центр, провайдер) создают обе угрозы сразу.

**Совместный отказ.** Cidon и соавт. показали, что случайная репликация почти гарантированно теряет данные при одновременном отказе многих узлов, например при отключении питания всего кластера [2]. [usenix](https://www.usenix.org/conference/atc13/technical-sessions/papers/cidon)

**Совместное чтение и сговор.** Даже в научных моделях угроз это допускается: в AEC-DS злонамеренные узлы могут сговариваться [5]. [arxiv](https://arxiv.org/html/2607.17460)

**Скрытая связь — реальная проблема.** Storj по умолчанию запрещает класть фрагменты одного сегмента на коррелированные узлы и определяет их по подсети /24 публичных IPv4. При этом операторы узлов обсуждают массовый обход этого правила через VPN [16]. Значит, связь узлов нельзя считать известной: её нужно выявлять. [storj](https://forum.storj.io/t/using-multiple-ips-on-the-same-physical-service/27738/22)[storj](https://forum.storj.io/t/discussion-on-scaling-the-network-a-multi-node-sno-tier-proposal/25217?page=2)

#### 1.4. Решения, которые зависят от риска узла

1. **Где хранить.** Например, в AEC-DS узлы для фрагментов выбираются по приоритетному баллу, который растёт с репутацией узла и важностью данных [5]. [arxiv](https://arxiv.org/html/2607.17460)
2. **Какую избыточность выбрать.** Там же число проверочных фрагментов пересчитывается по формуле от средней репутации узлов, хранящих объект [5]. [arxiv](https://arxiv.org/html/2607.17460)
3. **Сколько платить и какой залог брать.** В Filecoin залог вносится заранее и пропорционален объёму заявленного хранилища [12]. [filecoin](https://docs.filecoin.io/storage-provider/filecoin-economics/fil-collateral)

#### 1.5. Требования к решению

- **К1.** Оценка риска — вероятность отказа за период, а не балл.
- **К2.** Разумная оценка нового узла с короткой историей.
- **К3.** Учёт общих причин отказов (владелец, дата-центр, провайдер).
- **К4.** Оценка определяет цену и залог узла.
- **К5.** Расчёт воспроизводим и проверяем участниками сети.
- **К6.** Ни узел, ни группа связанных узлов не могут прочитать данные; ключ не хранится в одном месте.

#### 1.6. Обзор решений: надёжность

**Структурные правила.**

- **CRUSH** [1] — псевдослучайная функция распределения данных без центрального каталога, которая по заданным политикам разносит реплики по разным доменам отказов. [ucsc](https://www.ssrc.ucsc.edu/Papers/weil-sc06.pdf)
- **Copysets** [2] — метод репликации, который заметно снижает частоту потерь данных и даёт почти оптимальный компромисс между разбросом данных по узлам и вероятностью потери. [usenix](https://www.usenix.org/conference/atc13/technical-sessions/papers/cidon)

Оба решения считают узлы одинаковыми и задают правила структурно, без оценки риска.

**Залог и штрафы (Filecoin).** За каждый день офлайна сектора начисляется штраф, равный 3,51 дня ожидаемых наград за блоки [12]. Это единый тариф: ставка не зависит от риска конкретного провайдера. [filecoin](https://docs.filecoin.io/provide-storage/filecoin-economics/slashing)

**Эвристический скоринг (Sia).**

- Каждый арендатор считает балл хоста сам, единой оценки в сети нет [13]. [sia](https://docs.sia.tech/hosting/about-hosting-on-sia)
- Балл складывается из вручную подобранных множителей: если хост снижает цену вдвое, его балл растёт в 16 раз. [sia](https://docs.sia.tech/legacy/hosting/about-hosting-on-sia)
- Среди семи метрик, которые ведёт renterd, есть аптайм и возраст хоста. [sia](https://forum.sia.tech/t/hostscore-a-host-benchmarking-tool)

**Байесовская репутация (Storj).**

- Storj использует бета-модель репутации: балл узла — вероятность успешного случайного аудита, причём недавняя история весит больше [15, 17]. [storj](https://forum.storj.io/t/tuning-audit-scoring/14084/16)
- Порог дисквалификации — 0,96, начальная альфа — 1000. [github](https://github.com/storj/storj/blob/v1.80.7/satellite/reputation/config.go)
- Для новых узлов: непроверенный узел получает лишь 5% загрузок, пока не пройдёт 100 аудитов [16]. [storj](https://forum.storj.io/t/question-about-success-rates/23524/20)

**Научные схемы для блокчейна.**

- **Dynamic-EC** [4] работает в закрытой (permissioned) сети на Tendermint. Число проверочных фрагментов меняется по уровню риска всей системы, который определяется числом замеченных злонамеренных узлов. Личные оценки узлов агрегируются алгоритмом EigenTrust, и каждый узел относят к честным, рискованным или злонамеренным. [hep](https://academic.hep.com.cn/fcs/EN/10.1007/s11704-023-3209-3)[hep](https://academic.hep.com.cn/fcs/EN/10.1007/s11704-023-3209-3)
- **AEC-DS** [5] — самая близкая работа. Репутация обновляется только по результатам аудитов PDP как экспоненциальное скользящее среднее бинарного исхода с коэффициентом 0,1. По порогам репутации узлы делятся на три яруса, и новые узлы попадают в нижний ярус с самыми частыми аудитами. [arxiv](https://arxiv.org/html/2607.17460)[arxiv](https://arxiv.org/html/2607.17460)
- Ещё одна схема того же класса — DRD-EC [6].

**ML-прогноз отказов.** Botezatu и соавт. по SMART-атрибутам предсказывают необходимость замены диска с точностью до 98% [3]. Это централизованная модель по оборудованию, не связанная с размещением и ценой. [unpaywall](https://unpaywall.org/10.1145%2F2939672.2939699)

#### 1.7. Обзор решений: конфиденциальность

**Шифрование до кодирования.**

- **Storj:** данные шифруются на клиенте до того, как разбиваются стирающим кодом. Расшифровать собранные фрагменты может только клиент, потому что ключ есть только у него [14]. [storj](https://docs.storj.io/dcs/concepts/definitions)[storj](https://storj.dev/learn)
- **Sia:** каждый фрагмент шифруется ChaCha20 и отправляется на отдельный хост [13]. [sia](https://docs.sia.tech/renting/renting-storage)

**Без встроенного шифрования (Filecoin).** Сеть не шифрует данные автоматически, это задача пользователя [12]. Шифрование добавляют сторонние решения: Lighthouse использует пороговую криптографию BLS для ключей расшифровки [18]. [it](https://www.thecrypto.it.com/tokens/what-is-filecoin)[medium](https://nanditmehra123.medium.com/encryption-and-access-control-for-web3-using-lighthouse-71dc4e7d0632)

**Классические схемы** (обзор — [24]):

- **Схема Шамира** [19]: абсолютная секретность, но каждая доля размером с весь секрет.
- **IDA Рабина** [20]: экономное распределение без секретности.
- **SSMS** [21]: схема Кравчика даёт вычислительно стойкое пороговое разделение секрета с долями меньше теоретического предела, сочетая шифрование и IDA. [arxiv](https://arxiv.org/pdf/2502.02774)
- **AONT-RS** [23]: данные сначала проходят преобразование «всё или ничего» (идея Ривеста [22]), затем систематический код Рида–Соломона добавляет избыточные фрагменты; схема применяется в IBM Cloud Object Storage. [arxiv](https://arxiv.org/pdf/1706.05960)

**Почему «секретная карта размещения» сама по себе не защищает:**

1. Фрагмент систематического кода содержит открытый текст.
2. Секретность должна держаться на ключе, а не на тайне схемы (принцип Керкгоффса [25]).
3. Единственный хранитель карты становится единой точкой отказа.
4. Секретность карты противоречит проверяемости (К5).

#### 1.8. Сводная оценка (после проверки)

|Решение|К1|К2|К3|К4|К5|К6|
|---|---|---|---|---|---|---|
|CRUSH, copysets [1, 2]|нет|—|да (правилом)|нет|да|—|
|Filecoin [11, 12]|нет|нет|нет|нет (единый тариф)|да|нет|
|Sia [13]|нет|частично (возраст)|нет|частично|нет (свой балл у каждого)|да|
|Storj [14–16]|частично (аудит)|частично (общий приор, 100 аудитов)|частично (правило /24)|нет|— (централизовано)|да|
|Dynamic-EC [4]|нет|— (закрытая сеть)|нет|нет|частично|—|
|AEC-DS [5]|нет|частично (нижний ярус)|нет|нет|частично (смарт-контракт)|—|
|ML по SMART [3]|да|частично|нет|нет|нет|—|
|AONT-RS, SSMS [21, 23]|—|—|—|—|—|да (до сговора $k$ узлов)|

#### 1.9. Пробел и место работы

Проверка первоисточников усилила главный аргумент. И Dynamic-EC, и AEC-DS задают избыточность **эвристически**:

- Dynamic-EC выбирает её по трём уровням риска в зависимости от числа рискованных узлов;
- AEC-DS — по эмпирической формуле от средней репутации.

Ни одна из схем не подбирает избыточность под **целевую вероятность потери**, потому что у них нет вероятности отказа узла (К1).

Ни одно решение не закрывает все шесть требований. Новизна работы — тариф узла с двусторонним учётом кумуляции: одного кластера не должно хватать ни чтобы уничтожить объект, ни чтобы его собрать.

$$r \le \min(n - k, k - 1)$$

Криптография (AONT-RS или SSMS) берётся готовой.

#### 1.10. Обновление файла темы ✓

Сделано: конфиденциальность, уточнённая новизна и практическая значимость внесены в файл темы.

#### 1.11. Проверка по первоисточникам ✓

**AEC-DS прочитана полностью.** Три вывода для работы:

- **Модель отказов мягкая.** Учтены только два вида сбоев: временная недоступность узла с заданной вероятностью и византийское поведение части узлов. Коррелированные отказы не моделируются. [arxiv](https://arxiv.org/html/2607.17460)
- **Сохранность не различает методы.** Все четыре метода, включая статический код, сохранили 100% данных, и авторы сами объясняют это мягкостью своей модели отказов. Значит, в нашем эксперименте обязательны коррелированные отказы и сравнение избыточности при равном риске. [arxiv](https://arxiv.org/html/2607.17460)
- **Авторы признают ограничения.** Оценка только симуляционная, $k$ фиксирован, экономические стимулы не учтены, параметры подобраны вручную. Это прямо обосновывает К4. [arxiv](https://arxiv.org/html/2607.17460)

Важное уточнение формулировки: AEC-DS — это механизм замкнутого контура на смарт-контракте, а не ИИ-агент. В файле темы фразу «агент оценивает узлы» лучше заменить на «механизм оценивает узлы».

**Dynamic-EC.** Выходные данные взяты со страницы журнала. Обратите внимание: в списке литературы AEC-DS имена авторов Dynamic-EC указаны с ошибками, правильные — в [4].

**Storj.** Правило /24 подтверждено исходным кодом и разъяснениями сотрудника Storj.

**Sia.** Шифрование на клиенте подтверждено документацией.

**Не прочитаны** (взяты из чужих списков литературы, помечены *): [6], [10], [17]. Выходные данные [7], [19], [20], [22], [25] нужно сверить по оригиналу.

#### 1.12. Рабочий список литературы ✓

Собран рабочий список [1]–[25]. Открыто после пункта 1.11: прочитать [6], [10], [17] (помечены *), сверить выходные данные [7], [19], [20], [22], [25] по оригиналу. Список шага 2 продолжает нумерацию: [26]–[56].

_Распределённое хранение и надёжность_

1. Weil S.A., Brandt S.A., Miller E.L., Maltzahn C. CRUSH: Controlled, Scalable, Decentralized Placement of Replicated Data // Proc. SC'06. Tampa, 2006. DOI: 10.1145/1188455.1188582. [https://www.ssrc.ucsc.edu/Papers/weil-sc06.pdf](https://www.ssrc.ucsc.edu/Papers/weil-sc06.pdf)
2. Cidon A., Rumble S., Stutsman R., Katti S., Ousterhout J., Rosenblum M. Copysets: Reducing the Frequency of Data Loss in Cloud Storage // USENIX ATC '13. San Jose, 2013. [https://www.usenix.org/conference/atc13/technical-sessions/presentation/cidon](https://www.usenix.org/conference/atc13/technical-sessions/presentation/cidon)
3. Botezatu M.M., Giurgiu I., Bogojeska J., Wiesmann D. Predicting Disk Replacement towards Reliable Data Centers // KDD '16. 2016. P. 39–48. DOI: 10.1145/2939672.2939699.

_Блокчейн-хранение, кодирование, репутация_

4. Zhang M., Wu C., Li J., Guo M. Dynamic-EC: an efficient dynamic erasure coding method for permissioned blockchain systems // Frontiers of Computer Science. 2025. Vol. 19, No. 1. Art. 191101. DOI: 10.1007/s11704-023-3209-3.
5. Li S., Yu W., Wang K., Han M. AEC-DS: Adaptive Erasure Coding with PDP-Triggered Reputation and QoS-Aware Migration for Decentralized Storage. arXiv:2607.17460, 2026. [https://arxiv.org/abs/2607.17460](https://arxiv.org/abs/2607.17460)
6. Qiao J., Zhang J., Chen R. et al. DRD-EC: a dynamic risk-driven erasure coding method for IoT blockchains // Proc. SPIE. 2025. Vol. 13692. DOI: 10.1117/12.3069027. *
7. Kamvar S.D., Schlosser M.T., Garcia-Molina H. The EigenTrust algorithm for reputation management in P2P networks // WWW '03. 2003. P. 640–651.
8. Ateniese G. et al. Provable Data Possession at Untrusted Stores // ACM CCS '07. 2007. P. 598–609. DOI: 10.1145/1315245.1315318.
9. Juels A., Kaliski B.S. PORs: Proofs of Retrievability for Large Files // ACM CCS '07. 2007.
10. Li C. et al. SoK: Decentralized Storage Network // Blockchain: Research and Applications. 2024. [https://www.sciencedirect.com/science/article/pii/S2667295224000424](https://www.sciencedirect.com/science/article/pii/S2667295224000424) *

_Промышленные сети_

11. Protocol Labs. Filecoin: A Decentralized Storage Network. 2017. [https://research.protocol.ai/publications/filecoin-a-decentralized-storage-network/protocollabs2017a.pdf](https://research.protocol.ai/publications/filecoin-a-decentralized-storage-network/protocollabs2017a.pdf)
12. Filecoin Docs: FIL collateral; Slashing; Privacy & Access Control. [https://docs.filecoin.io/storage-providers/filecoin-economics/fil-collateral](https://docs.filecoin.io/storage-providers/filecoin-economics/fil-collateral) ; [https://docs.filecoin.io/storage-providers/filecoin-economics/slashing](https://docs.filecoin.io/storage-providers/filecoin-economics/slashing) ; [https://docs.filecoin.io/builder-cookbook/data-storage/privacy-and-access-control](https://docs.filecoin.io/builder-cookbook/data-storage/privacy-and-access-control)
13. Sia Docs: About Providing Storage; Renting Storage. [https://docs.sia.tech/hosting/about-hosting-on-sia](https://docs.sia.tech/hosting/about-hosting-on-sia) ; [https://docs.sia.tech/renting/renting-storage](https://docs.sia.tech/renting/renting-storage)
14. Storj Docs: Definitions; How Encryption is Implemented. [https://docs.storj.io/dcs/concepts/definitions](https://docs.storj.io/dcs/concepts/definitions) ; [https://docs.storj.io/dcs/concepts/encryption-key/how-encryption-is-implemented](https://docs.storj.io/dcs/concepts/encryption-key/how-encryption-is-implemented)
15. Storj: пакет reputation (исходный код сателлита) и обсуждение настройки. [https://pkg.go.dev/storj.io/storj@v1.111.5/satellite/reputation](https://pkg.go.dev/storj.io/storj@v1.111.5/satellite/reputation) ; [https://forum.storj.io/t/tuning-audit-scoring/14084/16](https://forum.storj.io/t/tuning-audit-scoring/14084/16)
16. Storj: выбор узлов по подсети /24. [https://github.com/storj/storj/blob/v1.27.1/satellite/nodeselection/selector.go](https://github.com/storj/storj/blob/v1.27.1/satellite/nodeselection/selector.go) ; [https://forum.storj.io/t/question-about-success-rates/23524/20](https://forum.storj.io/t/question-about-success-rates/23524/20)
17. Jøsang A., Ismail R. The Beta Reputation System // 15th Bled Electronic Commerce Conference. 2002. *
18. Lighthouse: Encryption and Access Control for Web3. [https://nanditmehra123.medium.com/encryption-and-access-control-for-web3-using-lighthouse-71dc4e7d0632](https://nanditmehra123.medium.com/encryption-and-access-control-for-web3-using-lighthouse-71dc4e7d0632)

_Конфиденциальность_

19. Shamir A. How to share a secret // Communications of the ACM. 1979. Vol. 22, No. 11. P. 612–613.
20. Rabin M.O. Efficient dispersal of information for security, load balancing, and fault tolerance // Journal of the ACM. 1989. Vol. 36, No. 2. P. 335–348.
21. Krawczyk H. Secret Sharing Made Short // CRYPTO '93. LNCS 773. Springer, 1993.
22. Rivest R.L. All-or-Nothing Encryption and the Package Transform // FSE '97. LNCS 1267. 1997.
23. Resch J.K., Plank J.S. AONT-RS: Blending Security and Performance in Dispersed Storage Systems // FAST '11. San Jose, 2011. [https://www.usenix.org/conference/fast11/aont-rs-blending-security-and-performance-dispersed-storage-systems](https://www.usenix.org/conference/fast11/aont-rs-blending-security-and-performance-dispersed-storage-systems)
24. Kapusta K., Memmi G. Data protection by means of fragmentation in various different distributed storage systems — a survey. arXiv:1706.05960. [https://arxiv.org/pdf/1706.05960](https://arxiv.org/pdf/1706.05960)
25. Kerckhoffs A. La cryptographie militaire // Journal des sciences militaires. 1883.