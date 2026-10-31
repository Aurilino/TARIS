"""Параметры среды (шаг 3, раздел «Параметры по умолчанию»).

Единица времени — сутки (эпоха). Длительности простоев задаются в часах и переводятся в сутки внутри модели.
"""
from dataclasses import dataclass, asdict, replace
import hashlib
import json

HOUR = 1.0 / 24.0
MINUTE = 1.0 / 1440.0
YEAR = 365.0


@dataclass(frozen=True)
class Config:
    seed: int = 1
    world_mode: str = "matched"            # "matched" — согласованный мир, "mismatched" — несогласованный
    horizon_days: int = 3 * 365            # горизонт симуляции
    train_days: int = 2 * 365              # обучение / проверка

    # --- 3.1. Сеть узлов
    n_initial_nodes: int = 1000
    node_join_rate: float = 0.6            # новых узлов в сутки (примерно уравновешивает уход)
    initial_age_mean_days: float = 365.0   # средний возраст узлов в начальный момент
    n_dc: int = 40
    n_providers: int = 10
    n_regions: int = 5
    operator_size_p: float = 0.6           # размер оператора ~ Geometric(p): P(1 узел) = p
    operator_same_dc_p: float = 0.7        # узлы одного оператора в одном дата-центре
    operator_same_subnet_p: float = 0.5    # ...и в одной подсети /24
    operator_join_gap_days: float = 10.0   # средний интервал регистрации узлов одного оператора
    hw_classes: tuple = ("consumer_hdd", "enterprise_hdd", "ssd")
    hw_probs: tuple = (0.5, 0.3, 0.2)
    beta_hw: tuple = (0.3, 0.0, -0.2)      # истинные эффекты класса на ln интенсивности
    beta_region_sd: float = 0.15           # разброс истинных эффектов регионов
    mean_event_rate: float = 1.0 / 30.0    # средняя по сети частота страховых случаев, в сутки (калибровка beta_0)
    heterogeneity_a: float = 0.5           # дисперсия истинного множителя Theta
    kappa_age: float = 0.8                 # форма по возрасту (согласованный мир)
    age_unit_days: float = 30.0            # возраст в формуле берётся как 1 + age / age_unit
    b_inf: float = 2.0                     # «ванна» (несогласованный мир): детская смертность
    A_inf_days: float = 30.0
    b_wear: float = 1.0                    # ...и износ
    A_wear_days: float = 3 * 365.0

    # --- 3.4. Тяжесть простоев
    franchise_d_hours: float = 0.25        # франшиза d = 15 минут
    T_rep_hours: float = 24.0              # порог переразмещения
    weibull_scale_hours: float = 5.0       # согласованный мир
    weibull_shape: float = 0.6
    alpha_hw: tuple = (0.3, 0.0, -0.2)     # эффект класса на ln масштаба длительности
    severity_node_sd: float = 0.3          # индивидуальный разброс тяжести
    lognorm_median_hours: float = 2.0      # несогласованный мир
    lognorm_sigma: float = 1.6

    # --- уход из сети
    churn_extra_new: float = 1.0 / 120.0   # добавка у новых узлов, в сутки
    churn_A_days: float = 60.0
    churn_base: float = 1.0 / (6 * 365.0)

    # --- микропрерывания
    micro_rate: float = 1.0                # в сутки
    micro_info_b: float = 0.5              # информативность b_mu
    micro_noise_sd: float = 0.5            # сигма логнормального шума epsilon
    micro_mean_minutes: float = 3.0

    # --- тихая потеря данных
    silent_rate: float = 1.0 / (5 * 365.0)

    # --- шоки кластеров (дата-центр, провайдер, регион)
    shocks_enabled: bool = True
    shock_rate_per_year: tuple = (0.5, 1.0 / 3.0, 0.1)
    shock_total_share: float = 0.10        # доля шоков с полной гибелью данных
    shock_scale_hours: tuple = (6.0, 4.0, 3.0)
    shock_shape: float = 0.8
    storm_multiplier: float = 40.0         # «буря» (несогласованный мир)
    storm_hours: float = 12.0

    # --- 3.9. Кризисы и предвестники
    # Для каждого типа: rate — частота nu_e (в год на кластер), pi — доля предсказуемых,
    # q — ложных тревог на один предсказуемый кризис, W_days — средняя длина окна предвестника,
    # s — доля узлов кластера-цели с предвестником, rho — рост интенсивности канала к моменту кризиса.
    crises_enabled: bool = False
    pre_gamma: float = 2.0                 # форма роста g(u) = u^gamma (согласованный мир)
    pre_k_W: float = 4.0                   # форма гамма-распределения длины окна (согласованный мир)
    pre_u0: float = 0.5                    # ступенька g(u) = 1[u >= u0] (несогласованный мир)
    pre_sigma_W: float = 1.0               # сигма логнормальной длины окна (несогласованный мир)
    mism_s_factor: float = 0.5             # несогласованный мир: s_e меньше...
    mism_q_factor: float = 2.0             # ...а ложных тревог больше
    # буря в дата-центре: исход — шок дата-центра (согласованный мир) или буря (несогласованный)
    cr_storm_rate: float = 0.5
    cr_storm_pi: float = 0.7
    cr_storm_q: float = 0.5
    cr_storm_W_days: float = 3.0
    cr_storm_s: float = 0.6
    cr_storm_rho: float = 5.0
    # каскад: вторичная буря в других дата-центрах того же провайдера
    cr_cascade_p: float = 0.3
    cr_cascade_delay_hours: float = 6.0
    cr_cascade_s: float = 0.6
    cr_cascade_rho: float = 5.0
    # износ партии дисков (класс оборудования × дата-центр)
    cr_wear_rate: float = 0.1
    cr_wear_pi: float = 0.9
    cr_wear_q: float = 0.5
    cr_wear_W_days: float = 30.0
    cr_wear_s: float = 0.8
    cr_wear_rho: float = 10.0
    cr_wear_p_death: float = 0.3           # вероятность гибели узла партии за период
    cr_wear_period_days: float = 30.0
    # массовый уход из региона
    cr_exodus_rate: float = 0.2
    cr_exodus_pi: float = 0.8
    cr_exodus_q: float = 0.5
    cr_exodus_W_days: float = 21.0
    cr_exodus_s: float = 0.5
    cr_exodus_rho: float = 5.0
    cr_exodus_rho_join: float = 0.3        # регистрации падают до этой доли обычного уровня
    cr_exodus_multiplier: float = 30.0     # Gamma_churn
    cr_exodus_period_days: float = 30.0
    # уязвимость ПО (узлы одной версии клиента); частота — на всю сеть.
    # Срабатывает через ~Exp(trigger_days) после выпуска версии и бьёт только по уже обновившимся.
    cr_software_rate: float = 1.0
    cr_software_trigger_days: float = 3.0
    cr_software_pi: float = 0.5
    cr_software_q: float = 0.5
    cr_software_W_days: float = 5.0
    cr_software_s: float = 0.8             # среди обновившихся узлов
    cr_software_rho: float = 5.0
    cr_software_L_mean_hours: float = 12.0
    cr_software_total_share: float = 0.05
    sw_release_every_days: float = 60.0    # новые версии клиента
    sw_update_delay_days: float = 10.0     # средняя задержка обновления узла
    # пожар в дата-центре: внезапная полная гибель
    cr_fire_rate: float = 0.05
    # атака Сибиллы — вместе с модулем adversary
    cr_sybil_enabled: bool = False

    # --- 3.10. Устройства пользователей (слабые нестабильные узлы вне дата-центров)
    devices_enabled: bool = False
    dev_ratio: float = 2.0                 # устройств на один узел в дата-центре (и в начале, и среди новых)
    dev_m_rate: float = 6.0                # множители к узлу с теми же признаками: частота простоев...
    dev_m_dur: float = 2.0                 # ...масштаб длительности простоя...
    dev_m_churn: float = 10.0              # ...уход из сети...
    dev_m_micro: float = 3.0               # ...микропрерывания...
    dev_m_silent: float = 3.0              # ...тихие потери
    dev_capacity_mean: float = 50.0        # средняя ёмкость, фрагментов
    dev_home_providers: int = 4            # домашние провайдеры (свои номера после провайдеров дата-центров)
    dev_multi_share: float = 0.1           # доля пользователей с 2–3 устройствами
    dev_hw_probs: tuple = (0.7, 0.0, 0.3)  # бытовой HDD, серверный HDD, SSD

    # --- шаг 2, пункт 2.4. Граф связей (модель TARIS; считается только по «взгляду сети»)
    graph_K_rho: float = 10.0              # кредибилити инфраструктурной связи: Z = n / (n + K)
    graph_b_wallet: float = 0.95           # доказательная сила сигналов владения (noisy-OR): общий кошелёк...
    graph_b_subnet: float = 0.6            # ...общая подсеть /24...
    graph_b_join: float = 0.2              # ...регистрация с разницей меньше graph_join_window_days
    graph_join_window_days: float = 1.0
    graph_store_min: float = 0.2           # пары слабее этого не хранятся (разреженный граф)
    graph_cluster_cut: float = 0.5         # кластеры: иерархическая кластеризация, связь не слабее этого
    graph_snapshot_days: int = 30          # шаг снимков графа в журнале

    # --- 3.2. Данные и кодирование
    blocks_per_epoch: int = 20
    block_growth_per_year: float = 0.0     # рост потока блоков
    object_size: float = 1.0               # условная единица объёма
    k: int = 10
    capacity_mean_fragments: float = 1000.0
    capacity_sd: float = 0.5               # сигма логнормального разброса ёмкости

    # --- 3.3. Наблюдение
    ping_interval_minutes: float = 5.0
    ping_noise: float = 0.001              # ложный провал пинга (шум на стороне сети)
    audits_per_epoch: int = 1
    audit_interval_days: int = 1           # аудит раз в столько суток (абляция H7: редкие аудиты; 1 — каждые сутки)
    alarm_step_hours: int = 24             # шаг тревог и переноса для политик с fast = True (24 — раз в сутки, как всё остальное)
    audit_noise: float = 0.005             # ложный провал аудита PDP

    # --- 3.4 / 3.6. Восстановление и учёт
    repair_hours: float = 2.0              # фиксированная подготовка операции восстановления
    transfer_rate_size_per_hour: float | None = None  # скорость передачи; None оставляет старую задержку
    track_objects: int = 300               # объектов для точного учёта недоступности

    @property
    def d(self) -> float:
        return self.franchise_d_hours * HOUR

    @property
    def T_rep(self) -> float:
        return self.T_rep_hours * HOUR

    @property
    def ping(self) -> float:
        return self.ping_interval_minutes * MINUTE

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    def hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


def scenario(name: str, **overrides) -> Config:
    """Готовые сценарии. S1, S3–S5 и S7 добавляются вместе с модулями validate и adversary."""
    base = {
        "S0": dict(shocks_enabled=False),
        "S2": dict(shocks_enabled=True),
        # S2, где шоки дата-центров заменены библиотекой кризисов (пункт 3.9)
        "S6": dict(shocks_enabled=True, crises_enabled=True,
                   shock_rate_per_year=(0.0,) + Config.shock_rate_per_year[1:]),
        # S6 + устройства пользователей (пункт 3.10)
        "S8": dict(shocks_enabled=True, crises_enabled=True, devices_enabled=True,
                   shock_rate_per_year=(0.0,) + Config.shock_rate_per_year[1:]),
    }
    if name not in base:
        raise ValueError(f"Сценарий {name} ещё не реализован")
    return replace(Config(), **{**base[name], **overrides})
