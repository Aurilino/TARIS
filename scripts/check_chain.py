"""Проверки модулей data и chain (шаг 3, пункты 3.2–3.3)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from taris_sim.config import scenario, MINUTE
from taris_sim.world import World
from taris_sim.chain import Chain, merge_intervals
from taris_sim.data import DataStream


def run(cfg):
    w = World(cfg); ch = Chain(w); ds = DataStream(cfg, len(w.nodes))
    c, nd = cfg, w.nodes
    rows = []
    # 1. Шум пингов
    expo = (nd.alive_to - nd.alive_from).clip(lower=0).sum()
    n_pings = expo / c.ping
    exp_noise = n_pings * c.ping_noise
    obs_noise = ch.n_noise_pings
    rows.append(("Ложные провалы пингов", obs_noise, exp_noise, (obs_noise - exp_noise) / np.sqrt(exp_noise)))
    # 2. Шум аудитов
    a = ch.audits
    exp_an = len(a) * c.audit_noise
    obs_an = int(a.noise_fail.sum())
    rows.append(("Ложные провалы аудитов", obs_an, exp_an, (obs_an - exp_an) / np.sqrt(exp_an * (1 - c.audit_noise))))
    # 3. Доля аудитов при недоступном узле = доля времени в простое (истина)
    td = merge_intervals(ch.true_down.drop(columns="source").assign(source="own")).drop(columns="source")
    end = np.minimum(td.end.to_numpy(), nd.alive_to.to_numpy()[td.node] if False else td.end.to_numpy())
    # время простоя внутри жизни узла (уход до конца горизонта засчитывается целиком, как и аудиты в это время не проводятся)
    alive_to = nd.alive_to.to_numpy()[td.node]
    down_time = np.clip(np.minimum(td.end.to_numpy(), alive_to) - td.start.to_numpy(), 0, None).sum()
    share_true = down_time / expo
    share_obs = 1 - a.available.mean()
    se = np.sqrt(share_true * (1 - share_true) / len(a))
    rows.append(("Аудит при недоступном узле, доля", share_obs, share_true, (share_obs - share_true) / se))
    df = pd.DataFrame(rows, columns=["Проверка", "Наблюдается", "Ожидается", "z"])
    df["Итог"] = np.where(df.z.abs() < 3, "OK", "ПРОВЕРИТЬ")
    # 4. Ошибка измерения длительности реальных простоев (в минутах): по модулю меньше интервала пинга
    u = ch.obs_unmerged[ch.obs_unmerged.source == "own"]
    err_start = ((u.end - u.start) - (u.true_end - u.true_start)) / MINUTE
    # 5. Поток блоков
    blocks_ok = len(ds.objects) == c.blocks_per_epoch * c.horizon_days
    # 6. Пошаговый интерфейс воспроизводим
    seeds1 = [Chain(w).step()["seed"] for _ in range(1)]
    ch2 = Chain(w); s1 = [ch2.step()["seed"] for _ in range(3)]
    ch2.reset(); s2 = [ch2.step()["seed"] for _ in range(3)]
    return df, err_start, blocks_ok, s1 == s2


if __name__ == "__main__":
    df, err, blocks_ok, rep_ok = run(scenario("S2"))
    print(df.round(4).to_string(index=False))
    print(f"\nОшибка измерения длительности реального простоя: от {err.min():.2f} до {err.max():.2f} мин, в среднем {err.mean():.2f} (должно быть в (-5, 5), в среднем около 0)")
    print(f"Поток блоков: {'OK' if blocks_ok else 'ОШИБКА'}; воспроизводимость step(): {'OK' if rep_ok else 'ОШИБКА'}")
