"""График I: профиль предвестника глазами сети (шаг 3, пункт 3.9; шаг Е плана кризисов).

Во сколько раз частота наблюдаемых событий у узлов-целей выше их фона в каждые сутки до кризиса —
отдельно для предсказуемых кризисов, ложных тревог и внезапных кризисов. Бури — короткие простои по пингам,
износ партии — проваленные аудиты. Строки — согласованный и несогласованный мир.

    python scripts/plot_crisis.py figures [число зёрен, по умолчанию 6]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from check_crisis import observed_events, excluded, background, window_excess

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
# проверенная категориальная палитра (слоты 1–3) — порядок фиксирован
GROUPS = [("предсказуемые", "#2a78d6"), ("ложные тревоги", "#eb6834"), ("внезапные", "#1baf7a")]
# окна [d, d + шаг) суток относительно начала кризиса; последнее кончается в момент кризиса — сам кризис не входит
PANELS = [("storm", "short", "Буря: короткие простои по пингам", np.arange(-8, 0)),
          ("wear", "audit", "Износ партии: проваленные аудиты, окна по 2 суток", np.arange(-30, 0, 2))]


def profile(ws, chains, typ, kind, offsets):
    """Сумма наблюдаемых и фоновых событий по группам эпизодов для каждых суток относительно момента кризиса."""
    acc = {g: (np.zeros(len(offsets)), np.zeros(len(offsets))) for g, _ in GROUPS}
    step = offsets[1] - offsets[0]
    for w, ch in zip(ws, chains):
        ev, ex = observed_events(w, ch), excluded(w)
        r0 = background(w, ev, ex)
        for e in w.crisis.episodes:
            if e["type"] != typ:
                continue
            g = "ложные тревоги" if e["false_alarm"] else ("предсказуемые" if e["predictable"] else "внезапные")
            O, E = acc[g]
            for j, d in enumerate(offsets):
                o, m = window_excess(w, ev, ex, r0, e, kind, e["tau"] + d, e["tau"] + d + step, exclude_window=False)
                O[j] += o; E[j] += m
    return {g: O / np.maximum(E, 1e-12) for g, (O, E) in acc.items()}


fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharey=False)
for row, mode in enumerate(("matched", "mismatched")):
    ws = [World(scenario("S6", world_mode=mode, seed=s)) for s in range(1, SEEDS + 1)]
    chains = [Chain(w) for w in ws]
    for col, (typ, kind, title, offs) in enumerate(PANELS):
        ax = axes[row, col]
        prof = profile(ws, chains, typ, kind, offs)
        ax.axhline(1.0, color="#9E9E9E", lw=1, ls="--", zorder=1)
        for g, c in GROUPS:
            ax.plot(offs, prof[g], color=c, lw=2, marker="o", ms=4, label=g, zorder=3)
        ax.set_ylim(0, max(2.0, 1.1 * max(np.nanmax(p) for p in prof.values())))
        # подписи у правого края; если они ближе 7% высоты, раздвигаем
        y0, y1 = ax.get_ylim()
        ends = sorted(((prof[g][-1], g) for g, _ in GROUPS))
        ys, gap = [], 0.07 * (y1 - y0)
        for y, g in ends:
            ys.append(max(y, ys[-1] + gap) if ys else y)
        for (y, g), ly in zip(ends, ys):
            ax.annotate(g, (offs[-1], y), xytext=(offs[-1] + 0.25 * (offs[1] - offs[0]) + 0.3, ly), textcoords="data",
                        va="center", fontsize=9, color="#333333")
        ax.set_title(f"{title}\n{'согласованный' if mode == 'matched' else 'несогласованный'} мир", fontsize=10, loc="left")
        ax.set_xlabel("начало окна, сутки до кризиса")
        ax.set_ylabel("частота к фону узлов-целей")
        ax.set_xlim(offs[0] - 0.5, offs[-1] + (3 if col == 0 else 9))
        ax.grid(axis="y", color="#E6E6E6", lw=0.8)
axes[0, 0].legend(frameon=False, loc="upper left")
fig.suptitle(f"I. Профиль предвестника глазами сети (S6, {SEEDS} мира на режим)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
path = os.path.join(OUT, "I_crisis_precursor_profile.png")
fig.savefig(path, dpi=150)
print("готово:", path)
