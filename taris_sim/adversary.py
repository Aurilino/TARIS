"""Противник: Сибилла, сговор, ложные заявления, тихие потери и перерегистрация.

План зависит от мира и отдельного rng.ADVERSARY, но не от решений политики.
Истина находится здесь; публичны только заявления и парные сигналы финансирования.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from . import rng as R


class Adversary:
    def __init__(self, world):
        self.w, self.cfg = world, world.cfg
        c = self.cfg
        for name in ("adv_sybil_share", "adv_pair_leak", "adv_bad_fraction",
                     "adv_false_claim_fraction", "adv_whitewash_fraction"):
            value = getattr(c, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name}: требуется число от 0 до 1")
        if c.adv_sybil_share >= 1:
            raise ValueError("adv_sybil_share должен быть меньше 1")
        for name in ("adv_wave_days", "adv_rate_multiplier", "adv_silent_multiplier"):
            if not np.isfinite(getattr(c, name)) or getattr(c, name) <= 0:
                raise ValueError(f"{name}: требуется положительное конечное число")
        if (not 0 <= c.adv_wave_width_days < c.adv_wave_days or
                not np.isfinite(c.adv_whitewash_gap_days) or c.adv_whitewash_gap_days <= 0):
            raise ValueError("Неверные параметры волн регистрации или задержки перерегистрации")
        if type(c.adv_colluding_owners) is not int or c.adv_colluding_owners < 0:
            raise ValueError("adv_colluding_owners: требуется целое >= 0")
        self.attack_t = float(c.adv_attack_day if c.adv_attack_day is not None else .7 * c.horizon_days)
        if not np.isfinite(self.attack_t) or not 0 < self.attack_t < c.horizon_days:
            raise ValueError("adv_attack_day должен быть внутри горизонта и больше 0")
        if c.adv_key_compromise_day is not None and (not np.isfinite(c.adv_key_compromise_day) or
                                                   not 0 <= c.adv_key_compromise_day < c.horizon_days):
            raise ValueError("adv_key_compromise_day должен быть внутри горизонта")
        self.primary = []
        self.whitewash = []
        self.public_funding_pairs = []
        self.attack_nodes = []

    def append_nodes(self):
        w, c = self.w, self.cfg
        nd = w.nodes
        self.base_n = len(nd)
        self.sybil_owner = int(nd.owner.max()) + 1
        self.actual_share = 0.
        for col, value in (("adversarial", False), ("sybil", False), ("bad_storage", False),
                           ("false_claim", False), ("whitewash_parent", -1)):
            nd[col] = value
        nd["dc_claimed"] = nd["dc"]
        nd["risk_join_t"] = nd["join_t"]
        owners = np.sort(nd.owner.unique())
        g = R.stream(c.seed, R.ADVERSARY, 2)
        self.colluders = list(map(int, g.choice(owners, min(len(owners), c.adv_colluding_owners), replace=False)))
        count = int(round(len(nd) * c.adv_sybil_share / max(1-c.adv_sybil_share, 1e-12)))
        if count == 0:
            self.coalition_owners = set(self.colluders)
            return
        rows = []
        g = R.stream(c.seed, R.ADVERSARY, 1)
        wallet0 = int(nd.wallet.max()) + 1
        dcs = np.resize(g.permutation(c.n_dc), count)
        for j in range(count):
            i = self.base_n + j
            dc = int(dcs[j])
            # Отдельная личность с возрастом/волной; риск и собственный поток не заимствуются у честного узла.
            base_join = float(g.choice(nd.join_t.to_numpy()))
            join = math.floor(base_join / c.adv_wave_days) * c.adv_wave_days + g.uniform(0, c.adv_wave_width_days)
            join = min(join, np.nextafter(float(c.horizon_days), 0.))
            hw = int(g.choice(len(c.hw_classes), p=c.hw_probs))
            a = c.heterogeneity_a
            theta = g.gamma(1/a, a) if c.world_mode == "matched" else np.exp(g.normal(-np.log1p(a)/2, np.sqrt(np.log1p(a))))
            bad = R.stream(c.seed, R.ADVERSARY, 3, i).random() < c.adv_bad_fraction
            claim_rng = R.stream(c.seed, R.ADVERSARY, 4, i)
            false = claim_rng.random() < c.adv_false_claim_fraction
            claimed_hw = int(np.argmin(c.beta_hw)) if false else hw
            # Ложный ДЦ остаётся в том же проверяемом провайдере и регионе, если есть альтернатива.
            possible = np.flatnonzero((w.dc_provider == w.dc_provider[dc]) & (w.dc_region == w.dc_region[dc]) & (np.arange(c.n_dc) != dc))
            claimed_dc = int(claim_rng.choice(possible)) if false and len(possible) else dc
            rows.append(dict(owner=self.sybil_owner, join_t=join, dc=dc, dc_claimed=claimed_dc,
                             subnet=f"a{dc}.{i}", provider=int(w.dc_provider[dc]), region=int(w.dc_region[dc]),
                             wallet=wallet0+j, hw_true=hw, hw_claimed=claimed_hw, theta=theta,
                             sev_u=g.normal(0., c.severity_node_sd),
                             micro_eps=np.exp(g.normal(-c.micro_noise_sd**2/2, c.micro_noise_sd)),
                             lin_pred=c.beta_hw[hw]+w.beta_region[w.dc_region[dc]],
                             lam_silent=c.silent_rate*(c.adv_silent_multiplier if bad else 1.),
                             is_device=False, m_rate=c.adv_rate_multiplier, m_dur=1., m_churn=1., m_micro=1., cap_scale=1.,
                             adversarial=True, sybil=True, bad_storage=bad, false_claim=false,
                             whitewash_parent=-1, risk_join_t=join))
        w.nodes = pd.concat([nd, pd.DataFrame(rows)], ignore_index=True)
        w.nodes.index.name = "node"
        self.primary = list(range(self.base_n, len(w.nodes)))
        self.actual_share = len(rows)/len(w.nodes)
        self.coalition_owners = {self.sybil_owner, *self.colluders}

    def append_whitewash(self):
        """Один новый псевдоним после первого собственного простоя > T_rep.

        Триггер — та же базовая шкала OUTAGE/DURATION, что у World; решение не
        зависит от политики. Прежняя личность заканчивает жизнь через T_rep.
        """
        w, c = self.w, self.cfg
        rows = []
        for parent in self.primary:
            if R.stream(c.seed, R.ADVERSARY, 5, parent).random() >= c.adv_whitewash_fraction:
                continue
            r = w.nodes.loc[parent].copy()
            a0, a1 = float(r.alive_from), float(r.alive_to_base)
            if a1 <= a0:
                continue
            grid = np.linspace(a0, a1, 200)
            maximum = float(w.rate(np.full(200, parent), grid).max())*1.05
            times = w._thin(R.stream(c.seed, R.OUTAGE, parent), lambda t: w.rate(np.full(len(t), parent), t), a0, a1, maximum)
            lengths = w.sample_duration(R.stream(c.seed, R.DURATION, parent), parent, len(times))
            candidates = times[lengths > c.T_rep] + c.T_rep
            candidates = candidates[(candidates + c.adv_whitewash_gap_days < a1) &
                                    (candidates + c.adv_whitewash_gap_days < self.attack_t if c.cr_sybil_enabled else True)]
            if not len(candidates):
                continue
            end = float(candidates.min())
            i = len(w.nodes) + len(rows)
            r["join_t"] = end + c.adv_whitewash_gap_days
            r["whitewash_parent"] = parent
            r["wallet"] = int(w.nodes.wallet.max()) + 1 + len(rows)
            r["subnet"] = f"w{int(r.dc)}.{i}"
            rows.append(r)
            self.whitewash.append({"parent": parent, "node": i, "leave_t": end, "join_t": float(r.join_t)})
        if rows:
            w.nodes = pd.concat([w.nodes, pd.DataFrame(rows)], ignore_index=True)
            w.nodes.index.name = "node"
        return bool(rows)

    def funding_signals(self):
        nodes = list(map(int, self.w.nodes.index[self.w.nodes.sybil]))
        for pos, i in enumerate(nodes):
            for j in nodes[pos+1:]:
                # Одно U для пары: p меняет только её наличие, не остальные пары/потоки.
                if R.stream(self.cfg.seed, R.ADVERSARY, 6, i, j).random() < self.cfg.adv_pair_leak:
                    self.public_funding_pairs.append((i, j))

    def apply_lifetimes(self):
        nd, c = self.w.nodes, self.cfg
        if "leave_reason" not in nd:
            nd["leave_reason"] = np.where(np.isfinite(nd.leave_t), "churn", "")
        for row in self.whitewash:
            i, t = row["parent"], row["leave_t"]
            if nd.at[i, "alive_from"] < t < nd.at[i, "alive_to"]:
                nd.at[i, "leave_t"] = t
                nd.at[i, "alive_to"] = t
                nd.at[i, "leave_reason"] = "whitewash"
        # Псевдоним не появляется, если родитель погиб раньше намеченной смены имени.
        for row in self.whitewash:
            parent, i = row["parent"], row["node"]
            if nd.at[parent, "alive_to"] < row["leave_t"]:
                if "cancelled" not in nd:
                    nd["cancelled"] = False
                nd.at[i, "cancelled"] = True
                nd.at[i, "join_t"] = np.inf
                nd.at[i, "alive_to"] = nd.at[i, "alive_from"]
                nd.at[i, "leave_t"] = np.inf
                nd.at[i, "leave_reason"] = ""
        if c.cr_sybil_enabled:
            for i in nd.index[nd.sybil]:
                if nd.at[i, "alive_from"] <= self.attack_t < nd.at[i, "alive_to"]:
                    nd.at[i, "leave_t"] = self.attack_t
                    nd.at[i, "alive_to"] = self.attack_t
                    nd.at[i, "leave_reason"] = "sybil"
                    self.attack_nodes.append(int(i))
                elif nd.at[i, "alive_from"] > self.attack_t:
                    if "cancelled" not in nd:
                        nd["cancelled"] = False
                    nd.at[i, "cancelled"] = True
                    nd.at[i, "join_t"] = np.inf
                    nd.at[i, "alive_to"] = nd.at[i, "alive_from"]
                    nd.at[i, "leave_t"] = np.inf
                    nd.at[i, "leave_reason"] = ""

    def security_scenario(self, horizon):
        """Фактическая коалиция, без утечки криптографических ключей по умолчанию."""
        events = []
        for owner in sorted(self.coalition_owners):
            if owner in set(self.w.nodes.owner) and horizon > 0:
                events.append(dict(kind="compromise_owner", t=0., attacker="adversary", owner=owner))
        key_time = self.cfg.adv_key_compromise_day
        if key_time is not None and key_time < horizon:
            events.append(dict(kind="compromise_access_key", t=float(key_time), attacker="adversary"))
        return {"schema": "taris-security-scenario/1", "events": events}

    def report(self, result):
        horizon = float(result.daily.epoch.max()+1) if len(result.daily) else 0.
        collected = {}
        for row in result.fragments.itertuples(index=False):
            if row.t_from >= min(row.t_to, row.t_dead):
                continue
            if self.w.nodes.at[int(row.node), "owner"] in self.coalition_owners:
                key = int(row.obj)
                by_index = collected.setdefault(key, {})
                by_index[int(row.idx)] = min(by_index.get(int(row.idx), math.inf), float(row.t_from))
        exposures = []
        for obj, indices in sorted(collected.items()):
            if len(indices) >= result.cfg.k:
                exposures.append(dict(obj=obj, t=sorted(indices.values())[result.cfg.k-1]))
        key_time = self.cfg.adv_key_compromise_day
        plaintext = [] if key_time is None else [dict(obj=r["obj"], t=max(r["t"], float(key_time)))
                                                  for r in exposures if max(r["t"], float(key_time)) < horizon]
        before_attack = np.nextafter(self.attack_t, 0.)
        nd = self.w.nodes
        living = (nd.alive_from <= before_attack) & (nd.alive_to > before_attack)
        live_share = float((living & nd.sybil).sum()/living.sum()) if living.any() else None
        return {"schema": "taris-adversary-report/1", "horizon_days": horizon,
                "primary_sybil_nodes": len(self.primary), "primary_table_share": self.actual_share,
                "live_sybil_share_before_attack": live_share if self.cfg.cr_sybil_enabled and self.attack_t < horizon else None,
                "aliases_planned": len(self.whitewash),
                "aliases_registered": int(sum(self.w.nodes.at[r["node"], "join_t"] < horizon for r in self.whitewash)),
                "attack_t": self.attack_t if self.cfg.cr_sybil_enabled and self.attack_t < horizon else None,
                "attack_nodes": self.attack_nodes if self.cfg.cr_sybil_enabled and self.attack_t < horizon else [],
                "losses_at_attack_time": int((result.losses.t == self.attack_t).sum()) if self.cfg.cr_sybil_enabled and self.attack_t < horizon else 0,
                "coalition_owners": sorted(self.coalition_owners),
                "ciphertext_exposures": exposures,
                "plaintext_disclosed_objects": len(plaintext), "plaintext_exposures": plaintext,
                "key_compromise_t": float(key_time) if key_time is not None and key_time < horizon else None,
                "key_model": "encrypted-manifest; explicit independent access-key event; no in-run rotation",
                "warning": "Сбор фрагментов не означает раскрытия текста; частоты атак синтетические."}

    def ownership_report(self, chain, t, threshold=.5):
        """H6-диагностика только по узлам коалиции, без честных посторонних пар."""
        from .graph import LinkGraph
        if not np.isfinite(t) or not 0 <= t <= self.cfg.horizon_days or not 0 < threshold <= 1:
            raise ValueError("Неверный момент или порог диагностики владения")
        graph = LinkGraph(self.w, chain)
        scores = graph.ownership(t)
        nd = self.w.nodes
        registered = graph.join <= t
        selected = registered & nd.owner.isin(self.coalition_owners).to_numpy()
        i, j = np.triu_indices(len(nd), 1)
        keep = selected[i] & selected[j]
        i, j = i[keep], j[keep]
        truth = nd.owner.to_numpy()[i] == nd.owner.to_numpy()[j]
        predicted = scores[i, j] >= threshold
        sybil = nd.sybil.to_numpy(dtype=bool)
        sp = sybil[i] & sybil[j]
        tp = int((truth & predicted).sum())
        fp = int((~truth & predicted).sum())
        positives = int(truth.sum())
        return {"t": float(t), "threshold": float(threshold), "pairs": len(i),
                "true_owner_pairs": positives, "true_positive": tp, "false_positive": fp,
                "precision": tp/(tp+fp) if tp+fp else None,
                "recall": tp/positives if positives else None,
                "sybil_pairs": int(sp.sum()), "sybil_pairs_detected": int((sp & predicted).sum()),
                "sybil_recall": float(predicted[sp].mean()) if sp.any() else None,
                "scope": "registered coalition nodes only; ownership signal, not infrastructure correlation"}
