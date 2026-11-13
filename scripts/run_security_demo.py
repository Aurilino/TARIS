"""Прогон TARIS и отдельный отчёт защищённого чтения со сверкой на байтах."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import UniformPolicy
from taris_sim.security import SecurityReplay, SecurityError, load, SCHEMA
from crypto_demo.replay import verify_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/security/demo"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--days", type=int, default=12)
    parser.add_argument("--scenario", choices=("S0", "S2", "S6"), default="S0")
    parser.add_argument("--attack-file", type=Path)
    parser.add_argument("--obj", type=int, default=0)
    parser.add_argument("--times", default="1,3")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.k < args.n <= 32 or args.days < 4:
        parser.error("Нужны 1 <= k < n <= 32 и days >= 4; для S6 — не менее 200 суток")
    if args.scenario == "S6" and args.days < 200:
        parser.error("S6 требует не менее 200 суток")
    try:
        times = [float(x) for x in args.times.split(",")]
        cfg = scenario(args.scenario, seed=args.seed, horizon_days=args.days,
                       train_days=max(1, args.days // 2), n_initial_nodes=50,
                       node_join_rate=.1, n_dc=10, n_providers=5, n_regions=3,
                       blocks_per_epoch=1, k=args.k)
        world = World(cfg)
        chain = Chain(world)
        data = DataStream(cfg, len(world.nodes))
        result = Ledger(world, chain, data).run(UniformPolicy(args.n))
        if args.attack_file:
            document = load(args.attack_file)
        else:
            # An explicit demonstration, not a frequency model of adversary behaviour.
            created = float(result.objects.at[args.obj, "created_epoch"])
            initial = result.fragments[(result.fragments.obj == args.obj) &
                                       (result.fragments.t_from == created)]
            nodes = list(map(int, initial.node[:args.k]))
            if len(nodes) < args.k:
                raise SecurityError("Не удалось выбрать k начальных держателей; задайте другой объект")
            document = {"schema": SCHEMA, "events": [
                {"kind": "compromise_node", "t": created, "attacker": "demo", "node": node}
                for node in nodes] + [{"kind": "compromise_access_key", "t": created + 2., "attacker": "demo"}]}
        replay = SecurityReplay(result, document)
        report = replay.report([(args.obj, t) for t in times])
        args.out.mkdir(parents=True, exist_ok=True)
        proofs = [verify_snapshot(result, document, args.obj, t, args.out) for t in times]
        for name, value in [("scenario.json", document), ("security_report.json", report),
                            ("byte_checks.json", proofs), ("config.json", json.loads(cfg.to_json()))]:
            (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Сценарий:", args.scenario, "; равномерная политика; мир seed =", args.seed)
        for query, proof in zip(report["queries"], proofs, strict=True):
            print(f"t={query['t']:g}: клиент={proof['client_result']}, "
                  f"собран шифротекст={any(a['ciphertext_collected'] for a in query['attackers'])}, "
                  f"раскрыт текст={any(a['plaintext_disclosed'] for a in query['attackers'])}")
        print("Отчёты:", args.out.resolve())
    except (SecurityError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
