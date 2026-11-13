"""Постфактум причины неудачных чтений; не данные обучения агента."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collections import Counter
import json
from taris_sim.config import Config
from taris_sim.world import World
from taris_sim.chain import Chain
from subprojects.taris_variants.run_queued import provenance


def main():
    directory = Path(__file__).with_name("series_i42")
    defaults, counts = Config(), {}
    for path in sorted(directory.glob("s[0-9]*.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["provenance"] == provenance()
        case, config = saved["case"], saved["case"]["config"]
        cfg = Config(**{key: tuple(value) if isinstance(getattr(defaults, key), tuple) else value for key, value in config.items()})
        chain = Chain(World(cfg))
        for profile, variants in case["results"].items():
            for variant, record in variants.items():
                bucket = counts.setdefault((profile, variant), Counter())
                for segment in record["segments"]:
                    if segment["phase"] != "read" or segment["end"] >= cfg.horizon_days:
                        continue
                    job = record["jobs"][segment["job"]]
                    bucket["completed_read_attempts"] += 1
                    if job["status"] != "read_failed":
                        bucket["successful_reads"] += 1
                        continue
                    bucket["failed_reads"] += 1
                    sources, destroyed = set(), False
                    for f in job["sources"]:
                        copy = record["physical_fragments"][f]
                        destroyed |= (copy["physical_end"] is not None and copy["physical_end"] < segment["end"])
                        down = chain.true_down
                        hits = down[(down.node == copy["node"]) & (down.start <= segment["end"]) & (down.end > segment["start"])]
                        sources.update(hits.source)
                    assert destroyed or sources, "Необъяснённая неудача чтения"
                    if destroyed:
                        bucket["copy_destroyed_during_read"] += 1
                    if sources == {"micro"} and not destroyed:
                        bucket["only_micro_outages"] += 1
                    if "micro" in sources:
                        bucket["includes_micro_outage"] += 1
                    if sources - {"micro"}:
                        bucket["includes_other_outage"] += 1
    report = {"schema": "taris-queue-read-diagnostics/1", "scope": "external_post_run",
        "rows": [{"profile": profile, "variant": variant, **dict(counter)} for (profile, variant), counter in counts.items()],
        "note": "Причины могут пересекаться; only_micro_outages исключает уничтожение и другие простои."}
    (directory/"read_diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
