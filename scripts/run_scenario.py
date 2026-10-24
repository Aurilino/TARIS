"""Исполнение файла сценария одной командой (визуализация, этап 6.2): прогон, журнал, страница просмотрщика.

python scripts/run_scenario.py scenarios/example_all_kinds.json [папка журналов, по умолчанию runs]

Журнал: <папка>/scenario_<имя файла>.taris.json.gz; страница: viewer/build/scenario_<имя файла>.html
(со встроенным журналом; локально открывается двойным щелчком или через python -m http.server).
"""
import sys, os, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from taris_sim import scenario as SC
from taris_sim.export import export_run

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "runs"
try:
    sc = SC.load(path)
except ValueError as err:
    print(err)
    sys.exit(1)
stem = os.path.splitext(os.path.basename(path))[0]
b = sc["base"]
print(f"Сценарий «{sc.get('name', stem)}»: {b.get('scenario', 'S6')}, {b.get('world', 'main')}, {b.get('policy', 'crisis_oracle13')}, "
      f"вмешательств {len(sc.get('interventions', []))}")
t0 = time.time()
w, ch, res = SC.run(sc)
log_path = os.path.join(out, f"scenario_{stem}.taris.json.gz")
export_run(w, ch, res, log_path, scenario=b.get("scenario", "S6"), agg_days=SC.WORLDS[b.get("world", "main")][1])
print(f"прогон и журнал: {time.time() - t0:.0f} с, {os.path.getsize(log_path) / 2**20:.2f} МБ — {log_path}")
for it in w.interventions:
    print("  вмешательство:", {k: v for k, v in it.items()})
here = os.path.dirname(os.path.abspath(__file__))
build = os.path.join(here, "..", "viewer", "build", f"scenario_{stem}")
subprocess.run([sys.executable, os.path.join(here, "build_viewer.py"), log_path, build], check=True)
print("страница:", os.path.normpath(os.path.join(build, "taris_viewer.html")))
