"""Библиотека готовых сценариев (визуализация, этап 6.4): прогон всех файлов scenarios/library/*.json
и одна страница просмотрщика со всеми журналами и выбором сценария.

python scripts/build_library.py            # считает только сценарии без журнала, затем собирает страницу
python scripts/build_library.py --rerun    # пересчитать все

Журналы: runs/library/<имя>.taris.json.gz; страница: viewer/build/library/taris_viewer.html
(и taris_viewer_artifact.html — для публикации). Сценарии считаются параллельно (по 4).
"""
import sys, os, glob, json, base64, time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def run_one(path: str) -> str:
    from taris_sim import scenario as SC
    from taris_sim.export import export_run
    sc = SC.load(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(ROOT, "runs", "library", f"{stem}.taris.json.gz")
    t0 = time.time()
    w, ch, res = SC.run(sc)
    b = sc["base"]
    export_run(w, ch, res, out, scenario=b.get("scenario", "S6"), agg_days=SC.WORLDS[b.get("world", "main")][1])
    warn = "".join(f"\n    ПРЕДУПРЕЖДЕНИЕ: {m}" for m in SC.empty_targets(w))
    return f"{stem}: {time.time() - t0:.0f} с, потеряно {len(res.losses)}, {os.path.getsize(out) / 2**20:.2f} МБ" + warn


def main():
    rerun = "--rerun" in sys.argv
    files = sorted(glob.glob(os.path.join(ROOT, "scenarios", "library", "*.json")))
    os.makedirs(os.path.join(ROOT, "runs", "library"), exist_ok=True)
    log_of = lambda p: os.path.join(ROOT, "runs", "library", os.path.splitext(os.path.basename(p))[0] + ".taris.json.gz")
    todo = [p for p in files if rerun or not os.path.exists(log_of(p))]
    if todo:
        print(f"считаю сценариев: {len(todo)}", flush=True)
        with ProcessPoolExecutor(max_workers=4) as ex:
            for msg in ex.map(run_one, todo):
                print(" ", msg, flush=True)
    lib = []
    for p in files:
        sc = json.load(open(p, encoding="utf-8"))
        lib.append({"name": sc.get("name", os.path.basename(p)), "description": sc.get("description", ""),
                    "file": os.path.basename(p), "b64": base64.b64encode(open(log_of(p), "rb").read()).decode("ascii")})
    tpl = open(os.path.join(ROOT, "viewer", "viewer_template.html"), encoding="utf-8").read()
    page = tpl.replace('<script id="log-lib" type="application/json">[]</script>',
                       '<script id="log-lib" type="application/json">' + json.dumps(lib, ensure_ascii=False) + "</script>")
    out_dir = os.path.join(ROOT, "viewer", "build", "library")
    os.makedirs(out_dir, exist_ok=True)
    wrapper = ('<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
               '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
    local = wrapper + page.replace("<style>", "<style>\n:root{padding-top:env(safe-area-inset-top,0px);"
                                              "padding-bottom:env(safe-area-inset-bottom,0px)}", 1)
    local = local.replace("</style>", "</style>\n</head>\n<body>", 1) + "\n</body>\n</html>\n"
    for name, text in (("taris_viewer.html", local), ("taris_viewer_artifact.html", page)):
        q = os.path.join(out_dir, name)
        open(q, "w", encoding="utf-8").write(text)
        print(f"{os.path.normpath(q)}: {os.path.getsize(q) / 2**20:.2f} МБ, сценариев {len(lib)}")


if __name__ == "__main__":
    main()
