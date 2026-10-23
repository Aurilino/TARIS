"""Сборка просмотрщика TARIS со встроенным журналом прогона (визуализация, этап 2).

python scripts/build_viewer.py runs/main_crisis_oracle13_matched.taris.json.gz [папка, по умолчанию viewer/build]

Шаблон viewer/viewer_template.html написан без <!doctype>/<html>/<head>/<body> — так его принимает
публикация Artifact. Скрипт пишет две страницы:
  taris_viewer.html          — для локального просмотра (с обёрткой документа), python -m http.server
  taris_viewer_artifact.html — для публикации (без обёртки)
Журнал (gzip) встраивается в base64; страница распаковывает его сама. Кнопка «Открыть журнал» открывает любой другой.
"""
import base64, os, sys

root = os.path.join(os.path.dirname(__file__), "..")
log_path = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "viewer", "build")
os.makedirs(out_dir, exist_ok=True)
tpl = open(os.path.join(root, "viewer", "viewer_template.html"), encoding="utf-8").read()
b64 = base64.b64encode(open(log_path, "rb").read()).decode("ascii")
page = tpl.replace("__LOG_B64__", b64)
wrapper = ('<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
local = wrapper + page.replace("<style>", "<style>\n:root{padding-top:env(safe-area-inset-top,0px);"
                                          "padding-bottom:env(safe-area-inset-bottom,0px)}", 1)
local = local.replace("</style>", "</style>\n</head>\n<body>", 1) + "\n</body>\n</html>\n"
for name, text in (("taris_viewer.html", local), ("taris_viewer_artifact.html", page)):
    p = os.path.join(out_dir, name)
    open(p, "w", encoding="utf-8").write(text)
    print(f"{p}: {os.path.getsize(p) / 2**20:.2f} МБ")
