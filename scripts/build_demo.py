"""Сборка интерактивной страницы среды: экспорт мира + подстановка в шаблон viewer/env_demo_template.html."""
import os, subprocess, sys
here = os.path.dirname(__file__)
out_html = sys.argv[1] if len(sys.argv) > 1 else "taris_env_demo.html"
subprocess.run([sys.executable, os.path.join(here, "export_demo.py"), "demo_world.json"], check=True)
tpl = open(os.path.join(here, "..", "viewer", "env_demo_template.html"), encoding="utf-8").read()
data = open("demo_world.json", encoding="utf-8").read()
open(out_html, "w", encoding="utf-8").write(tpl.replace("__DATA__", data))
print("готово:", out_html)
