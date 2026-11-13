"""CLI изолированного прототипа. Пароли вводятся скрыто через getpass."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from getpass import getpass
import json
from crypto_demo.storage import Client, DemoError, MAX_OBJECT, read_limited, atomic


def main():
    parser = argparse.ArgumentParser(description="Локальный прототип TARIS: зашифрованный манифест")
    parser.add_argument("--root", type=Path, default=Path("runs/crypto_demo/manual"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    store = sub.add_parser("store")
    store.add_argument("input", type=Path)
    store.add_argument("--k", type=int, default=10)
    store.add_argument("--n", type=int, default=13)
    read = sub.add_parser("read")
    read.add_argument("object_id")
    read.add_argument("output", type=Path)
    repair = sub.add_parser("repair")
    repair.add_argument("object_id")
    sub.add_parser("rotate")
    reencrypt = sub.add_parser("reencrypt")
    reencrypt.add_argument("object_id")
    for name in ("backup", "restore"):
        command = sub.add_parser(name)
        command.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        password = getpass("Пароль клиентского ключевого файла: ")
        if args.command == "init" and password != getpass("Повторите пароль: "):
            parser.error("Пароли не совпали")
        client = Client(args.root, password)
        if args.command == "init":
            client.initialize()
            print("Клиентское хранилище создано. Используйте только тестовые данные.")
        elif args.command == "store":
            raw = read_limited(args.input, MAX_OBJECT, "INPUT_UNAVAILABLE")
            print(json.dumps(client.store(raw, args.k, args.n), indent=2))
        elif args.command == "read":
            # Do not replace an existing file or client/storage state.
            if args.output.exists() or args.output.resolve().is_relative_to(args.root.resolve()):
                parser.error("Выберите новый выходной файл вне каталога прототипа")
            plaintext = client.read_latest(args.object_id)
            atomic(args.output, plaintext)
            print("Проверенный объект записан:", args.output.resolve())
        elif args.command == "repair":
            plan = client.repair_plan(args.object_id)
            client.agent.repair(plan)
            print("Фрагменты восстановлены по несекретному плану.")
        elif args.command == "rotate":
            client.rotate_access()
            print("Активный ключ доступа заменён; оболочки перепакованы.")
        elif args.command == "reencrypt":
            print(json.dumps(client.reencrypt(args.object_id), indent=2))
        elif args.command == "backup":
            if args.path.exists() or args.path.resolve().is_relative_to(args.root.resolve()):
                parser.error("Выберите новый файл копии вне каталога прототипа")
            client.backup_key(args.path)
            print("Зашифрованный ключевой файл скопирован; доверенные указатели резервируйте отдельно.")
        elif args.command == "restore":
            client.restore_key(args.path)
            print("Ключ восстановлен. Для новых записей создан новый ключ доступа.")
    except DemoError as error:
        print("Ошибка:", error.code, file=sys.stderr)
        return 1
    except OSError as error:
        print("Ошибка файловой операции:", type(error).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
