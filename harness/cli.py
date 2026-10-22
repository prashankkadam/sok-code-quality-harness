from __future__ import annotations

import argparse
import logging


def _list(s: str | None) -> list[str] | None:
    return s.split(",") if s else None


def main() -> None:
    p = argparse.ArgumentParser("harness")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lcbpro-build")
    sub.add_parser("rag-build")
    t = sub.add_parser("train")
    t.add_argument("model", choices=["g31", "g26"])
    for name in ("main", "extended", "agent", "h2h"):
        q = sub.add_parser(name)
        q.add_argument("--models")
        q.add_argument("--techniques")
        q.add_argument("--benches")
        q.add_argument("--budgets")
        q.add_argument("--seeds")
    b = sub.add_parser("h2h-bon-matched")
    b.add_argument("--ks")
    b.add_argument("--seeds")
    fc = sub.add_parser("fidelity")
    fc.add_argument("techniques")
    fc.add_argument("--model", default="g31")
    fc.add_argument("--seeds")
    sub.add_parser("tables")
    sub.add_parser("ledger")
    m = sub.add_parser("manual-sample")
    m.add_argument("--per-model", type=int, default=110)
    m.add_argument("--raters", default="A,B")
    a = sub.add_parser("manual-agreement")
    a.add_argument("rated", nargs="+")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    from . import run
    seeds = [int(x) for x in _list(getattr(args, "seeds", None)) or []] or None
    if args.cmd == "lcbpro-build":
        from .benchmarks.correctness import build
        build()
    elif args.cmd == "rag-build":
        from .benchmarks.base import get
        from .config import load
        from .techniques.rag import build
        cfg = load("benchmarks")
        benches = {b for d in ("COR", "RM", "EFF", "SEC") for b in cfg["main"][d] if b != "deepswe"} | set(cfg["h2h"])
        evals = [t for b in sorted(benches) for t in get(b).tasks()]
        print(build(evals))
    elif args.cmd == "train":
        print(run.train_main(args.model))
    elif args.cmd == "main":
        run.main_grid(_list(args.models), _list(args.techniques), [int(b) for b in _list(args.budgets) or []] or None,
                      _list(args.benches), seeds)
    elif args.cmd == "extended":
        run.extended_grid(_list(args.models), seeds)
    elif args.cmd == "agent":
        run.agent_grid(_list(args.models), seeds)
    elif args.cmd == "h2h":
        run.h2h(_list(args.techniques), seeds)
    elif args.cmd == "h2h-bon-matched":
        print(run.h2h_bon_matched([int(k) for k in _list(args.ks)] if args.ks else None, seeds))
    elif args.cmd == "fidelity":
        for hid in _list(args.techniques):
            print(run.fidelity_check(hid, args.model, seeds))
    elif args.cmd == "tables":
        from .tables import write_all
        print(write_all())
    elif args.cmd == "ledger":
        from .ledger import write
        print(write())
    elif args.cmd == "manual-sample":
        from .config import grid
        from .stats import draw_manual_sample
        print(draw_manual_sample("main", grid("main_grid"), args.per_model, raters=_list(args.raters)))
    elif args.cmd == "manual-agreement":
        from pathlib import Path
        from .stats import manual_agreement
        print(manual_agreement("main", [Path(x) for x in args.rated]))


if __name__ == "__main__":
    main()
