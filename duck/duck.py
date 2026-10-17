"""duck: search large image-caption datasets and write img2dataset-ready CSVs.

    run.bat --dataset <name> [options]      search one dataset
    run.bat --dataset <name> --help         list that dataset's options
    run.bat --list-datasets                 list the datasets this build knows

Dataset folders come from duck.conf next to this file (copy duck.conf.example).
"""

import argparse
import configparser
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import duck_engine  # noqa: E402
from duck_datasets import REGISTRY  # noqa: E402

CONF_NAME = "duck.conf"


def load_config():
    path = os.path.join(HERE, CONF_NAME)
    if not os.path.exists(path):
        duck_engine.fail(f"{CONF_NAME} not found in {HERE}\n"
                         f"Copy {CONF_NAME}.example to {CONF_NAME} and set the dataset folders.")
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(path, encoding="utf-8")
    return cp


def conf_path(cp, section, key, default=None):
    value = cp.get(section, key, fallback=default)
    if value is None:
        return None
    value = value.strip().strip('"')
    if not value:
        return default
    return value if os.path.isabs(value) else os.path.join(HERE, value)


def list_datasets():
    width = max(len(n) for n in REGISTRY)
    print("Datasets:")
    for name, spec in REGISTRY.items():
        print(f"  {name:<{width}}  {spec.title} ({spec.hf_id})")
        if spec.notes:
            print(f"  {'':<{width}}  {spec.notes}")


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dataset", default=None)
    pre.add_argument("--list-datasets", action="store_true")
    ns, rest = pre.parse_known_args()

    if ns.list_datasets:
        list_datasets()
        return
    if ns.dataset is None:
        print(__doc__.strip())
        print()
        list_datasets()
        sys.exit(0 if any(a in ("-h", "--help") for a in rest) else 2)
    spec = REGISTRY.get(ns.dataset)
    if spec is None:
        print(f"Unknown dataset: {ns.dataset!r}")
        print()
        list_datasets()
        sys.exit(2)

    parser = duck_engine.build_parser(spec, prog=f"run.bat --dataset {spec.name}")
    args = parser.parse_args(rest)

    cp = load_config()
    data_dir = args.dir if not duck_engine.unset(args.dir) else conf_path(cp, "paths", spec.name)
    if not data_dir:
        duck_engine.fail(f"no folder configured for {spec.name}.\n"
                         f"Add '{spec.name} = <folder>' under [paths] in {CONF_NAME}, or pass --dir.")
    results_dir = conf_path(cp, "output", "results", "results")

    duck_engine.run(spec, args, parser, data_dir, results_dir)


if __name__ == "__main__":
    main()
