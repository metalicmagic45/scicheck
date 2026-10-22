from __future__ import annotations

import argparse

import torch

from .config import load_config


def cmd_info(cfg, args):
    from .models import ARCHS, build_for_target, count_params, tracked_layers
    from .snapshots import feature_names

    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.device_count()} device(s))" if torch.cuda.is_available() else ""))
    print(f"{'arch':5} {'target':>7} {'width':>6} {'params':>7} {'snapshot dim':>13}")
    for a in cfg.architectures:
        for s in cfg.sizes:
            m, w = build_for_target(a, s)
            dim = len(feature_names(len(tracked_layers(m))))
            print(f"{a:5} {s:>7} {w:>6} {count_params(m):>7} {dim:>13}")
    print("snapshot features:", ", ".join(feature_names(ARCHS["mlp"].n_tracked)))


def cmd_download(cfg, args):
    from .data import download_all

    download_all(cfg.data_dir, cfg.data_seed)


def cmd_accuracy(cfg, args):
    from .experiments.accuracy import run_accuracy

    devices = args.devices.split(",") if args.devices else None
    df = run_accuracy(cfg, workers=args.workers, devices=devices, threads=args.threads)
    print(f"collected {len(df)} runs -> {cfg.out_path / 'runs.csv'}")


def cmd_performance(cfg, args):
    from .experiments.performance import run_performance

    run_performance(cfg, sweeps=[args.sweep] if args.sweep else None, device=args.device)


def cmd_report(cfg, args):
    from .report import build_report

    print(f"report -> {build_report(cfg)}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="scicheck")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("info", "download", "accuracy", "performance", "report"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default="configs/paper.yaml")
        sp.add_argument("--archs", default=None,
                        help="subset of the config's architectures, e.g. mlp or cnn,ae")
        if name == "accuracy":
            sp.add_argument("--workers", type=int, default=1,
                            help="parallel worker processes")
            sp.add_argument("--devices", default=None,
                            help="devices assigned round-robin to workers, e.g. cuda:0,cuda:1")
            sp.add_argument("--threads", type=int, default=None, help="torch CPU threads per worker")
        if name == "performance":
            sp.add_argument("--sweep", choices=["runtime", "memory"], default=None)
            sp.add_argument("--device", default=None, help="device for the runtime sweep (default: config)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.archs:
        wanted = args.archs.split(",")
        unknown = [a for a in wanted if a not in cfg.architectures]
        if unknown:
            p.error(f"--archs {unknown} not in the config's architectures {cfg.architectures}")
        cfg.architectures = wanted
    {"info": cmd_info, "download": cmd_download, "accuracy": cmd_accuracy,
     "performance": cmd_performance, "report": cmd_report}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
