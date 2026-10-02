"""Execute a declared model/seed matrix; shared data preparation is reused."""

import argparse
import json
import subprocess
import sys

from firerisk.config import load_config
from firerisk.storage import resolve_storage_root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2026])
    p.add_argument("--gpus", type=int, choices=[1, 2], default=1)
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the declared matrix on held-out test after training",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed runs and resume existing epoch checkpoints",
    )
    args = p.parse_args()
    runs = []
    for filename in args.configs:
        config = load_config(filename)
        root = resolve_storage_root(config["storage"])
        subprocess.run(
            [sys.executable, "-m", "firerisk", "prepare", "--config", filename], check=True
        )
        for seed in args.seeds:
            run = root / "runs" / f"{config['experiment']['name']}-seed{seed}"
            runs.append(str(run))
            summary = run / "summary.json"
            complete = (
                summary.exists() and json.loads(summary.read_text()).get("status") == "complete"
            )
            prefix = (
                [sys.executable, "-m", "firerisk"]
                if args.gpus == 1
                else [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=2",
                    "-m",
                    "firerisk",
                ]
            )
            command = prefix + ["train", "--config", filename, "--set", f"experiment.seed={seed}"]
            if args.resume and not complete and (run / "last.pt").exists():
                command += ["--resume", str(run / "last.pt")]
            if not (args.resume and complete):
                subprocess.run(command, check=True)
            if args.evaluate:
                subprocess.run(
                    [sys.executable, "-m", "firerisk", "evaluate", "--run-dir", str(run)],
                    check=True,
                )
    if args.evaluate:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "firerisk",
                "compare",
                "--runs",
                *runs,
                "--output",
                str(root / "reports" / "suite"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
