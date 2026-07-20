from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starfuzz.experiments.ablation import default_ablation_suite, run_ablation_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or run starfuzz ablation suite.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="execute the suite; default only writes the plan")
    args = parser.parse_args()
    suite = default_ablation_suite(args.output_dir)
    plan = run_ablation_suite(suite, dry_run=not args.run)
    print(f"saved ablation plan: {plan}")


if __name__ == "__main__":
    main()


