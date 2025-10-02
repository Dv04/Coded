import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="COMP 480/580 Assignment 1 Runner")
    parser.add_argument("--all", action="store_true", help="Run everything")
    parser.add_argument("--q1", action="store_true", help="Run Q1 (avalanche analysis)")
    parser.add_argument(
        "--q2", action="store_true", help="Run Q2 (turtle CI utilities demo)"
    )
    parser.add_argument("--q4", action="store_true", help="Run Q4 (Bloom filter demos)")
    parser.add_argument(
        "--plots", action="store_true", help="Also render Q1 heatmap PNGs"
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Run Q4 extended (AOL URLs) if available",
    )
    args = parser.parse_args()

    out = Path("outputs")
    out.mkdir(exist_ok=True)

    ran = False

    if args.all or args.q1:
        ran = True
        from q1_avalanche import run_q1

        run_q1()

        if args.plots or args.all:
            try:
                from q1_heatmaps import make_all

                make_all()
            except Exception as e:
                print(f"[Q1] Heatmap rendering skipped ({e})")

    if args.all or args.q4:
        ran = True
        from q4_bloom import warmup, extended

        warmup()
        if args.extended or args.all:
            extended()

            if (Path("outputs/q4") / "extended.csv").exists():
                try:
                    from q4_plots import main as plots_main

                    plots_main()
                except Exception as e:
                    print(f"[Q4] Plotting skipped ({e})")

    if not ran:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
