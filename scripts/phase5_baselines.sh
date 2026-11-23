#!/bin/bash
# Execute only the five predeclared Random baselines, once each, after approval.
set -euo pipefail
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
for seed in 0 1 2 3 4; do
  test ! -e "results/phase5/random_${seed}.json"
  python -c 'import random,runpy,sys; seed=int(sys.argv[1]); random.seed(seed); sys.argv=["src.run","--strategy","src.baseline_random.RandomStrategy","--data-dir","data/test","-o",f"results/phase5/random_{seed}.json","-v"]; runpy.run_module("src.run",run_name="__main__")' "$seed" > "results/phase5/random_${seed}_cli.log" 2>&1
  printf 'Completed Random seed %s\n' "$seed"
done
