test:
  uv run pytest

# Each command writes its CSV or figure to results/
baseline:
  uv run scripts/dependence.py baseline --out_file results/baseline.csv

ess-gamma:
  uv run scripts/dependence.py ess-gamma --out_file results/ess_gamma.csv

sieve:
  uv run scripts/dependence.py sieve --out_file results/sieve.csv

sieve-ncal:
  uv run scripts/dependence.py sieve-ncal --batch_size 250 --out_file results/sieve_ncal.csv

plateau:
  uv run scripts/dependence.py plateau

plot:
  uv run scripts/dependence.py plot

all: baseline ess-gamma sieve sieve-ncal plateau plot
