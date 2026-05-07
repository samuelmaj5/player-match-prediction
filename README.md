# player-match-prediction

Predicting soccer match outcomes from player-level stats. See [`plan.md`](plan.md) for scope and [`CLAUDE.md`](CLAUDE.md) for ML constraints.

## Setup

```bash
pip install -r requirements.txt
```

## Run the pipeline

Open the notebook:

```bash
jupyter notebook notebooks/01_pipeline.ipynb
```

Or run end-to-end from a shell:

```bash
python -c "
from src.load import load_football_data, load_team_xg
from src.join import join_sources
from src.features import rolling_xg_diff

fd = load_football_data('2022-2023')
xg = load_team_xg('2022-2023')
joined = join_sources(fd, xg)
feat = rolling_xg_diff(joined, xg, window=5)
print(feat.shape, '— non-null:', feat['xg_diff_5'].notna().sum())
"
```

First run downloads to `data/raw/` (gitignored); subsequent runs are cached.

## Layout

```
src/
  load.py      # Football-Data results+odds, Understat per-match xG
  join.py      # team-name map + match_id + source join
  features.py  # lag-safe rolling 5-game xG-diff
notebooks/
  01_pipeline.ipynb
```

Steps 4–5 (bookmaker baseline, logistic regression, evaluation) are not yet implemented.
