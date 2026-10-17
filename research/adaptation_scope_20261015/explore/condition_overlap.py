"""Exploratory (explore/README.md, item 1): how well the conditions outside a condition's group (meteorology, fire,
season, as in ablate_groups.py) predict it over the Chinese panel (R^2 of a linear fit on the standardised
conditions).
Usage: python explore/condition_overlap.py <out_dir>   Writes condition_overlap.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adapt_lib as al

C = al._C[al.COND].dropna()
Zc = ((C - C.mean()) / C.std()).to_numpy()
GROUPS = {"meteorology": [0, 1, 2, 3, 4], "fire": [5], "season": [6, 7]}
group_of = {j: g for g, idx in GROUPS.items() for j in idx}
out = {}
for j, name in enumerate(al.COND):
    outside = [i for i in range(len(al.COND)) if group_of[i] != group_of[j]]
    others = np.c_[np.ones(len(Zc)), Zc[:, outside]]
    beta = np.linalg.lstsq(others, Zc[:, j], rcond=None)[0]
    out[name] = round(float(1 - np.var(Zc[:, j] - others @ beta) / np.var(Zc[:, j])), 3)
Path(sys.argv[1], "condition_overlap.json").write_text(json.dumps(out, indent=1))
print(out)
