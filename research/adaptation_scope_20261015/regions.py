"""Fix the regional partition of Chinese FUAs and the locked confirmation region (run once, before any target fit).

Regions: k-means (10 clusters, n_init 20, random_state 5993) on centroid latitude/longitude of the Chinese FUAs
outside the 43-FUA YRD membership. Locked region: drawn with numpy default_rng(20261015) from the regions with
at least 20 FUAs observed in >= 80% of the 96 months. The locked region is excluded from all development work
and is used once, as target T3, after the development results are final.
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.cluster import KMeans

panel, member, out = sys.argv[1:4]
p = pd.read_csv(panel)
yrd = set(pd.read_csv(member).query("country_iso == 'CHN'").eFUA_ID)
c = p[p.country_iso.eq("CHN")]
f = c.groupby("eFUA_ID").agg(lat=("fua_centroid_lat", "first"), lon=("fua_centroid_lon", "first"),
                             name=("fua_name", "first"), months=("t", "size"))
nonyrd = f[~f.index.isin(yrd)].copy()
km = KMeans(n_clusters=10, n_init=20, random_state=5993).fit(nonyrd[["lat", "lon"]])
nonyrd["region"] = km.labels_
nonyrd["eligible80"] = nonyrd.months >= np.ceil(0.8 * p.t.nunique())
size = nonyrd.groupby("region").eligible80.sum()
candidates = sorted(size[size >= 20].index.tolist())
locked = int(np.random.default_rng(20261015).choice(candidates))
summary = {
    "regions": {int(r): {"fuas": int((nonyrd.region == r).sum()), "eligible80": int(size[r]),
                         "lat": round(float(nonyrd[nonyrd.region == r].lat.mean()), 2),
                         "lon": round(float(nonyrd[nonyrd.region == r].lon.mean()), 2),
                         "names": sorted(nonyrd[nonyrd.region == r].name.tolist())}
                for r in sorted(nonyrd.region.unique())},
    "candidates_for_locked": candidates, "locked_region": locked,
    "fua_region": {str(int(k)): int(v) for k, v in nonyrd.region.items()},
}
json.dump(summary, open(out, "w"), indent=1, ensure_ascii=False)
print("candidates", candidates, "locked", locked, "eligible FUAs in locked", int(size[locked]))
print("locked region sample:", summary["regions"][locked]["names"][:12])
