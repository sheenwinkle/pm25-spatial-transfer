"""Restrict source label age while retaining paired target budgets and gates."""
import argparse
import json
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from run_focused import (HERE, RESULTS, SEEDS, FLOORS, MODELS, MIX, Y, read, dump, sha,
                         load_panel, scope_mask, norm, tree, pos, embed, kernel,
                         select_locked, audit_lock)

SCOPE = 'latitude22_35_coastal150'
OUT = HERE / 'recency_results'


def make_predictions(d, old_folder, folder):
    meta = json.loads((old_folder / 'training_manifest.json').read_text(encoding='utf-8'))
    p = read(old_folder / 'predictions.csv.gz')
    src, tgt = meta['direction'].split('_to_')
    scoped = d[scope_mask(d, SCOPE)]
    target = scoped[scoped.country_iso.eq(tgt)]
    source = scoped[scoped.country_iso.eq(src)]
    target_dev = target[target.eFUA_ID.isin(meta['development_cities'])]
    u = target_dev[target_dev.year.le(2018)]
    features = meta['features']
    match_features = features[:6] + features[-3:]
    index = pd.MultiIndex.from_frame(p[['eFUA_ID', 't']])
    fits, selected = [], []
    for month in sorted(p.t.unique()):
        fit = target_dev[target_dev.t.between(month - 12, month - 1)]
        ev = target[target.t.eq(month)]
        s = source[source.t.between(month - 24, month - 1)]
        assert len(fit) >= 5 and len(s) >= 60
        assert s.t.max() < month and fit.t.max() < month
        ei = index.get_indexer(pd.MultiIndex.from_frame(ev[['eFUA_ID', 't']]))
        assert (ei >= 0).all()
        assert np.array_equal(p.loc[ei, 'fit_labels'].to_numpy(), np.repeat(len(fit), len(ev)))
        z = norm()
        sx = z.fit_transform(s[match_features])
        ux = z.transform(u[match_features])
        distance = cKDTree(ux).query(sx, workers=4)[0]
        cutoff = float(np.quantile(distance, .5))
        matched = s[distance <= cutoff]
        selected.append(s[['eFUA_ID', 't', 'country_iso']].assign(prediction_t=month,
                        matching_distance=distance, matching_cutoff=cutoff, matched=distance <= cutoff))
        x, ex, y = fit[features], ev[features], fit[Y].to_numpy()
        both = pd.concat([x, ex], ignore_index=True)
        for policy, sample in [('regional', s), ('matched', matched)]:
            m = tree(True).fit(sample[features], sample[Y])
            raw = pos(m.predict(x))
            eraw = pos(m.predict(ex))
            p.loc[ei, policy + '_raw'] = eraw
            scale = StandardScaler()
            rx, re = scale.fit_transform(raw[:, None]), scale.transform(eraw[:, None])
            p.loc[ei, policy + '_affine'] = pos(Ridge(alpha=10).fit(rx, y).predict(re))
            p.loc[ei, policy + '_residual'] = pos(eraw + tree().fit(x, y - raw).predict(ex))
            e = embed(m, both)
            a, b = e[:len(x)], e[len(x):]
            p.loc[ei, policy + '_kernel'] = kernel((a @ a.T).toarray(), (b @ a.T).toarray(), y)
            for seed in SEEDS:
                sm = tree(True).fit(sample[features], np.random.default_rng(seed).permutation(sample[Y].to_numpy()))
                e = embed(sm, both)
                a, b = e[:len(x)], e[len(x):]
                p.loc[ei, f'{policy}_shuffle_{seed}'] = kernel((a @ a.T).toarray(), (b @ a.T).toarray(), y)
        fits.append({'prediction_t': int(month), 'source_first_t': int(s.t.min()), 'source_last_t': int(s.t.max()),
                     'source_rows': len(s), 'matched_source_rows': len(matched), 'matching_cutoff': cutoff,
                     'target_first_t': int(fit.t.min()), 'target_last_t': int(fit.t.max()),
                     'target_labels': len(fit), 'target_fuas': int(fit.eFUA_ID.nunique())})
    assert np.isfinite(p[MODELS + [Y]].to_numpy()).all()
    p.to_csv(folder / 'predictions.csv.gz', index=False)
    pd.DataFrame(fits).to_csv(folder / 'fit_audit.csv', index=False)
    pd.concat(selected, ignore_index=True).to_csv(folder / 'source_selection.csv.gz', index=False)
    meta = {**meta, 'source_period_policy': 'rolling24_regional_and_matched',
            'pilot_protocol_sha256': sha(HERE / 'RECENCY_PROTOCOL.md'), 'pilot_script_sha256': sha(__file__),
            'parent_predictions_sha256': sha(old_folder / 'predictions.csv.gz')}
    dump(folder / 'training_manifest.json', meta)
    return p, meta


def run_config(d, direction, seed):
    old_folder = RESULTS / SCOPE / direction / str(seed)
    folder = OUT / SCOPE / direction / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        print('CACHED_RECENCY', direction, seed, flush=True)
        return
    if (folder / 'predictions.csv.gz').exists():
        p = read(folder / 'predictions.csv.gz')
        meta = json.loads((folder / 'training_manifest.json').read_text(encoding='utf-8'))
    else:
        p, meta = make_predictions(d, old_folder, folder)
    disc = p[p.development_city & p.year.eq(2019)].reset_index(drop=True)
    val = p[p.development_city & p.year.eq(2020)].reset_index(drop=True)
    locks, inherited = [], []
    for floor in FLOORS:
        percent = int(floor * 100)
        path = folder / f'lock_{percent}.json'
        if path.exists():
            lock = json.loads(path.read_text(encoding='utf-8'))
        else:
            lock, candidates, shortlist = select_locked(disc, val, floor)
            lock.update({'scope': SCOPE, 'direction': direction, 'seed': seed,
                         'source_period_policy': 'rolling24_regional_and_matched',
                         'protocol_sha256': sha(HERE / 'RECENCY_PROTOCOL.md')})
            dump(path, lock)
            for label, entries in [('search', candidates), ('shortlist', shortlist)]:
                z = pd.DataFrame(entries)
                if len(z):
                    z['rule'] = z.rule.map(json.dumps)
                z.to_csv(folder / f'{label}_{percent}.csv.gz', index=False)
        locks.append(lock)
        inherited.append(json.loads((old_folder / f'lock_{percent}.json').read_text(encoding='utf-8')))
    setting = {'scope': SCOPE, 'direction': direction, 'seed': seed,
               'source_period_policy': 'rolling24_regional_and_matched'}
    country = d[d.country_iso.eq(direction.split('_to_')[1])]
    for kind, items in [('rediscovered', locks), ('inherited_rules', inherited)]:
        gates, metrics = [], []
        for lock in items:
            a, b = audit_lock(p, lock, country, setting)
            gates.extend(a)
            metrics.extend(b)
        pd.DataFrame(gates).to_csv(folder / f'{kind}_gates.csv', index=False)
        pd.DataFrame(metrics).to_csv(folder / f'{kind}_metrics.csv.gz', index=False)
    dump(folder / 'complete.json', {**meta, 'completed': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                   'prediction_sha256': sha(folder / 'predictions.csv.gz')})
    print('COMPLETE_RECENCY', direction, seed, 'discovery_pass', sum(x['discovery_pass'] for x in locks), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    receipt = OUT / 'receipt.json'
    if not receipt.exists():
        dump(receipt, {'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
                       'protocol_sha256': sha(HERE / 'RECENCY_PROTOCOL.md'), 'script_sha256': sha(__file__),
                       'shared_helpers_sha256': sha(HERE / 'run_focused.py')})
    d = load_panel()
    n = 0
    for direction in ['CHN_to_AUS', 'AUS_to_CHN']:
        for seed in SEEDS:
            run_config(d, direction, seed)
            n += 1
            if args.limit and n >= args.limit:
                return
    for name in ['rediscovered_gates.csv', 'inherited_rules_gates.csv']:
        pd.concat([read(p) for p in OUT.glob(f'*/*/*/{name}')], ignore_index=True).to_csv(OUT / name, index=False)


if __name__ == '__main__':
    main()
