"""Fixed cohort comparison using shared city splits and observed monthly labels."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'thesis_completion_20260912'
FOCUSED = HERE.parent / 'focused_conditions_20260919'
sys.path.insert(0, str(FOCUSED))
from run_focused import (Y, SEEDS, FLOORS, POLICIES, LOCAL, TRANSFER, NEG,
                        REFS, MODELS, GATES, sha, dump, read, norm, tree,
                        embed, kernel, pos, select_locked, audit_lock, load_panel)

COHORTS = ['membership_all', 'documented_role']


def fit_predict(d, features, cohort, direction, seed, folder, national, ids):
    src, tgt = direction.split('_to_')
    s = d[d.country_iso.eq(src) & d.year.le(2018)].copy()
    t = d[d.country_iso.eq(tgt)].copy().reset_index(drop=True)
    dev = t.eFUA_ID.isin(ids).to_numpy()
    assert len(ids) >= 3 and t.loc[~dev, 'eFUA_ID'].nunique() >= 3
    t['development_city'] = dev
    t['new_weather_grid'] = ~t.weather_grid_id.isin(set(t.loc[dev, 'weather_grid_id']))
    u = t[dev & t.year.le(2018).to_numpy()]
    matching_features = features[:6] + features[-3:]
    scaler = norm()
    sx = scaler.fit_transform(s[matching_features])
    ux = scaler.transform(u[matching_features])
    distance = cKDTree(ux).query(sx, workers=4)[0]
    cutoff = float(np.quantile(distance, .5))
    matched = s[distance <= cutoff].copy()
    assert len(matched) >= 60
    t['regional_support_distance'] = cKDTree(sx).query(scaler.transform(t[matching_features]), workers=4)[0]
    source_sets = {'regional': s, 'matched': matched}
    models = {'national': national['models']['source']}
    for policy, rows in source_sets.items():
        models[policy] = tree(True).fit(rows[features], rows[Y])
    embeddings = {}
    for policy in POLICIES:
        embeddings[f'{policy}_kernel'] = embed(models[policy], t[features])
        t[f'{policy}_raw'] = pos(models[policy].predict(t[features]))
        for shuffle in SEEDS:
            key = f'{policy}_shuffle_{shuffle}'
            if policy == 'national':
                model = national['models'][f'shuffle_{shuffle}']
            else:
                rows = source_sets[policy]
                model = tree(True).fit(rows[features], np.random.default_rng(shuffle).permutation(rows[Y].to_numpy()))
            models[key] = model
            embeddings[key] = embed(model, t[features])
    for model in LOCAL + TRANSFER + NEG:
        t[model] = np.nan
    used = np.zeros(len(t), bool)
    audits = []
    for month in sorted(t.loc[t.year.ge(2019), 't'].unique()):
        fit = dev & t.t.between(month - 12, month - 1).to_numpy()
        ev = t.t.eq(month).to_numpy()
        if fit.sum() < 5:
            continue
        assert not (fit & ev).any() and not (fit & ~dev).any()
        assert t.loc[fit, 't'].max() < month
        used |= fit
        x, ex, y = t.loc[fit, features], t.loc[ev, features], t.loc[fit, Y].to_numpy()
        audits.append({'prediction_t': int(month), 'latest_fit_t': int(t.loc[fit, 't'].max()),
                       'earliest_fit_t': int(t.loc[fit, 't'].min()), 'labels': int(fit.sum()),
                       'fit_cities': sorted(t.loc[fit, 'eFUA_ID'].unique().tolist())})
        t.loc[ev, 'fit_labels'] = fit.sum()
        normal = norm()
        xx, ee = normal.fit_transform(x), normal.transform(ex)
        t.loc[ev, 'target_mean'] = y.mean()
        t.loc[ev, 'target_ridge'] = pos(Ridge(alpha=10).fit(xx, y).predict(ee))
        t.loc[ev, 'target_rbf'] = kernel(rbf_kernel(xx), rbf_kernel(ee, xx), y)
        for key, cls in [('target_rf', RandomForestRegressor), ('target_extra', ExtraTreesRegressor)]:
            model = cls(n_estimators=80, min_samples_leaf=3, max_features=1.0, n_jobs=4, random_state=5993).fit(xx, y)
            t.loc[ev, key] = pos(model.predict(ee))
        t.loc[ev, 'target_lgbm'] = pos(tree().fit(x, y).predict(ex))
        for policy in POLICIES:
            raw, eraw = t.loc[fit, f'{policy}_raw'].to_numpy(), t.loc[ev, f'{policy}_raw'].to_numpy()
            normal_raw = StandardScaler()
            rx, re = normal_raw.fit_transform(raw[:, None]), normal_raw.transform(eraw[:, None])
            t.loc[ev, f'{policy}_affine'] = pos(Ridge(alpha=10).fit(rx, y).predict(re))
            t.loc[ev, f'{policy}_residual'] = pos(eraw + tree().fit(x, y - raw).predict(ex))
        for key, embedding in embeddings.items():
            a, b = embedding[fit], embedding[ev]
            t.loc[ev, key] = kernel((a @ a.T).toarray(), (b @ a.T).toarray(), y)
    meta = {'cohort': cohort, 'direction': direction, 'seed': seed, 'window_months': 12,
            'development_cities': ids, 'features': features, 'source_rows': len(s),
            'matched_source_rows': len(matched), 'source_fuas': int(s.eFUA_ID.nunique()),
            'matched_source_fuas': int(matched.eFUA_ID.nunique()), 'source_max_year': int(s.year.max()),
            'source_matching_cutoff': cutoff, 'source_matching_features': matching_features,
            'target_rows': len(t), 'target_fuas': int(t.eFUA_ID.nunique()),
            'unique_target_calibration_labels': int(used.sum()), 'role_stability_verified': False,
            'protocol_sha256': sha(HERE / 'PROTOCOL.md'), 'script_sha256': sha(__file__),
            'helper_sha256': sha(FOCUSED / 'run_focused.py'),
            'panel_sha256': sha(HERE / f'data/{cohort}_panel.csv.gz')}
    dump(folder / 'training_manifest.json', meta)
    pd.DataFrame(audits).to_csv(folder / 'fit_audit.csv', index=False)
    s[['country_iso', 'eFUA_ID', 'year', 'month', 't']].assign(matching_distance=distance, matched=distance <= cutoff).to_csv(folder / 'source_selection.csv.gz', index=False)
    t.loc[used, ['eFUA_ID', 'year', 'month', 't']].to_csv(folder / 'calibration_rows.csv.gz', index=False)
    joblib.dump({'models': models, 'matching_scaler': scaler, 'features': features}, folder / 'source_models.joblib', compress=3)
    p = t[t.year.ge(2019)].copy()
    good = p[MODELS].notna().all(axis=1)
    p.loc[~good, ['eFUA_ID', 'year', 'month']].to_csv(folder / 'no_prediction.csv', index=False)
    columns = list(dict.fromkeys(['country_iso', 'eFUA_ID', 'fua_name', 'year', 'month', 't', 'quarter_id',
               'weather_grid_id', Y, 'station_count', 'development_city', 'new_weather_grid', 'fit_labels',
               *features, *GATES, *MODELS]))
    p = p.loc[good, columns].reset_index(drop=True)
    p.to_csv(folder / 'predictions.csv.gz', index=False)
    return p, meta


def run_config(d, cohort, direction, seed, national, splits, denominator):
    folder = HERE / 'results' / cohort / direction / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        print('CACHED', cohort, direction, seed, flush=True)
        return
    target = direction.split('_to_')[1]
    if (folder / 'predictions.csv.gz').exists():
        p = read(folder / 'predictions.csv.gz')
        meta = json.loads((folder / 'training_manifest.json').read_text(encoding='utf-8'))
        assert meta['script_sha256'] == sha(__file__)
        assert meta['protocol_sha256'] == sha(HERE / 'PROTOCOL.md')
    else:
        p, meta = fit_predict(d, national['features'], cohort, direction, seed, folder, national, splits[target]['seeds'][str(seed)])
    discovery = p[p.development_city & p.year.eq(2019)].reset_index(drop=True)
    validation = p[p.development_city & p.year.eq(2020)].reset_index(drop=True)
    setting = {'scope': cohort, 'cohort': cohort, 'direction': direction, 'seed': seed}
    locks = []
    for floor in FLOORS:
        pct = int(floor * 100)
        path = folder / f'lock_{pct}.json'
        if path.exists():
            lock = json.loads(path.read_text(encoding='utf-8'))
        else:
            lock, candidates, shortlist = select_locked(discovery, validation, floor)
            lock.update(setting)
            lock['protocol_sha256'] = sha(HERE / 'PROTOCOL.md')
            dump(path, lock)
            for label, rows in [('search', candidates), ('shortlist', shortlist)]:
                frame = pd.DataFrame(rows)
                if len(frame):
                    frame['rule'] = frame.rule.map(json.dumps)
                frame.to_csv(folder / f'{label}_{pct}.csv.gz', index=False)
        locks.append(lock)
    rows, models = [], []
    for lock in locks:
        a, b = audit_lock(p, lock, denominator[denominator.country_iso.eq(target)], setting)
        rows.extend(a)
        models.extend(b)
    pd.DataFrame(rows).to_csv(folder / 'gates.csv', index=False)
    pd.DataFrame(models).to_csv(folder / 'metrics.csv.gz', index=False)
    dump(folder / 'complete.json', {**meta, 'completed': time.strftime('%Y-%m-%dT%H:%M:%S'),
         'lock_count': len(locks), 'prediction_sha256': sha(folder / 'predictions.csv.gz')})
    print('COMPLETE', cohort, direction, seed, 'selection_passes', sum(l['discovery_pass'] for l in locks), flush=True)


def aggregate():
    result = HERE / 'results'
    for name, pattern in [('all_gates.csv', '*/*/*/gates.csv'), ('all_metrics.csv.gz', '*/*/*/metrics.csv.gz')]:
        paths = list(result.glob(pattern))
        if paths:
            pd.concat([read(p) for p in paths], ignore_index=True).to_csv(HERE / 'audit' / name, index=False)
    locks = [json.loads(p.read_text(encoding='utf-8')) for p in result.glob('*/*/*/lock_*.json')]
    dump(HERE / 'audit/locks.json', locks)
    dump(HERE / 'audit/completion.json', {'configurations': len(list(result.glob('*/*/*/complete.json'))),
         'expected_configurations': 12, 'locks': len(locks), 'expected_locks': 48})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()
    if args.aggregate_only:
        aggregate()
        return
    receipt = HERE / 'audit/run_receipt.json'
    if not receipt.exists():
        dump(receipt, {'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
             'protocol_sha256': sha(HERE / 'PROTOCOL.md'), 'script_sha256': sha(__file__),
             'split_sha256': sha(HERE / 'audit/fixed_city_splits.json')})
    else:
        previous = json.loads(receipt.read_text(encoding='utf-8'))
        assert previous['protocol_sha256'] == sha(HERE / 'PROTOCOL.md')
        assert previous['script_sha256'] == sha(__file__)
    splits = json.loads((HERE / 'audit/fixed_city_splits.json').read_text(encoding='utf-8'))
    denominator = load_panel()
    count = 0
    for cohort in COHORTS:
        d = read(HERE / f'data/{cohort}_panel.csv.gz')
        for direction in ['CHN_to_AUS', 'AUS_to_CHN']:
            national = joblib.load(OLD / 'results/all_networks/monthly_plus_mixing' / direction / 'source_models.joblib')
            for seed in SEEDS:
                print('START', cohort, direction, seed, flush=True)
                run_config(d, cohort, direction, seed, national, splits, denominator)
                aggregate()
                count += 1
                if args.limit and count >= args.limit:
                    return


if __name__ == '__main__':
    main()
