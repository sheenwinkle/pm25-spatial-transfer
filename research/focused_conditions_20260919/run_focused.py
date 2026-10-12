"""Geography, source selection and nested conditional-transfer exploration."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import hashlib
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
sys.path.insert(0, str(OLD))
from run_monitor_models import norm, tree, embed, kernel, pos, metrics, bootstrap

Y = 'observed_pm25'
SCOPES = ['coastal150', 'latitude22_35_coastal150', 'east_window']
SEEDS = [5993, 5994, 5995]
FLOORS = [.50, .25, .10, .05]
POLICIES = ['national', 'regional', 'matched']
LOCAL = ['target_mean', 'target_ridge', 'target_rbf', 'target_rf', 'target_extra', 'target_lgbm']
TRANSFER = [f'{p}_{m}' for p in POLICIES for m in ['affine', 'residual', 'kernel']]
NEG = [f'{p}_shuffle_{s}' for p in POLICIES for s in SEEDS]
REFS = LOCAL + NEG
MODELS = LOCAL + TRANSFER + NEG + [f'{p}_raw' for p in POLICIES]
MIX = ['era5_pbl_mean_m', 'era5_pbl_low_fraction', 'era5_surface_ventilation_mean_m2_s']
GATES = ['monthly_power_temperature_c', 'monthly_power_humidity_pct', 'monthly_power_wind_mps',
         'monthly_power_precip_mm_day', *MIX, 'distance_to_coast_km',
         'log_population_density', 'gate_log_fire', 'regional_support_distance']
ANCHORS = {'monthly_power_temperature_c': [10, 20, 25],
           'monthly_power_humidity_pct': [60, 80], 'monthly_power_wind_mps': [2, 4],
           'era5_pbl_mean_m': [500, 1000], 'gate_log_fire': [0]}
RESULTS = HERE / 'results'
AUDIT = HERE / 'audit'


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def read(path, **kw):
    return pd.read_csv(path, float_precision='round_trip', **kw)


def scope_mask(d, scope):
    coast = d.distance_to_coast_km.le(150)
    if scope == 'coastal150':
        return coast
    if scope == 'latitude22_35_coastal150':
        return coast & d.fua_centroid_lat.abs().between(22, 35)
    if scope == 'east_window':
        return ((d.country_iso.eq('CHN') & d.fua_centroid_lon.between(117, 123)
                 & d.fua_centroid_lat.between(28, 34)) |
                (d.country_iso.eq('AUS') & coast & d.fua_centroid_lon.between(140, 154)
                 & d.fua_centroid_lat.between(-39, -25)))
    raise ValueError(scope)


def rulemask(d, rule):
    keep = np.ones(len(d), dtype=bool)
    for a in rule:
        v = d[a['feature']].to_numpy(float)
        keep &= np.isfinite(v) & ((v <= a['threshold']) if a['op'] == '<=' else (v >= a['threshold']))
    return keep


def support(d, keep, floor):
    if not len(d) or not keep.any():
        return False
    for y in sorted(d.year.unique()):
        ym = d.year.eq(y).to_numpy()
        z = d[ym & keep]
        if ((ym & keep).sum() / ym.sum() + 1e-12 < floor or len(z) < 24
                or z.eFUA_ID.nunique() < 3 or z.t.nunique() < 6):
            return False
    return True


def old_support(d, keep):
    z = d[keep]
    return bool(len(d) and keep.mean() >= .25 and
                z.eFUA_ID.nunique() >= max(3, np.ceil(.25 * d.eFUA_ID.nunique())) and z.t.nunique() >= 12)


def score(d, model, keep):
    z = d[keep]
    if not len(z):
        return {'gain': -1e100, 'coverage': 0, 'reference': None}
    e = z[REFS].to_numpy() - z[Y].to_numpy()[:, None]
    rms = np.sqrt(np.mean(e * e, axis=0))
    i = int(np.argmin(rms))
    rmse = float(np.sqrt(np.mean((z[model].to_numpy() - z[Y].to_numpy()) ** 2)))
    return {'gain': float(1 - rmse / max(rms[i], 1e-12)), 'coverage': float(keep.mean()),
            'reference': REFS[i], 'rmse': rmse, 'reference_rmse': float(rms[i])}


def search_rules(d, floor):
    """Beam search sees only 2019 development-city outcomes."""
    atoms = []
    for feature in GATES:
        v = d[feature].to_numpy(float)
        if np.isfinite(v).sum() < 24:
            continue
        qs = list(np.nanquantile(v, [.1, .25, .5, .75, .9])) + ANCHORS.get(feature, [])
        for q in sorted(set(float(x) for x in qs)):
            for op in ['<=', '>=']:
                a = {'feature': feature, 'op': op, 'threshold': q}
                keep = rulemask(d, [a])
                if support(d, keep, floor) and keep.mean() < .999:
                    atoms.append(a)
    am = [rulemask(d, [a]) for a in atoms]
    # Array-based sufficient statistics avoid repeated DataFrame work in the beam.
    cities = d.eFUA_ID.to_numpy()
    times = d.t.to_numpy()
    errors = (d[MODELS].to_numpy() - d[Y].to_numpy()[:, None]) ** 2
    ri = [MODELS.index(m) for m in REFS]
    all_rows = []
    for model in TRANSFER:
        mi = MODELS.index(model)
        seen = set()
        rows = []

        def evaluate(ids, keep):
            key = np.packbits(keep).tobytes()
            if key in seen:
                return None
            seen.add(key)
            n = int(keep.sum())
            if (n < 24 or n / len(d) + 1e-12 < floor or
                    np.unique(cities[keep]).size < 3 or np.unique(times[keep]).size < 6):
                return None
            r = np.sqrt(errors[keep].mean(axis=0))
            j = min(ri, key=lambda i: r[i])
            row = {'model': model, 'rule': [atoms[i] for i in ids], 'complexity': len(ids),
                   'gain_2019': float(1 - r[mi] / max(r[j], 1e-12)),
                   'coverage_2019': n / len(d), 'reference_2019': MODELS[j],
                   'n_2019': n, 'fuas_2019': int(np.unique(cities[keep]).size)}
            rows.append(row)
            return row, tuple(ids), keep

        evaluate([], np.ones(len(d), bool))
        beam = [(None, (), np.ones(len(d), bool))]
        for depth in range(1, 5):
            expanded = []
            for _, ids, keep in beam:
                for i, a in enumerate(atoms):
                    if i in ids:
                        continue
                    if any(atoms[j]['feature'] == a['feature'] and atoms[j]['op'] == a['op'] for j in ids):
                        continue
                    new = evaluate(sorted((*ids, i)), keep & am[i])
                    if new:
                        expanded.append(new)
            beam = sorted(expanded, key=lambda x: (-x[0]['gain_2019'], -x[0]['coverage_2019']))[:12]
            if not beam:
                break
        all_rows.extend(rows)
    return all_rows


def select_locked(discovery, validation, floor):
    candidates = search_rules(discovery, floor)
    shortlist = []
    for model in TRANSFER:
        rows = [r for r in candidates if r['model'] == model]
        rows = sorted(rows, key=lambda r: (-r['gain_2019'], r['complexity'], -r['coverage_2019']))[:12]
        for r in rows:
            keep = rulemask(validation, r['rule'])
            s = score(validation, model, keep)
            ok = support(validation, keep, floor)
            shortlist.append({**r, 'gain_2020': s['gain'], 'coverage_2020': s['coverage'],
                              'reference_2020': s['reference'], 'support_2020': ok,
                              'worst_gain': min(r['gain_2019'], s['gain']),
                              'discovery_pass': bool(ok and min(r['gain_2019'], s['gain']) >= .05)})
    valid = [r for r in shortlist if r['support_2020']]
    if not valid:
        return {'status': 'insufficient_validation_support', 'discovery_pass': False, 'floor': floor}, candidates, shortlist
    passed = [r for r in valid if r['discovery_pass']]
    chosen = (sorted(passed, key=lambda r: (r['complexity'], -min(r['coverage_2019'], r['coverage_2020']), -r['worst_gain']))[0]
              if passed else max(valid, key=lambda r: (r['worst_gain'], -r['complexity'])))
    return {**chosen, 'status': 'locked', 'floor': floor}, candidates, shortlist


def load_panel():
    d = read(OLD / 'data/all_networks_model_panel.csv.gz')
    m = read(OLD / 'data/common_era5_month_2015_2022.csv.gz')
    d = d.merge(m[['eFUA_ID', 'year', 'month'] + MIX], on=['eFUA_ID', 'year', 'month'], validate='one_to_one')
    for c in [MIX[0], MIX[2]]:
        d['log_' + c] = np.log1p(d[c])
    d['gate_log_fire'] = d.log_fire_count_100km.where(d.fire_source_available.eq(1))
    assert not d.duplicated(['country_iso', 'eFUA_ID', 't']).any()
    return d


def fit_predict(d, features, scope, direction, seed, folder, national):
    src, tgt = direction.split('_to_')
    mask = scope_mask(d, scope)
    s = d[mask & d.country_iso.eq(src) & d.year.le(2018)].copy()
    t = d[mask & d.country_iso.eq(tgt)].copy().reset_index(drop=True)
    counts = t[t.year.eq(2018)].groupby('eFUA_ID').size()
    ids = np.array(sorted(counts[counts >= 6].index))
    np.random.default_rng(seed).shuffle(ids)
    ids = ids[:min(10, int(np.ceil(len(ids) / 2)))]
    assert len(ids) >= 3
    dev = t.eFUA_ID.isin(ids).to_numpy()
    t['development_city'] = dev
    t['new_weather_grid'] = ~t.weather_grid_id.isin(set(t.loc[dev, 'weather_grid_id']))
    u = t[dev & t.year.le(2018).to_numpy()]
    match_features = features[:6] + features[-3:]
    matching_scaler = norm()
    sx = matching_scaler.fit_transform(s[match_features])
    ux = matching_scaler.transform(u[match_features])
    dist = cKDTree(ux).query(sx, workers=4)[0]
    cutoff = float(np.quantile(dist, .5))
    matched = s[dist <= cutoff].copy()
    assert len(matched) >= 60
    t['regional_support_distance'] = cKDTree(sx).query(matching_scaler.transform(t[match_features]), workers=4)[0]
    source_sets = {'regional': s, 'matched': matched}
    models = {'national': national['models']['source']}
    for p, data in source_sets.items():
        models[p] = tree(True).fit(data[features], data[Y])
    embeddings = {}
    for policy in POLICIES:
        embeddings[f'{policy}_kernel'] = embed(models[policy], t[features])
        t[f'{policy}_raw'] = pos(models[policy].predict(t[features]))
        for sh in SEEDS:
            key = f'{policy}_shuffle_{sh}'
            if policy == 'national':
                m = national['models'][f'shuffle_{sh}']
            else:
                data = source_sets[policy]
                m = tree(True).fit(data[features], np.random.default_rng(sh).permutation(data[Y].to_numpy()))
            models[key] = m
            embeddings[key] = embed(m, t[features])
    for m in LOCAL + TRANSFER + NEG:
        t[m] = np.nan
    used = np.zeros(len(t), bool)
    fits = []
    for month in sorted(t.loc[t.year.ge(2019), 't'].unique()):
        fit = dev & t.t.between(month - 12, month - 1).to_numpy()
        ev = t.t.eq(month).to_numpy()
        if fit.sum() < 5:
            continue
        assert not (fit & ev).any() and not (fit & ~dev).any()
        assert t.loc[fit, 't'].max() < month
        used |= fit
        x, ex, y = t.loc[fit, features], t.loc[ev, features], t.loc[fit, Y].to_numpy()
        fits.append({'prediction_t': int(month), 'latest_fit_t': int(t.loc[fit, 't'].max()),
                     'earliest_fit_t': int(t.loc[fit, 't'].min()), 'labels': int(fit.sum()),
                     'fit_cities': sorted(t.loc[fit, 'eFUA_ID'].unique().tolist())})
        t.loc[ev, 'fit_labels'] = fit.sum()
        z = norm()
        xx, ee = z.fit_transform(x), z.transform(ex)
        t.loc[ev, 'target_mean'] = y.mean()
        t.loc[ev, 'target_ridge'] = pos(Ridge(alpha=10).fit(xx, y).predict(ee))
        t.loc[ev, 'target_rbf'] = kernel(rbf_kernel(xx), rbf_kernel(ee, xx), y)
        for key, cls in [('target_rf', RandomForestRegressor), ('target_extra', ExtraTreesRegressor)]:
            m = cls(n_estimators=80, min_samples_leaf=3, max_features=1.0, n_jobs=4, random_state=5993).fit(xx, y)
            t.loc[ev, key] = pos(m.predict(ee))
        t.loc[ev, 'target_lgbm'] = pos(tree().fit(x, y).predict(ex))
        for policy in POLICIES:
            raw, eraw = t.loc[fit, f'{policy}_raw'].to_numpy(), t.loc[ev, f'{policy}_raw'].to_numpy()
            zz = StandardScaler()
            rx, re = zz.fit_transform(raw[:, None]), zz.transform(eraw[:, None])
            t.loc[ev, f'{policy}_affine'] = pos(Ridge(alpha=10).fit(rx, y).predict(re))
            t.loc[ev, f'{policy}_residual'] = pos(eraw + tree().fit(x, y - raw).predict(ex))
        for key, e in embeddings.items():
            a, b = e[fit], e[ev]
            t.loc[ev, key] = kernel((a @ a.T).toarray(), (b @ a.T).toarray(), y)
    meta = {'scope': scope, 'direction': direction, 'seed': seed, 'window_months': 12,
            'development_cities': ids.tolist(), 'source_rows': len(s), 'matched_source_rows': len(matched),
            'source_fuas': int(s.eFUA_ID.nunique()), 'matched_source_fuas': int(matched.eFUA_ID.nunique()),
            'source_matching_cutoff': cutoff, 'source_matching_features': match_features,
            'target_rows': len(t), 'target_fuas': int(t.eFUA_ID.nunique()),
            'unique_target_calibration_labels': int(used.sum()), 'features': features,
            'protocol_sha256': sha(HERE / 'PROTOCOL.md'), 'script_sha256': sha(__file__)}
    dump(folder / 'training_manifest.json', meta)
    pd.DataFrame(fits).to_csv(folder / 'fit_audit.csv', index=False)
    s[['country_iso', 'eFUA_ID', 'year', 'month', 't']].assign(matching_distance=dist, matched=dist <= cutoff).to_csv(folder / 'source_selection.csv.gz', index=False)
    t.loc[used, ['eFUA_ID', 't', 'year', 'month']].to_csv(folder / 'calibration_rows.csv.gz', index=False)
    joblib.dump({'models': models, 'matching_scaler': matching_scaler, 'features': features}, folder / 'source_models.joblib', compress=3)
    p = t[t.year.ge(2019)].copy()
    good = p[MODELS].notna().all(axis=1)
    p.loc[~good, ['eFUA_ID', 'year', 'month']].to_csv(folder / 'no_prediction.csv', index=False)
    keep = list(dict.fromkeys(['country_iso', 'eFUA_ID', 'fua_name', 'year', 'month', 't', 'quarter_id',
                             'weather_grid_id', Y, 'station_count', 'development_city', 'new_weather_grid',
                             'fit_labels', *features, *GATES, *MODELS]))
    p = p.loc[good, keep].reset_index(drop=True)
    p.to_csv(folder / 'predictions.csv.gz', index=False)
    return p, meta


def audit_lock(p, lock, country_panel, setting):
    rows, model_rows = [], []
    if lock['status'] != 'locked':
        return rows, model_rows
    stages = {'discovery_2019': p.development_city & p.year.eq(2019),
              'validation_2020': p.development_city & p.year.eq(2020),
              'temporal_audit': p.development_city & p.year.ge(2021),
              'new_city_audit': ~p.development_city & p.year.ge(2021),
              'new_grid_audit': ~p.development_city & p.new_weather_grid & p.year.ge(2021)}
    for stage, use in stages.items():
        full = p[use]
        keep = rulemask(full, lock['rule'])
        accepted = full[keep]
        sc = score(full, lock['model'], keep)
        local_rms = {m: metrics(accepted, accepted[m])['rmse'] for m in LOCAL}
        local_best = min(local_rms, key=lambda m: local_rms[m] if local_rms[m] is not None else float('inf'))
        cis = {}
        if 'audit' in stage and len(accepted):
            for cluster in ['eFUA_ID', 'weather_grid_id']:
                for ref in REFS:
                    lo, hi = bootstrap(accepted, lock['model'], ref, setting['seed'], cluster)
                    cis[f'{cluster}_{ref}_low'], cis[f'{cluster}_{ref}_high'] = lo, hi
        okci = bool(cis) and all(v is not None and v > 0 for k, v in cis.items() if k.endswith('_low'))
        ok = support(full, keep, lock['floor'])
        years = sorted(full.year.unique())
        denominator = country_panel[country_panel.year.isin(years)]
        # Shares include all country's city-months in the same years, not only the chosen city split.
        row = {**setting, 'floor': lock['floor'], 'stage': stage, 'model': lock['model'],
               'rule': json.dumps(lock['rule']), 'complexity': lock['complexity'],
               'discovery_pass': lock['discovery_pass'], 'coverage': sc['coverage'],
               'country_panel_share': len(accepted) / len(denominator) if len(denominator) else 0,
               'full_scope_stage_rows': len(full), 'gain': sc['gain'], 'best_comparator': sc['reference'],
               'best_local_model': local_best, 'best_local_rmse': local_rms[local_best],
               'support_pass': ok, 'old_support_pass': old_support(full, keep), 'bootstrap_pass': okci,
               'screen_pass': bool(lock['discovery_pass'] and sc['gain'] >= .05 and ok and okci),
               **metrics(accepted, accepted[lock['model']]), **cis}
        rows.append(row)
        for selection, z in [('full_scope', full), ('accepted', accepted)]:
            for model in MODELS:
                model_rows.append({**setting, 'floor': lock['floor'], 'stage': stage,
                                   'selection': selection, 'model': model, **metrics(z, z[model])})
    return rows, model_rows


def run_config(d, features, scope, direction, seed, national):
    folder = RESULTS / scope / direction / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        print('CACHED', scope, direction, seed, flush=True)
        return
    if (folder / 'predictions.csv.gz').exists():
        p = read(folder / 'predictions.csv.gz')
        meta = json.loads((folder / 'training_manifest.json').read_text(encoding='utf-8'))
    else:
        p, meta = fit_predict(d, features, scope, direction, seed, folder, national)
    disc = p[p.development_city & p.year.eq(2019)].reset_index(drop=True)
    val = p[p.development_city & p.year.eq(2020)].reset_index(drop=True)
    settings = {'scope': scope, 'direction': direction, 'seed': seed}
    locks = []
    for floor in FLOORS:
        name = int(floor * 100)
        lock_path = folder / f'lock_{name}.json'
        if lock_path.exists():
            lock = json.loads(lock_path.read_text(encoding='utf-8'))
        else:
            lock, candidates, shortlist = select_locked(disc, val, floor)
            dump(lock_path, {**lock, **settings, 'protocol_sha256': sha(HERE / 'PROTOCOL.md')})
            for label, r in [('search', candidates), ('shortlist', shortlist)]:
                z = pd.DataFrame(r)
                if len(z):
                    z['rule'] = z.rule.map(json.dumps)
                z.to_csv(folder / f'{label}_{name}.csv.gz', index=False)
        locks.append(lock)
    # All four choices are frozen before any audit outcomes are used here.
    rows, models = [], []
    for lock in locks:
        a, b = audit_lock(p, lock, d[d.country_iso.eq(direction.split('_to_')[1])], settings)
        rows.extend(a)
        models.extend(b)
    pd.DataFrame(rows).to_csv(folder / 'gates.csv', index=False)
    pd.DataFrame(models).to_csv(folder / 'metrics.csv.gz', index=False)
    dump(folder / 'complete.json', {**meta, 'completed': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                   'lock_count': len(locks), 'prediction_sha256': sha(folder / 'predictions.csv.gz')})
    print('COMPLETE', scope, direction, seed, 'discovery_pass', sum(x['discovery_pass'] for x in locks), flush=True)


def aggregate():
    AUDIT.mkdir(exist_ok=True)
    paths = list(RESULTS.glob('*/*/*/gates.csv'))
    if not paths:
        return
    pd.concat([read(p) for p in paths], ignore_index=True).to_csv(AUDIT / 'all_gates.csv', index=False)
    pd.concat([read(p) for p in RESULTS.glob('*/*/*/metrics.csv.gz')], ignore_index=True).to_csv(AUDIT / 'all_metrics.csv.gz', index=False)
    locks = [json.loads(p.read_text(encoding='utf-8')) for p in RESULTS.glob('*/*/*/lock_*.json')]
    dump(AUDIT / 'locks.json', locks)
    dump(AUDIT / 'completion.json', {'completed_configurations': len(list(RESULTS.glob('*/*/*/complete.json'))),
                                   'expected_configurations': 18, 'locks': len(locks), 'expected_locks': 72})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()
    if args.aggregate_only:
        aggregate()
        return
    AUDIT.mkdir(parents=True, exist_ok=True)
    inputs = [OLD / 'data/all_networks_model_panel.csv.gz', OLD / 'data/common_era5_month_2015_2022.csv.gz']
    receipt_path = AUDIT / 'run_receipt.json'
    if not receipt_path.exists():
        dump(receipt_path, {'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'protocol_sha256': sha(HERE / 'PROTOCOL.md'),
                            'script_sha256': sha(__file__), 'inputs': {str(p): sha(p) for p in inputs}})
    d = load_panel()
    scope_rows = []
    for scope in SCOPES:
        z = d[scope_mask(d, scope)]
        for c, a in z.groupby('country_iso'):
            scope_rows.append({'scope': scope, 'country': c, 'rows': len(a), 'fuas': a.eFUA_ID.nunique(),
                               'national_rows': len(d[d.country_iso.eq(c)]), 'coverage': len(a) / len(d[d.country_iso.eq(c)])})
    pd.DataFrame(scope_rows).to_csv(AUDIT / 'geographic_scope.csv', index=False)
    d[['country_iso', 'eFUA_ID', 'fua_name', 'fua_centroid_lat', 'fua_centroid_lon', 'distance_to_coast_km']].drop_duplicates().to_csv(AUDIT / 'city_geography.csv', index=False)
    n = 0
    for scope in SCOPES:
        for direction in ['CHN_to_AUS', 'AUS_to_CHN']:
            national = joblib.load(OLD / 'results/all_networks/monthly_plus_mixing' / direction / 'source_models.joblib')
            for seed in SEEDS:
                run_config(d, national['features'], scope, direction, seed, national)
                n += 1
                if args.limit and n >= args.limit:
                    aggregate()
                    return
    aggregate()


if __name__ == '__main__':
    main()
