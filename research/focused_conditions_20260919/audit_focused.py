"""Recompute saved metrics, check source selection and target chronology."""
import json
from pathlib import Path, PureWindowsPath
import numpy as np
import pandas as pd
from run_focused import (HERE, OLD, RESULTS, AUDIT, SCOPES, SEEDS, FLOORS, LOCAL,
                         REFS, MODELS, Y, read, dump, sha, load_panel, scope_mask,
                         rulemask, support, old_support, score, metrics)


def stage_mask(p, stage):
    return {'discovery_2019': p.development_city & p.year.eq(2019),
            'validation_2020': p.development_city & p.year.eq(2020),
            'temporal_audit': p.development_city & p.year.ge(2021),
            'new_city_audit': ~p.development_city & p.year.ge(2021),
            'new_grid_audit': ~p.development_city & p.new_weather_grid & p.year.ge(2021)}[stage]


def main():
    completed = sorted(RESULTS.glob('*/*/*/complete.json'))
    assert len(completed) == 18, len(completed)
    d = load_panel()
    receipt = json.loads((AUDIT / 'run_receipt.json').read_text(encoding='utf-8'))
    assert receipt['protocol_sha256'] == sha(HERE / 'PROTOCOL.md')
    assert receipt['script_sha256'] == sha(HERE / 'run_focused.py')
    for file, digest in receipt['inputs'].items():
        assert sha(OLD / 'data' / PureWindowsPath(file).name) == digest
    max_error = 0.0
    metrics_checked = fits_checked = rules_checked = 0
    joint = []
    coverage_rows = []
    for complete in completed:
        folder = complete.parent
        meta = json.loads(complete.read_text(encoding='utf-8'))
        assert meta['script_sha256'] == receipt['script_sha256']
        p = read(folder / 'predictions.csv.gz')
        assert sha(folder / 'predictions.csv.gz') == meta['prediction_sha256']
        assert np.isfinite(p[MODELS + [Y]].to_numpy()).all()
        assert not p.duplicated(['eFUA_ID', 't']).any()
        assert set(p.loc[p.development_city, 'eFUA_ID']) == set(meta['development_cities'])
        country = meta['direction'].split('_to_')[1]
        raw = d[scope_mask(d, meta['scope']) & d.country_iso.eq(country)]
        eligible = raw[raw.year.eq(2018)].groupby('eFUA_ID').size()
        assert set(meta['development_cities']).issubset(eligible[eligible >= 6].index)
        fits = read(folder / 'fit_audit.csv')
        assert fits.latest_fit_t.lt(fits.prediction_t).all()
        assert fits.earliest_fit_t.ge(fits.prediction_t - 12).all()
        used = set()
        for fit in fits.itertuples():
            z = raw[raw.eFUA_ID.isin(meta['development_cities']) & raw.t.between(fit.prediction_t - 12, fit.prediction_t - 1)]
            assert len(z) == fit.labels
            assert sorted(z.eFUA_ID.unique()) == json.loads(fit.fit_cities)
            used.update(zip(z.eFUA_ID, z.t))
            fits_checked += 1
        saved = read(folder / 'calibration_rows.csv.gz')
        assert used == set(zip(saved.eFUA_ID, saved.t))
        assert len(used) == meta['unique_target_calibration_labels']
        s = read(folder / 'source_selection.csv.gz')
        assert s.year.le(2018).all()
        assert s.country_iso.eq(meta['direction'].split('_to_')[0]).all()
        assert np.array_equal(s.matched, s.matching_distance.le(meta['source_matching_cutoff']))
        assert int(s.matched.sum()) == meta['matched_source_rows']
        gates = read(folder / 'gates.csv')
        allm = read(folder / 'metrics.csv.gz')
        for floor in FLOORS:
            lock = json.loads((folder / f'lock_{int(floor * 100)}.json').read_text(encoding='utf-8'))
            if lock['status'] != 'locked':
                joint.append({**{k: meta[k] for k in ['scope', 'direction', 'seed']},
                              'floor': floor, 'locked': False, 'joint_screen': False})
                continue
            assert len(lock['rule']) <= 4
            for a in lock['rule']:
                assert a['feature'] != Y and a['feature'] in p
            disc, val = p[stage_mask(p, 'discovery_2019')], p[stage_mask(p, 'validation_2020')]
            assert support(disc, rulemask(disc, lock['rule']), floor)
            assert support(val, rulemask(val, lock['rule']), floor)
            for year, frame in [(2019, disc), (2020, val)]:
                actual = score(frame, lock['model'], rulemask(frame, lock['rule']))['gain']
                assert abs(actual - lock[f'gain_{year}']) < 1e-10
            assert lock['discovery_pass'] == (min(lock['gain_2019'], lock['gain_2020']) >= .05)
            fg = gates[np.isclose(gates.floor, floor)]
            for row in fg.to_dict('records'):
                full = p[stage_mask(p, row['stage'])]
                keep = rulemask(full, lock['rule'])
                z = full[keep]
                actual = metrics(z, z[lock['model']])
                assert row['n'] == len(z)
                for key in ['rmse', 'mae', 'r2', 'bias']:
                    if actual[key] is not None:
                        error = abs(row[key] - actual[key])
                        assert error < 1e-9, (folder, key, error)
                        max_error = max(max_error, error)
                assert row['support_pass'] == support(full, keep, floor)
                assert row['old_support_pass'] == old_support(full, keep)
                assert abs(row['gain'] - score(full, lock['model'], keep)['gain']) < 1e-10
                rules_checked += 1
                for year in sorted(full.year.unique()):
                    y = full[full.year.eq(year)]
                    take = rulemask(y, lock['rule'])
                    a = y[take]
                    coverage_rows.append({**{k: meta[k] for k in ['scope', 'direction', 'seed']},
                                          'floor': floor, 'stage': row['stage'], 'year': year,
                                          'n': len(a), 'fuas': a.eFUA_ID.nunique(), 'months': a.t.nunique(),
                                          'coverage': take.mean() if len(take) else 0})
                for selection, a in [('full_scope', full), ('accepted', z)]:
                    mm = allm[np.isclose(allm.floor, floor) & allm.stage.eq(row['stage']) & allm.selection.eq(selection)]
                    for r in mm.itertuples():
                        met = metrics(a, a[r.model])
                        assert r.n == met['n']
                        if met['rmse'] is not None:
                            err = abs(r.rmse - met['rmse'])
                            assert err < 1e-9
                            max_error = max(max_error, err)
                        metrics_checked += 1
            tm = fg[fg.stage.eq('temporal_audit')].iloc[0]
            nc = fg[fg.stage.eq('new_city_audit')].iloc[0]
            joint.append({**{k: meta[k] for k in ['scope', 'direction', 'seed']}, 'floor': floor,
                          'locked': True, 'model': lock['model'], 'rule': json.dumps(lock['rule']),
                          'discovery_pass': lock['discovery_pass'], 'worst_discovery_gain': lock['worst_gain'],
                          'temporal_gain': tm.gain, 'new_city_gain': nc.gain,
                          'temporal_coverage': tm.coverage, 'new_city_coverage': nc.coverage,
                          'temporal_support': bool(tm.support_pass), 'new_city_support': bool(nc.support_pass),
                          'temporal_bootstrap': bool(tm.bootstrap_pass), 'new_city_bootstrap': bool(nc.bootstrap_pass),
                          'temporal_r2': tm.r2, 'new_city_r2': nc.r2, 'new_city_rmse': nc.rmse,
                          'joint_screen': bool(tm.screen_pass and nc.screen_pass)})
    j = pd.DataFrame(joint)
    j.to_csv(AUDIT / 'joint_screens.csv', index=False)
    pd.DataFrame(coverage_rows).to_csv(AUDIT / 'annual_support.csv', index=False)
    summary = {'configurations': len(completed), 'locked_settings': int(j.locked.sum()),
               'settings_total': len(j), 'discovery_pass': int(j.discovery_pass.fillna(False).sum()),
               'joint_screen_pass': int(j.joint_screen.sum()), 'rules_reproduced': rules_checked,
               'model_metric_rows_checked': metrics_checked, 'calibration_fits_checked': fits_checked,
               'maximum_numeric_error': max_error, 'input_hashes_unchanged': True,
               'protocol_unchanged': True, 'runner_unchanged': True,
               'interpretation': 'Exploratory unadjusted screening, not independent confirmation.'}
    dump(AUDIT / 'verification.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
