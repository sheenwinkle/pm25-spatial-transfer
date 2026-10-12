"""Recompute targets, support, metrics, chronology and paired city membership."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'focused_conditions_20260919'))
from run_focused import read, sha, dump, MODELS, LOCAL, REFS, Y, rulemask, bootstrap

KEYS = ['country_iso', 'eFUA_ID', 'year', 'month']


def calculated_metrics(d, model):
    if not len(d):
        return {'n': 0, 'fuas': 0, 'months': 0, 'rmse': None, 'mae': None, 'r2': None, 'bias': None}
    y, pred = d[Y].to_numpy(), d[model].to_numpy()
    squared = np.square(y - pred)
    denominator = np.square(y - y.mean()).sum()
    return {'n': len(d), 'fuas': d.eFUA_ID.nunique(), 'months': d.t.nunique(),
            'rmse': np.sqrt(squared.mean()), 'mae': np.abs(y - pred).mean(),
            'r2': 1 - squared.sum() / denominator if denominator > 0 else None,
            'bias': (pred - y).mean()}


def stage_rows(p, stage):
    if stage == 'discovery_2019':
        return p[p.development_city & p.year.eq(2019)]
    if stage == 'validation_2020':
        return p[p.development_city & p.year.eq(2020)]
    use = p.year.ge(2021)
    use &= p.development_city if stage == 'temporal_audit' else ~p.development_city
    if stage == 'new_grid_audit':
        use &= p.new_weather_grid
    return p[use]


def support_independent(full, accepted, floor):
    if not len(accepted):
        return False
    for year, base in full.groupby('year'):
        z = accepted[accepted.year.eq(year)]
        if len(z) < 24 or z.eFUA_ID.nunique() < 3 or z.t.nunique() < 6 or len(z) / len(base) + 1e-12 < floor:
            return False
    return True


def main():
    station = read(HERE / 'data/station_month_with_roles.csv.gz')
    assert not station.duplicated(['site_key', 'year', 'month']).any()
    assert not (station.selected_role & ~station.selected_all).any()
    assert not station[station.selected_role].role.str.startswith('unknown').any()
    assert not station[station.selected_role].role.isin(['control_2012', 'PM25_peak_2023', 'designated_roadside']).any()
    panels = {}
    errors = []

    def same(a, b):
        if a is None or pd.isna(a):
            assert b is None or pd.isna(b), (a, b)
        else:
            assert b is not None and np.isfinite(b)
            errors.append(abs(float(a) - float(b)))
            assert np.isclose(a, b, rtol=1e-10, atol=1e-10), (a, b)

    for cohort, flag in [('membership_all', 'selected_all'), ('documented_role', 'selected_role')]:
        panel = read(HERE / f'data/{cohort}_panel.csv.gz')
        panels[cohort] = panel
        expected = station[station[flag]].groupby(KEYS).observed_pm25.mean().rename('expected').reset_index()
        joined = panel.merge(expected, on=KEYS, validate='one_to_one', how='outer', indicator=True)
        assert joined._merge.eq('both').all()
        for a, b in zip(joined.observed_pm25, joined.expected):
            same(a, b)
    assert panels['membership_all'].query("country_iso == 'CHN'").eFUA_ID.nunique() == 43
    splits = json.loads((HERE / 'audit/fixed_city_splits.json').read_text(encoding='utf-8'))
    for country, entry in splits.items():
        expected = []
        for panel in panels.values():
            counts = panel[panel.country_iso.eq(country) & panel.year.eq(2018)].groupby('eFUA_ID').size()
            expected.append(set(counts[counts >= 6].index))
        assert set(entry['eligible_common_2018']) == set.intersection(*expected)
    count_metrics = count_gates = count_fits = 0
    locks_summary, annual_rows = [], []
    folders = sorted((HERE / 'results').glob('*/*/*/complete.json'))
    assert len(folders) == 12, len(folders)
    for complete in folders:
        folder = complete.parent
        meta = json.loads(complete.read_text(encoding='utf-8'))
        assert meta['protocol_sha256'] == sha(HERE / 'PROTOCOL.md')
        assert meta['script_sha256'] == sha(HERE / 'run_regional.py')
        assert meta['panel_sha256'] == sha(HERE / f"data/{meta['cohort']}_panel.csv.gz")
        assert meta['source_max_year'] <= 2018
        src, target = meta['direction'].split('_to_')
        ids = splits[target]['seeds'][str(meta['seed'])]
        assert meta['development_cities'] == ids
        panel = panels[meta['cohort']]
        target_panel = panel[panel.country_iso.eq(target)]
        source = read(folder / 'source_selection.csv.gz')
        assert source.country_iso.eq(src).all() and source.year.le(2018).all()
        assert source.matched.sum() == meta['matched_source_rows']
        expected_source = panel[panel.country_iso.eq(src) & panel.year.le(2018)]
        assert set(zip(source.eFUA_ID, source.t)) == set(zip(expected_source.eFUA_ID, expected_source.t))
        used = set()
        p = read(folder / 'predictions.csv.gz')
        assert not p.duplicated(KEYS).any()
        assert p[MODELS].notna().all().all()
        assert (p.development_city == p.eFUA_ID.isin(ids)).all()
        comparison = p.merge(target_panel[KEYS + [Y] + meta['features']], on=KEYS, suffixes=('', '_original'), validate='one_to_one')
        assert len(comparison) == len(p)
        for col in [Y] + meta['features']:
            np.testing.assert_allclose(comparison[col], comparison[col + '_original'], rtol=1e-12, atol=1e-12, equal_nan=True)
        for fit in read(folder / 'fit_audit.csv').itertuples():
            z = target_panel[target_panel.eFUA_ID.isin(ids) & target_panel.t.between(fit.prediction_t - 12, fit.prediction_t - 1)]
            assert len(z) == fit.labels
            assert fit.latest_fit_t == z.t.max() < fit.prediction_t
            assert fit.earliest_fit_t == z.t.min()
            assert set(json.loads(fit.fit_cities)) == set(z.eFUA_ID)
            assert p[p.t.eq(fit.prediction_t)].fit_labels.eq(len(z)).all()
            used.update(zip(z.eFUA_ID, z.t))
            count_fits += 1
        assert len(used) == meta['unique_target_calibration_labels']
        cal = read(folder / 'calibration_rows.csv.gz')
        assert used == set(zip(cal.eFUA_ID, cal.t))
        reported = read(folder / 'metrics.csv.gz')
        gates = read(folder / 'gates.csv')
        for lock_path in sorted(folder.glob('lock_*.json')):
            lock = json.loads(lock_path.read_text(encoding='utf-8'))
            assert lock['protocol_sha256'] == sha(HERE / 'PROTOCOL.md')
            if lock['status'] != 'locked':
                locks_summary.append({**lock, 'joint_screen': False})
                continue
            relevant = gates[gates.floor.eq(lock['floor'])]
            for row in relevant.to_dict('records'):
                full = stage_rows(p, row['stage'])
                accepted = full[rulemask(full, lock['rule'])]
                for field, value in calculated_metrics(accepted, lock['model']).items():
                    same(row[field], value)
                same(row['coverage'], len(accepted) / len(full) if len(full) else 0)
                ok = support_independent(full, accepted, lock['floor'])
                assert row['support_pass'] == ok
                if len(accepted):
                    rm = {r: calculated_metrics(accepted, r)['rmse'] for r in REFS}
                    same(row['gain'], 1 - calculated_metrics(accepted, lock['model'])['rmse'] / max(min(rm.values()), 1e-12))
                    for cluster in ['eFUA_ID', 'weather_grid_id']:
                        if 'audit' not in row['stage']:
                            continue
                        for ref in REFS:
                            lo, hi = bootstrap(accepted, lock['model'], ref, meta['seed'], cluster)
                            same(row[f'{cluster}_{ref}_low'], lo)
                            same(row[f'{cluster}_{ref}_high'], hi)
                for year, whole in full.groupby('year'):
                    a = accepted[accepted.year.eq(year)]
                    annual_rows.append({'cohort': meta['cohort'], 'direction': meta['direction'], 'seed': meta['seed'],
                        'floor': lock['floor'], 'stage': row['stage'], 'year': year,
                        'full_rows': len(whole), 'n': len(a), 'fuas': a.eFUA_ID.nunique(),
                        'months': a.t.nunique(), 'coverage': len(a) / len(whole)})
                for selection, chosen in [('full_scope', full), ('accepted', accepted)]:
                    part = reported[reported.floor.eq(lock['floor']) & reported.stage.eq(row['stage']) & reported.selection.eq(selection)]
                    assert len(part) == len(MODELS)
                    for r in part.to_dict('records'):
                        for field, value in calculated_metrics(chosen, r['model']).items():
                            same(r[field], value)
                        count_metrics += 1
                count_gates += 1
            required = relevant[relevant.stage.isin(['temporal_audit', 'new_city_audit'])]
            record = {**lock, 'joint_screen': bool(len(required) == 2 and required.screen_pass.all())}
            for r in relevant.to_dict('records'):
                if 'audit' in r['stage']:
                    for col in ['n', 'fuas', 'rmse', 'r2', 'coverage', 'gain', 'support_pass', 'bootstrap_pass', 'screen_pass']:
                        record[r['stage'] + '_' + col] = r[col]
            locks_summary.append(record)
    ledger = pd.DataFrame(locks_summary)
    ledger['rule'] = ledger.rule.map(json.dumps)
    ledger.to_csv(HERE / 'audit/all_48_settings.csv', index=False)
    pd.DataFrame(annual_rows).to_csv(HERE / 'audit/annual_gate_support.csv', index=False)
    report = {'all_checks_passed': True, 'configurations': len(folders), 'settings': len(ledger),
              'joint_screens': int(ledger.joint_screen.sum()), 'metric_rows_recomputed': count_metrics,
              'gate_stages_recomputed': count_gates, 'rolling_fit_windows_checked': count_fits,
              'maximum_numeric_difference': max(errors), 'metadata_year_verification_complete': False,
              'checks': ['station-month uniqueness', 'exact target aggregation', 'role and membership exclusions',
                         'paired development cities', 'source chronology', 'target chronology and city isolation',
                         'features unchanged', 'all model metrics', 'per-year gate support', 'bootstrap reproducibility']}
    dump(HERE / 'audit/VERIFICATION.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
