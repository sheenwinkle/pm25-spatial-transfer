"""Validate rolling-source chronology and paired frozen-rule comparisons."""
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from run_focused import (HERE, RESULTS, MODELS, LOCAL, NEG, GATES, Y, load_panel, scope_mask,
                         read, dump, sha, norm, rulemask, support, score, metrics)
from audit_focused import stage_mask
from run_recency import OUT, SCOPE


def main():
    paths = list(OUT.glob('*/*/*/complete.json'))
    assert len(paths) == 6, len(paths)
    receipt = json.loads((OUT / 'receipt.json').read_text(encoding='utf-8'))
    assert receipt['script_sha256'] == sha(HERE / 'run_recency.py')
    assert receipt['protocol_sha256'] == sha(HERE / 'RECENCY_PROTOCOL.md')
    assert receipt['shared_helpers_sha256'] == sha(HERE / 'run_focused.py')
    d = load_panel()
    scoped = d[scope_mask(d, SCOPE)]
    source_fits = metric_rows = rules = 0
    paired, summary = [], []
    changed = {m for m in MODELS if m.startswith(('regional_', 'matched_'))}
    for path in paths:
        folder = path.parent
        meta = json.loads(path.read_text(encoding='utf-8'))
        parent = RESULTS / SCOPE / meta['direction'] / str(meta['seed'])
        p, before = read(folder / 'predictions.csv.gz'), read(parent / 'predictions.csv.gz')
        assert sha(folder / 'predictions.csv.gz') == meta['prediction_sha256']
        assert sha(parent / 'predictions.csv.gz') == meta['parent_predictions_sha256']
        unchanged = [c for c in p if c not in changed]
        pd.testing.assert_frame_equal(p[unchanged], before[unchanged], check_exact=True)
        source_country, target_country = meta['direction'].split('_to_')
        source = scoped[scoped.country_iso.eq(source_country)]
        target = scoped[scoped.country_iso.eq(target_country) & scoped.eFUA_ID.isin(meta['development_cities'])]
        u = target[target.year.le(2018)]
        mf = meta['features'][:6] + meta['features'][-3:]
        selection = read(folder / 'source_selection.csv.gz')
        for fit in read(folder / 'fit_audit.csv').itertuples():
            month = fit.prediction_t
            assert month - 24 <= fit.source_first_t <= fit.source_last_t < month
            assert month - 12 <= fit.target_first_t <= fit.target_last_t < month
            s = source[source.t.between(month - 24, month - 1)]
            t = target[target.t.between(month - 12, month - 1)]
            assert len(s) == fit.source_rows and len(t) == fit.target_labels
            z = norm()
            sx = z.fit_transform(s[mf])
            distance = cKDTree(z.transform(u[mf])).query(sx, workers=4)[0]
            cutoff = float(np.quantile(distance, .5))
            assert abs(cutoff - fit.matching_cutoff) < 1e-10
            recorded = selection[selection.prediction_t.eq(month)]
            assert list(zip(s.eFUA_ID, s.t)) == list(zip(recorded.eFUA_ID, recorded.t))
            np.testing.assert_allclose(distance, recorded.matching_distance.to_numpy(), rtol=0, atol=1e-10)
            assert np.array_equal(recorded.matched, distance <= cutoff)
            assert int(recorded.matched.sum()) == fit.matched_source_rows
            source_fits += 1
        gates = read(folder / 'rediscovered_gates.csv')
        inherited = read(folder / 'inherited_rules_gates.csv')
        for floor in [.5, .25, .1, .05]:
            lock = json.loads((folder / f'lock_{int(floor*100)}.json').read_text(encoding='utf-8'))
            assert lock['status'] == 'locked'
            for stage, year in [('discovery_2019', 2019), ('validation_2020', 2020)]:
                z = p[stage_mask(p, stage)]
                keep = rulemask(z, lock['rule'])
                assert support(z, keep, floor)
                assert abs(score(z, lock['model'], keep)['gain'] - lock[f'gain_{year}']) < 1e-10
            for kind, gs in [('rediscovered', gates), ('inherited_rules', inherited)]:
                oldlock = json.loads((parent / f'lock_{int(floor*100)}.json').read_text(encoding='utf-8'))
                active = lock if kind == 'rediscovered' else oldlock
                for r in gs[np.isclose(gs.floor, floor)].itertuples():
                    full = p[stage_mask(p, r.stage)]
                    keep = rulemask(full, active['rule'])
                    accepted = full[keep]
                    met = metrics(accepted, accepted[active['model']])
                    assert met['n'] == r.n
                    if met['rmse'] is not None:
                        assert abs(met['rmse'] - r.rmse) < 1e-10
                    rules += 1
                    if kind == 'inherited_rules' and 'audit' in r.stage:
                        b = before[stage_mask(before, r.stage)]
                        assert np.array_equal(keep, rulemask(b, active['rule']))
                        previous = metrics(b[keep], b.loc[keep, active['model']])
                        paired.append({'scope': SCOPE, 'direction': meta['direction'], 'seed': meta['seed'],
                                       'floor': floor, 'stage': r.stage, 'model': active['model'],
                                       'rule': json.dumps(active['rule']), 'n': r.n,
                                       'before_rmse': previous['rmse'], 'after_rmse': met['rmse'],
                                       'paired_rmse_reduction': 1 - met['rmse'] / previous['rmse'] if met['rmse'] is not None else None})
            fg = gates[np.isclose(gates.floor, floor)]
            tm, nc = fg[fg.stage.eq('temporal_audit')].iloc[0], fg[fg.stage.eq('new_city_audit')].iloc[0]
            summary.append({'scope': SCOPE, 'direction': meta['direction'], 'seed': meta['seed'], 'floor': floor,
                            'model': lock['model'], 'rule': json.dumps(lock['rule']), 'discovery_pass': lock['discovery_pass'],
                            'worst_discovery_gain': lock['worst_gain'], 'temporal_gain': tm.gain, 'new_city_gain': nc.gain,
                            'temporal_coverage': tm.coverage, 'new_city_coverage': nc.coverage,
                            'temporal_support': bool(tm.support_pass), 'new_city_support': bool(nc.support_pass),
                            'temporal_r2': tm.r2, 'new_city_r2': nc.r2, 'new_city_rmse': nc.rmse,
                            'joint_screen': bool(tm.screen_pass and nc.screen_pass)})
        for kind in ['rediscovered', 'inherited_rules']:
            allm = read(folder / f'{kind}_metrics.csv.gz')
            for r in allm.itertuples():
                active_path = (folder if kind == 'rediscovered' else parent) / f'lock_{int(r.floor*100)}.json'
                lock = json.loads(active_path.read_text(encoding='utf-8'))
                full = p[stage_mask(p, r.stage)]
                use = rulemask(full, lock['rule']) if r.selection == 'accepted' else np.ones(len(full), bool)
                actual = metrics(full[use], full.loc[use, r.model])
                assert r.n == actual['n']
                if actual['rmse'] is not None:
                    assert abs(r.rmse - actual['rmse']) < 1e-10
                metric_rows += 1
    s = pd.DataFrame(summary)
    s.to_csv(OUT / 'joint_screens.csv', index=False)
    pd.DataFrame(paired).to_csv(OUT / 'paired_fixed_rule_comparison.csv', index=False)
    verification = {'configurations': len(paths), 'locked_settings': len(s),
                    'discovery_pass': int(s.discovery_pass.sum()), 'joint_screen_pass': int(s.joint_screen.sum()),
                    'source_windows_verified': source_fits, 'source_matching_recomputed': source_fits,
                    'metric_rows_recomputed': metric_rows, 'rule_statistics_recomputed': rules,
                    'target_and_national_predictions_bitwise_unchanged': True,
                    'parent_predictions_unchanged': True, 'protocol_and_runner_unchanged': True}
    dump(OUT / 'verification.json', verification)
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
