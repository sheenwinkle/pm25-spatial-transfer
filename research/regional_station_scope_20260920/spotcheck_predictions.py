"""Refit one rolling month per configuration and reproduce all 27 predictions."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from run_regional import read, dump, norm, tree, embed, kernel, pos, POLICIES, SEEDS, MODELS

HERE = Path(__file__).resolve().parent


def main():
    rows = []
    for complete in sorted((HERE / 'results').glob('*/*/*/complete.json')):
        meta = json.loads(complete.read_text(encoding='utf-8'))
        folder = complete.parent
        panel = read(HERE / f"data/{meta['cohort']}_panel.csv.gz")
        target = meta['direction'].split('_to_')[1]
        panel = panel[panel.country_iso.eq(target)]
        saved = read(folder / 'predictions.csv.gz')
        month = int(saved[saved.year.eq(2021)].t.min())
        fit = panel[panel.eFUA_ID.isin(meta['development_cities']) & panel.t.between(month - 12, month - 1)]
        test = panel[panel.t.eq(month)].copy()
        features = meta['features']
        x, ex, y = fit[features], test[features], fit.observed_pm25.to_numpy()
        z = norm()
        xx, ee = z.fit_transform(x), z.transform(ex)
        predictions = {'target_mean': np.full(len(test), y.mean()),
            'target_ridge': pos(Ridge(alpha=10).fit(xx, y).predict(ee)),
            'target_rbf': kernel(rbf_kernel(xx), rbf_kernel(ee, xx), y),
            'target_lgbm': pos(tree().fit(x, y).predict(ex))}
        for key, cls in [('target_rf', RandomForestRegressor), ('target_extra', ExtraTreesRegressor)]:
            model = cls(n_estimators=80, min_samples_leaf=3, max_features=1.0, n_jobs=4, random_state=5993).fit(xx, y)
            predictions[key] = pos(model.predict(ee))
        models = joblib.load(folder / 'source_models.joblib')['models']
        for policy in POLICIES:
            model = models[policy]
            raw, eraw = pos(model.predict(x)), pos(model.predict(ex))
            predictions[policy + '_raw'] = eraw
            zz = StandardScaler()
            rx, re = zz.fit_transform(raw[:, None]), zz.transform(eraw[:, None])
            predictions[policy + '_affine'] = pos(Ridge(alpha=10).fit(rx, y).predict(re))
            predictions[policy + '_residual'] = pos(eraw + tree().fit(x, y - raw).predict(ex))
            for name, m in [(policy + '_kernel', model)] + [(f'{policy}_shuffle_{s}', models[f'{policy}_shuffle_{s}']) for s in SEEDS]:
                a, b = embed(m, x), embed(m, ex)
                predictions[name] = kernel((a @ a.T).toarray(), (b @ a.T).toarray(), y)
        actual = saved[saved.t.eq(month)].set_index('eFUA_ID').reindex(test.eFUA_ID)
        assert set(predictions) == set(MODELS)
        for name, values in predictions.items():
            np.testing.assert_allclose(values, actual[name], rtol=1e-10, atol=1e-10)
            rows.append({'cohort': meta['cohort'], 'direction': meta['direction'], 'seed': meta['seed'],
                         'model': name, 't': month, 'rows': len(test),
                         'max_absolute_error': float(np.max(abs(values - actual[name].to_numpy())))})
    pd.DataFrame(rows).to_csv(HERE / 'audit/prediction_refit_spotcheck.csv', index=False)
    result = {'all_checks_passed': True, 'rolling_month_refits': 12, 'model_comparisons': len(rows),
              'maximum_absolute_error': max(r['max_absolute_error'] for r in rows),
              'scope': 'refit target calibration and local models for first available 2021 month per configuration; source fits loaded unchanged'}
    dump(HERE / 'audit/PREDICTION_SPOTCHECK.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
