"""Recompute saved prediction metrics and audit frozen predictions against monitors."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from audit_inputs import HERE, PREVIOUS
from collect_monthly_power import dump, sha, PROCESSED

OUT = HERE / "audit"
Y = "pm25_mean_ug_m3"
BASELINES = ["target_mean", "target_ridge", "target_lgbm", "target_rbf", "shuffled_source_kernel"]
KEY = ["feature_set", "direction", "seed", "window_months"]


def score(frame, prediction, total, target=Y):
    if frame.empty:
        return dict(n=0, fuas=0, months=0, coverage=0., fua_coverage=0.,
                    rmse=np.nan, mae=np.nan, r2=np.nan, bias=np.nan, fua_macro_rmse=np.nan)
    err = frame[prediction].to_numpy(float) - frame[target].to_numpy(float)
    ss = np.sum((frame[target] - frame[target].mean()) ** 2)
    return dict(n=len(frame), fuas=frame.eFUA_ID.nunique(), months=frame.t.nunique(),
                coverage=len(frame)/len(total), fua_coverage=frame.eFUA_ID.nunique()/total.eFUA_ID.nunique(),
                rmse=np.sqrt(np.mean(err**2)), mae=np.mean(abs(err)), r2=1-np.sum(err**2)/ss if ss > 0 else np.nan,
                bias=err.mean(), fua_macro_rmse=pd.Series(err**2, index=frame.eFUA_ID).groupby(level=0).mean().pow(.5).mean())


def stages(p):
    return {
        "discovery": p.development_city & p.year.between(2019, 2020),
        "confirmation": p.development_city & p.year.between(2021, 2022),
        "spatial_temporal_audit": ~p.development_city & p.year.between(2023, 2024),
        "seen_city_temporal_audit": p.development_city & p.year.between(2023, 2024),
        "new_weather_grid_audit": ~p.development_city & p.new_weather_grid & p.year.between(2023, 2024),
    }


def assert_close(a, b):
    assert np.isclose(a, b, rtol=2e-10, atol=2e-10, equal_nan=True), (a, b)


def bootstrap_direct(p, seed, cluster):
    cities, quarters = pd.factorize(p[cluster])[0], pd.factorize(p.quarter_id)[0]
    nc, nt = len(np.unique(cities)), len(np.unique(quarters))
    if nc < 2 or nt < 2:
        return None
    rng = np.random.default_rng(seed)
    cw = rng.multinomial(nc, np.full(nc, 1/nc), size=1000)
    tw = rng.multinomial(nt, np.full(nt, 1/nt), size=1000)
    values = []
    a = (p.selected_prediction.to_numpy() - p[Y].to_numpy())**2
    b = (p.target_mean.to_numpy() - p[Y].to_numpy())**2
    # Direct row weights provide an implementation independent of grouped matrix sums.
    for c, t in zip(cw, tw):
        weights = c[cities]*t[quarters]
        if weights.sum():
            values.append(np.sqrt(np.average(b, weights=weights))-np.sqrt(np.average(a, weights=weights)))
    return np.quantile(values, [.025, .975])


def verify_models():
    manifest = json.loads((HERE/"results/run_manifest.json").read_text())
    assert manifest["script_sha256"] == sha(HERE/"run_models.py")
    assert manifest["protocol_sha256"] == sha(HERE/"protocol.json")
    assert manifest["base_helper_sha256"] == sha(PREVIOUS/"run_study.py")
    all_gates, all_metrics, budgets, failures = [], [], [], []
    metric_checks = stage_checks = ci_checks = 0
    for folder in sorted((HERE/"results").glob("*/*")):
        if not (folder/"gates.csv").exists():
            continue
        gates = pd.read_csv(folder/"gates.csv", float_precision="round_trip")
        metrics = pd.read_csv(folder/"model_metrics.csv", float_precision="round_trip")
        all_gates.append(gates); all_metrics.append(metrics)
        budgets.append(pd.read_csv(folder/"label_budgets.csv"))
        for file in sorted(folder.glob("predictions_*.csv.gz")):
            _, seed, window = file.name.split(".")[0].split("_")
            seed, window = int(seed), int(window)
            lock = json.loads((folder/f"lock_{seed}_{window}.json").read_text())
            p = pd.read_csv(file, float_precision="round_trip")
            assert not p.duplicated(["eFUA_ID", "t"]).any()
            assert np.isfinite(p[[Y, "selected_prediction"]+BASELINES]).all().all()
            assert p.t.eq(p.year*12+p.month).all()
            selected = np.ones(len(p), bool)
            for node in lock["rule"]:
                values = p[node["feature"]]
                selected &= values.le(node["threshold"]) if node["operator"] == "<=" else values.ge(node["threshold"])
            assert np.array_equal(selected, p.accepted)
            for stage, mask in stages(p).items():
                total, q = p.loc[mask], p.loc[mask & selected]
                row = gates.loc[gates.seed.eq(seed) & gates.window_months.eq(window) & gates.stage.eq(stage)].iloc[0]
                for key, value in score(q, "selected_prediction", total).items():
                    assert_close(value, row[key])
                assert q.weather_grid_id.nunique() == row.weather_grids
                for scope, part in [("all", total), ("frozen_rule", q)]:
                    for model in [lock["model"]]+BASELINES:
                        prediction = "selected_prediction" if model == lock["model"] else model
                        mr = metrics.loc[metrics.seed.eq(seed) & metrics.window_months.eq(window) & metrics.stage.eq(stage) & metrics.scope.eq(scope) & metrics.model.eq(model)].iloc[0]
                        for key, value in score(part, prediction, total).items():
                            assert_close(value, mr[key])
                        metric_checks += 1
                errors = {k: score(q, k, total)["rmse"] for k in BASELINES}
                for k, value in errors.items():
                    assert_close(value, row[k+"_rmse"])
                gain = 1-row.rmse/min(errors.values()) if len(q) else np.nan
                assert_close(gain, row.gain_vs_all_comparators)
                sufficient = row.coverage >= .25 and row.fua_coverage >= .25 and row.fuas >= 5 and row.months >= 12
                uncertainty = all(row[c+"_"+k+"_delta_low"] > 0 for c in ["fua", "grid"] for k in ["frozen", "mean", "shuffled"])
                passed = lock["discovery_pass"] and sufficient and gain >= .05 and uncertainty
                assert bool(row.stage_pass) == passed
                failures.append({**{k: row[k] for k in KEY}, "stage": stage,
                    "discovery_failed": not lock["discovery_pass"], "support_failed": not sufficient,
                    "gain_failed": not (gain >= .05), "uncertainty_failed": not uncertainty})
                stage_checks += 1
                if row.direction == "CHN_to_AUS" and seed == 5993 and window in [1, 6] and stage in ["confirmation", "spatial_temporal_audit"]:
                    for cluster, prefix in [("eFUA_ID", "fua"), ("weather_grid_id", "grid")]:
                        ci = bootstrap_direct(q, seed, cluster)
                        if ci is not None:
                            assert_close(ci[0], row[prefix+"_mean_delta_low"])
                            assert_close(ci[1], row[prefix+"_mean_delta_high"])
                            ci_checks += 1
    g, m = pd.concat(all_gates, ignore_index=True), pd.concat(all_metrics, ignore_index=True)
    g.to_csv(OUT/"all_gates.csv", index=False)
    m.to_csv(OUT/"all_model_metrics.csv", index=False)
    pd.concat(budgets).to_csv(OUT/"all_label_budgets.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT/"failure_components.csv", index=False)
    p = g.pivot(index=KEY, columns="stage", values="stage_pass")
    p["confirmation_and_new_cities_pass"] = p.confirmation & p.spatial_temporal_audit
    p.to_csv(OUT/"replication_by_configuration.csv")
    g.groupby(["feature_set", "direction", "stage"]).agg(configurations=("stage_pass", "size"), passed=("stage_pass", "sum"), discovery_qualified=("discovery_pass", "sum")).to_csv(OUT/"replication_counts.csv")
    q = m.loc[m.stage.eq("spatial_temporal_audit") & m.scope.eq("all")].copy()
    means = q.loc[q.model.eq("target_mean"), KEY+["rmse"]].rename(columns={"rmse": "mean_rmse"})
    q = q.merge(means, on=KEY, validate="many_to_one")
    q["gain_vs_mean"] = 1-q.rmse/q.mean_rmse
    q.groupby(["feature_set", "direction", "window_months", "model"]).agg(
        median_rmse=("rmse", "median"), min_rmse=("rmse", "min"), max_rmse=("rmse", "max"),
        median_r2=("r2", "median"), median_gain_vs_mean=("gain_vs_mean", "median"),
        min_gain_vs_mean=("gain_vs_mean", "min"), max_gain_vs_mean=("gain_vs_mean", "max")
    ).to_csv(OUT/"full_coverage_comparison.csv")
    verification = {"status": "passed", "configurations": len(p), "prediction_metric_records_recomputed": metric_checks,
                    "gate_stage_records_recomputed": stage_checks, "direct_bootstrap_intervals_recomputed": ci_checks,
                    "confirmation_and_new_cities_pass": int(p.confirmation_and_new_cities_pass.sum()),
                    "scope": "Selected prediction and all five saved comparators; other unselected candidate prediction arrays were not retained and are not independently recomputed here.",
                    "script_sha256": sha(__file__)}
    dump(OUT/"model_verification.json", verification)
    print(json.dumps(verification, indent=2), flush=True)


def regulatory_audit():
    monitor = pd.read_csv(PROCESSED/"regulatory_monthly_audit_input.csv.gz")
    eligible = monitor.loc[monitor.audit_cohort.ne("below_completeness") & monitor.monitor_pm25_ug_m3.notna()].copy()
    grouped = eligible.groupby(["audit_cohort", "eFUA_ID", "year", "month"], as_index=False).agg(
        monitor_mean=("monitor_pm25_ug_m3", "mean"), stations=("site_key", "nunique"),
        station_min=("monitor_pm25_ug_m3", "min"), station_max=("monitor_pm25_ug_m3", "max"))
    grouped.to_csv(OUT/"regulatory_fua_month_targets.csv", index=False)
    matches, scores, manifests = [], [], []
    for file in sorted((HERE/"results").glob("*/CHN_to_AUS/predictions_*.csv.gz")):
        _, seed, window = file.name.split(".")[0].split("_")
        setting = dict(feature_set=file.parent.parent.name, direction="CHN_to_AUS", seed=int(seed), window_months=int(window))
        p = pd.read_csv(file, float_precision="round_trip")
        p = p.loc[~p.development_city & p.year.between(2023, 2024)].copy()
        paired = p.merge(grouped, on=["eFUA_ID", "year", "month"], how="inner", validate="one_to_many")
        for key, value in setting.items(): paired[key] = value
        matches.append(paired)
        manifests.append({"file": str(file), "sha256": sha(file)})
        for cohort in ["recomputable_75pct", "official_monthly_no_day_counts"]:
            full = paired.loc[paired.audit_cohort.eq(cohort)]
            for scope in ["all_available", "frozen_rule"]:
                q = full if scope == "all_available" else full.loc[full.accepted]
                for model in ["selected_prediction"]+BASELINES+[Y]:
                    scores.append({**setting, "audit_cohort": cohort, "scope": scope, "prediction": model,
                        **score(q, model, full, target="monitor_mean"),
                        "all_target_rows": len(p), "monitor_available_rows": len(full),
                        "selected_over_all_target_rows": len(q)/len(p),
                        "interpretation": "product_agreement_not_predictive_baseline" if model == Y else "fitted_on_satpm_not_monitors"})
    s = pd.DataFrame(scores)
    s.to_csv(OUT/"regulatory_frozen_prediction_metrics.csv", index=False)
    pd.concat(matches).to_csv(OUT/"regulatory_prediction_matches.csv.gz", index=False, float_format="%.17g")
    s.groupby(["feature_set", "audit_cohort", "scope", "prediction"]).agg(
        configurations=("rmse", "size"), median_rmse=("rmse", "median"), min_rmse=("rmse", "min"), max_rmse=("rmse", "max"),
        median_r2=("r2", "median"), median_bias=("bias", "median"), min_rows=("n", "min"), max_rows=("n", "max"),
        min_fuas=("fuas", "min"), max_fuas=("fuas", "max")
    ).to_csv(OUT/"regulatory_summary.csv")
    dump(OUT/"regulatory_audit_manifest.json", {"status": "complete", "configurations": len(manifests),
        "metric_rows": len(s), "model_refits_on_monitor_labels": 0, "rule_reselections_on_monitor_labels": 0,
        "protocol_sha256": sha(HERE/"regulatory_audit_protocol.json"), "script_sha256": sha(__file__),
        "input_sha256": sha(PROCESSED/"regulatory_monthly_audit_input.csv.gz"), "prediction_files": manifests,
        "limitations": "Exploratory station-to-area label audit; NSW lacks day counts; upstream SatPM monitor overlap unknown. Cohort coverage uses matched monitor-available rows; also report denominator including every held target row."})
    print("REGULATORY_AUDIT_COMPLETE", len(s), flush=True)


if __name__ == "__main__":
    verify_models()
    regulatory_audit()
