"""Frozen within-China positive control for the observed-label pipeline."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import KernelCenterer, StandardScaler

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "thesis_completion_20260912"
BOUNDARY = HERE.parent / "boundary_regression_20261001"
sys.path.insert(0, str(OLD))

import run_monitor_models as core
from common import digest

DATA = HERE / "data"
AUDIT = HERE / "audit"
RESULTS = HERE / "results"
for folder in (DATA, AUDIT, RESULTS):
    folder.mkdir(parents=True, exist_ok=True)

SEEDS = (5993, 5994, 5995)
WINDOWS = (3, 6, 12)
Y = core.Y


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def positive_kernel(kk, kt, y):
    center = KernelCenterer()
    centered = center.fit_transform(kk)
    model = KernelRidge(alpha=1, kernel="precomputed").fit(centered, y - y.mean())
    return core.pos(y.mean() + model.predict(center.transform(kt)))


def load_domains():
    all_path = BOUNDARY / "data/all_networks_panel.csv.gz"
    membership_path = BOUNDARY / "data/membership_all_panel.csv.gz"
    panel = pd.read_csv(all_path)
    membership = pd.read_csv(membership_path)
    yrd_ids = set(membership.loc[membership.country_iso.eq("CHN"), "eFUA_ID"])
    target = panel[panel.country_iso.eq("CHN") & panel.eFUA_ID.isin(yrd_ids)].copy()
    source = panel[
        panel.country_iso.eq("CHN")
        & ~panel.eFUA_ID.isin(yrd_ids)
        & panel.year.le(2018)
    ].copy()
    assert set(source.eFUA_ID).isdisjoint(set(target.eFUA_ID))
    assert len(yrd_ids) == 43
    features = json.loads((OLD / "audit/build_summary.json").read_text(encoding="utf-8"))["features"]
    raw_gates = [name for name in core.GATES if name != "support_distance"]
    required = set(features + raw_gates + [Y, "eFUA_ID", "year", "month", "t",
                                           "quarter_id", "weather_grid_id"])
    assert required.issubset(panel.columns), sorted(required - set(panel.columns))
    receipt = {
        "all_network_panel_sha256": digest(all_path),
        "membership_panel_sha256": digest(membership_path),
        "protocol_sha256": digest(HERE / "PROTOCOL.md"),
        "source_rows": len(source),
        "source_fuas": int(source.eFUA_ID.nunique()),
        "target_rows": len(target),
        "target_fuas": int(target.eFUA_ID.nunique()),
        "source_target_overlap": 0,
        "features": features,
    }
    dump(AUDIT / "input_receipt.json", receipt)
    return source.reset_index(drop=True), target.reset_index(drop=True), features


def fit_static_models(source, target, features):
    source_model = core.tree(True).fit(source[features], source[Y])
    models = {"source": source_model}
    embeddings = {"source_kernel": core.embed(source_model, target[features])}
    target["source_only"] = core.pos(source_model.predict(target[features]))
    for seed in SEEDS:
        shuffled = core.tree(True).fit(
            source[features], np.random.default_rng(seed).permutation(source[Y].to_numpy())
        )
        models[f"shuffle_{seed}"] = shuffled
        embeddings[f"shuffled_kernel_{seed}"] = core.embed(shuffled, target[features])
    support_scaler = core.norm()
    source_x = support_scaler.fit_transform(source[features[:6]])
    target["support_distance"] = cKDTree(source_x).query(
        support_scaler.transform(target[features[:6]]), workers=4
    )[0]
    return models, embeddings, target


def target_predictions(target, source, features, dev, window, embeddings):
    predictions = target.copy()
    for model in core.MODELS:
        if model != "source_only":
            predictions[model] = np.nan
    predictions["fit_labels"] = 0
    used = np.zeros(len(target), dtype=bool)
    fit_audit = []
    for month in sorted(target.loc[target.year.ge(2019), "t"].unique()):
        fit = dev & target.t.between(month - window, month - 1).to_numpy()
        evaluate = target.t.eq(month).to_numpy()
        if fit.sum() < 5:
            continue
        assert not (fit & evaluate).any()
        assert not (fit & ~dev).any()
        assert target.loc[fit, "t"].max() < month
        used |= fit
        x = target.loc[fit, features]
        ex = target.loc[evaluate, features]
        y = target.loc[fit, Y].to_numpy()
        predictions.loc[evaluate, "fit_labels"] = int(fit.sum())
        fit_audit.append({
            "prediction_t": int(month),
            "latest_fit_t": int(target.loc[fit, "t"].max()),
            "labels": int(fit.sum()),
            "fit_fuas": int(target.loc[fit, "eFUA_ID"].nunique()),
        })

        normal = core.norm()
        xx = normal.fit_transform(x)
        ee = normal.transform(ex)
        predictions.loc[evaluate, "target_mean"] = y.mean()
        predictions.loc[evaluate, "target_ridge"] = core.pos(
            Ridge(alpha=10).fit(xx, y).predict(ee)
        )
        predictions.loc[evaluate, "target_rbf"] = positive_kernel(
            rbf_kernel(xx), rbf_kernel(ee, xx), y
        )
        for name, estimator in (
            ("target_rf", RandomForestRegressor),
            ("target_extra", ExtraTreesRegressor),
        ):
            fitted = estimator(
                n_estimators=80,
                min_samples_leaf=3,
                max_features=1.0,
                n_jobs=4,
                random_state=5993,
            ).fit(xx, y)
            predictions.loc[evaluate, name] = core.pos(fitted.predict(ee))
        predictions.loc[evaluate, "target_lgbm"] = core.pos(
            core.tree().fit(x, y).predict(ex)
        )

        raw = target.loc[fit, "source_only"].to_numpy()
        raw_eval = target.loc[evaluate, "source_only"].to_numpy()
        raw_scaler = StandardScaler()
        raw_x = raw_scaler.fit_transform(raw[:, None])
        raw_eval_x = raw_scaler.transform(raw_eval[:, None])
        predictions.loc[evaluate, "source_affine"] = core.pos(
            Ridge(alpha=10).fit(raw_x, y).predict(raw_eval_x)
        )
        predictions.loc[evaluate, "source_residual"] = core.pos(
            raw_eval + core.tree().fit(x, y - raw).predict(ex)
        )
        for name, embedding in embeddings.items():
            train_embedding = embedding[fit]
            eval_embedding = embedding[evaluate]
            predictions.loc[evaluate, name] = positive_kernel(
                (train_embedding @ train_embedding.T).toarray(),
                (eval_embedding @ train_embedding.T).toarray(),
                y,
            )
    return predictions, used, fit_audit


def evaluate_configuration(predictions, seed, window, dev, dev_grids):
    predictions["development_city"] = dev
    predictions["new_weather_grid"] = ~predictions.weather_grid_id.isin(dev_grids)
    predictions = predictions[predictions.year.ge(2019)].copy()
    complete = predictions[core.MODELS].notna().all(axis=1)
    predictions = predictions[complete].reset_index(drop=True)
    discovery = predictions[predictions.development_city & predictions.year.between(2019, 2020)]
    lock, search = core.discover(discovery)
    predictions["accepted"] = core.maskrule(predictions, lock["rule"])
    predictions["selected_prediction"] = predictions[lock["model"]]
    setting = {"control": "nonYRD_to_YRD", "seed": seed, "window_months": window}
    stages = {
        "discovery": predictions.development_city & predictions.year.le(2020),
        "temporal_audit": predictions.development_city & predictions.year.ge(2021),
        "new_city_audit": ~predictions.development_city & predictions.year.ge(2021),
        "new_weather_grid_audit": (
            ~predictions.development_city
            & predictions.new_weather_grid
            & predictions.year.ge(2021)
        ),
    }
    gates = []
    metrics = []
    for stage, mask in stages.items():
        full = predictions[mask]
        selected = full[full.accepted]
        for selection, rows in (("full", full), ("accepted", selected)):
            for model in core.MODELS:
                metrics.append({
                    **setting,
                    "stage": stage,
                    "selection": selection,
                    "model": model,
                    **core.metrics(rows, rows[model]),
                })
        if len(selected):
            relative_gain, best, _ = core.gain(full, lock["model"], full.accepted.to_numpy())
            references = list(dict.fromkeys([lock["reference"], "target_mean"] + core.NEG))
            intervals = {}
            for cluster in ("eFUA_ID", "weather_grid_id"):
                for reference in references:
                    low, high = core.bootstrap(selected, lock["model"], reference, seed, cluster)
                    intervals[f"{cluster}_{reference}_low"] = low
                    intervals[f"{cluster}_{reference}_high"] = high
            lower = [value for key, value in intervals.items() if key.endswith("_low")]
            bootstrap_pass = bool(lower and all(value is not None and value > 0 for value in lower))
            support_pass = core.supported(full, full.accepted.to_numpy())
            gates.append({
                **setting,
                "stage": stage,
                "selected_model": lock["model"],
                "rule": json.dumps(lock["rule"]),
                "discovery_pass": bool(lock["discovery_pass"]),
                "coverage": len(selected) / len(full),
                "relative_gain": relative_gain,
                "best_comparator": best,
                "support_pass": bool(support_pass),
                "bootstrap_pass": bootstrap_pass,
                "pass": bool(
                    lock["discovery_pass"]
                    and relative_gain >= 0.05
                    and support_pass
                    and bootstrap_pass
                ),
                **core.metrics(selected, selected[lock["model"]]),
                **intervals,
            })
    return predictions, lock, search, gates, metrics


def run():
    source, target, features = load_domains()
    static_models, static_embeddings, target = fit_static_models(source, target, features)
    eligible_counts = target[target.year.eq(2018)].groupby("eFUA_ID").size()
    eligible = np.array(sorted(eligible_counts[eligible_counts >= 6].index))
    assert len(eligible) == 39
    all_gates = []
    all_metrics = []
    splits = []
    for seed in SEEDS:
        ids = eligible.copy()
        np.random.default_rng(seed).shuffle(ids)
        ids = ids[: min(10, int(np.ceil(len(ids) / 2)))]
        dev = target.eFUA_ID.isin(ids).to_numpy()
        dev_grids = set(target.loc[dev, "weather_grid_id"])
        for city in sorted(target.eFUA_ID.unique()):
            splits.append({"seed": seed, "eFUA_ID": city, "development": city in ids})

        unlabelled_target = target[dev & target.year.le(2018).to_numpy()]
        normal = core.norm()
        domain_x = normal.fit_transform(
            pd.concat([source[features], unlabelled_target[features]], ignore_index=True)
        )
        domain_y = np.r_[np.zeros(len(source)), np.ones(len(unlabelled_target))]
        domain_model = LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=1000
        ).fit(domain_x, domain_y)
        probability = domain_model.predict_proba(domain_x[: len(source)])[:, 1]
        weights = np.clip(probability / np.maximum(1 - probability, 1e-9), 0.1, 10)
        weighted = core.tree(True).fit(source[features], source[Y], sample_weight=weights)
        embeddings = dict(static_embeddings)
        embeddings["weighted_source_kernel"] = core.embed(weighted, target[features])
        models = dict(static_models)
        models[f"weighted_{seed}"] = weighted
        seed_folder = RESULTS / str(seed)
        seed_folder.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": models, "features": features}, seed_folder / "source_models.joblib", compress=3)
        dump(seed_folder / "weighting.json", {
            "source_rows": len(source),
            "unlabelled_target_rows": len(unlabelled_target),
            "development_fuas": ids.tolist(),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "effective_source_rows": float(weights.sum() ** 2 / (weights ** 2).sum()),
        })

        for window in WINDOWS:
            folder = seed_folder / str(window)
            folder.mkdir(parents=True, exist_ok=True)
            complete_path = folder / "complete.json"
            if complete_path.exists():
                all_gates.extend(pd.read_csv(folder / "gates.csv").to_dict("records"))
                all_metrics.extend(pd.read_csv(folder / "metrics.csv.gz").to_dict("records"))
                print("CACHED", seed, window, flush=True)
                continue
            predictions, used, fit_audit = target_predictions(
                target, source, features, dev, window, embeddings
            )
            predictions, lock, search, gates, metrics = evaluate_configuration(
                predictions, seed, window, dev, dev_grids
            )
            keep = list(dict.fromkeys([
                "eFUA_ID", "fua_name", "year", "month", "t", "quarter_id",
                "weather_grid_id", Y, "station_count", "development_city",
                "new_weather_grid", "fit_labels", "accepted", "selected_prediction",
                *core.GATES, *core.MODELS,
            ]))
            predictions[keep].to_csv(folder / "predictions.csv.gz", index=False)
            pd.DataFrame(search).assign(rule=lambda frame: frame.rule.map(json.dumps)).to_csv(
                folder / "search.csv.gz", index=False
            )
            pd.DataFrame(gates).to_csv(folder / "gates.csv", index=False)
            pd.DataFrame(metrics).to_csv(folder / "metrics.csv.gz", index=False)
            pd.DataFrame(fit_audit).to_csv(folder / "fit_audit.csv", index=False)
            dump(folder / "lock.json", lock)
            dump(folder / "complete.json", {
                "completed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "seed": seed,
                "window_months": window,
                "development_fuas": ids.tolist(),
                "unique_target_calibration_labels": int(used.sum()),
                "protocol_sha256": digest(HERE / "PROTOCOL.md"),
                "script_sha256": digest(__file__),
                "predictions_sha256": digest(folder / "predictions.csv.gz"),
            })
            all_gates.extend(gates)
            all_metrics.extend(metrics)
            print(
                "POSITIVE_CONTROL",
                seed,
                window,
                "discovery_pass",
                lock["discovery_pass"],
                "discovery_gain",
                round(lock["gain"], 4),
                flush=True,
            )

    gates = pd.DataFrame(all_gates)
    metrics = pd.DataFrame(all_metrics)
    gates.to_csv(AUDIT / "all_gates.csv", index=False)
    metrics.to_csv(AUDIT / "all_metrics.csv.gz", index=False)
    pd.DataFrame(splits).to_csv(AUDIT / "city_splits.csv", index=False)
    joint = []
    for (seed, window), group in gates.groupby(["seed", "window_months"]):
        lookup = group.set_index("stage")["pass"].to_dict()
        joint.append({
            "seed": int(seed),
            "window_months": int(window),
            "discovery_pass": bool(group.discovery_pass.iloc[0]),
            "temporal_pass": bool(lookup.get("temporal_audit", False)),
            "new_city_pass": bool(lookup.get("new_city_audit", False)),
            "joint_pass": bool(
                lookup.get("temporal_audit", False) and lookup.get("new_city_audit", False)
            ),
        })
    summary = pd.DataFrame(joint).sort_values(["seed", "window_months"])
    summary.to_csv(AUDIT / "summary.csv", index=False)
    dump(AUDIT / "completion.json", {
        "status": "complete",
        "configurations": len(summary),
        "discovery_passes": int(summary.discovery_pass.sum()),
        "temporal_passes": int(summary.temporal_pass.sum()),
        "new_city_passes": int(summary.new_city_pass.sum()),
        "joint_passes": int(summary.joint_pass.sum()),
        "protocol_sha256": digest(HERE / "PROTOCOL.md"),
        "script_sha256": digest(__file__),
    })
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
