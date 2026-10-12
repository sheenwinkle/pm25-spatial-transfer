"""Known-effect synthetic recovery test for the frozen transfer pipeline."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import run_positive_control as pc

HERE = Path(__file__).resolve().parent
DATA = HERE / "synthetic_data"
AUDIT = HERE / "synthetic_audit"
RESULTS = HERE / "synthetic_results"
for folder in (DATA, AUDIT, RESULTS):
    folder.mkdir(parents=True, exist_ok=True)


def load_domains_without_writing_control_receipt():
    all_path = pc.BOUNDARY / "data/all_networks_panel.csv.gz"
    membership_path = pc.BOUNDARY / "data/membership_all_panel.csv.gz"
    panel = pd.read_csv(all_path)
    membership = pd.read_csv(membership_path)
    yrd_ids = set(membership.loc[membership.country_iso.eq("CHN"), "eFUA_ID"])
    target = panel[panel.country_iso.eq("CHN") & panel.eFUA_ID.isin(yrd_ids)].copy()
    source = panel[
        panel.country_iso.eq("CHN")
        & ~panel.eFUA_ID.isin(yrd_ids)
        & panel.year.le(2018)
    ].copy()
    features = json.loads(
        (pc.OLD / "audit/build_summary.json").read_text(encoding="utf-8")
    )["features"]
    assert set(source.eFUA_ID).isdisjoint(set(target.eFUA_ID))
    return source.reset_index(drop=True), target.reset_index(drop=True), features, all_path, membership_path


def generate_target(source, target, features, all_path, membership_path):
    generator = pc.core.tree(True).fit(source[features], source[pc.Y])
    signal = pc.core.pos(generator.predict(target[features]))
    noise_sd = float(0.25 * np.std(signal, ddof=1))
    noise = np.random.default_rng(2701).normal(0, noise_sd, len(target))
    target[pc.Y] = pc.core.pos(signal + noise)
    labels = target[["eFUA_ID", "year", "month", "t"]].copy()
    labels["source_signal"] = signal
    labels["synthetic_noise"] = noise
    labels["synthetic_pm25"] = target[pc.Y].to_numpy()
    label_path = DATA / "synthetic_target_labels.csv.gz"
    labels.to_csv(label_path, index=False)
    pc.dump(AUDIT / "generation_receipt.json", {
        "protocol_sha256": pc.digest(HERE / "SYNTHETIC_PROTOCOL.md"),
        "script_sha256": pc.digest(__file__),
        "all_network_panel_sha256": pc.digest(all_path),
        "membership_panel_sha256": pc.digest(membership_path),
        "label_sha256": pc.digest(label_path),
        "generator_seed": 5993,
        "noise_seed": 2701,
        "noise_sd_fraction": 0.25,
        "signal_sd": float(np.std(signal, ddof=1)),
        "noise_sd": noise_sd,
        "target_rows": len(target),
    })
    return target


def run():
    source, target, features, all_path, membership_path = load_domains_without_writing_control_receipt()
    target = generate_target(source, target, features, all_path, membership_path)
    static_models, static_embeddings, target = pc.fit_static_models(source, target, features)
    eligible_counts = target[target.year.eq(2018)].groupby("eFUA_ID").size()
    eligible = np.array(sorted(eligible_counts[eligible_counts >= 6].index))
    assert len(eligible) == 39
    all_gates = []
    all_metrics = []

    for seed in pc.SEEDS:
        ids = eligible.copy()
        np.random.default_rng(seed).shuffle(ids)
        ids = ids[: min(10, int(np.ceil(len(ids) / 2)))]
        dev = target.eFUA_ID.isin(ids).to_numpy()
        dev_grids = set(target.loc[dev, "weather_grid_id"])
        unlabelled_target = target[dev & target.year.le(2018).to_numpy()]
        normal = pc.core.norm()
        domain_x = normal.fit_transform(
            pd.concat([source[features], unlabelled_target[features]], ignore_index=True)
        )
        domain_y = np.r_[np.zeros(len(source)), np.ones(len(unlabelled_target))]
        domain_model = LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=1000
        ).fit(domain_x, domain_y)
        probability = domain_model.predict_proba(domain_x[: len(source)])[:, 1]
        weights = np.clip(probability / np.maximum(1 - probability, 1e-9), 0.1, 10)
        weighted = pc.core.tree(True).fit(
            source[features], source[pc.Y], sample_weight=weights
        )
        embeddings = dict(static_embeddings)
        embeddings["weighted_source_kernel"] = pc.core.embed(weighted, target[features])
        models = dict(static_models)
        models[f"weighted_{seed}"] = weighted
        seed_folder = RESULTS / str(seed)
        seed_folder.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": models, "features": features}, seed_folder / "source_models.joblib", compress=3)

        for window in pc.WINDOWS:
            folder = seed_folder / str(window)
            folder.mkdir(parents=True, exist_ok=True)
            predictions, used, fit_audit = pc.target_predictions(
                target, source, features, dev, window, embeddings
            )
            predictions, lock, search, gates, metrics = pc.evaluate_configuration(
                predictions, seed, window, dev, dev_grids
            )
            for row in gates:
                row["control"] = "synthetic_known_effect"
            for row in metrics:
                row["control"] = "synthetic_known_effect"
            keep = list(dict.fromkeys([
                "eFUA_ID", "fua_name", "year", "month", "t", "quarter_id",
                "weather_grid_id", pc.Y, "development_city", "new_weather_grid",
                "fit_labels", "accepted", "selected_prediction",
                *pc.core.GATES, *pc.core.MODELS,
            ]))
            predictions[keep].to_csv(folder / "predictions.csv.gz", index=False)
            pd.DataFrame(search).assign(rule=lambda frame: frame.rule.map(json.dumps)).to_csv(
                folder / "search.csv.gz", index=False
            )
            pd.DataFrame(gates).to_csv(folder / "gates.csv", index=False)
            pd.DataFrame(metrics).to_csv(folder / "metrics.csv.gz", index=False)
            pd.DataFrame(fit_audit).to_csv(folder / "fit_audit.csv", index=False)
            pc.dump(folder / "lock.json", lock)
            pc.dump(folder / "complete.json", {
                "completed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "seed": seed,
                "window_months": window,
                "development_fuas": ids.tolist(),
                "unique_target_calibration_labels": int(used.sum()),
                "protocol_sha256": pc.digest(HERE / "SYNTHETIC_PROTOCOL.md"),
                "script_sha256": pc.digest(__file__),
                "label_sha256": pc.digest(DATA / "synthetic_target_labels.csv.gz"),
                "predictions_sha256": pc.digest(folder / "predictions.csv.gz"),
            })
            all_gates.extend(gates)
            all_metrics.extend(metrics)
            print(
                "SYNTHETIC_RECOVERY",
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
    pc.dump(AUDIT / "completion.json", {
        "status": "complete",
        "configurations": len(summary),
        "discovery_passes": int(summary.discovery_pass.sum()),
        "temporal_passes": int(summary.temporal_pass.sum()),
        "new_city_passes": int(summary.new_city_pass.sum()),
        "joint_passes": int(summary.joint_pass.sum()),
        "protocol_sha256": pc.digest(HERE / "SYNTHETIC_PROTOCOL.md"),
        "script_sha256": pc.digest(__file__),
        "label_sha256": pc.digest(DATA / "synthetic_target_labels.csv.gz"),
    })
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
