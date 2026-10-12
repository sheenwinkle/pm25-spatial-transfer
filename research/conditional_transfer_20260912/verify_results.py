"""Recompute reported selected-rule metrics and predicates from saved predictions."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

import run_study as base

HERE=Path(__file__).resolve().parent
FRAME,_,_=joblib.load(HERE/"fua_month_cache.joblib")
CSV_DTYPES={c:str(FRAME[c].dtype) for c in [base.Y]+base.P["gate_features"] if c in FRAME.columns}


def restore_precision(frame):
    # CSV has no dtype metadata; restore original float32 rounding at exact gate boundaries.
    return frame.astype({c:v for c,v in CSV_DTYPES.items() if c in frame.columns})


def compare(frame, pred, reference, accepted, row):
    actual=base.statistics(frame.loc[accepted],pred[accepted],frame)
    other=base.statistics(frame.loc[accepted],reference[accepted],frame)
    for name in ["n","fuas","months","coverage","fua_coverage","rmse","mae","r2","bias","fua_macro_rmse"]:
        assert np.isclose(actual[name],row[name],atol=2e-5,rtol=2e-5,equal_nan=True),(name,actual[name],row[name])
    assert np.isclose(other["rmse"],row.baseline_rmse,atol=2e-5,rtol=2e-5)
    recomputed=base.apply_rule(frame,json.loads(row.rule))
    assert np.array_equal(recomputed,accepted)


def main():
    checks=0
    for direction in ["CHN_to_AUS","AUS_to_CHN"]:
        gates=pd.read_csv(base.OUT/f"gates_{direction}.csv")
        gates=gates[gates.gate_kind.eq("target_developed_rule")]
        for seed in base.P["seeds"]:
            t=restore_precision(pd.read_csv(base.OUT/f"predictions_{direction}_{seed}.csv.gz",float_precision="round_trip"))
            t["t"]=t.year*12+t.month
            dev=t.development_city.to_numpy()
            stages={"discovery":dev&t.year.between(2019,2020).to_numpy(),
                    "confirmation":dev&t.year.between(2021,2022).to_numpy(),
                    "spatial_temporal_audit":~dev&t.year.between(2023,2024).to_numpy(),
                    "seen_city_temporal_audit":dev&t.year.between(2023,2024).to_numpy()}
            for _,row in gates[gates.seed==seed].iterrows():
                budget=int(row.budget_months)
                frame=t.loc[stages[row.stage]].copy()
                compare(frame,frame[f"b{budget}_selected_prediction"].to_numpy(),
                        frame[f"b{budget}_baseline_prediction"].to_numpy(),
                        frame[f"b{budget}_accepted"].to_numpy(),row)
                checks+=1
    details=restore_precision(pd.read_csv(HERE/"sequential_results/selected_predictions.csv.gz",float_precision="round_trip"))
    details["t"]=details.year*12+details.month
    gates=pd.read_csv(HERE/"sequential_results/gates.csv")
    for keys,frame in details.groupby(["direction","seed","window_months","stage"]):
        direction,seed,window,stage=keys
        row=gates[(gates.direction==direction)&(gates.seed==seed)&(gates.window_months==window)&gates.stage.eq(stage)].iloc[0]
        assert not frame.duplicated(["eFUA_ID","year","month"]).any()
        compare(frame,frame.prediction.to_numpy(),frame.baseline_prediction.to_numpy(),frame.accepted.to_numpy(),row)
        checks+=1
    for directory,script,protocol in [(base.OUT,"run_study.py","protocol.json"),
                                       (HERE/"sequential_results","run_sequential.py","sequential_protocol.json")]:
        m=json.loads((directory/"run_manifest.json").read_text(encoding="utf-8"))
        assert m["status"]=="completed"
        assert m["script_sha256"]==base.sha(HERE/script)
        assert m["protocol_sha256"]==base.sha(HERE/protocol)
    summary={"status":"passed","selected_rule_stage_metric_checks":checks,
             "predicate_checks":checks,"duplicate_evaluation_key_failures":0,
             "protocol_and_script_hashes":"matched",
             "note":"Fitting-time and held-city isolation assertions were also executed in the sequential training run."}
    base.dump(HERE/"review/verification.json",summary)
    base.dump(HERE/"review/prediction_csv_schema.json",{
        "restored_numeric_dtypes":CSV_DTYPES,"other_numeric_columns":"float64 unless integer identifiers",
        "read_instruction":"Use pandas read_csv(float_precision='round_trip') and restore the specified original numeric dtypes before applying exact floating-point gate thresholds."})
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
