"""Independently trace recovered monthly values and compare them with station summaries."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from audit_inputs import HERE, ROOT, PREVIOUS
from collect_monthly_power import VARIABLES, META, PROCESSED, RAW_NEW, dump, sha


def main():
    out=HERE/"audit"
    manifest=json.loads((META/"data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["script_sha256"]==sha(HERE/"collect_monthly_power.py")
    assert manifest["fua_monthly_sha256"]==sha(PROCESSED/"fua_monthly_power_1998_2025.csv.gz")
    oldmanifest=json.loads((META/"inherited_raw_files.json").read_text(encoding="utf-8"))
    newmanifest=json.loads((META/"download_manifest.json").read_text(encoding="utf-8"))
    for entry in oldmanifest+newmanifest:assert sha(entry["path"])==entry["sha256"]
    weather=pd.read_csv(PROCESSED/"fua_monthly_power_1998_2025.csv.gz",float_precision="round_trip")
    assert len(weather)==523824 and weather.groupby("eFUA_ID").size().eq(336).all()
    assert weather[list(VARIABLES.values())].notna().all().all()
    grid=pd.read_csv(PROCESSED/"power_grid_month_1998_2025.csv.gz",float_precision="round_trip")
    check=weather.merge(grid,on=["power_lon","power_lat","year","month"],validate="many_to_one",suffixes=("","_grid"))
    for c in VARIABLES.values():assert np.array_equal(check[c],check[c+"_grid"])
    lookup=grid.set_index(["power_lon","power_lat","year","month"])
    count=0
    for entries,years in [(oldmanifest,[1998,2000,2018,2022]),(newmanifest,[2023,2024,2025])]:
        for entry in entries:
            raw=json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
            parameter=entry["parameter"]
            for f in raw["features"]:
                lon,lat=f["geometry"]["coordinates"][:2]
                if (lon,lat,years[0],1) not in lookup.index:continue
                for year in years:
                    for month in [1,6,12]:
                        direct=f["properties"]["parameter"][parameter][f"{year}{month:02d}"]
                        value=lookup.loc[(lon,lat,year,month),VARIABLES[parameter]]
                        assert direct==value,(entry["path"],lon,lat,year,month)
                        count+=1
                break
    before,_,_=joblib.load(PREVIOUS/"fua_month_cache.joblib")
    joined=before.merge(weather,on=["eFUA_ID","year","month"],validate="one_to_one",suffixes=("","_new"))
    comparisons=[]
    pairs=[("temperature",VARIABLES["T2M"],"surface_temperature_mean_c"),
           ("humidity",VARIABLES["RH2M"],"surface_relative_humidity_mean_pct"),
           ("wind",VARIABLES["WS10M"],"surface_wind_mean_mps")]
    for country,p in joined.groupby("country_iso"):
        for variable,new,obs in pairs:
            q=p.loc[p[obs].notna()&p[new].notna()]
            delta=q[new]-q[obs]
            comparisons.append({"country":country,"variable":variable,"paired_fua_months":len(q),
                "rmse_between_sources":float(np.sqrt(np.mean(delta**2))),"bias_reanalysis_minus_station":float(delta.mean()),
                "correlation":float(q[new].corr(q[obs])),"note":"Cross-source agreement, not independent validation of MERRA-2; observations may enter assimilation"})
    pd.DataFrame(comparisons).to_csv(out/"reanalysis_aviation_agreement.csv",index=False)
    regulatory=ROOT/"data/processed/modelling_panel/australia_multistate_regulatory_pm25_monthly_2015_2024.csv.gz"
    cols=["site_key","station_name","network_state","eFUA_ID","fua_name","year","month","monitor_pm25_ug_m3",
          "strict_monthly_valid","valid_days","calendar_days","monthly_coverage_fraction"]
    monitor=pd.read_csv(regulatory,usecols=cols)
    assert not monitor.duplicated(["site_key","year","month"]).any()
    monitor["completeness_recomputable"]=monitor.valid_days.notna()&monitor.calendar_days.gt(0)
    monitor["verified_day_fraction"]=monitor.valid_days/monitor.calendar_days
    strict=monitor.completeness_recomputable&monitor.verified_day_fraction.ge(.75)
    monitor["audit_cohort"]=np.where(strict,"recomputable_75pct",np.where(monitor.network_state.eq("NSW"),"official_monthly_no_day_counts","below_completeness"))
    q=monitor.groupby(["network_state","audit_cohort"]).agg(station_months=("site_key","size"),stations=("site_key","nunique"),fuas=("eFUA_ID","nunique")).reset_index()
    q.to_csv(out/"regulatory_quality_cohorts.csv",index=False)
    monitor.to_csv(PROCESSED/"regulatory_monthly_audit_input.csv.gz",index=False)
    result={"status":"passed","raw_hash_checks":len(oldmanifest)+len(newmanifest),
        "direct_raw_month_value_checks":count,"fua_monthly_rows":len(weather),"six_field_grid_to_fua_checks":len(weather)*6,
        "pm25_label_rows_unchanged":len(before),"new_weather_2025_rows":int(weather.year.eq(2025).sum()),
        "regulatory_input_sha256":sha(regulatory),"regulatory_quality_cohorts":q.to_dict("records"),
        "observation_note":"Official NSW monthly values have no valid-day counts in the inherited table; their legacy strict=True flag is not evidence of a locally verified 75% completeness threshold"}
    dump(out/"weather_verification.json",result)
    print(json.dumps(result,indent=2))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__=="__main__":main()
