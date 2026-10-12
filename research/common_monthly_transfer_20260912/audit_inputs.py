"""Inspect inherited raw monthly weather before choosing a recovery strategy."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=Path(r"C:\Users\USER\.codex\.chatgpt-projects\g-p-69aaacd3007481919301f520a145f3f4")
PREVIOUS=HERE.parent/"conditional_transfer_20260912"


def main():
    out=HERE/"audit";out.mkdir(exist_ok=True)
    raw=ROOT/"data/raw/meteorology/nasa_power_monthly_tiles_1998_2022"
    path=sorted(raw.glob("T2M_*.json"))[0]
    data=json.loads(path.read_text(encoding="utf-8"))
    feature=data["features"][0]
    values=feature["properties"]["parameter"]["T2M"]
    print("RAW_EXAMPLE",json.dumps({"file":str(path),"header":data.get("header"),
        "parameters":data.get("parameters"),"feature_count":len(data["features"]),
        "coordinates":feature["geometry"]["coordinates"],"keys":list(values)[:16],
        "year_2022":{k:v for k,v in values.items() if k.startswith("2022")}},ensure_ascii=False),flush=True)
    frame,_,_=joblib.load(PREVIOUS/"fua_month_cache.joblib")
    fuas=frame.groupby(["country_iso","eFUA_ID"]).first().reset_index()
    mapping=pd.read_csv(ROOT/"data/interim/meteorology/ghs_fua_to_nasa_power_grid_mapping.csv")
    assert not mapping.eFUA_ID.duplicated().any()
    mapping=fuas[["country_iso","eFUA_ID","fua_name","fua_centroid_lon","fua_centroid_lat"]].merge(mapping,
        on="eFUA_ID",how="left",validate="one_to_one",indicator=True)
    assert mapping._merge.eq("both").all()
    mapping=mapping.drop(columns="_merge")
    lon1=np.deg2rad(mapping.fua_centroid_lon);lat1=np.deg2rad(mapping.fua_centroid_lat)
    lon2=np.deg2rad(mapping.power_lon);lat2=np.deg2rad(mapping.power_lat)
    h=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    mapping["power_great_circle_distance_km"]=6371.0088*2*np.arcsin(np.sqrt(np.clip(h,0,1)))
    mapping.to_csv(out/"fua_existing_power_mapping.csv",index=False)
    counts=[]
    for country,p in mapping.groupby("country_iso"):
        counts.append({"country":country,"fuas":len(p),"unique_weather_cells":len(p[["power_lon","power_lat"]].drop_duplicates()),
            "median_centroid_distance_km":float(p.power_great_circle_distance_km.median()),
            "max_centroid_distance_km":float(p.power_great_circle_distance_km.max())})
    print("GRID_MAPPING",json.dumps(counts),flush=True)
    complete=frame.surface_temperature_mean_c.notna()&frame.surface_relative_humidity_mean_pct.notna()&frame.surface_wind_mean_mps.notna()
    q=frame[["country_iso","eFUA_ID","fua_name","year","month"]].copy()
    q["observed_temperature_humidity_wind"]=complete
    q.groupby(["country_iso","eFUA_ID","fua_name"]).observed_temperature_humidity_wind.agg(["sum","mean","size"]).reset_index().to_csv(out/"aviation_coverage_by_fua.csv",index=False)
    (out/"input_audit.json").write_text(json.dumps({"raw_files":len(list(raw.glob("*.json"))),
        "example_has_months":all(f"2022{m:02d}" in values for m in range(1,13)),"mapping":counts,
        "annual_extractor":"parse_annual_grid reads year + '13'; original raw responses retain months 01-12",
        "distance_note":"Existing grid assignment retained for comparability; distance recomputed geodesically, not in Web Mercator"},indent=2),encoding="utf-8")


if __name__=="__main__":main()
