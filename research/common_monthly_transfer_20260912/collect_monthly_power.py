"""Recover native monthly POWER values and extend the same grids through 2025."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import requests

from audit_inputs import HERE, ROOT

RAW_OLD=ROOT/"data/raw/meteorology/nasa_power_monthly_tiles_1998_2022"
RAW_NEW=HERE/"data/raw/power_tiles_2022_2025"
PROCESSED=HERE/"data/processed"
META=HERE/"data/metadata"
API="https://power.larc.nasa.gov/api/temporal/monthly/regional"
VARIABLES={"T2M":"monthly_power_temperature_c","T2M_MAX":"monthly_power_temperature_max_c",
    "T2M_MIN":"monthly_power_temperature_min_c","RH2M":"monthly_power_humidity_pct",
    "WS10M":"monthly_power_wind_mps","PRECTOTCORR":"monthly_power_precip_mm_day"}
UNITS={"T2M":"C","T2M_MAX":"C","T2M_MIN":"C","RH2M":"%","WS10M":"m/s","PRECTOTCORR":"mm/day"}
PATTERN=re.compile(r"(?P<parameter>.+)_(?P<country>AUS|CHN)_lat(?P<latmin>m?\d+)_(?P<latmax>m?\d+)_lon(?P<lonmin>m?\d+)_(?P<lonmax>m?\d+)\.json$")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding="utf-8")


def coords(feature):
    return tuple(map(float,feature["geometry"]["coordinates"][:2]))


def request_parameters(path):
    m=PATTERN.fullmatch(path.name)
    assert m,path
    g=m.groupdict()
    return {"parameters":g["parameter"],"latitude-min":int(g["latmin"].replace("m","-")),
        "latitude-max":int(g["latmax"].replace("m","-")),
        "longitude-min":int(g["lonmin"].replace("m","-")),"longitude-max":int(g["lonmax"].replace("m","-")),
        "community":"SB","start":2022,"end":2025,"format":"JSON"}


def validate_document(data,parameter,years):
    assert data.get("features"),"No grid features"
    assert data["parameters"][parameter]["units"]==UNITS[parameter],data.get("parameters")
    expected={f"{y}{m:02d}" for y in years for m in range(1,13)}
    for f in data["features"]:
        assert expected<=set(f["properties"]["parameter"][parameter]),"Missing calendar keys"


def parse_values(data,parameter,wanted,years):
    records=[]
    fill=float(data["header"]["fill_value"])
    for f in data["features"]:
        xy=coords(f)
        if xy not in wanted:continue
        values=f["properties"]["parameter"][parameter]
        for year in years:
            for month in range(1,14):
                value=values.get(f"{year}{month:02d}",fill)
                value=np.nan if value is None or value==fill else float(value)
                records.append((xy[0],xy[1],year,month,parameter,value))
    return records


def fetch(job):
    path=RAW_NEW/job["name"];parameter=job["parameter"]
    if path.exists():
        data=json.loads(path.read_text(encoding="utf-8"))
        validate_document(data,parameter,range(2022,2026))
        return job|{"status":"cached","path":str(path),"sha256":sha(path)}
    params=job["request_parameters"]
    for attempt in range(1,5):
        try:
            response=requests.get(API,params=params,timeout=(20,120))
            if response.status_code==429:
                retry=response.headers.get("Retry-After","60")
                seconds=float(retry) if retry.isdigit() else 60.
                if seconds>300:return job|{"status":"deferred_rate_limit","retry_after":retry}
                time.sleep(max(seconds,30.));continue
            response.raise_for_status()
            data=response.json();validate_document(data,parameter,range(2022,2026))
            tmp=path.with_suffix(".part")
            tmp.write_bytes(response.content);tmp.replace(path)
            record=job|{"status":"downloaded","path":str(path),"url":response.url,
                "sha256":sha(path),"bytes":path.stat().st_size,"retrieved_at_utc":datetime.now(timezone.utc).isoformat(),
                "api_header":data.get("header"),"attempt":attempt}
            dump(path.with_suffix(".request.json"),record)
            time.sleep(.2)
            return record
        except (requests.RequestException,ValueError,AssertionError,KeyError) as error:
            if attempt==4:return job|{"status":"failed","error":str(error),"attempt":attempt}
            time.sleep(2**attempt)
    return job|{"status":"failed","error":"rate limit persisted"}


def unique_values(records):
    data=pd.DataFrame.from_records(records,columns=["power_lon","power_lat","year","month","parameter","value"])
    key=["power_lon","power_lat","year","month","parameter"]
    dup=data[data.duplicated(key,keep=False)]
    if len(dup):
        ranges=dup.groupby(key).value.agg(["min","max"])
        assert (ranges["max"]-ranges["min"]).dropna().le(1e-9).all(),"Conflicting values on overlapping tiles"
    return data.drop_duplicates(key).reset_index(drop=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--probe",action="store_true")
    args=parser.parse_args()
    for p in [RAW_NEW,PROCESSED,META]:p.mkdir(parents=True,exist_ok=True)
    mapping=pd.read_csv(HERE/"audit/fua_existing_power_mapping.csv")
    wanted=set(map(tuple,mapping[["power_lon","power_lat"]].drop_duplicates().to_numpy()))
    jobs=[];old_records=[];provenance=[]
    for path in sorted(RAW_OLD.glob("*.json")):
        parameter=PATTERN.fullmatch(path.name)["parameter"]
        data=json.loads(path.read_text(encoding="utf-8"))
        validate_document(data,parameter,range(1998,2023))
        selected=wanted&{coords(f) for f in data["features"]}
        if not selected:continue
        old_records.extend(parse_values(data,parameter,selected,range(1998,2023)))
        jobs.append({"name":path.name,"parameter":parameter,"matched_cells":len(selected),"request_parameters":request_parameters(path)})
        provenance.append({"path":str(path),"sha256":sha(path),"parameter":parameter,"matched_cells":len(selected)})
    old=unique_values(old_records);del old_records
    old.to_csv(PROCESSED/"recovered_grid_1998_2022_long.csv.gz",index=False)
    dump(META/"inherited_raw_files.json",provenance)
    dump(META/"download_plan.json",{"api":API,"jobs":jobs,"concurrent_requests":3,
        "overlap_year":2022,"new_years":[2023,2024,2025],"resolution":"0.5 latitude x 0.625 longitude degrees",
        "notes":"Reuse saved native monthly values, exclude month 13 from monthly features; keep old 2022 after comparing overlap."})
    print("RECOVERED",len(old),"raw_values_including_annual","jobs",len(jobs),"cells",len(wanted),flush=True)
    if args.probe:
        print(json.dumps(fetch(jobs[0]),ensure_ascii=False,default=str),flush=True);return
    results=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(fetch,j):j for j in jobs}
        for i,f in enumerate(as_completed(pending),1):
            result=f.result();results.append(result)
            dump(META/"download_manifest.json",results)
            print("DOWNLOAD",i,"/",len(jobs),result["status"],result["name"],flush=True)
    failed=[r for r in results if r["status"] not in ["cached","downloaded"]]
    if failed:raise RuntimeError(f"{len(failed)} weather files unavailable; no complete-data claim or model run")
    records=[]
    for job in jobs:
        data=json.loads((RAW_NEW/job["name"]).read_text(encoding="utf-8"))
        records.extend(parse_values(data,job["parameter"],wanted,range(2022,2026)))
    new=unique_values(records);del records
    overlap=old[old.year.eq(2022)].merge(new[new.year.eq(2022)],on=["power_lon","power_lat","year","month","parameter"],suffixes=("_old","_new"),validate="one_to_one")
    overlap["absolute_difference"]=(overlap.value_old-overlap.value_new).abs()
    overlap.groupby("parameter").absolute_difference.agg(["count","mean","max"]).to_csv(META/"overlap_2022_comparison.csv")
    combined=pd.concat([old,new[new.year>=2023]],ignore_index=True)
    monthly=combined[combined.month.between(1,12)].pivot(index=["power_lon","power_lat","year","month"],columns="parameter",values="value").reset_index()
    monthly.columns.name=None;monthly=monthly.rename(columns=VARIABLES)
    assert len(monthly)==len(wanted)*28*12
    assert monthly[list(VARIABLES.values())].notna().all().all(),"Missing actual monthly weather values"
    limits={"monthly_power_temperature_c":(-80,60),"monthly_power_temperature_min_c":(-100,80),
        "monthly_power_temperature_max_c":(-100,80),"monthly_power_humidity_pct":(0,100),
        "monthly_power_wind_mps":(0,80),"monthly_power_precip_mm_day":(0,1000)}
    for c,(lo,hi) in limits.items():assert monthly[c].between(lo,hi).all(),(c,"value outside broad physical range")
    assert (monthly.monthly_power_temperature_min_c<=monthly.monthly_power_temperature_c).all()
    assert (monthly.monthly_power_temperature_c<=monthly.monthly_power_temperature_max_c).all()
    monthly.to_csv(PROCESSED/"power_grid_month_1998_2025.csv.gz",index=False)
    annual=combined[combined.month.eq(13)].pivot(index=["power_lon","power_lat","year"],columns="parameter",values="value").reset_index().rename(columns=VARIABLES)
    annual.to_csv(PROCESSED/"power_grid_annual_1998_2025.csv.gz",index=False)
    cols=["country_iso","eFUA_ID","fua_name","power_lon","power_lat","power_great_circle_distance_km"]
    fua=mapping[cols].merge(monthly,on=["power_lon","power_lat"],how="left",validate="many_to_many")
    assert not fua.duplicated(["eFUA_ID","year","month"]).any()
    assert fua.groupby("eFUA_ID").size().eq(336).all()
    fua["source_id"]="NASA_POWER_MERRA2_monthly"
    fua["source_kind"]="reanalysis_not_station_observation"
    fua.to_csv(PROCESSED/"fua_monthly_power_1998_2025.csv.gz",index=False)
    coverage=fua.assign(complete=fua[list(VARIABLES.values())].notna().all(axis=1)).groupby("country_iso").agg(
        fuas=("eFUA_ID","nunique"),rows=("eFUA_ID","size"),complete_rows=("complete","sum"),complete_fraction=("complete","mean"))
    coverage.to_csv(META/"coverage.csv")
    monthly_variation=fua.groupby(["country_iso","eFUA_ID","year"]).monthly_power_temperature_c.std()
    assert monthly_variation.gt(0).all(),"Annual repeated values detected in monthly temperature"
    dump(META/"data_manifest.json",{"status":"complete","source_url":API,"sources":["NASA POWER","MERRA-2"],
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"coverage":coverage.reset_index().to_dict("records"),
        "weather_grid_cells":len(wanted),"years":[1998,2025],"feature_variables":VARIABLES,"api_units":UNITS,
        "raw_inherited_manifest_sha256":sha(META/"inherited_raw_files.json"),"raw_new_manifest_sha256":sha(META/"download_manifest.json"),
        "fua_monthly_sha256":sha(PROCESSED/"fua_monthly_power_1998_2025.csv.gz"),
        "grid_monthly_sha256":sha(PROCESSED/"power_grid_month_1998_2025.csv.gz"),
        "mapping_sha256":sha(HERE/"audit/fua_existing_power_mapping.csv"),
        "script_sha256":sha(__file__),"weather_quality":"All six variables present, broad range and temperature-order checks passed",
        "limitations":"Native coarse grid means sampled at inherited centroid assignments; not new monitoring stations, PM2.5 labels, or boundary-layer-height measurements"})
    print("COLLECTION_COMPLETE",coverage.to_string(),flush=True)


if __name__=="__main__":main()
