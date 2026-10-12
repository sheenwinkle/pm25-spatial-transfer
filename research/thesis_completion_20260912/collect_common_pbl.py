"""Extract common coarse ERA5 mixing diagnostics from public WeatherBench2 Zarr."""
import concurrent.futures
import hashlib
import json
import time
import requests
import numpy as np
import pandas as pd
import numcodecs
import joblib
from common import HERE, DATA, AUDIT, MONTHLY, dump, digest

BASE='https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'
RAW=DATA/'raw_weatherbench2';RAW.mkdir(exist_ok=True)

def fetch(key):
    p=RAW/key.replace('/','__')
    if p.exists():return p.read_bytes()
    for attempt in range(5):
        try:
            r=requests.get(BASE+'/'+key,timeout=90);r.raise_for_status();data=r.content
            p.write_bytes(data);return data
        except requests.RequestException:
            if attempt==4:raise
            time.sleep(2**attempt)

def main():
    meta=json.loads(fetch('.zmetadata'))['metadata']
    def decode(var,key):
        cfg=meta[var+'/.zarray'];buf=fetch(var+'/'+key)
        raw=numcodecs.get_codec(cfg['compressor']).decode(buf)
        return np.frombuffer(raw,dtype=cfg['dtype']),buf
    coords={}
    for v in ['latitude','longitude','time']:
        cfg=meta[v+'/.zarray'];parts=[]
        for i in range(int(np.ceil(cfg['shape'][0]/cfg['chunks'][0]))):parts.append(decode(v,str(i))[0])
        coords[v]=np.concatenate(parts)[:cfg['shape'][0]]
    times=pd.to_datetime(coords['time'],unit='h',origin=pd.Timestamp('1959-01-01'))
    assert meta['time/.zattrs']['units']=='hours since 1959-01-01'
    panel,_,_=joblib.load(MONTHLY/'modelling_cache.joblib')
    fuas=panel[['eFUA_ID','country_iso','fua_centroid_lon','fua_centroid_lat']].drop_duplicates('eFUA_ID').sort_values('eFUA_ID')
    ix=np.abs(coords['longitude'][:,None]-(fuas.fua_centroid_lon.to_numpy()%360)[None,:]).argmin(0)
    iy=np.abs(coords['latitude'][:,None]-fuas.fua_centroid_lat.to_numpy()[None,:]).argmin(0)
    fuas['era5_grid_lon']=coords['longitude'][ix];fuas['era5_grid_lat']=coords['latitude'][iy]
    fuas.to_csv(AUDIT/'common_era5_fua_mapping.csv',index=False)
    names=['boundary_layer_height','10m_u_component_of_wind','10m_v_component_of_wind']
    ci=meta[names[0]+'/.zarray']['chunks'];assert ci==[8,240,121]
    for v in names:
        assert meta[v+'/.zarray']['chunks']==ci
        assert meta[v+'/.zattrs']['_ARRAY_DIMENSIONS']==['time','longitude','latitude']
    selected=np.flatnonzero((times.year>=2015)&(times.year<=2022));chunks=sorted(set(selected//8))
    outputs={v:{} for v in names};manifest=[]
    def worker(task):
        v,i=task;a,buf=decode(v,f'{i}.0.0');a=a.reshape(ci)
        return v,i,a[:,ix,iy].copy(),hashlib.sha256(buf).hexdigest(),len(buf)
    tasks=[(v,int(i)) for i in chunks for v in names]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for k,(v,i,a,h,n) in enumerate(pool.map(worker,tasks)):
            outputs[v][i]=a;manifest.append({'variable':v,'chunk':i,'sha256':h,'bytes':n})
            if (k+1)%300==0:print('ERA5_CHUNKS',k+1,'/',len(tasks),flush=True)
    allidx=np.concatenate([np.arange(i*8,i*8+8) for i in chunks]);tt=times[allidx]
    use=(tt.year>=2015)&(tt.year<=2022);tt=tt[use]
    arrays={v:np.concatenate([outputs[v][i] for i in chunks])[use] for v in names}
    blh=arrays[names[0]];vent=blh*np.hypot(arrays[names[1]],arrays[names[2]])
    assert np.isfinite(blh).all() and np.isfinite(vent).all() and (blh>=0).all()
    parts=[]
    for year in range(2015,2023):
        for month in range(1,13):
            m=(tt.year==year)&(tt.month==month);a=blh[m];b=vent[m];out=fuas.copy()
            out['year']=year;out['month']=month;out['era5_pbl_mean_m']=a.mean(0)
            out['era5_pbl_p10_m']=np.quantile(a,.1,axis=0);out['era5_pbl_low_fraction']=(a<500).mean(0)
            out['era5_surface_ventilation_mean_m2_s']=b.mean(0);out['era5_samples']=m.sum()
            parts.append(out)
    d=pd.concat(parts,ignore_index=True);d.to_csv(DATA/'common_era5_month_2015_2022.csv.gz',index=False)
    pd.DataFrame(manifest).to_csv(AUDIT/'weatherbench2_chunk_manifest.csv',index=False)
    dump(AUDIT/'common_era5_manifest.json',{'source':BASE,'rows':len(d),'fuas':len(fuas),
        'raw_chunk_count':len(tasks),'downloaded_bytes':sum(r['bytes'] for r in manifest),
        'resolution':'1.5 degree first-order conservative remapping; nearest remapped cell; six-hourly UTC',
        'interpretation':'Coarse contextual mixing proxy, not local boundary-layer measurement or vertically integrated ventilation.',
        'panel_sha256':digest(DATA/'common_era5_month_2015_2022.csv.gz'),'script_sha256':digest(__file__)})
    print('COMMON_ERA5_COMPLETE',len(d),flush=True)

if __name__=='__main__':main()
