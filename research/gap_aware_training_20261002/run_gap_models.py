"""A bounded valid-observation history replaces the fixed calendar mask."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import ast
import copy
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REGIONAL=HERE.parent/'regional_station_scope_20260920'
CORRECTED=HERE.parent/'boundary_regression_20261001'
WINDOWS=HERE.parent/'history_window_20261002'
BASE=HERE.parent/'thesis_completion_20260912'
sys.path.insert(0,str(REGIONAL))
import run_regional as old
read,dump,sha=old.read,old.dump,old.sha
for name in ['audit','data','results']:(HERE/name).mkdir(exist_ok=True)

def now():return datetime.now(timezone.utc).isoformat()

def select_history(frame,development,month,width,mode='valid'):
    if mode=='fixed':return development&frame.t.between(month-width,month-1).to_numpy()
    if mode!='valid':raise ValueError(mode)
    candidates=frame[development&frame.t.between(month-12,month-1).to_numpy()]
    chosen=candidates.sort_values(['t','eFUA_ID'],kind='stable').groupby('eFUA_ID',sort=False).tail(width)
    return frame.index.isin(chosen.index)

def test_selector():
    f=pd.DataFrame({'eFUA_ID':[1,1,1,2,2,2,9,1],
                    't':[1,3,4,1,2,4,4,5],'observed_pm25':np.arange(8.)})
    dev=f.eFUA_ID.isin([1,2]).to_numpy()
    a=select_history(f,dev,5,2)
    assert set(f[a].index)=={1,2,4,5}
    assert not a[6] and not a[7]
    shifted=f.copy();shifted['observed_pm25']=np.random.default_rng(99).normal(size=len(f))*1e6
    assert np.array_equal(a,select_history(shifted,dev,5,2))
    assert not select_history(f,dev,20,2).any()
    assert np.array_equal(select_history(f,dev,5,2,'fixed'),dev&f.t.between(3,4).to_numpy())
    regular=pd.DataFrame({'eFUA_ID':[1]*6+[2]*6,'t':list(range(1,7))*2})
    d=np.ones(len(regular),bool)
    assert np.array_equal(select_history(regular,d,7,3),select_history(regular,d,7,3,'fixed'))
    dump(HERE/'audit/selector_unit_tests.json',{'passed':True,'tests':['skips_missing_months','no_future_labels',
        'no_held_out_city','age_bound_12','outcome_invariance','per_city_count_cap','complete_history_equivalence']})

def fitter(width,root,mode):
    tree=ast.parse((REGIONAL/'run_regional.py').read_text(encoding='utf-8'))
    func=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_predict'))
    class Edit(ast.NodeTransformer):
        masks=0
        manifests=0
        def visit_Assign(self,node):
            self.generic_visit(node)
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='fit':
                assert isinstance(node.value,ast.BinOp) and isinstance(node.value.op,ast.BitAnd)
                node.value=ast.Call(func=ast.Name(id='SELECT_HISTORY',ctx=ast.Load()),
                    args=[ast.Name(id='t',ctx=ast.Load()),ast.Name(id='dev',ctx=ast.Load()),ast.Name(id='month',ctx=ast.Load())],keywords=[])
                self.masks+=1
            return node
        def visit_Dict(self,node):
            self.generic_visit(node)
            for i,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='window_months':node.values[i]=ast.Constant(value=width);self.manifests+=1
            return node
    edit=Edit();func=edit.visit(func);assert edit.masks==edit.manifests==1
    env=old.__dict__.copy();env['HERE']=root
    env['SELECT_HISTORY']=lambda t,d,m:select_history(t,d,m,width,mode)
    module=ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[]))
    exec(compile(module,str(REGIONAL/'run_regional.py'),'exec'),env)
    dump(root/'implementation_receipt.json',{'mode':mode,'requested_valid_months_per_city':width,
        'maximum_calendar_lookback':12 if mode=='valid' else width,'mask_replacements':edit.masks,
        'manifest_replacements':edit.manifests,'base_implementation_sha256':sha(REGIONAL/'run_regional.py')})
    return env['fit_predict']

def main():
    test_selector()
    files=[REGIONAL/'run_regional.py',old.FOCUSED/'run_focused.py',BASE/'run_monitor_models.py',
        REGIONAL/'audit/fixed_city_splits.json',CORRECTED/'data/corrected_australian_national_models.joblib',
        BASE/'results/all_networks/monthly_plus_mixing/CHN_to_AUS/source_models.joblib']
    files+=list((CORRECTED/'data').glob('*_panel.csv.gz'))
    for width in [1,3]:files+=list((WINDOWS/f'window_{width}/results').glob('*/*/*/predictions.csv.gz'))
    receipt={'protocol_sha256':sha(HERE/'PROTOCOL.md'),'script_sha256':sha(__file__),
        'inputs':[{'path':str(p),'sha256':sha(p)} for p in files]}
    rp=HERE/'audit/model_run_receipt.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==receipt
    else:dump(rp,receipt)
    split=json.loads((REGIONAL/'audit/fixed_city_splits.json').read_text())
    bundles={'CHN_to_AUS':joblib.load(BASE/'results/all_networks/monthly_plus_mixing/CHN_to_AUS/source_models.joblib'),
             'AUS_to_CHN':joblib.load(CORRECTED/'data/corrected_australian_national_models.joblib')}
    configs=[('fixed_check',1,'fixed')]+[(f'valid_{w}',w,'valid') for w in [3,1]]
    for tag,width,mode in configs:
        root=HERE/tag;(root/'data').mkdir(parents=True,exist_ok=True)
        shutil.copy2(HERE/'PROTOCOL.md',root/'PROTOCOL.md')
        for cohort in old.COHORTS:shutil.copy2(CORRECTED/f'data/{cohort}_panel.csv.gz',root/f'data/{cohort}_panel.csv.gz')
        fit=fitter(width,root,mode)
        for cohort in old.COHORTS:
            panel=read(root/f'data/{cohort}_panel.csv.gz')
            for direction,bundle in bundles.items():
                for seed in old.SEEDS:
                    if mode=='fixed' and (cohort,direction,seed)!=('membership_all','CHN_to_AUS',5993):continue
                    folder=root/f'results/{cohort}/{direction}/{seed}';folder.mkdir(parents=True,exist_ok=True)
                    if (folder/'complete.json').exists():continue
                    print('FIT',tag,cohort,direction,seed,flush=True)
                    target=direction.split('_to_')[1];ids=split[target]['seeds'][str(seed)]
                    p,meta=fit(panel,bundle['features'],cohort,direction,seed,folder,bundle,ids)
                    if mode=='fixed':
                        before=read(WINDOWS/f'window_1/results/{cohort}/{direction}/{seed}/predictions.csv.gz')
                        assert before[['eFUA_ID','t']].equals(p[['eFUA_ID','t']])
                        error=float(np.max(np.abs(p[old.MODELS].to_numpy()-before[old.MODELS].to_numpy())))
                        assert error<1e-8,error
                        dump(HERE/'audit/fixed_path_regression.json',{'passed':True,'maximum_prediction_error':error,'rows':len(p),'models':27})
                    t=panel[panel.country_iso.eq(target)].reset_index(drop=True)
                    dev=t.eFUA_ID.isin(ids).to_numpy();rows=[];labels=[]
                    for month in sorted(t[t.year.ge(2019)].t.unique()):
                        selected=t[select_history(t,dev,month,width,mode)]
                        age=month-selected.t
                        rows.append({'prediction_t':month,'selected_rows':len(selected),'fit_performed':len(selected)>=5,
                            'contributing_cities':selected.eFUA_ID.nunique(),'latest_age':age.min(),'oldest_age':age.max(),
                            'labels_older_than_nominal_window':int(age.gt(width).sum()),
                            'nominal_window':width,'mode':mode})
                        labels.append(selected[['country_iso','eFUA_ID','year','month','t']].assign(prediction_t=month,label_age=age.to_numpy(),fit_performed=len(selected)>=5))
                    pd.DataFrame(rows).to_csv(folder/'history_audit.csv',index=False)
                    pd.concat(labels,ignore_index=True).to_csv(folder/'selected_label_rows.csv.gz',index=False)
                    dump(folder/'complete.json',{'created_at':now(),'mode':mode,'width':width,'max_calendar_age':12 if mode=='valid' else width,
                        'prediction_sha256':sha(folder/'predictions.csv.gz'),'script_sha256':sha(__file__),
                        'protocol_sha256':sha(HERE/'PROTOCOL.md')})
                    print('DONE',tag,cohort,direction,seed,flush=True)
    for f in receipt['inputs']:assert sha(f['path'])==f['sha256']
    dump(HERE/'audit/model_completion.json',{'completed_at':now(),'new_gap_configurations':24,'fixed_regression_configurations':1,
        'targets_imputed':False,'quality_threshold_changed':False,'fresh_external_confirmation':False})

if __name__=='__main__':main()
