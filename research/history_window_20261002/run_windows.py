"""Change the historical label window, keeping the prior fit implementation."""
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

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'regional_station_scope_20260920'
CORRECTED=HERE.parent/'boundary_regression_20261001'
BASE=HERE.parent/'thesis_completion_20260912'
sys.path.insert(0,str(PREVIOUS))
import run_regional as old
read,dump,sha=old.read,old.dump,old.sha
WINDOWS=[1,3,6,12]
PRIMARY='regional_kernel'
for name in ['audit','data','results']:(HERE/name).mkdir(exist_ok=True)

def now():return datetime.now(timezone.utc).isoformat()

def fitter(months,root):
    module=ast.parse((PREVIOUS/'run_regional.py').read_text(encoding='utf-8'))
    func=copy.deepcopy(next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='fit_predict'))
    class WindowChange(ast.NodeTransformer):
        masks=0
        manifests=0
        def visit_BinOp(self,node):
            self.generic_visit(node)
            if isinstance(node.op,ast.Sub) and isinstance(node.left,ast.Name) and node.left.id=='month' and isinstance(node.right,ast.Constant) and node.right.value==12:
                node.right=ast.Constant(value=months);self.masks+=1
            return node
        def visit_Dict(self,node):
            self.generic_visit(node)
            for i,k in enumerate(node.keys):
                if isinstance(k,ast.Constant) and k.value=='window_months':
                    assert isinstance(node.values[i],ast.Constant) and node.values[i].value==12
                    node.values[i]=ast.Constant(value=months);self.manifests+=1
            return node
    edit=WindowChange();func=edit.visit(func)
    assert edit.masks==edit.manifests==1
    env=old.__dict__.copy();env['HERE']=root
    compiled=ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[]))
    exec(compile(compiled,str(PREVIOUS/'run_regional.py'),'exec'),env)
    dump(root/'ast_change_receipt.json',{'window':months,'training_mask_changes':edit.masks,
        'manifest_changes':edit.manifests,'source_sha256':sha(PREVIOUS/'run_regional.py'),
        'changed_ast':ast.dump(compiled,include_attributes=False)})
    return env['fit_predict']

def main():
    inputs=[PREVIOUS/'run_regional.py',old.FOCUSED/'run_focused.py',BASE/'run_monitor_models.py',
            PREVIOUS/'audit/fixed_city_splits.json',CORRECTED/'data/corrected_australian_national_models.joblib',
            BASE/'results/all_networks/monthly_plus_mixing/CHN_to_AUS/source_models.joblib']
    inputs+=list((CORRECTED/'data').glob('*_panel.csv.gz'))
    inputs+=list((CORRECTED/'results').glob('*/*/*/predictions.csv.gz'))
    receipt={'protocol_sha256':sha(HERE/'PROTOCOL.md'),'script_sha256':sha(__file__),
             'inputs':[{'path':str(p),'sha256':sha(p)} for p in inputs]}
    rp=HERE/'audit/run_receipt.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==receipt
    else:dump(rp,receipt)
    splits=json.loads((PREVIOUS/'audit/fixed_city_splits.json').read_text(encoding='utf-8'))
    bundles={'CHN_to_AUS':joblib.load(BASE/'results/all_networks/monthly_plus_mixing/CHN_to_AUS/source_models.joblib'),
             'AUS_to_CHN':joblib.load(CORRECTED/'data/corrected_australian_national_models.joblib')}
    for months in [12,3,1,6]:
        root=HERE/f'window_{months}';(root/'data').mkdir(parents=True,exist_ok=True)
        shutil.copy2(HERE/'PROTOCOL.md',root/'PROTOCOL.md')
        for cohort in old.COHORTS:shutil.copy2(CORRECTED/f'data/{cohort}_panel.csv.gz',root/f'data/{cohort}_panel.csv.gz')
        fit=fitter(months,root)
        for cohort in old.COHORTS:
            panel=read(root/f'data/{cohort}_panel.csv.gz')
            for direction,bundle in bundles.items():
                for seed in old.SEEDS:
                    folder=root/f'results/{cohort}/{direction}/{seed}';folder.mkdir(parents=True,exist_ok=True)
                    complete=folder/'complete.json'
                    if complete.exists():
                        cached=json.loads(complete.read_text(encoding='utf-8'))
                        assert cached['prediction_sha256']==sha(folder/'predictions.csv.gz')
                        continue
                    check_case=months==12 and cohort=='membership_all' and direction=='CHN_to_AUS' and seed==5993
                    if months==12 and not check_case:
                        original=CORRECTED/f'results/{cohort}/{direction}/{seed}'
                        for name in ['predictions.csv.gz','fit_audit.csv','calibration_rows.csv.gz','training_manifest.json','no_prediction.csv']:
                            shutil.copy2(original/name,folder/name)
                        mode='reused_verified_corrected_12month'
                    else:
                        print('FIT',months,cohort,direction,seed,flush=True)
                        target=direction.split('_to_')[1]
                        p,meta=fit(panel,bundle['features'],cohort,direction,seed,folder,bundle,splits[target]['seeds'][str(seed)])
                        mode='new_fit'
                        if check_case:
                            before=read(CORRECTED/f'results/{cohort}/{direction}/{seed}/predictions.csv.gz')
                            assert len(before)==len(p)
                            assert p[['eFUA_ID','t']].equals(before[['eFUA_ID','t']])
                            error=float(np.max(np.abs(p[old.MODELS].to_numpy()-before[old.MODELS].to_numpy())))
                            assert error<1e-8,error
                            dump(HERE/'audit/12month_regression.json',{'passed':True,'models':len(old.MODELS),'rows':len(p),'max_error':error})
                    dump(complete,{'finished_at':now(),'window_months':months,'mode':mode,
                        'prediction_sha256':sha(folder/'predictions.csv.gz'),'protocol_sha256':sha(HERE/'PROTOCOL.md'),
                        'driver_sha256':sha(__file__)})
                    print('DONE',months,cohort,direction,seed,mode,flush=True)
    for item in receipt['inputs']:assert sha(item['path'])==item['sha256']
    dump(HERE/'audit/completion.json',{'completed_at':now(),'configurations':48,
        'new_fitted_configurations':37,'reused_configurations':11,'primary_window':3,'primary_model':PRIMARY,
        'new_external_labels_used':False,'new_condition_search':False})

if __name__=='__main__':main()
