"""Inspect provenance and identifiers, never recompute prior test performance."""
import json
from pathlib import Path
import pandas as pd
from common import HERE, ROOT, WORK, PREVIOUS, digest, dump, now

def main():
    paths={
        'au2025_manifest':ROOT/'model_outputs/australia_2025_external_test_20260829/metadata/run_metadata.json',
        'au2025_code':ROOT/'thesis_workspace/run_australia_2025_external_test.py',
        'au2025_predictions':ROOT/'model_outputs/australia_2025_external_test_20260829/tables/external_2025_static_predictions.csv.gz',
        'au2021_2024_predictions':ROOT/'model_outputs/australia_multistate_refinement_20260824/static_test_predictions_2021_2024.csv.gz',
        'cn_yrd_code':ROOT/'thesis_workspace/run_yrd_reference_alignment_audit.py',
        'cn_yrd_metrics':ROOT/'model_outputs/yrd_reference_alignment_20260823/tables/yrd_final_monthly_model_metrics_2021_2024.csv',
        'current_model_protocol':PREVIOUS/'PROTOCOL.md',
        'current_model_code':PREVIOUS/'run_regional.py',
    }
    evidence=[]
    for key,path in paths.items():
        assert path.exists(),path
        evidence.append({'id':key,'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)})
    manifest=json.loads(paths['au2025_manifest'].read_text(encoding='utf-8'))
    rows=[]
    for year in (2023,2024,2025):
        for state in ('NSW','QLD','VIC'):
            rows.append({'country_iso':'AUS','network':state,'year':year,'target':'regulatory_PM25',
                'status':'prior_model_outcomes_evaluated','fresh_validation_allowed':False,
                'evidence_ids':'au2025_manifest|au2025_code|au2025_predictions' if year==2025 else 'au2021_2024_predictions',
                'permitted_use':'retrospective_time_extension_or_development; not untouched confirmation'})
        if year<2025:
            rows.append({'country_iso':'CHN','network':'YRD_satellite_grid','year':year,'target':'SatPM_model_estimate',
                'status':'prior_model_outcomes_evaluated','fresh_validation_allowed':False,
                'evidence_ids':'cn_yrd_code|cn_yrd_metrics','permitted_use':'retrospective; not regulatory ground truth'})
        rows.append({'country_iso':'CHN','network':'CNEMC','year':year,'target':'regulatory_PM25',
            'status':'full_acquisition_and_project_exposure_not_yet_certified','fresh_validation_allowed':False,
            'evidence_ids':'china_availability_probes.json','permitted_use':'candidate only; three fixed dates probed, not full years'})
        for state in ('SA','WA'):
            rows.append({'country_iso':'AUS','network':state,'year':year,'target':'regulatory_PM25_candidate',
                'status':'candidate_prior_exposure_not_proven_absent','fresh_validation_allowed':False,
                'evidence_ids':'candidate_downloads.json' if state=='SA' else 'source_ledger.json',
                'permitted_use':'catalogue/schema audit; no model evaluation before new lock'})
    pd.DataFrame(rows).to_csv(HERE/'audit/outcome_exposure_ledger.csv',index=False)
    # Text/name search has limited evidential strength; do not call no hits proof.
    terms=['netley','cowandilla','christie downs','air_quality_monitoring_stations_sa',
        'south australian airsheds','west australian monitor','wa_air_quality']
    hits=[];scanned=0
    for base in [ROOT/'thesis_workspace',ROOT/'model_outputs',ROOT/'data/processed',WORK/'research']:
        for path in base.rglob('*'):
            if not path.is_file() or HERE in path.parents or path.suffix.lower() not in ('.py','.json','.md','.txt','.tex'):
                continue
            if path.stat().st_size>10_000_000:continue
            scanned+=1
            text=path.read_text(encoding='utf-8',errors='replace').lower()
            found=[t for t in terms if t in text or t in path.name.lower()]
            if found:hits.append({'path':str(path),'terms':found})
    dump(HERE/'audit/prior_network_search.json',{'files_scanned':scanned,'terms':terms,'hits':hits,
        'limitation':'Text/name search excludes binary and large files and cannot certify absence of historical human or tool exposure.'})
    dump(HERE/'audit/exposure_evidence.json',{'created_at':now(),'files':evidence,
        'earlier_2025_test_run':{k:manifest[k] for k in ['created_at_utc','training_rows','external_rows','external_stations','external_states']},
        'old_test_claim_not_retroactively_invalidated':True,
        'correction':'The 2025 test may have been untouched before its first evaluation on 2026-08-29, but cannot be reused as fresh confirmation after subsequent adaptive research.'})
    print('Exposure rows',len(rows),'prior files scanned',scanned,'candidate-network text hits',len(hits),flush=True)

if __name__=='__main__':main()
