"""Finish the bounded recovery audit without modifying model inputs."""
import io
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from fetch_sources import HERE, dump, sha

PREV = HERE.parent / 'gap_aware_training_20261002'
ROOT = Path.home() / '.codex/.chatgpt-projects/g-p-69aaacd3007481919301f520a145f3f4'


def historical_time_gate(records, start, end):
    """Fail closed on an unparseable timestamp or any out-of-period row."""
    if not records:
        return {'accepted': False, 'reason': 'empty_response', 'rows': 0}
    times = []
    for row in records:
        match = re.fullmatch(r'/Date\((-?\d+)\)/', str(row.get('TimePoint', '')))
        if not match:
            return {'accepted': False, 'reason': 'unparseable_timestamp', 'rows': len(records)}
        times.append(pd.Timestamp(int(match.group(1)), unit='ms', tz='UTC'))
    stamp = pd.DatetimeIndex(times).tz_convert('Asia/Shanghai')
    start = pd.Timestamp(start).tz_localize('Asia/Shanghai')
    end = pd.Timestamp(end).tz_localize('Asia/Shanghai')
    valid = (stamp >= start) & (stamp <= end)
    return {'accepted': bool(valid.all()), 'reason': 'in_period' if valid.all() else 'wrong_period',
            'rows': len(records), 'rows_in_requested_period': int(valid.sum()),
            'returned_min_asia_shanghai': str(stamp.min()),
            'returned_max_asia_shanghai': str(stamp.max())}


def check_gate():
    def row(date):
        return {'TimePoint': f'/Date({pd.Timestamp(date, tz="Asia/Shanghai").value // 10**6})/'}
    a, b = '2018-12-22 00:00', '2018-12-22 23:59:59'
    cases = [('in_range', [row('2018-12-22 12:00')], True),
             ('wrong_year', [row('2026-10-03 15:00')], False),
             ('malformed', [{'TimePoint': 'not-a-time'}], False),
             ('empty', [], False),
             ('mixed_years', [row('2018-12-22 12:00'), row('2026-10-03 15:00')], False),
             ('before_day', [row('2018-12-21 23:00')], False),
             ('after_day', [row('2018-12-23 00:00')], False)]
    results = []
    for name, records, expected in cases:
        result = historical_time_gate(records, a, b)
        assert result['accepted'] == expected, (name, result)
        results.append({'test': name, 'expected': expected, 'passed': True})
    dump(HERE / 'audit/historical_time_gate_tests.json', results)


def verify_candidate_independently():
    # Decode with netCDF4 directly, then compare keyed long records rather than wide alignment.
    source = HERE / 'evidence/li_201812_nc.body'
    archive = ROOT / 'data/raw/cnemc_zenodo_10836299/20180101-20181231_CNEMC.zip'
    with netCDF4.Dataset(source) as ds:
        codes = [str(x).strip() for x in netCDF4.chartostring(np.ma.filled(ds['station_code'][:], b''))]
        dates = [str(x).strip() for x in netCDF4.chartostring(np.ma.filled(ds['time_point'][:], b''))]
        values = np.ma.filled(ds['pm2_5'][:], np.nan).astype(float)
    expected = pd.date_range('2018-12-01', '2018-12-31 23:00', freq='h')
    parsed = pd.to_datetime(dates, format='%Y-%m-%dT%H:%M:%SZ', utc=True, errors='coerce').tz_localize(None)
    valid_slots = np.asarray(parsed == expected)
    valid_value = np.isfinite(values) & (values >= 0) & (values <= 1000)
    ii, jj = np.nonzero(valid_slots[:, None] & valid_value)
    new = pd.DataFrame({'time': expected[ii], 'station_code': np.asarray(codes)[jj],
                        'candidate': values[ii, jj]})
    rows = []
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            match = re.search(r'china_sites_(201812\d{2})\.csv$', name)
            if not match:
                continue
            raw = z.read(name)
            if not raw.startswith(b'date,hour,type,'):
                continue
            d = pd.read_csv(io.BytesIO(raw), low_memory=False)
            d = d[d.type.eq('PM2.5')].copy()
            d['time'] = pd.to_datetime(d.date.astype(str), format='%Y%m%d') + pd.to_timedelta(d.hour, unit='h')
            station_columns = [x for x in d if re.fullmatch(r'\d+[A-Z]', x)]
            d = d.melt(id_vars=['time'], value_vars=station_columns,
                       var_name='station_code', value_name='original')
            d.original = pd.to_numeric(d.original, errors='coerce')
            rows.append(d[np.isfinite(d.original) & d.original.between(0, 1000)])
    old = pd.concat(rows, ignore_index=True)
    assert not old.duplicated(['time', 'station_code']).any()
    merged = new.merge(old, on=['time', 'station_code'], how='left', validate='one_to_one')
    absent = merged.original.isna()
    conflict = ~absent & merged.original.ne(merged.candidate)
    overlap = merged[~absent]
    table = pd.read_csv(HERE / 'data/li_missing_only_candidates.csv')
    expected_keys = set(zip(pd.to_datetime(table.source_clock), table.station_code, table.pm25_ug_m3))
    actual_keys = set(merged.loc[absent, ['time', 'station_code', 'candidate']].itertuples(index=False, name=None))
    assert expected_keys == actual_keys
    overlay = pd.concat([old, merged.loc[absent, ['time', 'station_code', 'candidate']].rename(columns={'candidate': 'original'})])
    def day_counts(frame):
        f = frame.assign(day=frame.time.dt.normalize())
        return f.groupby(['station_code', 'day']).time.nunique()
    before, after = day_counts(old), day_counts(overlay)
    added_days = int(after.ge(20).sum() - before.ge(20).sum())
    old_month = before.ge(20).groupby(level=0).sum()
    new_month = after.ge(20).groupby(level=0).sum()
    added_months = int(new_month.ge(24).sum() - old_month.ge(24).sum())
    report = json.loads((HERE / 'audit/li_candidate_audit.json').read_text())
    checks = {'overlap': len(overlap), 'equal': int(overlap.original.eq(overlap.candidate).sum()),
              'conflicts': int(conflict.sum()), 'candidate_additions': int(absent.sum()),
              'added_eligible_station_days': added_days, 'added_eligible_station_months': added_months,
              'timestamp_slots': int(valid_slots.sum()), 'original_archive_sha256': sha(archive),
              'source_sha256': sha(source)}
    assert checks['overlap'] == report['clock_tests'][1]['overlap_station_hours']
    assert checks['equal'] == report['clock_tests'][1]['exact_equal']
    assert checks['conflicts'] == report['overlap_conflicts']
    assert checks['candidate_additions'] == report['missing_only_candidate_station_hours']
    assert added_days == report['added_eligible_station_days_hypothetical'] == 0
    assert added_months == report['new_eligible_station_months_hypothetical'] == 0
    assert checks['timestamp_slots'] == report['valid_timestamps_used']
    assert checks['original_archive_sha256'] == report['original_archive_sha256']
    assert checks['source_sha256'] == report['source_sha256']
    return checks


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    check_gate()
    response = json.loads((HERE / 'evidence/cnemc_historical_probe.body').read_text(encoding='utf-8-sig'))
    gate = historical_time_gate(response, '2018-12-22 00:00', '2018-12-22 23:59:59')
    assert not gate['accepted'] and gate['rows_in_requested_period'] == 0
    gate.update({'request_date_parameter': '2018-12-22 12:00:00',
                 'response_sha256': sha(HERE / 'evidence/cnemc_historical_probe.body'),
                 'scope': 'Only the tested public request; not a claim that no historical service exists.'})
    dump(HERE / 'audit/official_historical_time_audit.json', gate)
    with zipfile.ZipFile(HERE / 'evidence/pangaea_processed_zip.body') as z:
        members = [{'path': x.filename, 'bytes': x.file_size} for x in z.infolist()]
        data = pd.read_csv(io.BytesIO(z.read('PM2.5 data in China/Homogenized_PM25_data.txt')),
                           sep=r'\s+', skiprows=4)
        geo = pd.read_csv(io.BytesIO(z.read('PM2.5 data in China/Goelocation.txt')), sep=r'\s+', skiprows=3)
    processed = {'members': members, 'daily_rows': len(data), 'station_columns': len(data.columns)-1,
                 'geolocation_rows': len(geo), 'date_min': str(data.iloc[:, 0].min()),
                 'date_max': str(data.iloc[:, 0].max()), 'columns_first': data.columns[:6].tolist(),
                 'observed_imputed_cell_flags_in_public_zip': False,
                 'raw_hourly_files_in_public_zip': False, 'admitted_as_raw': False,
                 'reason': 'Homogenized daily product, not flag-resolved raw hourly measurements.',
                 'sha256': sha(HERE / 'evidence/pangaea_processed_zip.body')}
    dump(HERE / 'audit/pangaea_payload_audit.json', processed)
    day = pd.read_csv(PREV / 'audit/raw_day_availability.csv')
    gap = day[day.maximum_station_valid_hours.lt(20)].copy()
    gap['requested_variable'] = 'Observed hourly PM2.5 mass concentration; not AQI, interpolation or reanalysis'
    gap['request_scope'] = 'National CNEMC station codes if available; prioritise Yangtze River Delta'
    gap['timezone_required'] = 'Explicit UTC or Asia/Shanghai and hour start/end convention'
    gap.to_csv(HERE / 'audit/priority_missing_dates.csv', index=False)
    inventory = []
    for path in sorted((HERE / 'evidence').glob('*.json')):
        receipt = json.loads(path.read_text(encoding='utf-8'))
        body = path.with_suffix('.body')
        if 'sha256' in receipt:
            assert body.exists() and sha(body) == receipt['sha256'], path
        inventory.append({'key': receipt['key'], 'url': receipt['url'], 'http_status': receipt.get('status'),
                          'bytes': receipt.get('bytes'), 'complete_body': receipt.get('complete_body'),
                          'error': receipt.get('error', ''), 'sha256': receipt.get('sha256', '')})
    pd.DataFrame(inventory).to_csv(HERE / 'audit/request_inventory.csv', index=False)
    prior = json.loads((PREV / 'audit/model_run_receipt.json').read_text())
    unchanged = []
    for item in prior['inputs']:
        actual = sha(item['path'])
        assert actual == item['sha256'], item['path']
        unchanged.append({**item, 'verified_unchanged': True})
    dump(HERE / 'audit/unchanged_model_inputs.json', unchanged)
    independent = verify_candidate_independently()
    result = {'completed_client_date': '2026-10-04', 'verified_utc': datetime.now(timezone.utc).isoformat(),
              'passed': True, 'time_gate_tests': 7, 'original_model_inputs_unchanged': len(unchanged),
              'request_receipts_verified': len(inventory), 'independent_candidate_recheck': independent,
              'priority_dates': len(gap), 'accepted_new_station_hours': 0,
              'added_eligible_station_days': 0, 'added_eligible_station_months': 0,
              'models_refit_this_branch': 0, 'original_inputs_modified': False,
              'recovery_resolved': False, 'bounded_search_and_audit_complete': True}
    dump(HERE / 'audit/VERIFICATION.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
