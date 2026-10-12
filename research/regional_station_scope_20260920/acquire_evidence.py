"""Archive primary documents used for geography and historical station roles."""
import hashlib
import json
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
SOURCES = {
    'yrd_plan_2016': 'https://www.ndrc.gov.cn/xxgk/zcfb/ghwb/201606/W020190905497826154295.pdf',
    'china_network_2012': 'https://www.mee.gov.cn/gkml/hbb/bwj/201204/W020221214657415435410.pdf',
    'qld_nepm_2023': 'https://www.qld.gov.au/__data/assets/pdf_file/0032/68657/air-monitoring-report-2023.pdf',
    'nsw_nepm_2022': 'https://www.environment.nsw.gov.au/sites/default/files/2025-02/nsw-air-quality-monitoring-report-2022-NEPM-250027.pdf',
    'nsw_nepm_2019': 'https://www.environment.nsw.gov.au/sites/default/files/national-environment-protection-measure-ambient-air-quality-nsw-compliance-report-2019-210093.pdf',
    'vic_nepm_2015': 'https://www.epa.vic.gov.au/-/media/epa/files/publications/1632-1.pdf',
    'vic_nepm_2021': 'https://www.epa.vic.gov.au/sites/default/files/epa/publications/2052_nepm_compliance_air-quality-report.pdf',
    'admin_codes_2012': 'https://hudong.moe.gov.cn/srcsite/A16/s3342/201203/W020220704498224866735.pdf',
}


def main():
    folder = HERE / 'evidence'
    folder.mkdir(parents=True, exist_ok=True)
    ledger = []
    for key, url in SOURCES.items():
        path = folder / (key + '.pdf')
        row = {'id': key, 'url': url, 'retrieved_date': '2026-09-20'}
        try:
            if not path.exists():
                r = requests.get(url, timeout=90)
                r.raise_for_status()
                if not r.content.startswith(b'%PDF'):
                    raise ValueError('Response is not a PDF')
                path.write_bytes(r.content)
            row.update(status='downloaded', bytes=path.stat().st_size,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except Exception as e:
            row.update(status='failed', error=str(e))
        ledger.append(row)
        print(key, row['status'], flush=True)
        (folder / 'source_ledger.json').write_text(json.dumps(ledger, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
