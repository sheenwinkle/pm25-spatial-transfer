"""Preserve supporting official metadata pages and the representativeness paper."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
SOURCES = {
    'jiangsu_county_codes_2013': 'https://jszd.stats.gov.cn/gjtjjjsdczd/tjbz/content/d0361b20-928c-48ad-ad25-3a003f4073ec.html',
    'zhejiang_yearbook_2016_table2_2': 'https://zjjcmspublic.oss-cn-hangzhou-zwynet-d01-a.internet.cloud.zj.gov.cn/jcms_files/jcms1/web3077/site/flash/tjj/Reports1/2016年统计年鉴光盘20190415/html/cn/二、人口和就业人员/2-2 各市、县总户数和总人口数(2015年底).html',
    'nsw_newcastle_network': 'https://www.epa.nsw.gov.au/Your-environment/Air/regional-air-quality/newcastle-local-air-quality-monitoring-network',
    'su_2021_representativeness': 'https://html.rhhz.net/hjkxxb/html/20210211001.htm',
}


def main():
    ledger = []
    for name, url in SOURCES.items():
        path = HERE / 'evidence' / (name + '.html')
        row = {'id': name, 'url': url, 'retrieved_date': datetime.now().date().isoformat()}
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            body = soup.get_text(' ', strip=True)
            assert len(body) > 100
            path.write_bytes(response.content)
            path.with_suffix('.txt').write_text(body, encoding='utf-8')
            row.update(status='downloaded', sha256=hashlib.sha256(response.content).hexdigest(),
                       final_url=response.url, bytes=len(response.content))
        except Exception as exc:
            row.update(status='failed', error=str(exc))
        ledger.append(row)
        print(name, row['status'], flush=True)
    (HERE / 'evidence/web_source_ledger.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
