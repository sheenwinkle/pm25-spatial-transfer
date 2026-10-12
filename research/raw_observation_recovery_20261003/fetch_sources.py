"""Bounded, cached public-source retrieval with byte-level provenance."""
import argparse
import hashlib
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
import requests

HERE=Path(__file__).resolve().parent
for name in ['evidence','data','audit']:(HERE/name).mkdir(exist_ok=True)


def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def fetch(key,url,method='GET',data=None,limit=32_000_000):
    out=HERE/'evidence'/f'{key}.body';receipt=HERE/'evidence'/f'{key}.json'
    if receipt.exists():return json.loads(receipt.read_text(encoding='utf-8')),out
    row={'key':key,'url':url,'method':method,'request_form':data,'retrieved_utc':datetime.now(timezone.utc).isoformat()}
    try:
        with requests.request(method,url,data=data,headers={'User-Agent':'Academic-air-quality-archive-audit/1.0'},
            timeout=(12,40),stream=True) as response:
            row.update({'status':response.status_code,'final_url':response.url,'content_type':response.headers.get('Content-Type'),
                'content_length_header':response.headers.get('Content-Length'),'etag':response.headers.get('ETag')})
            n=0
            with open(out,'wb') as f:
                for block in response.iter_content(262144):
                    if not block:continue
                    if n+len(block)>limit:raise ValueError(f'Payload exceeds {limit} byte cap; partial evidence only')
                    f.write(block);n+=len(block)
            row.update({'bytes':n,'sha256':sha(out),'complete_body':True})
    except (requests.RequestException,ValueError) as exc:
        row['error']=str(exc);row['complete_body']=False
        if out.exists():row.update({'bytes':out.stat().st_size,'sha256':sha(out)})
    dump(receipt,row);print(key,row.get('status'),row.get('bytes'),row.get('error',''),flush=True)
    return row,out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sources={
        'zenodo_6404603':'https://zenodo.org/api/records/6404603',
        'zenodo_10836299':'https://zenodo.org/api/records/10836299',
        'openaq_root':'https://openaq-data-archive.s3.amazonaws.com/?list-type=2&delimiter=/&max-keys=50',
        'casearth_645':'https://data.casearth.cn/thematic/CS/645',
        'cnemc_live_portal':'https://air.cnemc.cn:18007/',
        'cnemc_history_notice':'https://www.cnemc.cn/gzdt/wjtz/201603/t20160315_648691.shtml',
        'gnss_reorganised':'https://www.gnss-isseg.top/index.php/data/data1/',
        'eia_history':'https://eia-data.com/pm_history-html/',
        'openaq_docs':'https://docs.openaq.org/aws/about',
        'aqicn_download':'https://aqicn.org/data-platform/query/cn/',
    }
    for key,url in sources.items():fetch(key,url)


if __name__=='__main__':main()
