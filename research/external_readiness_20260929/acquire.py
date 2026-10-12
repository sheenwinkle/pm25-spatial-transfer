"""Archive public catalogues and primary documents, without reading test metrics."""
import json
import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from urllib.parse import urlparse
from common import HERE, digest, dump, now

SOURCES = {
    'cnemc_zenodo_catalogue': 'https://zenodo.org/api/records/10836299',
    'quotsoft_documentation': 'https://quotsoft.net/air/',
    'sa_netley_catalogue': 'https://data.sa.gov.au/data/api/3/action/package_show?id=netley-air-quality-monitoring-station-particle-data',
    'sa_air_search': 'https://data.sa.gov.au/data/api/3/action/package_search?q=air%20quality&rows=100',
    'wa_official_landing': 'https://www.wa.gov.au/service/environment/environment-information-services/air-quality',
    'mee_station_changes_2022': 'https://www.mee.gov.cn/xxgk2018/xxgk/xxgk13/202301/t20230117_1013520.html',
    'jiangsu_network_changes_2022': 'https://sthjt.jiangsu.gov.cn/art/2022/7/26/art_83601_10553588.html',
}

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ledger_path = HERE/'evidence/source_ledger.json'
    ledger = json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else []
    sources = dict(SOURCES)
    if len(sys.argv)>1:
        sources.update(json.loads(open(sys.argv[1],encoding='utf-8').read()))
    for key, url in sources.items():
        for prior in ledger:
            if prior['id']==key and urlparse(url).path.lower().endswith('.pdf') and not prior.get('path','').endswith('.pdf'):
                prior.update(status='failed', error='PDF URL returned HTML, not a verified document')
        if any(r['id']==key and r['status']=='downloaded' for r in ledger):
            continue
        row = {'id':key, 'url':url, 'retrieved_at':now()}
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            content = r.content
            if urlparse(url).path.lower().endswith('.pdf') and not content.startswith(b'%PDF'):
                raise ValueError('PDF URL returned non-PDF content')
            suffix = '.pdf' if content.startswith(b'%PDF') else '.json' if ('json' in r.headers.get('Content-Type','') or content[:1] in [b'{',b'[']) else '.html'
            path = HERE/'evidence'/f'{key}{suffix}'
            path.write_bytes(content)
            row.update(status='downloaded', final_url=r.url, bytes=len(content), sha256=digest(path), path=path.name, content_type=r.headers.get('Content-Type'))
            if suffix=='.html':
                soup=BeautifulSoup(content,'html.parser')
                (path.with_suffix('.txt')).write_text(soup.get_text('\n',strip=True),encoding='utf-8')
                links=[{'text':a.get_text(' ',strip=True),'url':urljoin(r.url,a['href'])} for a in soup.select('a[href]')]
                dump(path.with_name(key+'_links.json'),links)
            if suffix=='.json':
                data=r.json()
                if key=='cnemc_zenodo_catalogue':
                    print(key, json.dumps({'links':data.get('links'),'files':[(x['key'],x['size']) for x in data.get('files',[])]}),flush=True)
                elif key=='sa_air_search':
                    print(key,[(x['name'],len(x['resources'])) for x in data.get('result',{}).get('results',[])],flush=True)
                elif key=='sa_netley_catalogue':
                    print(key,[(x['name'],x['url']) for x in data.get('result',{}).get('resources',[])],flush=True)
        except Exception as error:
            row.update(status='failed', error=str(error))
        ledger.append(row)
        dump(ledger_path,ledger)
        print(key,row['status'],flush=True)

if __name__=='__main__':main()
