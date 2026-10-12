"""Extract metadata tables; manually verify page layout before treating as evidence."""
import json
import pdfplumber
from common import HERE, dump

for path in sorted((HERE/'evidence').glob('*.pdf')):
    if path.with_suffix('.txt').exists():continue
    with pdfplumber.open(path) as doc:
        pages=[p.extract_text() or '' for p in doc.pages]
        path.with_suffix('.txt').write_text('\n'.join(f'PAGE {i+1}\n{t}' for i,t in enumerate(pages)),encoding='utf-8')
        candidates=[]
        for i,text in enumerate(pages):
            if ('Monitoring Network' in text or 'monitoring network' in text) and ('Start' in text or 'Trend' in text or 'Performance' in text) and ('Table ' in text):
                candidates.append({'pdf_page':i+1,'text':text,'tables':doc.pages[i].extract_tables()})
        dump(path.with_name(path.stem+'_metadata_tables.json'),candidates)
        print(path.name,[(x['pdf_page'],len(x['tables'])) for x in candidates],flush=True)
