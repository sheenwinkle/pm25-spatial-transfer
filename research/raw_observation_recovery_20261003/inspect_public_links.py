"""Inspect saved public HTML links and relevant public app endpoint strings."""
import re
import sys
from html.parser import HTMLParser
from fetch_sources import HERE


class Links(HTMLParser):
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='script' and 'src' in a:print('SCRIPT',a['src'])
        if tag=='a' and 'href' in a and 'pangaea' in a['href']:print('LINK',a['href'])


sys.stdout.reconfigure(encoding='utf-8')
for key in ['openaq_explorer','pangaea_917557']:
    print(key);Links().feed((HERE/f'evidence/{key}.body').read_text(encoding='utf-8'))
s=(HERE/'evidence/casearth_item_script.body').read_text(encoding='utf-8')
print('CASEARTH public item endpoints')
for match in sorted(set(re.findall(r'.{0,70}(?:/api/|/dataset/|download|fileList|login|itemDetail).{0,130}',s)))[:45]:print(match)
