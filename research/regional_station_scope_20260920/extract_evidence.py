"""Extract official city-specific station roles from table columns, not names."""
import csv
import json
import re
from pathlib import Path
import pdfplumber

HERE = Path(__file__).resolve().parent
CITIES = '上海 南京 无锡 常州 苏州 南通 盐城 扬州 镇江 泰州 杭州 宁波 嘉兴 湖州 绍兴 金华 舟山 台州 合肥 芜湖 马鞍山 铜陵 安庆 滁州 池州 宣城'.split()


def clean(value):
    return re.sub(r'\s+', '', value or '')


def main():
    rows = []
    with pdfplumber.open(HERE / 'evidence/china_network_2012.pdf') as doc:
        for index in [5, 10, 11, 12, 13]:
            table = doc.pages[index].extract_tables()[0]
            merged = []
            for row in table[1:]:
                if row[1]:
                    merged.append([clean(v) for v in row])
                elif merged:
                    # Several cells on page 13 are split by PDF extraction lines.
                    for j, v in enumerate(row):
                        merged[-1][j] += clean(v)
            for row in merged:
                city = row[1].removesuffix('市')
                if city not in CITIES:
                    continue
                names = [(row[3], 'control_2012')] + [(v, 'network_regular_2012') for v in row[4:]]
                names = [(n, role) for n, role in names if n]
                assert len(names) == int(row[2]), (city, row[2], names)
                rows.extend({'city_zh': city, 'station_name_zh': name, 'role': role,
                             'evidence_year': 2012, 'pdf_page': index + 1,
                             'source_id': 'china_network_2012'} for name, role in names)
    assert set(r['city_zh'] for r in rows) == set(CITIES)
    assert len({(r['city_zh'], r['station_name_zh']) for r in rows}) == len(rows)
    folder = HERE / 'data'
    folder.mkdir(exist_ok=True)
    with (folder / 'china_official_roles_2012.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (folder / 'yrd26_city_names.json').write_text(json.dumps(CITIES, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Extracted', len(rows), 'stations in', len(CITIES), 'cities')


if __name__ == '__main__':
    main()
