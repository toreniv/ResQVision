import csv
import json
import pathlib
import sys

SEARCH_DIRS = ['outputs', 'output', '.']
OUT_DIR = pathlib.Path('frontend/public/data')
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = ['benchmark_results.csv', 'risk_ranking.csv', 'attention_stats.csv']

for filename in FILES:
    found = None
    for d in SEARCH_DIRS:
        p = pathlib.Path(d) / filename
        if p.exists():
            found = p
            break
    if not found:
        print(f'[MISSING] {filename} not found in {SEARCH_DIRS}', file=sys.stderr)
        continue
    with open(found, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames
    out_path = OUT_DIR / filename.replace('.csv', '.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    print(f'[OK] {found} -> {out_path} ({len(rows)} rows)')
    print(f'     Columns: {columns}')
