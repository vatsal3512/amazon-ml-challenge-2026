"""Checks matching_results.tsv (and optionally candidate_pairs.tsv) against the README rules.
usage: python validate.py --matching M.tsv [--candidate C.tsv] --test-dir DIR"""
import argparse, sys, csv
import pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument('--matching', required=True); ap.add_argument('--candidate'); ap.add_argument('--test-dir', required=True)
a = ap.parse_args()
rd = lambda p: pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False)
s1 = set(rd(f'{a.test_dir}/test_source1.tsv').entity_id)
pool = set(rd(f'{a.test_dir}/test_source2.tsv').entity_id) | set(rd(f'{a.test_dir}/test_source3.tsv').entity_id)
issues = []

def check(path, col):
    df = rd(path)
    if list(df.columns) != ['source1_entity_id', col]:
        issues.append(f'{path}: columns are {list(df.columns)}, expected source1_entity_id,{col}')
        return None
    if df.source1_entity_id.duplicated().any():
        issues.append(f'{path}: {int(df.source1_entity_id.duplicated().sum())} duplicate source1_entity_id rows')
    ids = set(df.source1_entity_id)
    if ids != s1:
        issues.append(f'{path}: missing {len(s1 - ids)} S1 ids, unknown {len(ids - s1)} ids')
    out, bad_dup, bad_id, bad_prefix = {}, 0, 0, 0
    for k, v in zip(df.source1_entity_id, df[col]):
        l = [x for x in v.split(',') if x]
        if len(l) != len(set(l)): bad_dup += 1
        bad_id += sum(x not in pool for x in l)
        bad_prefix += sum(not x.startswith(('S2-', 'S3-')) for x in l)
        out[k] = set(l)
    if bad_dup: issues.append(f'{path}: {bad_dup} rows contain duplicate ids in the list')
    if bad_id: issues.append(f'{path}: {bad_id} listed ids are not in the test S2/S3 files')
    if bad_prefix: issues.append(f'{path}: {bad_prefix} ids are not S2-/S3- ids')
    print(f'{path}: {len(df)} rows, mean list length {sum(map(len, out.values())) / max(len(out), 1):.2f}, '
          f'empty rows {sum(not v for v in out.values()) / max(len(out), 1):.1%}')
    return out

m = check(a.matching, 'matched_entity_ids')
if a.candidate:
    c = check(a.candidate, 'candidate_entity_ids')
    if m is not None and c is not None:
        n = sum(len(v - c.get(k, set())) for k, v in m.items())
        if n: issues.append(f'{n} matched ids never appear in candidate_pairs (pipeline bug)')
if issues:
    print('FAIL'); [print(f'{i + 1}. {x}') for i, x in enumerate(issues)]; sys.exit(1)
print('PASS')
