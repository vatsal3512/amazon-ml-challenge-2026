"""Assemble one submission row per test Source 1 entity: India/US rows from the India/US prediction file (predict_final.py),
France rows from france_s2.py's france_rows.tsv (no header: source1_entity_id <tab> comma-separated matches).
usage: python merge_france.py INDIA_US.tsv FRANCE_ROWS.tsv OUT.tsv [--data data]"""
import sys, pandas as pd
iu, frr, out = sys.argv[1:4]
data = sys.argv[sys.argv.index('--data') + 1] if '--data' in sys.argv else 'data'
rd = lambda f: pd.read_csv(f, sep='\t', dtype=str, keep_default_na=False)
s1 = rd(f'{data}/test/test_source1.tsv')[['entity_id', 'country']]
m = rd(iu); iu_rows = dict(zip(m.source1_entity_id, m.matched_entity_ids))
f = pd.read_csv(frr, sep='\t', header=None, names=['q', 'v'], dtype=str, keep_default_na=False); fr_rows = dict(zip(f.q, f.v))
rows = [(q, fr_rows.get(q, '') if c == 'France' else iu_rows.get(q, '')) for q, c in zip(s1.entity_id, s1.country)]
pd.DataFrame(rows, columns=['source1_entity_id', 'matched_entity_ids']).to_csv(out, sep='\t', index=False)
print(f'wrote {out}: {len(rows)} rows ({sum(c == "France" for c in s1.country)} France rows from {frr})')
