"""Rule-based whole-city blocking (France recall): an UNOWNED pool record goes to an S1 when the normalised core name is identical (any order,
abbreviations mapped), legal forms are compatible, house number and street agree (or the pool address is empty), and EXACTLY ONE S1 qualifies.
mode 'eval'  : precision/volume of the rule on US or India TRAIN labels (all pool records of a random sample, model ignored)
mode 'apply' : add rule matches to a submission's France rows.  usage: python fr_block.py eval US | python fr_block.py apply IN.tsv OUT.tsv"""
import sys, collections, numpy as np, pandas as pd
from rapidfuzz import fuzz
from nameops import split
from street import parse
from textnorm import NAME_MAP
MAP = dict(NAME_MAP, **{'st': 'saint', 'ste': 'sainte', 'ets': 'etablissements', 'etablissement': 'etablissements', 'cie': 'compagnie', 'frs': 'freres'})
CTRY = __import__('os').environ.get('CTRY', 'France')
rd = lambda f: pd.read_csv(f, sep='\t', dtype=str, keep_default_na=False)
def key(n):
    s = split(n)
    return (None, set()) if s is None or not s[0] else (tuple(sorted(MAP.get(w, w) for w in s[0])), s[1])
def run(s1, pool, candidates):
    by = collections.defaultdict(list); info = {}
    for q, n, ad in zip(s1.index, s1.business_name, s1.business_address):
        k, lg = key(n)
        if k: by[k].append(q); num, stn, _ = parse(ad); info[q] = (lg, num, stn)
    res = []
    for t in candidates:
        k, lg = key(pool.at[t, 'business_name'])
        if not k or k not in by: continue
        ta = pool.at[t, 'business_address'].strip(); tn, ts, _ = parse(ta) if ta else ('', '', False)
        ok = []
        for q in by[k]:
            ql, qn, qs = info[q]
            if ql and lg and not (ql & lg): continue
            if not ta or (tn and qn and tn == qn and qs and ts and fuzz.ratio(qs, ts) >= 85): ok.append(q)
        if len(ok) == 1: res.append((ok[0], t, 'empty' if not ta else 'addr'))
    return res
if __name__ != '__main__':
    pass
elif sys.argv[1] == 'eval':
    c = sys.argv[2]; s1 = rd('data/train/train_source1.tsv'); s1 = s1[s1.country == c].set_index('entity_id')
    pool = pd.concat([rd('data/train/train_source2.tsv'), rd('data/train/train_source3.tsv')]); pool = pool[pool.country == c].set_index('entity_id')
    gt = rd('data/train/train_ground_truth.tsv'); owner = {t: q for q, v in zip(gt.source1_entity_id, gt.matched_entity_ids) for t in v.split(',') if t}
    cand = pool.index[np.random.RandomState(0).rand(len(pool)) < 0.15]
    r = pd.DataFrame(run(s1, pool, cand), columns=['q', 't', 'kind']); r['y'] = [owner.get(t) == q for q, t in zip(r.q, r.t)]
    print(f'[{c} train, 15% pool sample] rule fires on {len(r)} records ({len(r) / len(cand):.3f} of pool); precision {r.y.mean():.4f}'); print(r.groupby('kind').y.agg(['size', 'mean']).round(4).to_string())
else:
    src, dst = sys.argv[2], sys.argv[3]
    s1 = rd('data/test/test_source1.tsv'); frs = s1[s1.country == CTRY].set_index('entity_id')
    pool = pd.concat([rd('data/test/test_source2.tsv'), rd('data/test/test_source3.tsv')]); pool = pool[pool.country == CTRY].set_index('entity_id')
    m = rd(src); out = dict(zip(m.source1_entity_id, m.matched_entity_ids)); used = set(x for v in out.values() for x in v.split(',') if x)
    r = run(frs, pool, [t for t in pool.index if t not in used]); add = collections.defaultdict(list)
    for q, t, kind in r: add[q].append(t)
    for q, ts in add.items(): out[q] = ','.join([x for x in out[q].split(',') if x] + ts)
    pd.DataFrame({'source1_entity_id': m.source1_entity_id, 'matched_entity_ids': [out[q] for q in m.source1_entity_id]}).to_csv(dst, sep='\t', index=False)
    kinds = collections.Counter(k for _, _, k in r); print(f'France: added {len(r)} records ({len(r) / len(frs):.4f}/S1) {dict(kinds)} to {len(add)} S1')
