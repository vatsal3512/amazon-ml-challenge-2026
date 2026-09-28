"""France post-processing on top of a submission (steps reported separately):
 A  abbreviation-normalised descriptor filter: core words mapped through the French abbreviation map (st->saint, ets->etablissements, ...)
    before deciding a pure descriptor swap (extras only, never the top match)
 B  cross-business competition: drop an extra match whose normalised core name equals ANOTHER France S1's core name but not this S1's
 C  empty-address assignment: an empty-address France pool record, unused by anyone, whose normalised core name matches exactly one France S1,
    is given to that S1 (one owner per record)
usage: python fr_compete.py BASE_UNFILTERED.tsv OUT.tsv [steps: ABC]"""
import sys, collections, pandas as pd
from nameops import split
from textnorm import NAME_MAP
src, dst = sys.argv[1], sys.argv[2]; steps = sys.argv[3] if len(sys.argv) > 3 else 'ABC'
EXTRA_MAP = dict(NAME_MAP, **{'st': 'saint', 'ste': 'sainte', 'ets': 'etablissements', 'etablissement': 'etablissements', 'cie': 'compagnie', 'frs': 'freres'})
def core(n):
    s = split(n)
    return None if s is None else tuple(sorted(EXTRA_MAP.get(w, w) for w in s[0]))
rd = lambda f: pd.read_csv(f, sep='\t', dtype=str, keep_default_na=False)
s1 = rd('data/test/test_source1.tsv').set_index('entity_id'); frs = s1[s1.country == 'France']
pool = pd.concat([rd('data/test/test_source2.tsv'), rd('data/test/test_source3.tsv')]); pool = pool[pool.country == 'France'].set_index('entity_id')
qcore = {q: core(n) for q, n in frs.business_name.items()}
cnt = collections.Counter(); [cnt.update(set(c)) for c in qcore.values() if c]
vocab = {w for w, c in cnt.items() if c >= 30 and len(w) > 2}
by_core = collections.defaultdict(list)
for q, c in qcore.items():
    if c: by_core[c].append(q)
pc = {}
def pcore(t):
    if t not in pc: pc[t] = core(pool.at[t, 'business_name'])
    return pc[t]
m = rd(src); fr = set(frs.index); nA = nB = nC = 0; out = {}
for q, v in zip(m.source1_entity_id, m.matched_entity_ids):
    if q not in fr or not v: out[q] = v; continue
    ids = v.split(','); keep = [ids[0]]; a = qcore[q]
    for t in ids[1:]:
        b = pcore(t)
        if a and b:
            qa, tb = set(a), set(b); added, dropped = tb - qa, qa - tb
            if 'A' in steps and added and dropped and (qa & tb) and added <= vocab and dropped <= vocab: nA += 1; continue
            if 'B' in steps and b != a and any(o != q for o in by_core.get(b, ())): nB += 1; continue
        keep.append(t)
    out[q] = ','.join(keep)
if 'C' in steps:
    used = set(x for v in out.values() for x in v.split(',') if x)
    emp = pool.index[pool.business_address.str.strip() == '']
    for t in emp:
        if t in used: continue
        b = pcore(t); owners = by_core.get(b, []) if b else []
        if len(owners) == 1 and out.get(owners[0]):
            out[owners[0]] = out[owners[0]] + ',' + t; used.add(t); nC += 1
res = pd.DataFrame({'source1_entity_id': m.source1_entity_id, 'matched_entity_ids': [out[q] for q in m.source1_entity_id]})
res.to_csv(dst, sep='\t', index=False)
n = len(fr); tot = sum(len([x for x in out[q].split(',') if x]) for q in fr)
print(f'steps {steps}: A descriptor-swap removed {nA} ({nA / n:.3f}/S1), B competition removed {nB} ({nB / n:.3f}/S1), C empty-address added {nC} ({nC / n:.4f}/S1); France {tot / n:.3f}/S1')
