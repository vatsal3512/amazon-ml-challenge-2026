"""Name/number 'operation' profile of pairs: which edit turns the S1 record into the pool record.
Country-independent generator => true pairs in France should show the same operation rates per S1 as US/India true pairs.
Excess of an operation among France accepted pairs = sibling businesses (false matches).
usage: python nameops.py truth            -> rates on train true pairs (India, US)
       python nameops.py SUB.tsv[.gz]     -> rates on accepted test pairs per country; writes nameops_fr_pairs.tsv (France pairs + op)"""
import sys, re, unicodedata, numpy as np, pandas as pd
from rapidfuzz import fuzz
from street import parse

LEGAL = {'pvt', 'private', 'ltd', 'limited', 'llp', 'llc', 'inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 'plc', 'lp', 'pc', 'pllc',
         'sarl', 'sasu', 'sas', 'eurl', 'sa', 'sci', 'snc', 'scop', 'gmbh', 'cie', 'compagnie', 'ltda', 'selarl', 'sca', 'scs', 'gie'}
STOP = {'de', 'des', 'du', 'la', 'le', 'les', 'l', 'd', 'et', 'and', 'of', 'the', 'fils', 'freres', 'frs', 'sons', 'brothers', 'bros', 'amp'}
TOK = re.compile(r'[a-z0-9]+')

def fold(s):
    s = s.replace('�', '').lower()
    s = unicodedata.normalize('NFKD', s); s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'\b([a-z])\.(?=[a-z]\.)', r'\1', s)             # s.a.s. -> sas. ; e.u.r.l. -> eurl.
    s = re.sub(r'\b([a-z])\.([a-z])\.?\b', r'\1\2', s)
    return s

def split(name):
    s = fold(name)
    if re.search(r'\b(dba|trading as|t/a|aka)\b', s) or re.search(r'\.(com|fr|net|org|in|co)\b', s): return None
    t = TOK.findall(s.replace('&', ' '))
    return [x for x in t if x not in LEGAL and x not in STOP], set(x for x in t if x in LEGAL)

def op(qn, tn):
    a, b = split(qn), split(tn)
    if a is None or b is None: return 'dba/domain'
    (qc, ql), (tc, tl) = a, b
    if not tc: return 'no-core'
    used, added = set(), 0
    for x in tc:
        j = next((i for i, y in enumerate(qc) if i not in used and (x == y or (len(x) > 3 and fuzz.ratio(x, y) >= 75) or (len(x) >= 3 and (y.startswith(x) or x.startswith(y))))), None)
        if j is None:
            j2 = next((i for i, y in enumerate(qc) if i not in used and x in ''.join(qc)), None)
            if j2 is None: added += 1; continue
            j = j2
        used.add(j)
    missing = len(qc) - len(used)
    core = 'core=' if added == 0 and missing == 0 else ('core-del' if added == 0 else ('core+add' if missing == 0 else 'core-swap'))
    leg = 'legal-swap' if ql and tl and not (ql & tl) else 'legal-ok'
    return f'{core} {leg}'

def numop(qa, ta):
    if not ta.strip(): return 'addr-empty'
    n1, n2 = parse(qa)[0], parse(ta)[0]
    return 'num?' if not n1 or not n2 else ('num=' if n1 == n2 else 'num!=')

rd = lambda f: pd.read_csv(f, sep='\t', dtype=str, keep_default_na=False)
def profile(pairs, s1, pool, nS1, label):
    ops = [(op(s1.at[q, 'business_name'], pool.at[t, 'business_name']), numop(s1.at[q, 'business_address'], pool.at[t, 'business_address'])) for q, t in pairs]
    df = pd.DataFrame(ops, columns=['name', 'num'])
    tab = (df.groupby(['name', 'num']).size() / nS1).unstack(fill_value=0).round(4)
    print(f'\n[{label}] {len(pairs)} pairs from {nS1} S1: operations per S1 (rows name op, cols number op)'); print(tab.to_string())
    return df

if __name__ != '__main__':
    pass
elif sys.argv[1] == 'truth':
    s1 = rd('data/train/train_source1.tsv').set_index('entity_id'); pool = pd.concat([rd('data/train/train_source2.tsv'), rd('data/train/train_source3.tsv')]).set_index('entity_id')
    gt = rd('data/train/train_ground_truth.tsv'); gt['country'] = s1.loc[gt.source1_entity_id, 'country'].values
    for c in ('India', 'US'):
        x = gt[gt.country == c].sample(20000, random_state=0)
        profile([(q, t) for q, v in zip(x.source1_entity_id, x.matched_entity_ids) for t in v.split(',') if t], s1, pool, len(x), f'{c} TRUE')
else:
    s1 = rd('data/test/test_source1.tsv').set_index('entity_id'); pool = pd.concat([rd('data/test/test_source2.tsv'), rd('data/test/test_source3.tsv')]).set_index('entity_id')
    m = rd(sys.argv[1]); m['country'] = s1.loc[m.source1_entity_id, 'country'].values
    for c in ('US', 'France'):
        x = m[m.country == c].sample(20000, random_state=0)
        pr = [(q, t) for q, v in zip(x.source1_entity_id, x.matched_entity_ids) for t in v.split(',') if t]
        df = profile(pr, s1, pool, len(x), f'{c} ACCEPTED')
        if c == 'France':
            df['q'] = [p[0] for p in pr]; df['t'] = [p[1] for p in pr]; df.to_csv('nameops_fr_pairs.tsv', sep='\t', index=False)
