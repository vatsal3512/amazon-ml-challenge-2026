"""Additional pair features aligned with feat2 chunks: transliteration-skeleton similarity, address-component / state match,
empty-address flags, and contention (how many other S1 records also retrieve this candidate, and how strongly).
usage: python features_add.py --split train|test --out OUT --full FULL_DIR [--k 8 --workers 24 --countries India,US]
  OUT  : dir with topk_<split>_<c>.npz + feat2/ (the query lists the pairs come from)
  FULL : dir with topk_<split>_<c>.npz covering ALL S1 queries (train: out_full; test: same as OUT)
Writes OUT/feat_add/<same chunk names> with X (n, len(ADD_NAMES)).
"""
import argparse, os, sys, glob, time, logging, re, numpy as np, pandas as pd
from multiprocessing import Pool
from rapidfuzz import fuzz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translit import skeleton

SUFFIX_SK = {'prvt', 'lmtd', 'ltd', 'pvt', 'inks', 'inkrprtd', 'lk', 'llk', 'kmpni', 'kmpn', 'kri', 'ko', 'pk', 'llp', 'srl', 'sas',
             'sasu', 'ers', 'sk', 'pk', 'in', 'ink'}
ADD_NAMES = ['sk_n_ratio', 'sk_n_tset', 'sk_n_c3', 'sk_a_ratio', 'sk_a_tset', 'comp_a', 'comp_b', 'state_sim', 'state_eq',
             'a1_empty', 'a2_empty', 'n2_nonlatin', 'a2_nonlatin', 'sk_n_exact',
             'c_nclaim', 'c_rank', 'c_gap', 'c_gap_next']
INDIA_ST = {'ap': 'andhra pradesh', 'ar': 'arunachal pradesh', 'as': 'assam', 'br': 'bihar', 'cg': 'chhattisgarh', 'ct': 'chhattisgarh',
            'ga': 'goa', 'gj': 'gujarat', 'hr': 'haryana', 'hp': 'himachal pradesh', 'jh': 'jharkhand', 'jk': 'jammu kashmir',
            'ka': 'karnataka', 'kl': 'kerala', 'mp': 'madhya pradesh', 'mh': 'maharashtra', 'mn': 'manipur', 'ml': 'meghalaya',
            'mz': 'mizoram', 'nl': 'nagaland', 'od': 'odisha', 'or': 'odisha', 'pb': 'punjab', 'rj': 'rajasthan', 'sk': 'sikkim',
            'tn': 'tamil nadu', 'tg': 'telangana', 'ts': 'telangana', 'tr': 'tripura', 'up': 'uttar pradesh', 'uk': 'uttarakhand',
            'ut': 'uttarakhand', 'wb': 'west bengal', 'dl': 'delhi', 'ch': 'chandigarh', 'py': 'puducherry', 'la': 'ladakh'}
_ST_SK = {k: skeleton(v).replace(' ', '') for k, v in INDIA_ST.items()}
_NONLAT = re.compile(r'[^\x00-ɏ]')
_cache = {}


def sk(s):
    r = _cache.get(s)
    if r is None:
        if len(_cache) > 400000:
            _cache.clear()
        r = _cache[s] = skeleton(s)
    return r


def name_key(s):
    t = [w for w in sk(s).split() if w not in SUFFIX_SK]
    return ' '.join(t) if t else sk(s)


def comps(addr):
    return [c for c in (sk(x).replace(' ', '') for x in addr.split(',')) if c]


def state_of(addr):
    parts = [p.strip() for p in addr.split(',') if p.strip()]
    if not parts:
        return ''
    last = parts[-1]
    if len(last) <= 3 and last.lower() in INDIA_ST:
        return _ST_SK[last.lower()]
    return sk(last).replace(' ', '')


def c3(s):
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else set()


def jac(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def comp_frac(A, B):
    if not A or not B:
        return 0.0
    return sum(any(fuzz.ratio(x, y) >= 80 for y in B) for x in A) / len(A)


def pair_feats(n1, a1, n2, a2):
    k1, k2 = name_key(n1), name_key(n2)
    s1, s2 = sk(a1), sk(a2)
    A, B = comps(a1), comps(a2)
    st1, st2 = state_of(a1), state_of(a2)
    e1, e2 = float(not a1.strip()), float(not a2.strip())
    both = not (e1 or e2)
    k1n, k2n = k1.replace(' ', ''), k2.replace(' ', '')
    return [fuzz.ratio(k1n, k2n) / 100, fuzz.token_set_ratio(k1, k2) / 100, jac(c3(k1n), c3(k2n)),
            fuzz.ratio(s1, s2) / 100 if both else -1.0, fuzz.token_set_ratio(s1, s2) / 100 if both else -1.0,
            comp_frac(A, B) if both else -1.0, comp_frac(B, A) if both else -1.0,
            fuzz.ratio(st1, st2) / 100 if st1 and st2 else -1.0, float(bool(st1) and st1 == st2),
            e1, e2, float(bool(_NONLAT.search(n2))), float(bool(_NONLAT.search(a2))), float(k1n == k2n and k1n != '')]


_G = {}


def _init(q, p):
    _G['q'], _G['p'] = q, p


def _work(args):
    qi, cs = args
    n1, a1 = _G['q'][qi]
    return [pair_feats(n1, a1, *_G['p'][ci]) for ci in cs]


def idint(ids):
    ids = np.asarray(ids).astype('U16')
    return np.array([int(x[3:]) for x in ids], np.int64) + np.where(np.char.startswith(ids, 'S3'), 10 ** 10, 0)


def contention_table(full_file, k):
    """Per (S1 id, pool idx) pair in the top-k lists of ALL S1 queries: n claimants, my rank among claimants, score gaps."""
    z = np.load(full_file)
    q = idint(z['q_ids'])
    cand = z['cand'][:, :k]
    sc = z['score'][:, :k].astype(np.float32)
    Q = np.repeat(q, cand.shape[1])
    C = cand.reshape(-1).astype(np.int64)
    S = sc.reshape(-1)
    o = np.lexsort((-S, C))
    Q, C, S = Q[o], C[o], S[o]
    first = np.ones(len(C), bool)
    first[1:] = C[1:] != C[:-1]
    gid = np.cumsum(first) - 1
    starts = np.flatnonzero(first)
    cnt = np.diff(np.append(starts, len(C)))
    rank = np.arange(len(C)) - starts[gid]
    best = S[starts][gid]
    second = np.where(cnt[gid] > 1, S[np.minimum(starts + 1, len(C) - 1)][gid], -1.0)
    gap = np.where(rank == 0, S - second, S - best)             # winner: margin over runner-up; others: deficit to the winner
    nxt = np.zeros(len(C), np.float32)
    nxt[:-1] = S[1:]
    same_next = np.append(~first[1:], False)
    nxt = np.where(same_next, S - nxt, S)                       # margin over the next-weaker claimant
    key = (Q << 24) | C
    oo = np.argsort(key)
    return key[oo], np.column_stack([cnt[gid], rank, gap, nxt]).astype(np.float32)[oo]


def run(split, out, full, k, workers, countries, data):
    os.makedirs(f'{out}/feat_add', exist_ok=True)
    rd = lambda p: pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False)
    d = f'{data}/{split}'
    s1 = rd(f'{d}/{split}_source1.tsv').set_index('entity_id')
    pool = pd.concat([rd(f'{d}/{split}_source2.tsv'), rd(f'{d}/{split}_source3.tsv')]).set_index('entity_id')
    for c in countries:
        z = np.load(f'{out}/topk_{split}_{c}.npz')
        q_ids, p_ids = z['q_ids'], z['pool_ids']
        qi_int = idint(q_ids)
        ckey, cval = contention_table(f'{full}/topk_{split}_{c}.npz', k)
        logging.info(f'[{c}] contention table {len(ckey)} pairs')
        qrec = list(zip(s1.loc[q_ids, 'business_name'], s1.loc[q_ids, 'business_address']))
        prec = list(zip(pool.loc[p_ids, 'business_name'], pool.loc[p_ids, 'business_address']))
        with Pool(workers, initializer=_init, initargs=(qrec, prec)) as pl:
            for f in sorted(glob.glob(f'{out}/feat2/{split}_{c}_*.npz')):
                dst = f'{out}/feat_add/{os.path.basename(f)}'
                if os.path.exists(dst):
                    continue
                t0 = time.time()
                dd = np.load(f)
                qi, ci = dd['q_idx'], dd['c_idx']
                bounds = np.flatnonzero(np.diff(qi)) + 1
                qs, cs = np.split(qi, bounds), np.split(ci, bounds)
                res = pl.map(_work, [(int(a[0]), b.tolist()) for a, b in zip(qs, cs)], chunksize=200)
                Xs = np.array([r for rows in res for r in rows], dtype=np.float32)
                key = (qi_int[qi] << 24) | ci.astype(np.int64)
                pos = np.searchsorted(ckey, key)
                pos[pos >= len(ckey)] = 0
                hit = ckey[pos] == key
                cont = np.where(hit[:, None], cval[pos], np.zeros(4, np.float32)).astype(np.float32)
                X = np.hstack([Xs, cont])
                assert X.shape[1] == len(ADD_NAMES) and len(X) == len(qi)
                np.savez_compressed(dst + '.tmp.npz', X=X)
                os.replace(dst + '.tmp.npz', dst)
                logging.info(f'[{c}] {os.path.basename(f)} ({len(X) / (time.time() - t0):,.0f} pairs/s, in claim table {hit.mean():.1%})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--full', required=True)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--countries', default='India,US')
    ap.add_argument('--data', default='data')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f'{a.out}/features_add_{a.split}.log')])
    run(a.split, a.out, a.full, a.k, a.workers, a.countries.split(','), a.data)
