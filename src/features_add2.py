"""Street / house-number pair features, aligned with feat2 chunks (written to OUT/feat_add2/<same names>).
usage: python features_add2.py --split train|test --out OUT --countries India,US [--workers 16] [--data data]
Columns: st_ratio (street-name similarity, -1 unknown), st_num_eq (1/0, -1 unknown), st_match (street>=0.85 and number equal),
st_both (both addresses have a parsed street), st_num_diff (log1p |number difference|, -1 unknown)."""
import argparse, os, sys, glob, time, logging, numpy as np, pandas as pd
from multiprocessing import Pool
from rapidfuzz import fuzz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from street import parse

ST_NAMES = ['st_ratio', 'st_num_eq', 'st_match', 'st_both', 'st_num_diff']
_cache = {}


def cparse(a):
    r = _cache.get(a)
    if r is None:
        if len(_cache) > 300000:
            _cache.clear()
        r = _cache[a] = parse(a)
    return r


def pair_feats(a1, a2):
    n1, s1, h1 = cparse(a1)
    n2, s2, h2 = cparse(a2)
    both = h1 and h2 and bool(s1) and bool(s2)
    sr = fuzz.ratio(s1, s2) / 100.0 if both else -1.0
    known = bool(n1) and bool(n2)
    ne = (1.0 if n1 == n2 else 0.0) if known else -1.0
    nd = -1.0
    if known and n1.isdigit() and n2.isdigit():
        nd = float(np.log1p(abs(int(n1) - int(n2))))
    return [sr, ne, float(both and sr >= 0.85 and ne == 1.0), float(both), nd]


_G = {}


def _init(q, p):
    _G['q'], _G['p'] = q, p


def _work(args):
    qi, cs = args
    a1 = _G['q'][qi]
    return [pair_feats(a1, _G['p'][ci]) for ci in cs]


def run(split, out, countries, workers, data):
    os.makedirs(f'{out}/feat_add2', exist_ok=True)
    rd = lambda p: pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False)
    d = f'{data}/{split}'
    s1 = rd(f'{d}/{split}_source1.tsv').set_index('entity_id')
    pool = pd.concat([rd(f'{d}/{split}_source2.tsv'), rd(f'{d}/{split}_source3.tsv')]).set_index('entity_id')
    for c in countries:
        z = np.load(f'{out}/topk_{split}_{c}.npz')
        q_ids, p_ids = z['q_ids'], z['pool_ids']
        qrec = s1.loc[q_ids, 'business_address'].tolist()
        prec = pool.loc[p_ids, 'business_address'].tolist()
        with Pool(workers, initializer=_init, initargs=(qrec, prec)) as pl:
            for f in sorted(glob.glob(f'{out}/feat2/{split}_{c}_*.npz')):
                if f.endswith('.tmp.npz'):
                    continue
                dst = f'{out}/feat_add2/{os.path.basename(f)}'
                if os.path.exists(dst):
                    continue
                t0 = time.time()
                dd = np.load(f)
                qi, ci = dd['q_idx'], dd['c_idx']
                bounds = np.flatnonzero(np.diff(qi)) + 1
                qs, cs = np.split(qi, bounds), np.split(ci, bounds)
                res = pl.map(_work, [(int(a[0]), b.tolist()) for a, b in zip(qs, cs)], chunksize=200)
                X = np.array([r for rows in res for r in rows], dtype=np.float32)
                assert X.shape == (len(qi), len(ST_NAMES))
                np.savez_compressed(dst + '.tmp.npz', X=X)
                os.replace(dst + '.tmp.npz', dst)
                logging.info(f'[{c}] {os.path.basename(f)} ({len(X) / (time.time() - t0):,.0f} pairs/s)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--countries', default='India,US')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--data', default='data')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f'{a.out}/features_add2_{a.split}.log')])
    run(a.split, a.out, a.countries.split(','), a.workers, a.data)
