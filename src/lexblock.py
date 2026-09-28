"""Lexical + address blocking: hashed TF-IDF over name tokens, name 4-grams, address tokens and house numbers.
Uses the query/pool order saved by dense_topk.py (OUT/work/<split>_<country>/ids.npz) so candidate indices align.
usage: python lexblock.py --split train|test --data DATA --out OUT [--k 50] [--workers 28]
Writes OUT/topk_lex_<split>_<country>.npz (q_ids, pool_ids, cand int32 (-1 pads), score float16); checkpoints per chunk.
"""
import argparse, os, sys, time, glob, logging, numpy as np, pandas as pd, scipy.sparse as sp
from multiprocessing import Pool
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import HashingVectorizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textnorm import name_tokens, addr_tokens, fold

NF = 1 << 23

def analyzer(rec):
    name, addr = rec.split('\x01', 1)
    t = ['n:' + x for x in name_tokens(name)]
    at = addr_tokens(addr)
    t += ['a:' + x for x in at] + ['d:' + x for x in at if x.isdigit()]
    f = ''.join(ch for ch in fold(name) if ch.isalnum())
    t += ['g:' + f[i:i + 4] for i in range(max(len(f) - 3, 0))]
    return t

HV = HashingVectorizer(analyzer=analyzer, n_features=NF, alternate_sign=False, norm=None, binary=True, dtype=np.float32)

def _hash(recs): return HV.transform(recs)

def hash_all(recs, workers):
    step = 20000
    with Pool(workers) as pl:
        parts = list(tqdm(pl.imap(_hash, [recs[i:i + step] for i in range(0, len(recs), step)]),
                          total=(len(recs) + step - 1) // step, desc='hash', mininterval=15))
    return sp.vstack(parts).tocsr()

_G = {}
def _init(Q, PT):
    _G['Q'], _G['PT'] = Q, PT

def _search(args):
    st, en, k = args
    S = (_G['Q'][st:en] @ _G['PT']).tocsr()
    n = en - st
    cand = np.full((n, k), -1, np.int32); score = np.zeros((n, k), np.float16)
    ip, ix, dt = S.indptr, S.indices, S.data
    for i in range(n):
        a, b = ip[i], ip[i + 1]
        if b == a: continue
        d = dt[a:b]; kk = min(k, b - a)
        top = np.argpartition(-d, kk - 1)[:kk] if b - a > kk else np.arange(b - a)
        top = top[np.argsort(-d[top])]
        cand[i, :kk] = ix[a:b][top]; score[i, :kk] = d[top]
    return st, cand, score

def run(split, data, out, k, workers, countries, chunk, max_df_frac, min_max_df):
    rd = lambda p: pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False)
    d = f'{data}/{split}'
    s1 = rd(f'{d}/{split}_source1.tsv').set_index('entity_id')
    pool = pd.concat([rd(f'{d}/{split}_source2.tsv'), rd(f'{d}/{split}_source3.tsv')]).set_index('entity_id')
    for c in countries:
        dst = f'{out}/topk_lex_{split}_{c}.npz'
        ids = f'{out}/work/{split}_{c}/ids.npz'
        if os.path.exists(dst): logging.info(f'[{c}] exists, skip'); continue
        if not os.path.exists(ids): logging.info(f'[{c}] no {ids}, skip'); continue
        z = np.load(ids); q_ids, p_ids = z['q_ids'], z['pool_ids']
        work = f'{out}/work/lex_{split}_{c}'; os.makedirs(work, exist_ok=True)
        t0 = time.time()
        logging.info(f'[{c}] hashing pool ({len(p_ids)}) and queries ({len(q_ids)})')
        mk = lambda df, i: [a + '\x01' + b for a, b in zip(df.loc[i, 'business_name'], df.loc[i, 'business_address'])]
        P = hash_all(mk(pool, p_ids), workers); Q = hash_all(mk(s1, q_ids), workers)
        dfc = np.asarray(P.getnnz(axis=0)).ravel()
        maxdf = max(min_max_df, int(len(p_ids) * max_df_frac))
        keep = (dfc >= 1) & (dfc <= maxdf)
        idf = np.zeros(NF, np.float32); idf[keep] = np.log(len(p_ids) / dfc[keep])
        logging.info(f'[{c}] kept {int(keep.sum())} terms (max df {maxdf})')
        def weight(M):
            M = M @ sp.diags(idf); M.eliminate_zeros()
            nrm = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()); nrm[nrm == 0] = 1
            return (sp.diags(1 / nrm) @ M).tocsr().astype(np.float32)
        P, Q = weight(P), weight(Q)
        PT = P.T.tocsr()
        del P
        starts = list(range(0, len(q_ids), chunk))
        todo = [(s, min(s + chunk, len(q_ids)), k) for s in starts if not os.path.exists(f'{work}/part_{s:08d}.npz')]
        logging.info(f'[{c}] {len(todo)}/{len(starts)} chunks to search')
        with Pool(workers, initializer=_init, initargs=(Q, PT)) as pl:
            for st, cand, score in tqdm(pl.imap_unordered(_search, todo), total=len(todo), desc=f'lex {c}', mininterval=15):
                tmp = f'{work}/part_{st:08d}.npz.tmp.npz'
                np.savez_compressed(tmp, cand=cand, score=score); os.replace(tmp, f'{work}/part_{st:08d}.npz')
        parts = [np.load(f) for f in sorted(glob.glob(f'{work}/part_*.npz'))]
        np.savez_compressed(dst + '.tmp.npz', q_ids=q_ids, pool_ids=p_ids,
                            cand=np.concatenate([x['cand'] for x in parts]), score=np.concatenate([x['score'] for x in parts]))
        os.replace(dst + '.tmp.npz', dst)
        logging.info(f'[{c}] DONE in {(time.time() - t0) / 60:.1f} min -> {dst}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=50); ap.add_argument('--workers', type=int, default=28)
    ap.add_argument('--chunk', type=int, default=1000); ap.add_argument('--countries', default='India,US,France')
    ap.add_argument('--max_df_frac', type=float, default=0.0005); ap.add_argument('--min_max_df', type=int, default=50)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f'{a.out}/lexblock_{a.split}.log')])
    run(a.split, a.data, a.out, a.k, a.workers, a.countries.split(','), a.chunk, a.max_df_frac, a.min_max_df)
