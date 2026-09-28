"""Pair features for (S1 query, candidate). Every string feature is computed on raw and normalized text.
Chunked, multiprocess, and cached to disk so a crash never restarts the job.
usage: python features.py --split train|test --data DATA --out OUT [--k 50] [--workers 28]
Writes OUT/feat/<split>_<country>_<chunk>.npz  (X float32, q_idx int32, c_idx int32) for each chunk of queries.
"""
import argparse, os, sys, time, logging, numpy as np, pandas as pd
from multiprocessing import Pool
from tqdm.auto import tqdm
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textnorm import name_tokens, addr_tokens, fold

NAMES = ['n_ratio', 'n_partial', 'n_tsort', 'n_tset', 'n_jw', 'n_jac_tok', 'n_jac_c3',
         'nn_ratio', 'nn_tset', 'nn_jw', 'nn_jac_tok', 'nn_jac_c3', 'nn_exact',
         'a_ratio', 'a_partial', 'a_tsort', 'a_jac_tok', 'a_jac_c3',
         'an_ratio', 'an_tset', 'an_jac_tok', 'an_jac_c3', 'an_exact',
         'num_jac', 'num_first_eq', 'num_any', 'both_addr', 'len_n1', 'len_n2', 'len_a1', 'len_a2',
         'bi_score', 'bi_rank', 'bi_gap']


def jac(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def c3(s): return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else set()
def nums(toks): return [t for t in toks if t.isdigit()]

def prep(name, addr):
    nt, at = name_tokens(name), addr_tokens(addr)
    nraw, araw = fold(name), fold(addr)
    nn, an = ' '.join(nt), ' '.join(at)
    return dict(nraw=nraw, araw=araw, nn=nn, an=an, nts=set(nt), ats=set(at), nnum=nums(at),
                c3n=c3(nraw), c3a=c3(araw), c3nn=c3(nn), c3an=c3(an), n_t=set(nraw.split()), a_t=set(araw.split()))

def pair(p, q, score, rank, gap):
    a, b = p, q
    n1n, n2n = a['nn'], b['nn']
    a1, a2 = a['an'], b['an']
    na, nb = a['nnum'], b['nnum']
    return [
        fuzz.ratio(a['nraw'], b['nraw']) / 100, fuzz.partial_ratio(a['nraw'], b['nraw']) / 100,
        fuzz.token_sort_ratio(a['nraw'], b['nraw']) / 100, fuzz.token_set_ratio(a['nraw'], b['nraw']) / 100,
        JaroWinkler.similarity(a['nraw'], b['nraw']), jac(a['n_t'], b['n_t']), jac(a['c3n'], b['c3n']),
        fuzz.ratio(n1n, n2n) / 100, fuzz.token_set_ratio(n1n, n2n) / 100, JaroWinkler.similarity(n1n, n2n),
        jac(a['nts'], b['nts']), jac(a['c3nn'], b['c3nn']), float(n1n == n2n and n1n != ''),
        fuzz.ratio(a['araw'], b['araw']) / 100, fuzz.partial_ratio(a['araw'], b['araw']) / 100,
        fuzz.token_sort_ratio(a['araw'], b['araw']) / 100, jac(a['a_t'], b['a_t']), jac(a['c3a'], b['c3a']),
        fuzz.ratio(a1, a2) / 100, fuzz.token_set_ratio(a1, a2) / 100, jac(a['ats'], b['ats']),
        jac(a['c3an'], b['c3an']), float(a1 == a2 and a1 != ''),
        jac(set(na), set(nb)), float(bool(na) and bool(nb) and na[0] == nb[0]), float(bool(set(na) & set(nb))),
        float(bool(a1) and bool(a2)), len(n1n), len(n2n), len(a1), len(a2), score, rank, gap]

_G = {}
def _init(qrec, prec):
    _G['q'], _G['p'] = qrec, prec

def _work(args):
    qi, cands, scores = args
    a = prep(*_G['q'][qi]); out = []
    for r, (ci, s) in enumerate(zip(cands, scores)):
        out.append(pair(a, prep(*_G['p'][ci]), float(s), r, float(scores[0] - s)))
    return out

def run(split, data, out, k, workers, chunk, countries):
    os.makedirs(f'{out}/feat', exist_ok=True)
    rd = lambda p: pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False)
    d = f'{data}/{split}'
    s1 = rd(f'{d}/{split}_source1.tsv').set_index('entity_id')
    pool = pd.concat([rd(f'{d}/{split}_source2.tsv'), rd(f'{d}/{split}_source3.tsv')]).set_index('entity_id')
    for c in countries:
        f = f'{out}/topk_{split}_{c}.npz'
        if not os.path.exists(f):
            logging.info(f'[{c}] no {f}, skipping'); continue
        z = np.load(f); q_ids, p_ids = z['q_ids'], z['pool_ids']
        cand, score = z['cand'][:, :k], z['score'][:, :k].astype(np.float32)
        qrec = list(zip(s1.loc[q_ids, 'business_name'], s1.loc[q_ids, 'business_address']))
        prec = list(zip(pool.loc[p_ids, 'business_name'], pool.loc[p_ids, 'business_address']))
        logging.info(f'[{c}] {len(q_ids)} queries x {cand.shape[1]} cands')
        with Pool(workers, initializer=_init, initargs=(qrec, prec)) as pl:
            for st in range(0, len(q_ids), chunk):
                dst = f'{out}/feat/{split}_{c}_{st:08d}.npz'
                if os.path.exists(dst):
                    continue
                en = min(st + chunk, len(q_ids)); t0 = time.time()
                jobs = [(i, cand[i], score[i]) for i in range(st, en)]
                res = pl.map(_work, jobs, chunksize=200)
                X = np.array([r for rows in res for r in rows], dtype=np.float32)
                qi = np.repeat(np.arange(st, en, dtype=np.int32), cand.shape[1])
                ci = cand[st:en].reshape(-1).astype(np.int32)
                np.savez_compressed(dst + '.tmp.npz', X=X, q_idx=qi, c_idx=ci); os.replace(dst + '.tmp.npz', dst)
                logging.info(f'[{c}] chunk {st}:{en} saved ({len(X) / (time.time() - t0):,.0f} pairs/s)')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=50); ap.add_argument('--workers', type=int, default=28)
    ap.add_argument('--chunk', type=int, default=50000); ap.add_argument('--countries', default='India,US,France')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f'{a.out}/features_{a.split}.log')])
    run(a.split, a.data, a.out, a.k, a.workers, a.chunk, a.countries.split(','))
