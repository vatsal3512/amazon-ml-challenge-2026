"""Union candidates (dense top-KD + lexical top-KL) and pair features. Chunked, multiprocess, resumable.
usage: python features2.py --split train|test --data DATA --out OUT [--kd 30 --kl 30 --workers 28]
Writes OUT/feat2/<split>_<country>_<start>.npz with X, q_idx, c_idx (indices into q_ids / pool_ids of the dense file).
"""
import argparse, os, sys, time, glob, logging, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import prep, pair, NAMES as _N
BASE = _N[:31]
EXTRA = ['bi_score', 'bi_rank', 'bi_gap', 'd_rank', 'l_rank', 'l_score', 'l_gap', 'n_cands', 'in_both']
NAMES = BASE + EXTRA

PREFIX_OK = os.environ.get('ER_PREFIX_OK') == '1'
_G = {}
def _init(qrec, prec):
    _G['q'], _G['p'] = qrec, prec

def _work(args):
    qi, cands = args
    a = prep(*_G['q'][qi]); out = []
    for ci in cands:
        out.append(pair(a, prep(*_G['p'][ci]), 0.0, 0, 0.0)[:31])
    return out

def load_emb(work, prefix):
    fs = sorted(glob.glob(f'{work}/emb_{prefix}_*.npy'))
    return np.concatenate([np.load(f) for f in fs])

def run(split, data, out, kd, kl, workers, chunk, countries):
    os.makedirs(f'{out}/feat2', exist_ok=True)
    rd = lambda p: pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False)
    d = f'{data}/{split}'
    s1 = rd(f'{d}/{split}_source1.tsv').set_index('entity_id')
    pool = pd.concat([rd(f'{d}/{split}_source2.tsv'), rd(f'{d}/{split}_source3.tsv')]).set_index('entity_id')
    for c in countries:
        fd, fl = f'{out}/topk_{split}_{c}.npz', f'{out}/topk_lex_{split}_{c}.npz'
        if not (os.path.exists(fd) and os.path.exists(fl)):
            logging.info(f'[{c}] missing inputs, skip'); continue
        zd, zl = np.load(fd), np.load(fl)
        q_ids, p_ids = zd['q_ids'], zd['pool_ids']
        dc, ds, lc, ls = zd['cand'], zd['score'].astype(np.float32), zl['cand'], zl['score'].astype(np.float32)
        work = f'{out}/work/{split}_{c}'
        NOEMB = not glob.glob(f'{work}/emb_p_*.npy')   # no embeddings on this machine: bi-encoder columns become 0 (unused by the France model)
        if not NOEMB:
            QE, PE = load_emb(work, 'q'), load_emb(work, 'p')
            if PREFIX_OK and len(QE) > len(q_ids):
                QE = QE[:len(q_ids)]
            assert len(QE) == len(q_ids) and len(PE) == len(p_ids), 'embedding shards do not match ids'
        qrec = list(zip(s1.loc[q_ids, 'business_name'], s1.loc[q_ids, 'business_address']))
        prec = list(zip(pool.loc[p_ids, 'business_name'], pool.loc[p_ids, 'business_address']))
        logging.info(f'[{c}] {len(q_ids)} queries')
        with Pool(workers, initializer=_init, initargs=(qrec, prec)) as pl:
            for st in range(0, len(q_ids), chunk):
                dst = f'{out}/feat2/{split}_{c}_{st:08d}.npz'
                if os.path.exists(dst): continue
                en = min(st + chunk, len(q_ids)); t0 = time.time()
                qi_l, ci_l, dr_l, lr_l, ls_l = [], [], [], [], []
                jobs = []
                for i in range(st, en):
                    dpos = {int(x): r for r, x in enumerate(dc[i])}
                    lpos = {int(x): r for r, x in enumerate(lc[i]) if x >= 0}
                    keep = list(dict.fromkeys([int(x) for x in dc[i, :kd]] + [int(x) for x in lc[i, :kl] if x >= 0]))
                    jobs.append((i, keep))
                    qi_l.append(np.full(len(keep), i, np.int32)); ci_l.append(np.array(keep, np.int32))
                    dr_l.append(np.array([dpos.get(x, 99) for x in keep], np.float32))
                    lr_l.append(np.array([lpos.get(x, 99) for x in keep], np.float32))
                    ls_l.append(np.array([ls[i, lpos[x]] if x in lpos else 0.0 for x in keep], np.float32))
                res = pl.map(_work, jobs, chunksize=200)
                B = np.array([r for rows in res for r in rows], dtype=np.float32)
                qi, ci = np.concatenate(qi_l), np.concatenate(ci_l)
                dr, lr, lsc = np.concatenate(dr_l), np.concatenate(lr_l), np.concatenate(ls_l)
                bi = np.zeros(len(qi), np.float32) if NOEMB else np.einsum('nd,nd->n', QE[qi].astype(np.float32), PE[ci].astype(np.float32))
                # per-query rank / gap / count
                order = np.lexsort((-bi, qi)); rank = np.empty(len(qi), np.float32)
                _, first, cnt = np.unique(qi[order], return_index=True, return_counts=True)
                rank[order] = np.arange(len(qi)) - np.repeat(first, cnt)
                best = np.maximum.reduceat(bi[order], first); bgap = np.empty(len(qi), np.float32)
                bgap[order] = np.repeat(best, cnt) - bi[order]
                lbest = np.maximum.reduceat(lsc[order], first); lgap = np.empty(len(qi), np.float32)
                lgap[order] = np.repeat(lbest, cnt) - lsc[order]
                ncand = np.empty(len(qi), np.float32); ncand[order] = np.repeat(cnt, cnt).astype(np.float32)
                both = ((dr < 99) & (lr < 99)).astype(np.float32)
                X = np.column_stack([B, bi, rank, bgap, dr, lr, lsc, lgap, ncand, both]).astype(np.float32)
                np.savez_compressed(dst + '.tmp.npz', X=X, q_idx=qi, c_idx=ci); os.replace(dst + '.tmp.npz', dst)
                logging.info(f'[{c}] chunk {st}:{en} saved ({len(X) / (time.time() - t0):,.0f} pairs/s, {len(X) / (en - st):.1f} cands/query)')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--kd', type=int, default=30); ap.add_argument('--kl', type=int, default=30)
    ap.add_argument('--workers', type=int, default=28); ap.add_argument('--chunk', type=int, default=50000)
    ap.add_argument('--countries', default='India,US,France')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f'{a.out}/features2_{a.split}.log')])
    run(a.split, a.data, a.out, a.kd, a.kl, a.workers, a.chunk, a.countries.split(','))
