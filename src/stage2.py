"""Stage-2 'extension' model: re-scores every candidate with per-query context (friend's insight #3 + #5 as features).
Stage-1 probabilities for training rows are 5-fold out-of-fold (group by S1); for test they come from the full stage-1 model.
Context per candidate: rank, P, gap to top-1, 2nd-best P, confident count, name/address similarity to the top-1 record and to the
other confident candidates, same-source flags, identical-name counts among candidates, exact (legal forms kept) name == S1 name,
empty address flags, number of S1 records sharing the S1 name.
Evaluation: held-out val queries (hash % 10 == 0), exact macro F0.5 with a tuned two-threshold rule, stage-1 vs stage-2.
usage: python stage2.py --data DATA --out OUT [--model OUT/lgb/model.txt] [--test]  -> OUT/stage2/{model.txt,decision.json,test_<c>.npz}"""
import argparse, glob, os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from multiprocessing import Pool
from rapidfuzz import fuzz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featio import load_chunk
from textnorm import name_tokens, norm_addr

ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--model', default='')
ap.add_argument('--countries', default='India,US'); ap.add_argument('--threads', type=int, default=28); ap.add_argument('--test', action='store_true')
ap.add_argument('--folds', type=int, default=5); ap.add_argument('--dest', default=''); ap.add_argument('--test_countries', default=''); a = ap.parse_args()
D = a.dest or f'{a.out}/stage2'; os.makedirs(D, exist_ok=True)
T0 = time.time(); log = lambda s: print(f'[{(time.time() - T0) / 60:5.1f} min] {s}', flush=True)
stage1 = lgb.Booster(model_file=a.model or f'{a.out}/lgb/model.txt'); ROUNDS = stage1.num_trees()
rd = lambda p: pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False)
CTX = ['s2_rank', 's2_P', 's2_P1', 's2_gap1', 's2_P2', 's2_nconf', 's2_sumP', 's2_name_top1', 's2_addr_top1', 's2_src_top1', 's2_name_conf', 's2_addr_conf',
       's2_eq_q', 's2_cempty', 's2_t1empty', 's2_nsame', 's2_nconf_src', 's2_s1namecnt', 's2_nconf_same']
nk = lambda s: ' '.join(name_tokens(s, strip_suffix=False))

def idint(ids):
    ids = np.asarray(ids).astype('U16'); return np.array([int(x[3:]) for x in ids], np.int64) + np.where(np.char.startswith(ids, 'S3'), 10 ** 10, 0)

G = {}
def ctx_block(rng):
    lo, hi = rng; st, P, nm, ad, src, qeq, qcnt = G['starts'], G['P'], G['nm'], G['ad'], G['src'], G['qeq'], G['qcnt']
    out = np.zeros((st[hi] - st[lo], len(CTX)), np.float32); base = st[lo]
    for g in range(lo, hi):
        s, e = st[g], st[g + 1]; p = P[s:e]; k = e - s; conf = np.flatnonzero(p >= 0.5)
        P2 = p[1] if k > 1 else 0.0
        for i in range(k):
            r = s + i; o = out[r - base]
            j = 1 if i == 0 and k > 1 else 0
            o[0] = i; o[1] = p[i]; o[2] = p[0]; o[3] = p[0] - p[i]; o[4] = P2; o[5] = len(conf); o[6] = p.sum()
            if j != i:
                o[7] = fuzz.token_sort_ratio(nm[r], nm[s + j]) / 100
                o[8] = fuzz.token_sort_ratio(ad[r], ad[s + j]) / 100 if ad[r] and ad[s + j] else -1
                o[9] = src[r] == src[s + j]
            else:
                o[7] = o[8] = -1; o[9] = -1
            bn, ba, nsrc, nsame_c = -1.0, -1.0, 0, 0
            for c in conf:
                if c == i: continue
                rc = s + c
                bn = max(bn, fuzz.token_sort_ratio(nm[r], nm[rc]) / 100)
                if ad[r] and ad[rc]: ba = max(ba, fuzz.token_sort_ratio(ad[r], ad[rc]) / 100)
                nsrc += src[r] == src[rc]; nsame_c += nm[r] == nm[rc]
            o[10] = bn; o[11] = ba; o[12] = qeq[r]; o[13] = ad[r] == ''; o[14] = ad[s] == ''
            o[15] = sum(nm[r] == nm[s + t] for t in range(k) if t != i); o[16] = nsrc; o[17] = qcnt[r]; o[18] = nsame_c
    return out

def load(split, c, s1, pool, truth):
    z = np.load(f'{a.out}/topk_{split}_{c}.npz'); q_ids, p_ids = z['q_ids'], z['pool_ids']; Q, C, X = [], [], []
    for f in sorted(glob.glob(f'{a.out}/feat2/{split}_{c}_*.npz')):
        if f.endswith('.tmp.npz'): continue
        d = load_chunk(f); Q.append(d['q_idx']); C.append(d['c_idx']); X.append(d['X'])
    Q, C, X = np.concatenate(Q), np.concatenate(C), np.concatenate(X)
    qid, pid = q_ids[Q], p_ids[C]
    y = np.array([p in truth.get(q, ()) for q, p in zip(qid, pid)], np.int8) if truth is not None else None
    return dict(Q=Q, C=C, X=X, qid=qid, pid=pid, y=y)

def add_ctx(R, P, s1, pool, s1cnt):
    o = np.lexsort((-P, R['Q'])); inv = np.empty_like(o); inv[o] = np.arange(len(o))
    Qs = R['Q'][o]; first = np.ones(len(o), bool); first[1:] = Qs[1:] != Qs[:-1]; starts = np.append(np.flatnonzero(first), len(o))
    up = pd.unique(R['pid']); pn = pool.loc[up]; pnm = dict(zip(up, pn.business_name.map(nk))); pad = dict(zip(up, pn.business_address.map(norm_addr)))
    uq = pd.unique(R['qid']); qn = dict(zip(uq, s1.loc[uq, 'business_name'].map(nk)))
    pid_s, qid_s = R['pid'][o], R['qid'][o]
    G.update(starts=starts, P=P[o].astype(np.float32), nm=[pnm[x] for x in pid_s], ad=[pad[x] for x in pid_s], src=[x[:2] for x in pid_s],
             qeq=np.array([pnm[p] == qn[q] for p, q in zip(pid_s, qid_s)], np.float32),
             qcnt=np.log1p(np.array([s1cnt(q, qn[q]) for q in qid_s], np.float32)))
    ng = len(starts) - 1; step = 5000; blocks = [(i, min(i + step, ng)) for i in range(0, ng, step)]
    with Pool(a.threads) as pl: out = np.concatenate(pl.map(ctx_block, blocks))
    return np.hstack([R['X'], out[inv]])

def macro_tuned(P, q, y, ntrue_q):
    uq, inv = np.unique(q, return_inverse=True); tn = np.array([ntrue_q[u] for u in uq], float)
    o = np.lexsort((-P, inv)); first = np.ones(len(o), bool); first[1:] = inv[o][1:] != inv[o][:-1]
    rank = np.empty(len(P), int); starts = np.flatnonzero(first); cnt = np.diff(np.append(starts, len(o))); rank[o] = np.arange(len(o)) - np.repeat(starts, cnt)
    p1 = np.empty(len(uq)); p1[inv[o][starts]] = P[o][starts]; p1r = p1[inv]
    def macro(g, e):
        acc = ((rank == 0) & (P >= g)) | ((rank > 0) & (P >= e) & (p1r >= g))
        npd = np.bincount(inv[acc], minlength=len(uq)).astype(float); tpc = np.bincount(inv[acc & (y == 1)], minlength=len(uq)).astype(float)
        pr = np.divide(tpc, npd, out=np.zeros_like(tpc), where=npd > 0); rc = np.divide(tpc, tn, out=np.zeros_like(tpc), where=tn > 0)
        f = np.divide(1.25 * pr * rc, 0.25 * pr + rc, out=np.zeros_like(tpc), where=(pr + rc) > 0)
        return np.where(tn == 0, (npd == 0).astype(float), f).mean()
    grid = np.round(np.arange(0.05, 0.96, 0.025), 3)
    return max((macro(g, e), g, e) for g in grid for e in grid)

params = dict(objective='binary', learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              max_bin=255, num_threads=a.threads, verbose=-1, seed=1)
s1 = rd(f'{a.data}/train/train_source1.tsv').set_index('entity_id')
pool = pd.concat([rd(f'{a.data}/train/train_source2.tsv'), rd(f'{a.data}/train/train_source3.tsv')]).set_index('entity_id')
gt = rd(f'{a.data}/train/train_ground_truth.tsv'); truth = {q: set(x for x in v.split(',') if x) for q, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
ntrue = {q: len(v) for q, v in truth.items()}
parts = {}
n_test = pd.read_csv(f'{a.data}/test/test_source1.tsv', sep='\t', dtype=str, keep_default_na=False, usecols=['country']).country.value_counts()
for c in a.countries.split(','):
    R = load('train', c, s1, pool, truth); qi = idint(R['qid']); fold = (qi * 2654435761 % 10) % a.folds
    P = np.zeros(len(qi), np.float32)
    for k in range(a.folds):
        tr = fold != k
        m = lgb.train(params, lgb.Dataset(R['X'][tr], R['y'][tr]), num_boost_round=ROUNDS); P[~tr] = m.predict(R['X'][~tr], num_threads=a.threads)
    # S1-name counts on a random train subsample the size of the test country, so the count distribution matches test (US train is 2x test)
    sc = s1[s1.country == c]; sub = sc.sample(n=min(len(sc), int(n_test[c])), random_state=0); subset = set(sub.index)
    vc = sub.business_name.map(nk).value_counts().to_dict()
    s1cnt = lambda q, name, vc=vc, subset=subset: vc.get(name, 0) + (q not in subset)
    X2 = add_ctx(R, P, s1, pool, s1cnt); parts[c] = (X2, R['y'], qi, P, np.array([ntrue.get(q, 0) for q in R['qid']]), R['qid'])
    log(f'[{c}] {len(qi)} rows, OOF stage-1 + context done, X2 {X2.shape}')
val = {c: (parts[c][2] * 2654435761 % 10) == 0 for c in parts}
es = {c: (parts[c][2] * 2654435761 % 10) == 1 for c in parts}
trm = {c: ~val[c] & ~es[c] for c in parts}
cat = lambda key, msk: np.concatenate([parts[c][key][msk[c]] for c in parts])
dtr = lgb.Dataset(cat(0, trm), cat(1, trm)); des = lgb.Dataset(cat(0, es), cat(1, es), reference=dtr)
p2 = dict(params, learning_rate=0.05, num_leaves=63)
m2 = lgb.train(p2, dtr, num_boost_round=2000, valid_sets=[des], valid_names=['es'], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
m2.save_model(f'{D}/model.txt'); log(f'stage-2 trained, best iteration {m2.best_iteration}')
imp = m2.feature_importance('gain'); nx = parts[a.countries.split(',')[0]][0].shape[1] - len(CTX)
names = [f'x{i}' for i in range(nx)] + CTX; top = np.argsort(-imp)[:15]; log('top features: ' + ', '.join(f'{names[i]}={imp[i]:.0f}' for i in top))
dec = {}
for c in parts:
    X2, y, qi, P, nt, qid = parts[c]; v = val[c]; ntq = dict(zip(qi[v], nt[v]))
    full = stage1.predict(X2[v][:, :nx], num_threads=a.threads)
    b_full = macro_tuned(full, qi[v], y[v], ntq); b_oof = macro_tuned(P[v], qi[v], y[v], ntq); s2 = macro_tuned(m2.predict(X2[v], num_threads=a.threads), qi[v], y[v], ntq)
    log(f'[{c}] val macro F0.5: stage-1 full model {b_full[0]:.4f} | stage-1 OOF {b_oof[0]:.4f} | STAGE-2 {s2[0]:.4f} (gate {s2[1]}, extra {s2[2]})  gain vs full {s2[0] - b_full[0]:+.4f}')
    dec[c] = dict(gate=float(s2[1]), extra=float(s2[2]), f_s2=float(s2[0]), f_s1=float(b_full[0]))
json.dump(dec, open(f'{D}/decision.json', 'w'), indent=1)
if a.test:
    del parts
    s1 = rd(f'{a.data}/test/test_source1.tsv').set_index('entity_id')
    pool = pd.concat([rd(f'{a.data}/test/test_source2.tsv'), rd(f'{a.data}/test/test_source3.tsv')]).set_index('entity_id')
    for c in (a.test_countries or a.countries).split(','):
        R = load('test', c, s1, pool, None); P = stage1.predict(R['X'], num_threads=a.threads).astype(np.float32)
        vc = s1[s1.country == c].business_name.map(nk).value_counts().to_dict(); s1cnt = lambda q, name, vc=vc: vc.get(name, 1)
        X2 = add_ctx(R, P, s1, pool, s1cnt); P2 = m2.predict(X2, num_threads=a.threads).astype(np.float32)
        np.savez(f'{D}/test_{c}.npz', q_idx=R['Q'], c_idx=R['C'], P=P2); log(f'[{c}] test stage-2 scores saved ({len(P2)} rows)')
log('STAGE2 DONE')
