"""France: self-trained stage 1 (no text-model features) + stage-2 context model trained on India/US out-of-fold rows. Validated on the US->India
rehearsal (xfer_s2.py: self-training 0.9136 -> +stage 2 0.9180, label-free thresholds). Thresholds tuned on India/US held-out queries.
usage: python france_s2.py --out out_bi --dest out_bi/france_s2   -> DEST/france_rows.tsv (one owner per pool record)"""
import argparse, glob, os, sys, numpy as np, pandas as pd, lightgbm as lgb
from multiprocessing import Pool
from rapidfuzz import fuzz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featio import load_chunk, feature_names
from textnorm import name_tokens, norm_addr
ap = argparse.ArgumentParser(); ap.add_argument('--out', default='out_bi'); ap.add_argument('--data', default='data'); ap.add_argument('--dest', default='out_bi/france_s2')
ap.add_argument('--drop', default='bi_score,bi_rank,bi_gap,ce_logit,ce_rank,ce_gap_best,ce_margin,ce_mean_q'); ap.add_argument('--threads', type=int, default=12)
a = ap.parse_args(); os.makedirs(a.dest, exist_ok=True)
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


NAMES = feature_names(a.out); cols = [i for i, n in enumerate(NAMES) if n not in set(a.drop.split(','))]
s1 = rd(f'{a.data}/train/train_source1.tsv').set_index('entity_id')
pool = pd.concat([rd(f'{a.data}/train/train_source2.tsv'), rd(f'{a.data}/train/train_source3.tsv')]).set_index('entity_id')
gt = rd(f'{a.data}/train/train_ground_truth.tsv'); truth = {q: set(x for x in v.split(',') if x) for q, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
params = dict(objective='binary', learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, num_threads=a.threads, verbose=-1, seed=1)
def cnt_fn(frame, c, n):
    sc = frame[frame.country == c]; sub = sc.sample(n=min(len(sc), n), random_state=0); subset = set(sub.index); vc = sub.business_name.map(nk).value_counts().to_dict()
    return lambda q, name: vc.get(name, 0) + (q not in subset)
def rule_eval(P, Q, Y, g, e):
    uq, inv = np.unique(Q, return_inverse=True); tn = np.array([len(truth.get(u, ())) for u in uq], float)
    o = np.lexsort((-P, inv)); first = np.ones(len(o), bool); first[1:] = inv[o][1:] != inv[o][:-1]; st = np.flatnonzero(first)
    rank = np.empty(len(P), int); rank[o] = np.arange(len(o)) - np.repeat(st, np.diff(np.append(st, len(o)))); p1 = np.empty(len(uq)); p1[inv[o][st]] = P[o][st]
    acc = ((rank == 0) & (P >= g)) | ((rank > 0) & (P >= e) & (p1[inv] >= g))
    npd = np.bincount(inv[acc], minlength=len(uq)).astype(float); tp = np.bincount(inv[acc & (Y == 1)], minlength=len(uq)).astype(float)
    pr = np.divide(tp, npd, out=np.zeros_like(tp), where=npd > 0); rc = np.divide(tp, tn, out=np.zeros_like(tp), where=tn > 0)
    f = np.divide(1.25 * pr * rc, 0.25 * pr + rc, out=np.zeros_like(tp), where=(pr + rc) > 0); return np.where(tn == 0, (npd == 0).astype(float), f).mean()
grid = np.round(np.arange(0.2, 0.96, 0.05), 2)
def tune(P, Q, Y): return max((rule_eval(P, Q, Y, g, e), g, e) for g in grid for e in grid)
S = {c: load('train', c, s1, pool, truth) for c in ('India', 'US')}
for c in S:
    R = S[c]; R['X'] = R['X'][:, cols]; R['h'] = idint(R['qid']) * 2654435761 % 10
X_all = np.concatenate([S[c]['X'] for c in S]); Y_all = np.concatenate([S[c]['y'] for c in S]); H_all = np.concatenate([S[c]['h'] for c in S]); V_all = H_all == 0
P_oof = np.zeros(len(X_all), np.float32)
for k in range(5):
    tr = (H_all % 5) != k; m = lgb.train(params, lgb.Dataset(X_all[tr], Y_all[tr]), num_boost_round=150); P_oof[~tr] = m.predict(X_all[~tr], num_threads=a.threads)
off = 0; X2 = []
for c in S:
    n = len(S[c]['y']); X2.append(add_ctx(S[c], P_oof[off:off + n], s1, pool, cnt_fn(s1, c, 260000))); off += n
X2 = np.concatenate(X2); Q_all = np.concatenate([S[c]['qid'] for c in S])
m2 = lgb.train(dict(params, learning_rate=0.05, num_leaves=63), lgb.Dataset(X2[~V_all], Y_all[~V_all]), num_boost_round=300); m2.save_model(f'{a.dest}/stage2.txt')
g2 = tune(m2.predict(X2[V_all], num_threads=a.threads), Q_all[V_all], Y_all[V_all]); print(f'India/US held-out stage-2 {g2[0]:.4f} (gate {g2[1]}, extra {g2[2]})', flush=True)
# France: self-trained stage 1 on test rows, then stage 2
t1 = rd(f'{a.data}/test/test_source1.tsv').set_index('entity_id'); tpool = pd.concat([rd(f'{a.data}/test/test_source2.tsv'), rd(f'{a.data}/test/test_source3.tsv')]).set_index('entity_id')
F = load('test', 'France', t1, tpool, None); Xf = F['X'][:, cols]
X, Y = X_all[~V_all], Y_all[~V_all]
for r in range(3):
    Pf = lgb.train(params, lgb.Dataset(X, Y), num_boost_round=150).predict(Xf, num_threads=a.threads).astype(np.float32)
    if r == 2: break
    k = (Pf >= 0.95) | (Pf <= 0.05); X = np.concatenate([X_all[~V_all], Xf[k]]); Y = np.concatenate([Y_all[~V_all], (Pf[k] >= 0.95).astype(np.int8)])
X2f = add_ctx(dict(F, X=Xf), Pf, t1, tpool, cnt_fn(t1, 'France', 260000)); P2 = m2.predict(X2f, num_threads=a.threads)
Qf, Cf = F['qid'], F['pid']; g, e = g2[1], g2[2]
o = np.lexsort((-P2, Qf)); first = np.ones(len(o), bool); first[1:] = Qf[o][1:] != Qf[o][:-1]; st = np.flatnonzero(first)
rank = np.empty(len(P2), int); rank[o] = np.arange(len(o)) - np.repeat(st, np.diff(np.append(st, len(o))))
uq, inv = np.unique(Qf, return_inverse=True); p1 = np.zeros(len(uq)); np.maximum.at(p1, inv, P2)
keep = ((rank == 0) & (P2 >= g)) | ((rank > 0) & (P2 >= e) & (p1[inv] >= g))
idx = np.flatnonzero(keep); oo = idx[np.lexsort((-P2[idx], Cf[idx]))]; f1 = np.ones(len(oo), bool); f1[1:] = Cf[oo][1:] != Cf[oo][:-1]; sel = oo[f1]
rows = pd.DataFrame({'q': Qf[sel], 'p': Cf[sel], 's': P2[sel]}).sort_values(['q', 's'], ascending=[True, False]).groupby('q').p.apply(','.join)
print(f'France: {len(sel) / len(uq):.3f} pairs/S1, S1 with >=1 match {len(rows) / len(uq):.3f}', flush=True)
rows.to_csv(f'{a.dest}/france_rows.tsv', sep='	', header=False)
np.savez(f'{a.dest}/france_scores.npz', q=Qf, p=Cf, s=P2.astype(np.float32), gate=g, extra=e); print('FRANCE_S2 DONE', flush=True)
