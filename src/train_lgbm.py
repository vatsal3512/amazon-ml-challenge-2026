"""Train LightGBM on union-candidate pair features with a group (per-S1) split; tune per-country thresholds on macro F0.5.
usage: python train_lgbm.py --data DATA --out OUT [--countries India,US] [--rounds 600]
Artifacts in OUT/lgb/: model_ckpt.txt (every 50 rounds), model.txt, thresholds.json, val_report.json
"""
import argparse, os, sys, glob, json, time, logging, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featio import load_chunk, feature_names

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--gt', default=''); ap.add_argument('--countries', default='India,US')
ap.add_argument('--rounds', type=int, default=600); ap.add_argument('--threads', type=int, default=28)
ap.add_argument('--lr', type=float, default=0.1); ap.add_argument('--leaves', type=int, default=127)
a = ap.parse_args()
os.makedirs(f'{a.out}/lgb', exist_ok=True)
NAMES = feature_names(a.out)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f'{a.out}/lgb/train.log')])
log = logging.info

def idint(ids):
    ids = np.asarray(ids).astype('U16')
    num = np.array([int(x[3:]) for x in ids], np.int64)
    return num + np.where(np.char.startswith(ids, 'S3'), 10 ** 10, 0)

gt = pd.read_csv(a.gt or f'{a.data}/train/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
ntrue = dict(zip(gt.source1_entity_id, gt.matched_entity_ids.map(lambda x: len([i for i in x.split(',') if i]))))
pairs = gt.assign(m=gt.matched_entity_ids.str.split(',')).explode('m')
pairs = pairs[pairs.m != '']
tpairs = pd.DataFrame({'q': idint(pairs.source1_entity_id.values), 'p': idint(pairs.m.values), 'y': np.int8(1)})
log(f'truth pairs: {len(tpairs)}')

Xs, ys, qs, cs = [], [], [], []   # cs: country index per row; qs: global query int id
countries = a.countries.split(',')
for ci, c in enumerate(countries):
    z = np.load(f'{a.out}/topk_train_{c}.npz'); q_ids, p_ids = z['q_ids'], z['pool_ids']
    qi_int, pi_int = idint(q_ids), idint(p_ids)
    files = sorted(glob.glob(f'{a.out}/feat2/train_{c}_*.npz'))
    log(f'[{c}] {len(files)} feature chunks')
    for f in files:
        d = load_chunk(f)
        df = pd.DataFrame({'q': qi_int[d['q_idx']], 'p': pi_int[d['c_idx']]})
        y = df.merge(tpairs, on=['q', 'p'], how='left').y.fillna(0).values.astype(np.int8)
        Xs.append(d['X']); ys.append(y); qs.append(df.q.values); cs.append(np.full(len(df), ci, np.int8))
X, y, q, cty = np.concatenate(Xs), np.concatenate(ys), np.concatenate(qs), np.concatenate(cs)
del Xs
log(f'rows={len(X)} positives={int(y.sum())} ({y.mean():.3%}) queries={len(np.unique(q))}')

val = (q * 2654435761 % 10) == 0   # group split: whole queries go to one side
log(f'train rows {int((~val).sum())}, val rows {int(val.sum())}')
dtr = lgb.Dataset(X[~val], y[~val], feature_name=NAMES, free_raw_data=True)
dva = lgb.Dataset(X[val], y[val], feature_name=NAMES, reference=dtr)
params = dict(objective='binary', learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, max_bin=255, num_threads=a.threads, verbose=-1, seed=1)

def ckpt(env):
    if (env.iteration + 1) % 50 == 0:
        env.model.save_model(f'{a.out}/lgb/model_ckpt.txt')
        log(f'checkpoint saved at iteration {env.iteration + 1}')

t0 = time.time()
model = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva], valid_names=['val'],
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(20), ckpt])
model.save_model(f'{a.out}/lgb/model.txt')
log(f'trained in {(time.time() - t0) / 60:.1f} min, best iteration {model.best_iteration}')
imp = sorted(zip(NAMES, model.feature_importance('gain')), key=lambda t: -t[1])
log('top features: ' + ', '.join(f'{n}={g:.0f}' for n, g in imp[:12]))

# ---- exact macro F0.5 on the validation queries (singletons and uncovered matches included)
pv = model.predict(X[val], num_threads=a.threads)
qv, yv, cv = q[val], y[val], cty[val]
uq, inv = np.unique(qv, return_inverse=True)
truth_n = np.array([ntrue[f'S1-{int(u)}'] for u in uq], np.float64)
qcty = np.zeros(len(uq), np.int8); qcty[inv] = cv
def macro(t, mask):
    sel = pv >= t
    npred = np.bincount(inv[sel], minlength=len(uq)).astype(np.float64)
    tp = np.bincount(inv[sel & (yv == 1)], minlength=len(uq)).astype(np.float64)
    prec = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
    rec = np.divide(tp, truth_n, out=np.zeros_like(tp), where=truth_n > 0)
    f = np.divide(1.25 * prec * rec, 0.25 * prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    f = np.where(truth_n == 0, (npred == 0).astype(np.float64), f)
    return f[mask].mean()
report, thr = {}, {}
for ci, c in enumerate(countries):
    m = qcty == ci
    ts = np.arange(0.05, 0.96, 0.01); sc = [macro(t, m) for t in ts]
    b = int(np.argmax(sc)); thr[c] = float(ts[b]); report[c] = dict(threshold=thr[c], val_macro_f05=float(sc[b]), val_queries=int(m.sum()))
    log(f'[{c}] best threshold {ts[b]:.2f} -> validation macro F0.5 = {sc[b]:.4f} on {int(m.sum())} queries')
thr['France'] = float(max(thr.values()) + 0.0)   # unseen country: conservative fallback (higher-precision threshold)
json.dump(thr, open(f'{a.out}/lgb/thresholds.json', 'w')); json.dump(report, open(f'{a.out}/lgb/val_report.json', 'w'), indent=1)
log(f'thresholds: {thr}')
