"""Final predictions: per-country LightGBM (features incl. cross-encoder), two-threshold rule, contested-candidate rule.
Writes matching_results.tsv and candidate_pairs.tsv (all pairs scored). Countries can use different models.
usage: python predict_final.py --data DATA --out OUT --name NAME [--model_map France=path,India=path,US=path] [--decision OUT/lgb/decision.json] [--override France=0.5:0.9]"""
import argparse, os, sys, glob, json, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featio import load_chunk

ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--name', required=True)
ap.add_argument('--model_map', default=''); ap.add_argument('--decision', default=''); ap.add_argument('--override', default='')
ap.add_argument('--countries', default='India,US,France'); ap.add_argument('--threads', type=int, default=28); ap.add_argument('--no_assign', action='store_true'); ap.add_argument('--target_per_s1', type=float, default=0.0); ap.add_argument('--target_countries', default='France')
ap.add_argument('--pred_dir', default='')   # precomputed test scores (e.g. OUT/stage2) used instead of the model when test_<c>.npz exists
a = ap.parse_args()
models = {}
for kv in (a.model_map.split(',') if a.model_map else []):
    k, v = kv.split('='); models[k] = v
dec = json.load(open(a.decision or f'{a.out}/lgb/decision.json'))
rules = {c: (dec[c]['gate'], dec[c]['extra']) for c in dec}
for kv in (a.override.split(',') if a.override else []):
    k, v = kv.split('='); g, e = v.split(':'); rules[k] = (float(g), float(e))
dst = f'{a.out}/submissions/{a.name}'; os.makedirs(dst, exist_ok=True)
s1 = pd.read_csv(f'{a.data}/test/test_source1.tsv', sep='\t', dtype=str, keep_default_na=False)
matches = {i: '' for i in s1.entity_id}; cands = {i: '' for i in s1.entity_id}
for c in a.countries.split(','):
    model = lgb.Booster(model_file=models.get(c, f'{a.out}/lgb/model.txt')); g, e = rules.get(c, (0.7, 0.7)); t0 = time.time()
    z = np.load(f'{a.out}/topk_test_{c}.npz'); q_ids, p_ids = z['q_ids'], z['pool_ids']; Q, C, P = [], [], []
    pf = f'{a.pred_dir}/test_{c}.npz' if a.pred_dir else ''
    if pf and os.path.exists(pf):
        s = np.load(pf); Q, C, P = [s['q_idx']], [s['c_idx']], [s['P']]; print(f'[{c}] scores from {pf}', flush=True)
    else:
        for f in sorted(glob.glob(f'{a.out}/feat2/test_{c}_*.npz')):
            if f.endswith('.tmp.npz'): continue
            d = load_chunk(f); Q.append(d['q_idx']); C.append(d['c_idx']); P.append(model.predict(d['X'], num_threads=a.threads).astype(np.float32))
    Q, C, P = np.concatenate(Q), np.concatenate(C), np.concatenate(P)
    o = np.lexsort((-P, Q)); Qs, Cs, Ps = Q[o], C[o], P[o]; first = np.ones(len(o), bool); first[1:] = Qs[1:] != Qs[:-1]
    starts = np.flatnonzero(first); cnt = np.diff(np.append(starts, len(o))); rank = np.arange(len(o)) - np.repeat(starts, cnt); p1 = np.repeat(Ps[starts], cnt)
    if a.target_per_s1 > 0 and c in a.target_countries.split(','):
        best_t, best_d = 0.5, 1e9
        for t in np.arange(0.30, 0.995, 0.005):
            k = ((rank == 0) & (Ps >= t)) | ((rank > 0) & (Ps >= t) & (p1 >= t)); d = abs(k.sum() * 0.995 / len(q_ids) - a.target_per_s1)
            if d < best_d: best_d, best_t = d, t
        g = e = float(best_t); print(f'[{c}] threshold chosen so predicted matches per S1 ~ {a.target_per_s1}: {best_t:.3f}', flush=True)
    keep = ((rank == 0) & (Ps >= g)) | ((rank > 0) & (Ps >= e) & (p1 >= g)); n_rule = int(keep.sum())
    if not a.no_assign:
        idx = np.flatnonzero(keep); oo = idx[np.lexsort((-Ps[idx], Cs[idx]))]; f1 = np.ones(len(oo), bool); f1[1:] = Cs[oo][1:] != Cs[oo][:-1]
        keep = np.zeros(len(Ps), bool); keep[oo[f1]] = True
    b = np.flatnonzero(np.diff(Qs)) + 1
    for qs_, cs_ in zip(np.split(Qs, b), np.split(Cs, b)): cands[q_ids[qs_[0]]] = ','.join(p_ids[cs_])
    Qk, Ck = Qs[keep], Cs[keep]
    if len(Qk):
        b = np.flatnonzero(np.diff(Qk)) + 1
        for qs_, cs_ in zip(np.split(Qk, b), np.split(Ck, b)): matches[q_ids[qs_[0]]] = ','.join(p_ids[cs_])
    print(f'[{c}] gate {g:.2f} extra {e:.2f}: pairs {len(Ps)}, by rule {n_rule}, after contested rule {int(keep.sum())} ({keep.sum() / len(q_ids):.3f} per S1), '
          f'empty {np.mean([matches[q] == "" for q in q_ids]):.3f}, {(time.time() - t0) / 60:.1f} min', flush=True)
with open(f'{dst}/matching_results.tsv', 'w', newline='') as f:
    f.write('source1_entity_id\tmatched_entity_ids\n')
    for i in s1.entity_id: f.write(f'{i}\t{matches[i]}\n')
with open(f'{dst}/candidate_pairs.tsv', 'w', newline='') as f:
    f.write('source1_entity_id\tcandidate_entity_ids\n')
    for i in s1.entity_id: f.write(f'{i}\t{cands[i]}\n')
n = np.array([0 if matches[i] == '' else matches[i].count(',') + 1 for i in s1.entity_id]); print(f'wrote {dst}: mean matches {n.mean():.2f}, empty {np.mean(n == 0):.1%}')
