"""Per-pair cross-encoder features aligned with feat2 chunks: ce_logit, ce_rank, ce_gap_best, ce_margin, ce_mean_q.
usage: python features_ce.py --split train|test --out OUT --countries India,US [--score_dir OUT/ce2]  -> OUT/feat_ce/<chunk names>"""
import argparse, glob, os, sys, numpy as np

CE_NAMES = ['ce_logit', 'ce_rank', 'ce_gap_best', 'ce_margin', 'ce_mean_q']


def ce_features(qi, ce):
    """qi: query index per pair (contiguous groups), ce: cross-encoder logit per pair."""
    order = np.lexsort((-ce, qi))
    qs, cs = qi[order], ce[order]
    first = np.ones(len(qs), bool)
    first[1:] = qs[1:] != qs[:-1]
    starts = np.flatnonzero(first)
    cnt = np.diff(np.append(starts, len(qs)))
    gid = np.repeat(np.arange(len(starts)), cnt)
    rank = np.arange(len(qs)) - starts[gid]
    best = cs[starts][gid]
    second = np.where(cnt[gid] > 1, cs[np.minimum(starts + 1, len(qs) - 1)][gid], -12.0)
    mean = (np.bincount(gid, weights=cs) / cnt)[gid]
    out = np.empty((len(qs), 5), np.float32)
    out[:, 0] = cs
    out[:, 1] = rank
    out[:, 2] = best - cs
    out[:, 3] = cs - np.where(rank == 0, second, best)
    out[:, 4] = mean
    res = np.empty_like(out)
    res[order] = out
    return res


def run(split, out, countries, score_dir, dest='feat_ce'):
    os.makedirs(f'{out}/{dest}', exist_ok=True)
    for c in countries:
        z = np.load(f'{score_dir}/scores_{split}_{c}.npz')
        logit, zq, zc = z['logit'], z['q_idx'], z['c_idx']
        off = 0
        files = [f for f in sorted(glob.glob(f'{out}/feat2/{split}_{c}_*.npz')) if not f.endswith('.tmp.npz')]
        for f in files:
            d = np.load(f)
            qi, ci = d['q_idx'], d['c_idx']
            n = len(qi)
            assert (zq[off:off + n] == qi).all() and (zc[off:off + n] == ci).all(), f'score order mismatch in {f}'
            dst = f'{out}/{dest}/{os.path.basename(f)}'
            if not os.path.exists(dst):
                np.savez_compressed(dst + '.tmp.npz', X=ce_features(qi, logit[off:off + n]))
                os.replace(dst + '.tmp.npz', dst)
            off += n
        assert off == len(logit), 'chunk pairs do not add up to the CE score count'
        print(f'[{c}] {len(files)} chunks done', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True); ap.add_argument('--out', required=True); ap.add_argument('--countries', required=True); ap.add_argument('--score_dir', default=''); ap.add_argument('--dest', default='feat_ce')
    a = ap.parse_args()
    run(a.split, a.out, a.countries.split(','), a.score_dir or f'{a.out}/ce2', a.dest)
