def f05(pred, true):
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_map, true_map):
    """Leaderboard metric: mean per-S1 F0.5 over every S1 in true_map (singletons included)."""
    return sum(f05(set(pred_map.get(k, ())), set(v)) for k, v in true_map.items()) / max(len(true_map), 1)
