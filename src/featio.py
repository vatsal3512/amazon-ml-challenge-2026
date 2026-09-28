"""Load feature chunks: feat2 columns plus feat_add, feat_add2 and feat_ce columns when those directories exist next to them."""
import os
import numpy as np
from features2 import NAMES
from features_add import ADD_NAMES
from features_add2 import ST_NAMES
from features_ce import CE_NAMES

SIDES = (('feat_add', ADD_NAMES), ('feat_add2', ST_NAMES), ('feat_ce', CE_NAMES), ('feat_cex', ['x' + n for n in CE_NAMES]))   # feat_cex: 2nd cross-encoder (XLM-R), optional


def _side(f, name):
    return os.path.join(os.path.dirname(os.path.dirname(f)), name, os.path.basename(f))


def _has(out, name):
    p = os.path.join(out, name)
    return os.path.isdir(p) and len(os.listdir(p)) > 0


def has_add(out):
    return _has(out, 'feat_add')


def feature_names(out):
    names = list(NAMES)
    for side, nm in SIDES:
        if _has(out, side):
            names += nm
    return names


def load_chunk(f):
    d = np.load(f)
    X, q, c = d['X'], d['q_idx'], d['c_idx']
    parts = [X]
    for side, _ in SIDES:
        p = _side(f, side)
        if os.path.exists(p):
            parts.append(np.load(p)['X'])
    if len(parts) > 1:
        X = np.hstack(parts)
    return {'X': X, 'q_idx': q, 'c_idx': c}
