"""Street / house-number extraction and comparison (language-agnostic heuristics; no external data)."""
import re
from rapidfuzz import fuzz
from textnorm import addr_tokens

STREET_WORDS = {'rue', 'avenue', 'boulevard', 'allee', 'chemin', 'route', 'cours', 'quai', 'place', 'impasse', 'passage', 'square',
                'sentier', 'esplanade', 'street', 'road', 'drive', 'lane', 'court', 'way', 'circle', 'terrace', 'highway', 'parkway',
                'trail', 'loop', 'pike', 'plaza', 'path', 'alley', 'row', 'walk', 'lieu'}
FILL = {'de', 'du', 'des', 'la', 'le', 'les', 'l', 'd', 'of', 'the', 'et', 'aux', 'au', 'en', 'sur', 'saint', 'sainte', 'st', 'ste', 'bis', 'ter',
        'number', 'no', 'n', 'nº', 'unit', 'apartment', 'suite', 'floor', 'north', 'south', 'east', 'west'}


def parse(addr):
    """Return (house_number, street_name_string, has_street) from a raw address string."""
    best = None
    for comp in addr.split(','):
        t = addr_tokens(comp)
        if any(w in STREET_WORDS for w in t):
            best = t
            break
    if best is None:                       # no street-type word: still recover a house number from the first component that has digits
        for comp in addr.split(','):
            t = addr_tokens(comp)
            for w in t:
                if w.isdigit():
                    return w, '', False
        return '', '', False
    num = ''
    for w in best:
        if w.isdigit():
            num = w
            break
    if not num:
        for w in addr_tokens(addr):
            if w.isdigit():
                num = w
                break
    name = [w for w in best if not w.isdigit() and w not in STREET_WORDS and w not in FILL and len(w) > 1]
    return num, ''.join(name), True


def street_sim(a1, a2):
    """(street_ratio in [0,1] or -1 if unknown, number_equal in {1,0,-1 unknown})."""
    n1, s1, h1 = parse(a1)
    n2, s2, h2 = parse(a2)
    sr = fuzz.ratio(s1, s2) / 100.0 if (h1 and h2 and s1 and s2) else -1.0
    ne = (1.0 if n1 == n2 else 0.0) if (n1 and n2) else -1.0
    return sr, ne
