"""Script-agnostic phonetic transliteration for Indic scripts using only Unicode character names (stdlib, no external data).
translit(s) -> Latin phonetic string;  skeleton(s) -> consonant skeleton for fuzzy comparison across scripts and spellings.
"""
import re, unicodedata

_VOWELS = {'A': 'a', 'AA': 'aa', 'I': 'i', 'II': 'i', 'U': 'u', 'UU': 'u', 'VOCALIC R': 'ri', 'VOCALIC L': 'li', 'E': 'e', 'EE': 'e',
           'AI': 'ai', 'O': 'o', 'OO': 'o', 'AU': 'au', 'SHORT E': 'e', 'SHORT O': 'o', 'CANDRA E': 'e', 'CANDRA O': 'o',
           'CANDRA A': 'a', 'SHORT A': 'a', 'AW': 'au', 'VOCALIC RR': 'ri', 'VOCALIC LL': 'li'}
_CONS = {'K': 'k', 'KA': 'k', 'KHA': 'kh', 'GA': 'g', 'GHA': 'gh', 'NGA': 'n', 'CA': 'ch', 'CHA': 'chh', 'JA': 'j', 'JHA': 'jh',
         'NYA': 'n', 'TTA': 't', 'TTHA': 'th', 'DDA': 'd', 'DDHA': 'dh', 'NNA': 'n', 'TA': 't', 'THA': 'th', 'DA': 'd', 'DHA': 'dh',
         'NA': 'n', 'PA': 'p', 'PHA': 'ph', 'BA': 'b', 'BHA': 'bh', 'MA': 'm', 'YA': 'y', 'RA': 'r', 'RRA': 'r', 'LA': 'l',
         'LLA': 'l', 'LLLA': 'l', 'VA': 'v', 'WA': 'v', 'SHA': 'sh', 'SSA': 'sh', 'SA': 's', 'HA': 'h', 'FA': 'f', 'ZA': 'z',
         'QA': 'k', 'KHHA': 'kh', 'GHHA': 'g', 'ZHA': 'l', 'NNNA': 'n', 'DDDHA': 'r', 'RHA': 'r', 'YYA': 'y', 'JJA': 'j',
         'TTTA': 't', 'KSSA': 'ksh', 'JNYA': 'gn', 'KSHA': 'ksh'}
_SIGNS = {'ANUSVARA': 'n', 'CANDRABINDU': 'n', 'VISARGA': 'h', 'NUKTA': '', 'ANUSVARA ABOVE': 'n', 'AVAGRAHA': '', 'LENGTH MARK': '',
          'AU LENGTH MARK': '', 'AI LENGTH MARK': ''}
_DIGITS = {'ZERO': '0', 'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4', 'FIVE': '5', 'SIX': '6', 'SEVEN': '7', 'EIGHT': '8', 'NINE': '9'}
_DROP_FINAL_A = ('DEVANAGARI', 'BENGALI', 'GUJARATI', 'GURMUKHI', 'ORIYA')
_cache = {}

def _char(c):
    r = _cache.get(c)
    if r is not None:
        return r
    try:
        nm = unicodedata.name(c)
    except ValueError:
        nm = ''
    kind, val, script = 'other', c, ''
    if ' LETTER ' in nm:
        script, syl = nm.split(' LETTER ', 1)
        if syl in _CONS: kind, val = 'cons', _CONS[syl]
        elif syl in _VOWELS: kind, val = 'vowel', _VOWELS[syl]
        else: kind, val = 'other', ''
    elif ' VOWEL SIGN ' in nm:
        script, syl = nm.split(' VOWEL SIGN ', 1)
        kind, val = 'sign', _VOWELS.get(syl, '')
    elif nm.endswith(' VIRAMA') or nm.endswith(' SIGN VIRAMA') or 'PULLI' in nm:
        script = nm.split(' ')[0]; kind, val = 'virama', ''
    elif ' SIGN ' in nm:
        script, syl = nm.split(' SIGN ', 1); kind, val = 'mark', _SIGNS.get(syl, '')
    elif ' DIGIT ' in nm:
        script, syl = nm.split(' DIGIT ', 1); kind, val = 'digit', _DIGITS.get(syl, '')
    _cache[c] = (kind, val, script)
    return _cache[c]

def translit(s):
    s = s.replace('‌', '').replace('‍', '')
    out, pending_a, cur_script = [], False, ''
    for c in s:
        if ord(c) < 0x900:
            pending_a = False; out.append(c); continue
        kind, val, script = _char(c)
        if kind == 'cons':
            out.append(val); out.append('a'); pending_a = True; cur_script = script
        elif kind == 'sign':
            if pending_a and out and out[-1] == 'a': out.pop()
            out.append(val); pending_a = False
        elif kind == 'virama':
            if pending_a and out and out[-1] == 'a': out.pop()
            pending_a = False
        elif kind == 'vowel' or kind == 'mark' or kind == 'digit':
            out.append(val); pending_a = False
        else:
            if pending_a and cur_script in _DROP_FINAL_A and out and out[-1] == 'a': out.pop()   # word-final schwa deletion
            pending_a = False; out.append(' ' if kind == 'other' and not val else val)
    if pending_a and cur_script in _DROP_FINAL_A and out and out[-1] == 'a': out.pop()
    return ''.join(out)

_PH = [('ph', 'f'), ('sh', 's'), ('chh', 'c'), ('ch', 'c'), ('kh', 'k'), ('gh', 'g'), ('th', 't'), ('dh', 'd'), ('bh', 'b'), ('jh', 'j'),
       ('ck', 'k'), ('q', 'k'), ('x', 'ks'), ('w', 'v'), ('z', 'j'), ('y', 'i')]

def skeleton(s):
    """Lower-case Latin skeleton: transliterate, fold spelling variants, drop vowels and repeats."""
    s = translit(s).lower()
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    for a, b in _PH: s = s.replace(a, b)
    s = s.replace('c', 'k')
    s = re.sub(r'(?<=[a-z])[aeiou]+', '', s)      # keep a leading vowel of each word, drop the rest
    s = re.sub(r'(.)\1+', r'\1', s)
    return re.sub(r'\s+', ' ', s).strip()
