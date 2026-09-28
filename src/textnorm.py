import os, re, unicodedata
FRENCH = os.environ.get('ER_LOCALE', '') == 'France'   # French abbreviation rules apply only when building France features

SUFFIX = {
    'pvt','private','ltd','limited','llp','llc','inc','incorporated','corp','corporation','co','company',
    'plc','lp','pc','pllc','sarl','sasu','sas','eurl','sa','sci','snc','scop','gmbh','cie','fils','et',
}
ADDR_MAP = {
    'st':'street','str':'street','rd':'road','ave':'avenue','av':'avenue','dr':'drive','ln':'lane','ct':'court',
    'blvd':'boulevard','bd':'boulevard','bld':'boulevard','hwy':'highway','pkwy':'parkway','apt':'apartment',
    'ste':'suite','pl':'place','sq':'square','cir':'circle','ter':'terrace','nr':'near','opp':'opposite',
    'r':'rue','all':'allee','fl':'floor','bldg':'building','no':'number','nagar':'nagar','mkt':'market',
    'n':'north','s':'south','e':'east','w':'west',
}
NAME_MAP = {'st': 'saint', 'ste': 'sainte', 'frs': 'freres', 'fr': 'freres', 'ets': 'etablissements', 'etablissement': 'etablissements',
            'etabl': 'etablissements', 'compagnie': 'cie'}
FR_STREET = {'rue', 'avenue', 'boulevard', 'allee', 'chemin', 'route', 'quai', 'cours', 'impasse', 'place', 'r', 'bd', 'av', 'all'}
LANDMARK = {'near','opposite','behind','beside','nr','opp','next','adjacent'}
STATE_US = {
 'alabama':'al','alaska':'ak','arizona':'az','arkansas':'ar','california':'ca','colorado':'co','connecticut':'ct',
 'delaware':'de','florida':'fl','georgia':'ga','hawaii':'hi','idaho':'id','illinois':'il','indiana':'in','iowa':'ia',
 'kansas':'ks','kentucky':'ky','louisiana':'la','maine':'me','maryland':'md','massachusetts':'ma','michigan':'mi',
 'minnesota':'mn','mississippi':'ms','missouri':'mo','montana':'mt','nebraska':'ne','nevada':'nv','new hampshire':'nh',
 'new jersey':'nj','new mexico':'nm','new york':'ny','north carolina':'nc','north dakota':'nd','ohio':'oh','oklahoma':'ok',
 'oregon':'or','pennsylvania':'pa','rhode island':'ri','south carolina':'sc','south dakota':'sd','tennessee':'tn',
 'texas':'tx','utah':'ut','vermont':'vt','virginia':'va','washington':'wa','west virginia':'wv','wisconsin':'wi',
 'wyoming':'wy','district of columbia':'dc',
}
_STATE_RE = re.compile(r'\b(' + '|'.join(sorted(STATE_US, key=len, reverse=True)) + r')\b')
_TOK = re.compile(r'[a-z0-9]+')

def fold(s):
    s = s.replace('�', '').lower()
    if all(ord(c) < 0x250 for c in s):
        s = unicodedata.normalize('NFKD', s)
        s = ''.join(c for c in s if not unicodedata.combining(c))
    return s.replace('&', ' and ').replace('+', ' and ')

def name_tokens(s, strip_suffix=True):
    t = [NAME_MAP.get(x, x) if FRENCH else x for x in _TOK.findall(fold(s))]
    if strip_suffix:
        u = [x for x in t if x not in SUFFIX]
        t = u or t
    return t

def addr_tokens(s):
    s = fold(s)
    s = _STATE_RE.sub(lambda m: STATE_US[m.group(1)], s)
    out, skip = [], False
    raw = _TOK.findall(s)
    french = FRENCH and any(w in FR_STREET for w in raw)
    for x in raw:
        if french and x in ('st', 'ste'):
            x = 'saint' if x == 'st' else 'sainte'
        else:
            x = ADDR_MAP.get(x, x)
        if x.isdigit():
            x = x.lstrip('0') or '0'
        out.append(x)
    return [x for x in out if x not in LANDMARK]

def norm_name(s): return ' '.join(name_tokens(s))
def norm_addr(s): return ' '.join(addr_tokens(s))
