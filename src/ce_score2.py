"""Score ALL pairs of a feat2 set with the cross-encoder (GPU), resumable in shards. Output: OUT/ce2/scores_<split>_<country>.npz (q_idx, c_idx, logit)
in feat2 chunk order.  usage: python ce_score2.py --data DATA --out OUT --split train|test --countries India,US [--model_dir OUT/ce2/model]"""
import argparse, glob, os, sys, time, logging, unicodedata, numpy as np, pandas as pd, torch
from tqdm.auto import tqdm

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--split', required=True); ap.add_argument('--countries', required=True)
ap.add_argument('--model_dir', default=''); ap.add_argument('--bs', type=int, default=1024); ap.add_argument('--shard', type=int, default=2_000_000)
ap.add_argument('--maxlen', type=int, default=128); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--score_dir', default='')
a = ap.parse_args()
D = a.score_dir or f'{a.out}/ce2'; os.makedirs(D, exist_ok=True); md = a.model_dir or f'{a.out}/ce2/model'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f'{D}/score_{a.split}.log')])
log = logging.info

def fold(s):
    s = s.replace('�', '').lower()
    if all(ord(c) < 0x250 for c in s):
        s = unicodedata.normalize('NFKD', s); s = ''.join(c for c in s if not unicodedata.combining(c))
    return s

from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(md); model = AutoModelForSequenceClassification.from_pretrained(md).to('cuda').eval().to(torch.bfloat16)
rd = lambda p: pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False)
sp = a.split
s1 = rd(f'{a.data}/{sp}/{sp}_source1.tsv').set_index('entity_id')
pool = pd.concat([rd(f'{a.data}/{sp}/{sp}_source2.tsv'), rd(f'{a.data}/{sp}/{sp}_source3.tsv')]).set_index('entity_id')

class DS(torch.utils.data.Dataset):
    def __init__(self, A, B): self.A, self.B = A, B
    def __len__(self): return len(self.A)
    def __getitem__(self, i): return self.A[i], self.B[i]
def collate(b):
    A, B = zip(*b); return tok(list(A), list(B), truncation=True, max_length=a.maxlen, padding=True, return_tensors='pt')

@torch.no_grad()
def score(TA, TB, desc):
    order = np.argsort([len(x) + len(y) for x, y in zip(TA, TB)], kind='stable')
    dl = torch.utils.data.DataLoader(DS([TA[i] for i in order], [TB[i] for i in order]), batch_size=a.bs, collate_fn=collate, num_workers=a.workers, shuffle=False, prefetch_factor=4)
    out = np.empty(len(TA), np.float32); k = 0
    for enc in tqdm(dl, desc=desc, file=sys.stdout, mininterval=30, unit='batch'):
        lg = model(**{n: v.to('cuda') for n, v in enc.items()}).logits.squeeze(-1).float().cpu().numpy(); out[order[k:k + len(lg)]] = lg; k += len(lg)
    return out

for c in a.countries.split(','):
    dst = f'{D}/scores_{sp}_{c}.npz'
    if os.path.exists(dst): log(f'[{c}] exists, skip'); continue
    z = np.load(f'{a.out}/topk_{sp}_{c}.npz'); q_ids, p_ids = z['q_ids'], z['pool_ids']
    files = [f for f in sorted(glob.glob(f'{a.out}/feat2/{sp}_{c}_*.npz')) if not f.endswith('.tmp.npz')]
    Q = np.concatenate([np.load(f)['q_idx'] for f in files]); C = np.concatenate([np.load(f)['c_idx'] for f in files])
    log(f'[{c}] {len(Q)} pairs in {len(files)} chunks')
    qn = s1.loc[q_ids]; pn = pool.loc[p_ids]
    qtext = (qn.business_name.map(fold) + ' | ' + qn.business_address.map(fold)).values; ptext = (pn.business_name.map(fold) + ' | ' + pn.business_address.map(fold)).values
    lg = np.empty(len(Q), np.float32); t0 = time.time()
    for st in range(0, len(Q), a.shard):
        spf = f'{D}/shard_{sp}_{c}_{st:010d}.npy'; sl = slice(st, min(st + a.shard, len(Q)))
        if os.path.exists(spf): lg[sl] = np.load(spf); continue
        part = score(qtext[Q[sl]], ptext[C[sl]], f'[{c}] {st}/{len(Q)}'); np.save(spf + '.tmp.npy', part); os.replace(spf + '.tmp.npy', spf); lg[sl] = part
        log(f'[{c}] shard {st}:{sl.stop} saved ({sl.stop / (time.time() - t0):,.0f} pairs/s overall)')
    np.savez_compressed(dst + '.tmp.npz', q_idx=Q.astype(np.int32), c_idx=C.astype(np.int32), logit=lg); os.replace(dst + '.tmp.npz', dst); log(f'[{c}] DONE -> {dst}')
