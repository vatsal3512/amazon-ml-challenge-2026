"""Contrastively fine-tune the bi-encoder on matched (S1, S2/S3) pairs with in-batch same-country negatives. GPU; checkpointed.
Uses only S1 records NOT among the queries used for LightGBM/CE training (no leakage into retrieval features).
usage: python bi_train.py --data DATA --out OUT [--n_s1 400000 --bs 512 --epochs 1]
Writes OUT/bi/model (SentenceTransformer format; pass to dense_topk.py --model OUT/bi/model)
"""
import argparse, os, sys, time, logging, unicodedata, numpy as np, pandas as pd, torch
import torch.nn.functional as Fn
from tqdm.auto import tqdm

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--gt', default='')
ap.add_argument('--n_s1', type=int, default=400_000); ap.add_argument('--bs', type=int, default=512)
ap.add_argument('--epochs', type=int, default=1); ap.add_argument('--lr', type=float, default=3e-5)
ap.add_argument('--scale', type=float, default=20.0); ap.add_argument('--ckpt_every', type=int, default=500)
ap.add_argument('--model', default='paraphrase-multilingual-MiniLM-L12-v2'); ap.add_argument('--max_steps', type=int, default=0)
ap.add_argument('--allow_cpu', action='store_true'); ap.add_argument('--maxlen', type=int, default=64)
a = ap.parse_args()
os.makedirs(f'{a.out}/bi', exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f'{a.out}/bi/train.log')])
log = logging.info
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cpu' and not a.allow_cpu: log('no GPU, aborting'); sys.exit(2)

def fold(s):
    s = s.replace('�', '').lower()
    if all(ord(c) < 0x250 for c in s):
        s = unicodedata.normalize('NFKD', s); s = ''.join(c for c in s if not unicodedata.combining(c))
    return s.replace('&', ' and ').replace('+', ' and ')

pf = f'{a.out}/bi/pairs.npz'
if not os.path.exists(pf):
    rd = lambda p: pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False)
    s1 = rd(f'{a.data}/train/train_source1.tsv').set_index('entity_id')
    pool = pd.concat([rd(f'{a.data}/train/train_source2.tsv'), rd(f'{a.data}/train/train_source3.tsv')]).set_index('entity_id')
    gt = rd(a.gt or f'{a.data}/train/train_ground_truth.tsv'); gt = gt[gt.matched_entity_ids != '']
    used = set()
    for c in ('India', 'US'):
        f = f'{a.out}/topk_train_{c}.npz'
        if os.path.exists(f): used |= set(np.load(f)['q_ids'])
    gt = gt[~gt.source1_entity_id.isin(used)].sample(frac=1, random_state=0)
    gt = gt.groupby(s1.loc[gt.source1_entity_id, 'country'].values, group_keys=False).head(a.n_s1 // 2)   # balanced per country
    rows = []
    rng = np.random.RandomState(0)
    for q, m in zip(gt.source1_entity_id, gt.matched_entity_ids):
        ms = [x for x in m.split(',') if x]
        for x in rng.choice(ms, size=min(2, len(ms)), replace=False): rows.append((q, x))
    P = pd.DataFrame(rows, columns=['q', 'p'])
    qa = s1.loc[P.q]; pb = pool.loc[P.p]
    ta = (qa.business_name.map(fold) + ' | ' + qa.business_address.map(fold)).values
    tb = (pb.business_name.map(fold) + ' | ' + pb.business_address.map(fold)).values
    np.savez_compressed(pf + '.tmp.npz', ta=ta.astype(object), tb=tb.astype(object), country=qa.country.values.astype(object))
    os.replace(pf + '.tmp.npz', pf)
z = np.load(pf, allow_pickle=True); TA, TB, CT = z['ta'], z['tb'], z['country']
log(f'{len(TA)} training pairs; countries {dict(zip(*np.unique(CT, return_counts=True)))}')

from sentence_transformers import SentenceTransformer
st = SentenceTransformer(a.model, device=dev); st.max_seq_length = a.maxlen
opt = torch.optim.AdamW(st.parameters(), lr=a.lr, weight_decay=0.01)
# batches are country-homogeneous so in-batch negatives are same-language, same-style records
idx_by_c = {c: np.flatnonzero(CT == c) for c in np.unique(CT)}
batches = []
for ep in range(a.epochs):
    rng = np.random.RandomState(ep)
    for c, ix in idx_by_c.items():
        ix = rng.permutation(ix)
        batches += [ix[i:i + a.bs] for i in range(0, len(ix) - a.bs + 1, a.bs)]
order = np.random.RandomState(7).permutation(len(batches)); batches = [batches[i] for i in order]
total = len(batches); step = 0; ck = f'{a.out}/bi/ckpt'
if os.path.exists(f'{ck}/step.txt'):
    step = int(open(f'{ck}/step.txt').read()); st = SentenceTransformer(ck, device=dev); st.max_seq_length = a.maxlen
    opt = torch.optim.AdamW(st.parameters(), lr=a.lr, weight_decay=0.01); log(f'resumed at step {step}')
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.05 * total)) * max(0.0, (total - s) / total))
for _ in range(step): sched.step()
def embed(texts):
    f = st.tokenize(list(texts)); f = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in f.items()}
    return Fn.normalize(st(f)['sentence_embedding'].float(), dim=-1)
st.train(); t0 = time.time(); run = 0.0; start = step
bar = tqdm(total=total, initial=step, desc='bi train', unit='step', file=sys.stdout, mininterval=20)
while step < total and not (a.max_steps and step >= a.max_steps):
    b = batches[step]
    with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == 'cuda')):
        ea, eb = embed(TA[b]), embed(TB[b])
    sim = ea @ eb.T * a.scale; lab = torch.arange(len(b), device=dev)
    loss = (Fn.cross_entropy(sim, lab) + Fn.cross_entropy(sim.T, lab)) / 2
    loss.backward(); torch.nn.utils.clip_grad_norm_(st.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    step += 1; bar.update(1); run += loss.item()
    if step % 50 == 0:
        log(f'step {step}/{total} loss {run / 50:.4f} (eta {(total - step) * (time.time() - t0) / max(step - start, 1) / 60:.1f} min)'); run = 0.0
    if step % a.ckpt_every == 0:
        os.makedirs(ck, exist_ok=True); st.save(ck); open(f'{ck}/step.txt', 'w').write(str(step)); log(f'checkpoint saved at step {step}')
st.save(f'{a.out}/bi/model'); log('bi-encoder saved to bi/model')
