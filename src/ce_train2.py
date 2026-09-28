"""Cross-encoder trained on millions of K=8 candidate pairs from train S1 records that are DISJOINT from the LightGBM sample queries
(out/topk_train_*.npz q_ids), so its scores on the LightGBM's rows are genuinely out-of-sample. Multi-worker tokenization; resumable.
usage: python ce_train2.py --data DATA --out out_bi --full out_full [--n_pairs 4500000]"""
import argparse, os, sys, time, logging, unicodedata, numpy as np, pandas as pd, torch
from tqdm.auto import tqdm

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--full', required=True)
ap.add_argument('--n_pairs', type=int, default=4_500_000); ap.add_argument('--bs', type=int, default=256); ap.add_argument('--lr', type=float, default=4e-5)
ap.add_argument('--maxlen', type=int, default=128); ap.add_argument('--workers', type=int, default=10); ap.add_argument('--ckpt_every', type=int, default=3000)
ap.add_argument('--model', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); ap.add_argument('--max_steps', type=int, default=0)
ap.add_argument('--k', type=int, default=8); ap.add_argument('--countries', default='India,US'); ap.add_argument('--cipher', type=float, default=0.0); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--dir_name', default='ce2')
a = ap.parse_args()
D = f'{a.out}/{a.dir_name}'; os.makedirs(D, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f'{D}/train.log')])
log = logging.info

def fold(s):
    s = s.replace('�', '').lower()
    if all(ord(c) < 0x250 for c in s):
        s = unicodedata.normalize('NFKD', s); s = ''.join(c for c in s if not unicodedata.combining(c))
    return s

pf = f'{D}/pairs.npz'
if not os.path.exists(pf):
    rd = lambda p: pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False)
    s1 = rd(f'{a.data}/train/train_source1.tsv').set_index('entity_id')
    pool = pd.concat([rd(f'{a.data}/train/train_source2.tsv'), rd(f'{a.data}/train/train_source3.tsv')]).set_index('entity_id')
    gt = rd(f'{a.data}/train/train_ground_truth.tsv'); truth = {q: set(x for x in v.split(',') if x) for q, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    rng = np.random.RandomState(a.seed); QA, PA, YA = [], [], []
    CS = a.countries.split(',')
    for c in CS:
        zf = np.load(f'{a.full}/topk_train_{c}.npz'); q_ids, p_ids, cand = zf['q_ids'], zf['pool_ids'], zf['cand'][:, :a.k]
        smp = set(np.load(f'{a.out}/topk_train_{c}.npz')['q_ids'])
        idx = np.array([i for i, q in enumerate(q_ids) if q not in smp]); rng.shuffle(idx); idx = idx[:a.n_pairs // (len(CS) * a.k)]
        log(f'[{c}] {len(idx)} disjoint queries selected ({len(smp)} sample queries excluded)')
        for i in idx:
            t = truth.get(q_ids[i], ())
            for j in range(a.k):
                pid = p_ids[cand[i, j]]; QA.append(q_ids[i]); PA.append(pid); YA.append(1.0 if pid in t else 0.0)
    need_q, need_p = set(QA), set(PA)
    qn = s1[s1.index.isin(need_q)]; pn = pool[pool.index.isin(need_p)]
    qt = dict(zip(qn.index, (qn.business_name.map(fold) + ' | ' + qn.business_address.map(fold)))); pt = dict(zip(pn.index, (pn.business_name.map(fold) + ' | ' + pn.business_address.map(fold))))
    perm = rng.permutation(len(QA))
    TA = np.array([qt[QA[i]] for i in perm], dtype=object); TB = np.array([pt[PA[i]] for i in perm], dtype=object); Y = np.array([YA[i] for i in perm], np.float32)
    np.savez_compressed(pf + '.tmp.npz', ta=TA, tb=TB, y=Y); os.replace(pf + '.tmp.npz', pf)
z = np.load(pf, allow_pickle=True); TA, TB, Y = z['ta'], z['tb'], z['y']
log(f'{len(Y)} training pairs, {Y.mean():.1%} positive')

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
tok = AutoTokenizer.from_pretrained(a.model)
dev = 'cuda'
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).to(dev)
steps = len(Y) // a.bs
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01); sch = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)
step = 0; ck = f'{D}/ckpt.pt'
if os.path.exists(ck):
    s = torch.load(ck, map_location=dev); model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sch.load_state_dict(s['sch']); step = s['step']; log(f'resumed at step {step}')

LET = 'abcdefghijklmnopqrstuvwxyz'
class BatchDS(torch.utils.data.Dataset):
    def __len__(self): return steps
    def __getitem__(self, b):
        sl = slice(b * a.bs, (b + 1) * a.bs); ta, tb = list(TA[sl]), list(TB[sl])
        if a.cipher > 0:   # domain randomization: the same random letter substitution on both sides of a pair
            rg = np.random.default_rng(b)
            for i in range(len(ta)):
                if rg.random() < a.cipher:
                    t = str.maketrans(LET, ''.join(rg.permutation(list(LET)))); ta[i] = ta[i].translate(t); tb[i] = tb[i].translate(t)
        enc = tok(ta, tb, truncation=True, max_length=a.maxlen, padding=True, return_tensors='pt')
        enc['labels'] = torch.from_numpy(Y[sl]); return dict(enc)
dl = torch.utils.data.DataLoader(BatchDS(), batch_size=None, sampler=range(step, steps), num_workers=a.workers, prefetch_factor=6, persistent_workers=False)
loss_fn = torch.nn.BCEWithLogitsLoss(); model.train(); t0 = time.time(); run = 0.0; start = step
bar = tqdm(total=steps, initial=step, desc='ce2 train', unit='step', file=sys.stdout, mininterval=30)
for enc in dl:
    lab = enc.pop('labels').to(dev); enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logit = model(**enc).logits.squeeze(-1)
    loss = loss_fn(logit.float(), lab); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1; bar.update(1); run += loss.item()
    if step % 200 == 0:
        el = time.time() - t0; log(f'step {step}/{steps} loss {run / 200:.4f} ({(step - start) * a.bs / el:,.0f} pairs/s, eta {(steps - step) * el / max(step - start, 1) / 60:.1f} min)'); run = 0.0
    if step % a.ckpt_every == 0:
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), step=step), ck + '.tmp'); os.replace(ck + '.tmp', ck); log(f'checkpoint at step {step}')
    if a.max_steps and step >= a.max_steps: break
model.save_pretrained(f'{D}/model'); tok.save_pretrained(f'{D}/model'); log('CE2 training done -> ce2/model')
