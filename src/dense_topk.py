"""Stage 1: dense top-k retrieval with per-stage checkpoints. Rerunning resumes from the last saved artifact.
usage: python dense_topk.py --split test|train --data DIR --out DIR [--k 50] [--max_queries N]

Artifacts under OUT/work/<split>_<country>/ :
  ids.npz            q_ids, pool_ids (the exact query sample and pool order)
  text_q.pkl/text_p.pkl   normalized text that gets encoded
  emb_{p,q}_00000.npy...  fp16 embedding shards (one per --enc_chunk records)
  part_00000000.npz...    top-k results (one per --q_ckpt queries)
Final result: OUT/topk_<split>_<country>.npz
"""
import os, sys, time, glob, pickle, logging, argparse, unicodedata, numpy as np, pandas as pd, torch
from tqdm.auto import tqdm

def fold(s):
    s = s.replace('�', '').lower()
    if all(ord(c) < 0x250 for c in s):
        s = unicodedata.normalize('NFKD', s)
        s = ''.join(c for c in s if not unicodedata.combining(c))
    return s.replace('&', ' and ').replace('+', ' and ')

ap = argparse.ArgumentParser()
ap.add_argument('--split', required=True)
ap.add_argument('--data', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--k', type=int, default=50)
ap.add_argument('--max_queries', type=int, default=0)
ap.add_argument('--countries', default='')
ap.add_argument('--batch', type=int, default=2048)
ap.add_argument('--enc_chunk', type=int, default=200_000)
ap.add_argument('--q_ckpt', type=int, default=100_000)
ap.add_argument('--pool_block', type=int, default=1_000_000)
ap.add_argument('--q_block', type=int, default=2048)
ap.add_argument('--model', default='/kaggle/input/er-minilm' if os.path.isdir('/kaggle/input/er-minilm')
                else 'paraphrase-multilingual-MiniLM-L12-v2')
ap.add_argument('--allow_cpu', action='store_true')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S',
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f'{a.out}/dense_{a.split}.log')])
log = logging.info
PB = dict(file=sys.stdout, mininterval=20, dynamic_ncols=False, ncols=100)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dt = torch.float16 if dev == 'cuda' else torch.float32
if dev == 'cpu' and not a.allow_cpu:
    log('ERROR: no GPU visible; aborting to avoid a days-long CPU run (use --allow_cpu to override).')
    sys.exit(2)
log(f'model={a.model} device={dev} split={a.split} k={a.k} max_queries={a.max_queries} batch={a.batch}')
if dev == 'cuda':
    log(f'gpu={torch.cuda.get_device_name(0)} mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')


def atomic(path, writer):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        writer(f)
    os.replace(tmp, path)

def save_npy(path, arr): atomic(path, lambda f: np.save(f, arr))
def save_npz(path, **kw): atomic(path, lambda f: np.savez_compressed(f, **kw))
def save_pkl(path, obj): atomic(path, lambda f: pickle.dump(obj, f, protocol=4))
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def rd(p):
    log(f'reading {p}')
    return pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False)

d = os.path.join(a.data, a.split)
s1 = rd(f'{d}/{a.split}_source1.tsv')
pool = pd.concat([rd(f'{d}/{a.split}_source2.tsv'), rd(f'{d}/{a.split}_source3.tsv')], ignore_index=True)
countries = a.countries.split(',') if a.countries else sorted(s1.country.unique())

model = None
def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(a.model, device=dev)
        model.max_seq_length = 64
        if dev == 'cuda':
            model.half()
    return model

@torch.no_grad()
def embed(texts, tag, work, prefix):
    """Encode in shards; existing shards are reused. Returns a device tensor (n, dim)."""
    n = len(texts); shards = list(range(0, n, a.enc_chunk))
    t0 = time.time(); done_new = 0; parts = []
    with tqdm(total=n, desc=f'encode {tag}', unit='rec', **PB) as bar:
        for i in shards:
            sp = f'{work}/emb_{prefix}_{i // a.enc_chunk:05d}.npy'
            if os.path.exists(sp):
                arr = np.load(sp)
                bar.update(len(arr)); parts.append(torch.from_numpy(arr).to(dev, dt)); continue
            e = get_model().encode(texts[i:i + a.enc_chunk], batch_size=a.batch, convert_to_tensor=True,
                                   normalize_embeddings=True, show_progress_bar=False).to(dt)
            save_npy(sp, e.cpu().numpy().astype(np.float16))
            parts.append(e); bar.update(len(e)); done_new += len(e)
            rate = done_new / (time.time() - t0)
            log(f'encode {tag}: {min(i + a.enc_chunk, n)}/{n} saved shard ({rate:,.0f} rec/s, '
                f'eta {(n - min(i + a.enc_chunk, n)) / rate / 60:.1f} min)')
    return torch.cat(parts, 0)

for c in countries:
    dst = f'{a.out}/topk_{a.split}_{c}.npz'
    if os.path.exists(dst):
        log(f'[{c}] final result exists, skipping'); continue
    t0 = time.time()
    work = f'{a.out}/work/{a.split}_{c}'; os.makedirs(work, exist_ok=True)

    # stage 1: query sample, pool order, normalized text
    if os.path.exists(f'{work}/text_p.pkl') and os.path.exists(f'{work}/text_q.pkl') and os.path.exists(f'{work}/ids.npz'):
        z = np.load(f'{work}/ids.npz'); q_ids, p_ids = z['q_ids'], z['pool_ids']
        tq, tp = load_pkl(f'{work}/text_q.pkl'), load_pkl(f'{work}/text_p.pkl')
        log(f'[{c}] loaded cached text: {len(tq)} queries, {len(tp)} pool')
    else:
        q = s1[s1.country == c]
        if a.max_queries and len(q) > a.max_queries:
            q = q.sample(a.max_queries, random_state=0)
        p = pool[pool.country == c].reset_index(drop=True)
        log(f'[{c}] normalizing text: {len(q)} queries, {len(p)} pool')
        mk = lambda df, t: (df.business_name.map(fold) + ' | ' + df.business_address.map(fold)).tolist()
        tq, tp = mk(q, 'q'), mk(p, 'p')
        q_ids, p_ids = q.entity_id.values.astype('U16'), p.entity_id.values.astype('U16')
        save_pkl(f'{work}/text_q.pkl', tq); save_pkl(f'{work}/text_p.pkl', tp)
        save_npz(f'{work}/ids.npz', q_ids=q_ids, pool_ids=p_ids)
        log(f'[{c}] saved normalized text + ids')

    # stage 2: embeddings (sharded)
    P = embed(tp, f'{c}-pool', work, 'p')
    Q = embed(tq, f'{c}-queries', work, 'q')

    # stage 3: top-k search, checkpointed every q_ckpt queries
    K = a.k; n = len(tq)
    for cs0 in range(0, n, a.q_ckpt):
        part = f'{work}/part_{cs0:08d}.npz'
        if os.path.exists(part):
            log(f'[{c}] search part {cs0} exists, skipping'); continue
        ce = min(cs0 + a.q_ckpt, n)
        top_s = torch.empty((ce - cs0, K), dtype=torch.float32, device=dev)
        top_i = torch.empty((ce - cs0, K), dtype=torch.int32, device=dev)
        for qs in tqdm(range(cs0, ce, a.q_block), desc=f'search {c} [{cs0}:{ce}]', unit='blk', **PB):
            qb = Q[qs:min(qs + a.q_block, ce)]
            bs = torch.full((len(qb), K), -2.0, dtype=torch.float32, device=dev)
            bi = torch.zeros((len(qb), K), dtype=torch.int32, device=dev)
            for ps in range(0, len(tp), a.pool_block):
                sc = (qb @ P[ps:ps + a.pool_block].T).float()
                v, ix = sc.topk(min(K, sc.shape[1]), dim=1)
                cs = torch.cat([bs, v], 1); ci = torch.cat([bi, (ix + ps).int()], 1)
                bs, o = cs.topk(K, dim=1); bi = ci.gather(1, o)
            top_s[qs - cs0:qs - cs0 + len(qb)] = bs; top_i[qs - cs0:qs - cs0 + len(qb)] = bi
        save_npz(part, cand=top_i.cpu().numpy(), score=top_s.cpu().numpy().astype(np.float16))
        log(f'[{c}] saved search part {cs0}:{ce} of {n}')

    # stage 4: merge parts into the final file
    parts = [np.load(f) for f in sorted(glob.glob(f'{work}/part_*.npz'))]
    save_npz(dst, q_ids=q_ids, pool_ids=p_ids, cand=np.concatenate([x['cand'] for x in parts]),
             score=np.concatenate([x['score'] for x in parts]))
    del P, Q
    if dev == 'cuda':
        torch.cuda.empty_cache()
    log(f'[{c}] DONE in {(time.time() - t0) / 60:.1f} min -> {dst}')
