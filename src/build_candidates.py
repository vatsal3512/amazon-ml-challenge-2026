"""Build candidate_pairs.tsv for the final submission (one row per Source 1 entity, comma-separated candidate ids).
Candidate generation (blocking) = what the pipeline actually used:
  1. dense retrieval: top-8 pool records by fine-tuned bi-encoder cosine (all countries)            -> DENSE file (top-8 lists)
  2. France only: lexical retrieval top-8 (hashed TF-IDF over name tokens / char 4-grams / address)  -> topk_lex_test_France.npz
  3. France only: exact-key blocking (normalised name + house number + street, domain/acronym/pseudo-name keys) -> pairs present in
     the final matches that are not in 1-2
  4. India/US: the matcher scores retrieval ranks 1-20; ranks 9-20 are kept as candidates only when the matcher accepted them (pruning)
Every matched pair is therefore contained in the candidate set (checked at the end).
usage: python build_candidates.py --matches v25.tsv --dense candidate_pairs_k8.tsv.gz --lex out_bi/topk_lex_test_France.npz --out candidate_pairs.tsv"""
import argparse, gzip, numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--matches', required=True); ap.add_argument('--dense', required=True)
ap.add_argument('--lex', required=True); ap.add_argument('--data', default='data'); ap.add_argument('--out', required=True); ap.add_argument('--kl', type=int, default=8)
a = ap.parse_args()
rd = lambda f: pd.read_csv(f, sep='\t', dtype=str, keep_default_na=False)
s1 = rd(f'{a.data}/test/test_source1.tsv')[['entity_id', 'country']]
m = rd(a.matches); match = dict(zip(m.source1_entity_id, m.matched_entity_ids))
d = pd.read_csv(a.dense, sep='\t', dtype=str, keep_default_na=False, compression='gzip' if a.dense.endswith('.gz') else None)
dense = dict(zip(d.source1_entity_id, d.candidate_entity_ids))
z = np.load(a.lex, allow_pickle=True); lq, lp, lc = z['q_ids'], z['pool_ids'], z['cand'][:, :a.kl]
lex = {q: [lp[j] for j in row if j >= 0] for q, row in zip(lq, lc)}
out, sizes, missing = [], {}, 0
for q, c in zip(s1.entity_id, s1.country):
    cand = [x for x in dense.get(q, '').split(',') if x]
    if c == 'France': cand += lex.get(q, [])
    cand += [x for x in match.get(q, '').split(',') if x]
    cand = list(dict.fromkeys(cand))
    ms = set(x for x in match.get(q, '').split(',') if x); missing += len(ms - set(cand))
    out.append((q, ','.join(cand))); sizes.setdefault(c, []).append(len(cand))
assert missing == 0, f'{missing} matched pairs are missing from the candidate set'
pd.DataFrame(out, columns=['source1_entity_id', 'candidate_entity_ids']).to_csv(a.out, sep='\t', index=False)
allz = np.concatenate([np.array(v) for v in sizes.values()])
print(f'wrote {a.out}: {len(out)} rows, {allz.sum()} candidate pairs, mean {allz.mean():.2f} per S1 | ' + ', '.join(f'{c} {np.mean(v):.2f}' for c, v in sizes.items()) + ' | all matched pairs contained: yes')
