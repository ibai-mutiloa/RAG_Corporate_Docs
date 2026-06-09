#!/usr/bin/env python3
"""Local RAGAS-like evaluator for dataset JSON files.

Usage: python3 scripts/evaluate_ragas.py tests/ragas_eval/RAGAS_dataset.json

Computes per-sample and aggregate metrics: exact match, token-F1, ROUGE-L, BLEU-4, lexical precision.
Saves results to tests/ragas_eval/ragas_results_local_TIMESTAMP.json
"""
import json
import sys
import math
import re
from collections import Counter
from datetime import datetime


def normalize(text: str):
    if text is None:
        return []
    text = text.lower().strip()
    # keep words and numbers
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    return tokens


def token_f1(ref, pred):
    ref_toks = normalize(ref)
    pred_toks = normalize(pred)
    if len(ref_toks) == 0 and len(pred_toks) == 0:
        return 1.0
    if len(ref_toks) == 0 or len(pred_toks) == 0:
        return 0.0
    common = Counter(ref_toks) & Counter(pred_toks)
    common_count = sum(common.values())
    prec = common_count / len(pred_toks)
    rec = common_count / len(ref_toks)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def lcs_len(a, b):
    # classic dynamic programming LCS on token lists
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def rouge_l_f1(ref, pred):
    ref_toks = normalize(ref)
    pred_toks = normalize(pred)
    if len(ref_toks) == 0 or len(pred_toks) == 0:
        return 0.0
    lcs = lcs_len(ref_toks, pred_toks)
    prec = lcs / len(pred_toks)
    rec = lcs / len(ref_toks)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def bleu_4(ref, pred):
    ref_toks = normalize(ref)
    pred_toks = normalize(pred)
    if len(pred_toks) == 0:
        return 0.0
    precisions = []
    for n in range(1,5):
        ref_ngrams = Counter(ngrams(ref_toks, n))
        pred_ngrams = Counter(ngrams(pred_toks, n))
        if len(pred_ngrams) == 0:
            precisions.append(0.0)
            continue
        common = sum((ref_ngrams & pred_ngrams).values())
        precisions.append(common / sum(pred_ngrams.values()))
    # geometric mean with smoothing
    smooth = 1e-9
    log_sum = 0.0
    for p in precisions:
        log_sum += math.log(p + smooth)
    geo_mean = math.exp(log_sum / 4)
    # brevity penalty
    ref_len = len(ref_toks)
    pred_len = len(pred_toks)
    if pred_len == 0:
        bp = 0.0
    elif pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / pred_len)
    return bp * geo_mean


def lexical_precision(ref, pred):
    ref_toks = normalize(ref)
    pred_toks = normalize(pred)
    if len(pred_toks) == 0:
        return 0.0
    common = Counter(ref_toks) & Counter(pred_toks)
    return sum(common.values()) / len(pred_toks)


def evaluate(dataset):
    results = []
    agg = {
        'exact_match': 0.0,
        'token_f1': 0.0,
        'rouge_l': 0.0,
        'bleu_4': 0.0,
        'lexical_precision': 0.0,
    }
    n = len(dataset)
    for item in dataset:
        user_input = item.get('user_input')
        pred = item.get('response', '')
        ref = item.get('ground_truth') or item.get('reference') or ''
        em = 1.0 if (pred or '').strip() == (ref or '').strip() and len(pred.strip())>0 else 0.0
        tf1 = token_f1(ref, pred)
        rl = rouge_l_f1(ref, pred)
        b4 = bleu_4(ref, pred)
        lp = lexical_precision(ref, pred)
        agg['exact_match'] += em
        agg['token_f1'] += tf1
        agg['rouge_l'] += rl
        agg['bleu_4'] += b4
        agg['lexical_precision'] += lp
        results.append({
            'user_input': user_input,
            'reference': ref,
            'response': pred,
            'exact_match': em,
            'token_f1': tf1,
            'rouge_l': rl,
            'bleu_4': b4,
            'lexical_precision': lp,
        })
    # averages
    summary = {k: (v / n if n>0 else 0.0) for k, v in agg.items()}
    return results, summary


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 scripts/evaluate_ragas.py tests/ragas_eval/RAGAS_dataset.json')
        sys.exit(1)
    path = sys.argv[1]
    with open(path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    results, summary = evaluate(dataset)
    now = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_path = f'tests/ragas_eval/ragas_results_local_{now}.json'
    out = {'summary': summary, 'results': results}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('Wrote', out_path)
    print('Summary:')
    for k, v in summary.items():
        print(f'  {k}: {v:.6f}')


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import json
import re
import sys
from collections import Counter
import csv
import csv
import math
from math import sqrt

# normalize and token utilities (always available)
def normalize(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def tokens(text):
    return normalize(text).split()


# try to import sentence-transformers for embedding-based relevancy if available
EMBEDDINGS_AVAILABLE = False
model = None
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer('all-MiniLM-L6-v2')
    EMBEDDINGS_AVAILABLE = True
except Exception:
    model = None
    EMBEDDINGS_AVAILABLE = False


def exact_match(a, b):
    return normalize(a) == normalize(b)


def f1_score(pred, ref):
    p_tok = tokens(pred)
    r_tok = tokens(ref)
    if not p_tok and not r_tok:
        return 1.0
    if not p_tok or not r_tok:
        return 0.0
    common = Counter(p_tok) & Counter(r_tok)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p_tok)
    rec = num_same / len(r_tok)
    return 2 * prec * rec / (prec + rec)


def lcs_length(a, b):
    # classic DP LCS length
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, n + 1):
            tmp = dp[j]
            if ai == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def rouge_l_score(pred, ref):
    p_tok = tokens(pred)
    r_tok = tokens(ref)
    lcs = lcs_length(p_tok, r_tok)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p_tok) if p_tok else 0.0
    rec = lcs / len(r_tok) if r_tok else 0.0
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def ngram_counts(tokens_list, n):
    counts = Counter()
    for i in range(len(tokens_list) - n + 1):
        counts[tuple(tokens_list[i:i + n])] += 1
    return counts


def sentence_bleu(pred, ref, max_n=4, smooth=1.0):
    p_tok = tokens(pred)
    r_tok = tokens(ref)
    if not p_tok:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        p_counts = ngram_counts(p_tok, n)
        r_counts = ngram_counts(r_tok, n)
        if not p_counts:
            precisions.append(0.0)
            continue
        match = 0
        total = 0
        for ng, cnt in p_counts.items():
            match += min(cnt, r_counts.get(ng, 0))
            total += cnt
        # add-one smoothing
        precisions.append((match + smooth) / (total + smooth))

    # geometric mean
    log_sum = 0.0
    for p in precisions:
        if p <= 0:
            log_sum += -9999
        else:
            log_sum += math.log(p)
    geo_mean = math.exp(log_sum / max_n)

    # brevity penalty
    ref_len = len(r_tok)
    pred_len = len(p_tok)
    if pred_len == 0:
        bp = 0.0
    elif pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / pred_len)
    return bp * geo_mean


def lexical_precision(pred, ref):
    p_tok = tokens(pred)
    r_set = set(tokens(ref))
    if not p_tok:
        return 0.0
    in_ref = sum(1 for t in p_tok if t in r_set)
    return in_ref / len(p_tok)


def context_recall(pred, ref):
    # fraction of reference tokens present in prediction
    p_set = set(tokens(pred))
    r_tok = tokens(ref)
    if not r_tok:
        return 0.0
    in_pred = sum(1 for t in r_tok if t in p_set)
    return in_pred / len(r_tok)


def jaccard_similarity(a, b):
    sa = set(tokens(a))
    sb = set(tokens(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union)


def embedding_similarity(a, b):
    if not EMBEDDINGS_AVAILABLE or not a or not b:
        return None
    emb = model.encode([a, b], convert_to_numpy=True)
    v1, v2 = emb[0], emb[1]
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def answer_relevancy(resp, user_input, gold=None):
    # Prefer embedding similarity between response and reference (gold) when available.
    # If gold is not provided, fall back to comparing to the user_input. If embeddings
    # are unavailable, fallback to Jaccard token similarity.
    target = gold if gold else user_input
    sim = None
    if EMBEDDINGS_AVAILABLE:
        sim = embedding_similarity(resp, target)
    if sim is None:
        sim = jaccard_similarity(resp, target)
    return sim


def faithfulness_proxy(resp, gold):
    # primary: lexical precision; if embeddings available, use similarity to gold as secondary
    lex = lexical_precision(resp, gold)
    if EMBEDDINGS_AVAILABLE:
        emb = embedding_similarity(resp, gold)
        if emb is not None:
            # combine lexical and embedding similarity
            return 0.6 * lex + 0.4 * emb
    return lex


def main(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    em = 0
    f1s = []
    rouge_ls = []
    bleus = []
    lex_prec = []
    human_counts = Counter()

    out_rows = []
    ctx_recs = []
    relevs = []
    faiths = []
    for i, ex in enumerate(data, 1):
        resp = ex.get("response", "")
        gold = ex.get("ground_truth", "")
        user_q = ex.get("user_input", "")
        em_i = 1 if exact_match(resp, gold) else 0
        f1_i = f1_score(resp, gold)
        rouge_i = rouge_l_score(resp, gold)
        bleu_i = sentence_bleu(resp, gold)
        lex_i = lexical_precision(resp, gold)
        ctx_rec_i = context_recall(resp, gold)
        relev_i = answer_relevancy(resp, user_q, gold)
        faith_i = faithfulness_proxy(resp, gold)

        em += em_i
        f1s.append(f1_i)
        rouge_ls.append(rouge_i)
        bleus.append(bleu_i)
        lex_prec.append(lex_i)
        ctx_recs.append(ctx_rec_i)
        relevs.append(relev_i if relev_i is not None else 0.0)
        faiths.append(faith_i if faith_i is not None else 0.0)
        human_counts[ex.get("evaluation", "")] += 1

        out_rows.append({
            "idx": i,
            "user_input": user_q,
            "response": resp,
            "ground_truth": gold,
            "EM": em_i,
            "F1": f1_i,
            "ROUGE-L": rouge_i,
            "BLEU": bleu_i,
            "lexical_precision": lex_i,
            "context_recall": ctx_rec_i,
            "answer_relevancy": relev_i,
            "faithfulness": faith_i,
            "human_eval": ex.get("evaluation", ""),
        })

    avg_f1 = sum(f1s) / total if total else 0.0
    avg_rouge = sum(rouge_ls) / total if total else 0.0
    avg_bleu = sum(bleus) / total if total else 0.0
    avg_lex = sum(lex_prec) / total if total else 0.0
    em_rate = em / total if total else 0.0
    avg_ctx_rec = sum(ctx_recs) / total if total else 0.0
    avg_relev = sum(relevs) / total if total else 0.0
    avg_faith = sum(faiths) / total if total else 0.0

    print(f"Examples: {total}")
    print(f"Exact Match: {em} / {total} = {em_rate:.3f}")
    print(f"Average token F1: {avg_f1:.3f}")
    print(f"Average ROUGE-L: {avg_rouge:.3f}")
    print(f"Average BLEU-4: {avg_bleu:.3f}")
    print(f"Average lexical precision (precision proxy): {avg_lex:.3f}")
    print(f"Average context recall: {avg_ctx_rec:.3f}")
    print(f"Average answer relevancy: {avg_relev:.3f}")
    print(f"Average faithfulness (combined proxy): {avg_faith:.3f}")
    print("Human evaluation label distribution:")
    for k, v in human_counts.most_common():
        print(f" - {k}: {v}")

    # write per-sample CSV
    csv_path = 'scripts/ragas_eval_results.csv'
    keys = ["idx", "user_input", "response", "ground_truth", "EM", "F1", "ROUGE-L", "BLEU", "lexical_precision", "context_recall", "answer_relevancy", "faithfulness", "human_eval"]
    with open(csv_path, 'w', encoding='utf-8', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=keys)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"Per-sample results written to {csv_path}")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'tests/ragas_eval/RAGAS_dataset.json'
    main(path)
