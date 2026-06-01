#!/usr/bin/env python3
"""Call local API /search for each item in RAGAS dataset and evaluate responses.

Usage: python3 scripts/run_ragas_via_api.py tests/ragas_eval/RAGAS_dataset.json

Outputs:
 - tests/ragas_eval/ragas_results_api_TIMESTAMP.json (raw API responses)
 - prints a short evaluation summary using token-F1, ROUGE-L, BLEU-4, lexical precision
"""
import sys
import json
import time
from datetime import datetime
import urllib.request
import urllib.error
import urllib.parse
import ssl

try:
    import requests
except Exception:
    requests = None

from evaluate_ragas import normalize, token_f1, rouge_l_f1, bleu_4, lexical_precision


def post_json(url, payload, timeout=30):
    headers = {'Content-Type': 'application/json'}
    if requests:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    else:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.load(resp)


def evaluate_responses(dataset, responses):
    n = len(dataset)
    agg = {'token_f1': 0.0, 'rouge_l': 0.0, 'bleu_4': 0.0, 'lexical_precision': 0.0, 'exact_match': 0.0}
    for item, resp in zip(dataset, responses):
        ref = item.get('ground_truth') or item.get('reference') or ''
        # API may return 'answer' or fallback to empty
        ans = ''
        if isinstance(resp, dict):
            ans = (resp.get('answer') or '')
            # sometimes answer may be nested under 'data' key
            if not ans and 'data' in resp and isinstance(resp['data'], dict):
                ans = resp['data'].get('answer', '')

        agg['token_f1'] += token_f1(ref, ans)
        agg['rouge_l'] += rouge_l_f1(ref, ans)
        agg['bleu_4'] += bleu_4(ref, ans)
        agg['lexical_precision'] += lexical_precision(ref, ans)
        agg['exact_match'] += 1.0 if (ref or '').strip() and ref.strip() == (ans or '').strip() else 0.0

    summary = {k: (v / n if n>0 else 0.0) for k, v in agg.items()}
    return summary


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 scripts/run_ragas_via_api.py tests/ragas_eval/RAGAS_dataset.json')
        sys.exit(1)

    dataset_path = sys.argv[1]
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    url = 'http://127.0.0.1:5000/search'
    responses = []
    for i, item in enumerate(dataset, start=1):
        q = item.get('user_input') or item.get('question') or ''
        payload = {'question': q, 'generate_answer': True, 'top_k': 6}
        try:
            resp = post_json(url, payload, timeout=60)
        except Exception as e:
            resp = {'error': str(e)}
        responses.append(resp)
        print(f'[{i}/{len(dataset)}] Question sent. status: {"error" in resp and "ERR" or "OK"}')
        time.sleep(0.4)

    now = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_path = f'tests/ragas_eval/ragas_results_api_{now}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'dataset_path': dataset_path, 'results': responses}, f, ensure_ascii=False, indent=2)

    print('Wrote raw API responses to', out_path)

    summary = evaluate_responses(dataset, responses)
    print('Evaluation summary:')
    print('  token_f1: %.6f' % summary['token_f1'])
    print('  rouge_l:  %.6f' % summary['rouge_l'])
    print('  bleu_4:   %.6f' % summary['bleu_4'])
    print('  lexical_precision: %.6f' % summary['lexical_precision'])
    print('  exact_match: %.6f' % summary['exact_match'])


if __name__ == '__main__':
    main()
