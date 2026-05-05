"""
CNN/DailyMail and XSum data pipeline.

Loads dataset, takes a stratified sample by sentence count, splits 64/16/20
into train/val/test, sentence-splits each article and reference summary,
builds oracle extractive labels, and saves everything to a .pt cache.

Tokenization is deferred to training time so the same cache works with
BERT, RoBERTa, DistilBERT, ALBERT, and DeBERTa without rebuilding.

Run (Colab):
    !python src/data.py --dataset cnndm --sample_size 10000
    !python src/data.py --dataset xsum  --sample_size 10000

Output:
    {SAVE_DIR}/{dataset}_{sample_size}.pt
"""

import os
import re
import random
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import RobertaTokenizerFast


SEED    = 42
MAX_SENTS = 50


def setup_save_dir():
    """Return save directory. On Colab, mount Drive manually in a cell first."""
    drive_root = '/content/drive/MyDrive'
    if os.path.exists(drive_root):
        return os.path.join(drive_root, 'GaussianBERTSum/data')
    return './data'


def fast_sent_count(text):
    return max(len(re.findall(r'[.!?]+', text)), 1)


def stratified_sample(articles, sample_size, seed):
    counts = [fast_sent_count(a) for a in articles]
    df = pd.DataFrame({'idx': np.arange(len(articles)), 'n_sents': counts})
    df['bin'] = pd.qcut(df['n_sents'], q=100, labels=False, duplicates='drop')

    bin_counts = df['bin'].value_counts().sort_index()
    quotas = (bin_counts / bin_counts.sum() * sample_size).round().astype(int)

    sampled = []
    for bin_id, n in quotas.items():
        pool = df[df['bin'] == bin_id]
        n = min(n, len(pool))
        sampled.extend(pool.sample(n=n, random_state=seed)['idx'].tolist())

    sampled = sampled[:sample_size]
    random.Random(seed).shuffle(sampled)
    return sampled


def split_indices(n, train_n, val_n, seed):
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order[:train_n], order[train_n:train_n + val_n], order[train_n + val_n:]


def split_sents(text, cap):
    return sent_tokenize(text)[:cap]


def create_oracle_labels(article_sents, summary_sents):
    """Top-3 source sentences by avg ROUGE-1/2/L per reference sentence."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    labels = [0] * len(article_sents)
    for ref in summary_sents:
        scores = []
        for src in article_sents:
            s = scorer.score(src, ref)
            avg_f = (s['rouge1'].fmeasure + s['rouge2'].fmeasure + s['rougeL'].fmeasure) / 3.0
            scores.append(avg_f)
        top3 = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
        for i in top3:
            labels[i] = 1
    return labels


def _oracle_worker(pair):
    """Top-level wrapper so multiprocessing can pickle it."""
    return create_oracle_labels(pair[0], pair[1])


def format_bertsum(sentences, tokenizer, max_len=512):
    """
    Build BERTSum input: [CLS] S1 [SEP] [CLS] S2 [SEP] ...
    Works for BERT / RoBERTa / DistilBERT / ALBERT / DeBERTa by reading
    tokenizer.cls_token_id and sep_token_id directly.
    """
    cls = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id

    input_ids, token_type_ids, cls_positions = [], [], []
    seg = 0
    for sent in sentences:
        sent_ids = tokenizer.encode(sent, add_special_tokens=False)
        if len(input_ids) + len(sent_ids) + 2 > max_len:
            break
        cls_positions.append(len(input_ids))
        input_ids.append(cls)
        token_type_ids.append(seg)
        input_ids.extend(sent_ids)
        token_type_ids.extend([seg] * len(sent_ids))
        input_ids.append(sep)
        token_type_ids.append(seg)
        seg = 1 - seg

    return {
        'input_ids':      input_ids,
        'attention_mask': [1] * len(input_ids),
        'token_type_ids': token_type_ids,
        'cls_positions':  cls_positions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',     default='cnndm',
                        choices=['cnndm', 'xsum', 'multi_news'])
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Number of docs to sample (e.g. 1000 or 10000)')
    parser.add_argument('--workers',     type=int, default=2,
                        help='Parallel workers for oracle label computation')
    args = parser.parse_args()

    # Derive split sizes from sample_size (64/16/20 ratio)
    train_size = int(args.sample_size * 0.64)
    val_size   = int(args.sample_size * 0.16)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    nltk.download('punkt',     quiet=True)
    nltk.download('punkt_tab', quiet=True)

    save_dir = setup_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    print(f'Save dir: {save_dir}  dataset: {args.dataset}  sample_size: {args.sample_size}')

    if args.dataset == 'cnndm':
        print('\n[1/7] Loading CNN/DailyMail 3.0.0 train split...')
        ds        = load_dataset('cnn_dailymail', '3.0.0', split='train')
        articles  = ds['article']
        summaries = ds['highlights']
        source_tag = 'cnn_dailymail/3.0.0'
    elif args.dataset == 'xsum':
        print('\n[1/7] Loading XSum train split...')
        ds        = load_dataset('xsum', split='train')
        articles  = ds['document']
        summaries = ds['summary']
        source_tag = 'xsum'
    elif args.dataset == 'multi_news':
        print('\n[1/7] Loading Multi-News train split...')
        ds        = load_dataset('multi_news', split='train')
        articles  = ds['document']
        summaries = ds['summary']
        source_tag = 'multi_news'
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    print(f'      {len(articles)} train docs available')

    print(f'\n[2/7] Stratified sampling {args.sample_size} docs...')
    sampled_idx       = stratified_sample(articles, args.sample_size, SEED)
    sampled_articles  = [articles[i]  for i in sampled_idx]
    sampled_summaries = [summaries[i] for i in sampled_idx]
    actual_n = len(sampled_idx)
    print(f'      sampled {actual_n} docs')

    test_n = actual_n - train_size - val_size
    print(f'\n[3/7] Splitting {train_size}/{val_size}/{test_n}...')
    train_idx, val_idx, test_idx = split_indices(actual_n, train_size, val_size, SEED)
    print(f'      train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}')

    print(f'\n[4/7] Sentence-splitting (cap at {MAX_SENTS} sentences)...')
    articles_sents  = [split_sents(a, MAX_SENTS) for a in sampled_articles]
    summaries_sents = [split_sents(s, MAX_SENTS) for s in sampled_summaries]
    art_lens = [len(s) for s in articles_sents]
    print(f'      article sents: mean={np.mean(art_lens):.1f}  max={max(art_lens)}')

    print('\n[5/7] Verifying BERTSum formatter with RoBERTa tokenizer...')
    tok  = RobertaTokenizerFast.from_pretrained('roberta-base')
    demo = format_bertsum(articles_sents[0], tok, max_len=512)
    for p in demo['cls_positions']:
        assert demo['input_ids'][p] == tok.cls_token_id
    print(f'      OK — encoded {len(demo["cls_positions"])} sentences into '
          f'{len(demo["input_ids"])} tokens')

    n_workers = min(args.workers, mp.cpu_count())
    print(f'\n[6/7] Building oracle extractive labels ({n_workers} workers)...')
    pairs = list(zip(articles_sents, summaries_sents))
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            oracle_labels = list(tqdm(
                pool.imap(_oracle_worker, pairs, chunksize=20),
                total=actual_n,
            ))
    else:
        oracle_labels = [_oracle_worker(p) for p in tqdm(pairs)]
    pos_rate = np.mean([sum(lbl) / max(len(lbl), 1) for lbl in oracle_labels])
    print(f'      mean positive rate per doc: {pos_rate:.3f}')

    print('\n[7/7] Saving cache...')
    data = {
        'articles_sents':  articles_sents,
        'summaries_sents': summaries_sents,
        'oracle_labels':   oracle_labels,
        'train_idx':       train_idx,
        'val_idx':         val_idx,
        'test_idx':        test_idx,
        'sample_size':     actual_n,
        'max_sents':       MAX_SENTS,
        'seed':            SEED,
        'source_dataset':  source_tag,
    }
    save_path = os.path.join(save_dir, f'{args.dataset}_{actual_n}.pt')
    torch.save(data, save_path)
    print(f'      saved -> {save_path}')
    print(f'      size: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB')

    print('\n--- Verify ---')
    reloaded = torch.load(save_path, weights_only=False)
    doc_id   = reloaded['train_idx'][0]
    sents    = reloaded['articles_sents'][doc_id]
    labels   = reloaded['oracle_labels'][doc_id]
    print(f'Train doc #{doc_id}  (sentences={len(sents)}  positives={sum(labels)})')
    for i, (s, lbl) in enumerate(zip(sents[:8], labels[:8])):
        marker = '[+]' if lbl else '   '
        print(f'  {marker} S{i:02d}: {s[:90]}')

    print(f'\n[done] {args.dataset} {actual_n}-doc cache complete.')


if __name__ == '__main__':
    main()
