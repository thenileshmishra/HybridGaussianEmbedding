"""
CNN/DailyMail data pipeline.

Loads CNN/DM, takes a 1000-doc stratified sample by sentence count, splits
64/16/20 into train/val/test, sentence-splits each article and reference
summary, and saves everything to a single .pt cache.

Tokenization is deferred to training time so the same cache works with
RoBERTa, DistilBERT, ALBERT, and DeBERTa without rebuilding.

Run (Colab):
    !python src/data.py

Run (local):
    python src/data.py

Output:
    {SAVE_DIR}/cnndm_1000.pt
"""

import os
import re
import random
import argparse
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import RobertaTokenizerFast


SEED = 42
SAMPLE_SIZE = 1000
MAX_SENTS = 50
TRAIN_SIZE = 640
VAL_SIZE = 160


def setup_save_dir():
    """Return save directory. On Colab, mount Drive manually in a cell first."""
    drive_root = '/content/drive/MyDrive'
    if os.path.exists(drive_root):
        return os.path.join(drive_root, 'GaussianBERTSum/data')
    return './data'


def fast_sent_count(text):
    """Cheap regex-based sentence count for stratification."""
    return max(len(re.findall(r'[.!?]+', text)), 1)


def stratified_sample(articles, sample_size, seed):
    """
    Bin documents by sentence count (100 quantile bins) and sample
    proportionally so the subset has the same length distribution as
    the full set.
    """
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
    """
    BERTSum-standard oracle extractive labels.

    For each reference summary sentence, pick the top-3 source sentences by
    average of ROUGE-1, ROUGE-2, ROUGE-L F-measure. Mark those source
    sentences as 1, rest as 0. This binary vector is the BCE training target.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    labels = [0] * len(article_sents)
    for ref in summary_sents:
        scores = []
        for src in article_sents:
            s = scorer.score(src, ref)
            avg_f = (s['rouge1'].fmeasure
                     + s['rouge2'].fmeasure
                     + s['rougeL'].fmeasure) / 3.0
            scores.append(avg_f)
        top3 = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
        for i in top3:
            labels[i] = 1
    return labels


def format_bertsum(sentences, tokenizer, max_len=512):
    """
    Build BERTSum input format: [CLS] S1 [SEP] [CLS] S2 [SEP] ...

    Reads tokenizer.cls_token_id and sep_token_id directly, so the same
    function works for RoBERTa / DistilBERT / ALBERT / DeBERTa. Truncates
    by dropping later sentences if total length would exceed max_len.

    Returns:
        dict with input_ids, attention_mask, token_type_ids, cls_positions
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
        'input_ids': input_ids,
        'attention_mask': [1] * len(input_ids),
        'token_type_ids': token_type_ids,
        'cls_positions': cls_positions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cnndm', choices=['cnndm', 'xsum'],
                        help='Dataset to process: cnndm or xsum')
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    save_dir = setup_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    print(f'Save dir: {save_dir}  dataset: {args.dataset}')

    if args.dataset == 'cnndm':
        print('\n[1/7] Loading CNN/DailyMail 3.0.0 train split...')
        ds = load_dataset('cnn_dailymail', '3.0.0', split='train')
        articles  = ds['article']
        summaries = ds['highlights']
        source_tag = 'cnn_dailymail/3.0.0'
    else:
        print('\n[1/7] Loading XSum train split...')
        ds = load_dataset('xsum', split='train')
        articles  = ds['document']
        summaries = ds['summary']
        source_tag = 'xsum'
    print(f'      {len(articles)} train docs')

    print('\n[2/7] Stratified sampling 1000 docs...')
    sampled_idx = stratified_sample(articles, SAMPLE_SIZE, SEED)
    sampled_articles  = [articles[i]  for i in sampled_idx]
    sampled_summaries = [summaries[i] for i in sampled_idx]
    actual_n = len(sampled_idx)
    print(f'      sampled {actual_n} docs')

    test_n = actual_n - TRAIN_SIZE - VAL_SIZE
    print(f'\n[3/7] Splitting {TRAIN_SIZE}/{VAL_SIZE}/{test_n}...')
    train_idx, val_idx, test_idx = split_indices(actual_n, TRAIN_SIZE, VAL_SIZE, SEED)
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
    print(f'      OK - encoded {len(demo["cls_positions"])} sentences into '
          f'{len(demo["input_ids"])} tokens')

    print('\n[6/7] Building oracle extractive labels (top-3 ROUGE)...')
    oracle_labels = [
        create_oracle_labels(a, s)
        for a, s in tqdm(list(zip(articles_sents, summaries_sents)), total=actual_n)
    ]
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
    save_path = os.path.join(save_dir, f'{args.dataset}_1000.pt')
    torch.save(data, save_path)
    print(f'      saved -> {save_path}')
    print(f'      size: {os.path.getsize(save_path) / 1024:.1f} KB')

    print('\n--- Verify ---')
    reloaded = torch.load(save_path, weights_only=False)
    doc_id = reloaded['train_idx'][0]
    sents  = reloaded['articles_sents'][doc_id]
    labels = reloaded['oracle_labels'][doc_id]
    print(f'Train doc #{doc_id}  (sentences={len(sents)}  positives={sum(labels)})')
    for i, (s, lbl) in enumerate(zip(sents, labels)):
        marker = '[+]' if lbl else '   '
        print(f'  {marker} S{i:02d}: {s[:90]}')
    print(f'\nReference summary ({len(reloaded["summaries_sents"][doc_id])} sentences):')
    for i, s in enumerate(reloaded['summaries_sents'][doc_id]):
        print(f'  S{i}: {s}')

    print(f'\n[done] {args.dataset} data pipeline complete.')


if __name__ == '__main__':
    main()
