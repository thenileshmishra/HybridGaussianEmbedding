"""
ParaCNN training loop — faithful re-implementation of the notebook
(see important/ParaCNN_PosEmbed.ipynb).

Loads the cache produced by data.py (sentence-split documents + oracle
extractive labels) and trains the ParaCNNExt model per-document, using
AdamW lr=0.002 + StepLR(step=2, gamma=0.9) + BCEWithLogitsLoss exactly as
in the notebook.  Top-3 sentence extraction at eval, ROUGE/BERTScore on
held-out test split.

Run (Colab):
    python src/train.py --variant hybrid --epochs 2 --batch_size 8 --sample_size 10000

Variants:
    sinu    — sinusoidal-only baseline (s_max=s_min=mu=sigma=0)
    hybrid  — sinusoidal + windowed Gaussian (paper's GA-tuned defaults)

Output:
    {SAVE_DIR}/../logs/{variant}_paracnn{ds_suffix}{size_suffix}.csv
    {SAVE_DIR}/../checkpoints/{variant}_paracnn{ds_suffix}{size_suffix}_best.pt
"""

import os
import csv
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from transformers import RobertaTokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm

from data import setup_save_dir
from model import ParaCNNExt


SEED = 42


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def select_summary(logits, sentences, k=3):
    """Top-k sentences by logit, returned in document order."""
    n = logits.size(-1)
    if n == 0:
        return ''
    kk = min(k, n)
    top_idx = sorted(torch.topk(logits.squeeze(0), kk).indices.tolist())
    return ' '.join(sentences[i] for i in top_idx if i < len(sentences))


def evaluate(model, sentences_list, labels_list, references_list, device,
             s_max, s_min, mu, sigma, collect=False):
    """ROUGE-1/2/L means (and optionally per-doc lists + texts)."""
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    r1, r2, rl = [], [], []
    all_preds, all_refs = [], []
    with torch.no_grad():
        for sents, labels, ref in zip(sentences_list, labels_list, references_list):
            if len(sents) == 0:
                continue
            logits, n_fit = model(sents, s_max, s_min, mu, sigma)
            if logits is None or n_fit == 0:
                pred = ''
            else:
                pred = select_summary(logits.cpu(), sents[:n_fit], k=3)
            s = scorer.score(ref, pred)
            r1.append(s['rouge1'].fmeasure)
            r2.append(s['rouge2'].fmeasure)
            rl.append(s['rougeL'].fmeasure)
            if collect:
                all_preds.append(pred)
                all_refs.append(ref)
    if collect:
        return (float(np.mean(r1)), float(np.mean(r2)), float(np.mean(rl)),
                r1, r2, rl, all_preds, all_refs)
    return float(np.mean(r1)), float(np.mean(r2)), float(np.mean(rl))


def variant_to_params(variant):
    """Map variant name -> (s_max, s_min, mu, sigma)."""
    if variant == 'sinu':
        return 0.0, 0.0, 0, 0
    if variant == 'hybrid':
        return 2.0, 0.3, 260, 65
    raise ValueError(f'Unknown variant: {variant}  (use sinu | hybrid)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant',     default='hybrid', choices=['sinu', 'hybrid'])
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--epochs',      type=int,   default=2)
    parser.add_argument('--lr',          type=float, default=2e-3)
    parser.add_argument('--max_len',     type=int,   default=800)
    parser.add_argument('--num_inter_layers', type=int, default=2)
    parser.add_argument('--heads',       type=int,   default=6)
    parser.add_argument('--dropout',     type=float, default=0.1)
    parser.add_argument('--max_position_embeddings', type=int, default=1024)
    parser.add_argument('--dataset',     default='cnndm', choices=['cnndm', 'xsum'])
    parser.add_argument('--sample_size', type=int,   default=1000,
                        help='Must match the sample_size used in data.py')
    parser.add_argument('--eval_only',   action='store_true',
                        help='Skip training, load best checkpoint and run test eval only')
    parser.add_argument('--paracnn_strict', action='store_true',
                        help='Match the notebook EXACTLY (per-doc optimizer.step, '
                             'inter_encoder excluded from optimizer, '
                             'gradient clip only on RoBERTa params). '
                             'Reproduces the original notebook including its known bugs.')
    # ParaCNN Gaussian hyperparameters (override variant defaults if provided)
    parser.add_argument('--s_max', type=float, default=None)
    parser.add_argument('--s_min', type=float, default=None)
    parser.add_argument('--mu',    type=int,   default=None)
    parser.add_argument('--sigma', type=int,   default=None)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    s_max, s_min, mu, sigma = variant_to_params(args.variant)
    if args.s_max is not None: s_max = args.s_max
    if args.s_min is not None: s_min = args.s_min
    if args.mu    is not None: mu    = args.mu
    if args.sigma is not None: sigma = args.sigma
    print(f'Variant: {args.variant}   s_max={s_max}  s_min={s_min}  mu={mu}  sigma={sigma}')

    # ---- Load cached data ----
    save_dir = setup_save_dir()
    cache_path = os.path.join(save_dir, f'{args.dataset}_{args.sample_size}.pt')
    print(f'Loading cache: {cache_path}')
    data = torch.load(cache_path, weights_only=False)

    articles  = data['articles_sents']
    summaries = data['summaries_sents']
    labels    = data['oracle_labels']

    train_idx = data['train_idx']
    val_idx   = data['val_idx']
    test_idx  = data['test_idx']

    def _slice(idx):
        sents = [articles[i] for i in idx]
        labs  = [labels[i]   for i in idx]
        refs  = [' '.join(summaries[i]) for i in idx]
        return sents, labs, refs

    train_sents, train_labs, train_refs = _slice(train_idx)
    val_sents,   val_labs,   val_refs   = _slice(val_idx)
    test_sents,  test_labs,  test_refs  = _slice(test_idx)

    print(f'Train docs: {len(train_sents)}  Val: {len(val_sents)}  Test: {len(test_sents)}')

    # ---- Build model ----
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = ParaCNNExt(
        max_len=args.max_len,
        d=768,
        num_inter_layers=args.num_inter_layers,
        heads=args.heads,
        dropout=args.dropout,
        max_position_embeddings=args.max_position_embeddings,
        tokenizer=tokenizer,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: paracnn  trainable params: {n_params/1e6:.1f}M')

    # ---- Optimizer / scheduler / loss (matches notebook) ----
    if args.paracnn_strict:
        # Notebook excludes inter_encoder from the optimizer (a bug in the original).
        # Only RoBERTa wrapper + logit head are trained; inter_encoder stays frozen at random init.
        optim_params = list(model.embedder.parameters()) \
                     + list(model.roberta.parameters()) \
                     + list(model.logit_layer.parameters())
        clip_params = list(model.roberta.parameters())   # notebook clips only RoBERTa
    else:
        optim_params = list(model.parameters())
        clip_params  = list(model.parameters())
    optimizer = AdamW(optim_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.9)
    loss_fn = nn.BCEWithLogitsLoss()
    print(f'Optimizer: {len(optim_params)} param tensors  '
          f'(strict={args.paracnn_strict}, inter_encoder '
          f'{"excluded" if args.paracnn_strict else "included"})')

    # ---- Paths ----
    project_root = os.path.dirname(save_dir)
    log_dir  = os.path.join(project_root, 'logs')
    ckpt_dir = os.path.join(project_root, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    ds_suffix   = f'_{args.dataset}' if args.dataset != 'cnndm' else ''
    size_suffix = f'_{args.sample_size}' if args.sample_size != 1000 else ''
    tag = f'{args.variant}_paracnn{ds_suffix}{size_suffix}'
    csv_path  = os.path.join(log_dir, f'{tag}.csv')
    ckpt_path = os.path.join(ckpt_dir, f'{tag}_best.pt')

    if not args.eval_only:
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_r1', 'val_r2', 'val_rl',
                                     's_max', 's_min', 'mu', 'sigma'])

    # ---- Training ----
    best_rl = -1.0
    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses = []
            order = list(range(len(train_sents)))
            random.shuffle(order)

            pbar = tqdm(range(0, len(order), args.batch_size),
                        desc=f'Epoch {epoch}/{args.epochs}')
            for start in pbar:
                batch_order = order[start:start + args.batch_size]
                if not args.paracnn_strict:
                    optimizer.zero_grad()
                batch_loss_total = 0.0
                n_in_batch = 0
                for j in batch_order:
                    sents = train_sents[j]
                    lbls  = train_labs[j]
                    if len(sents) == 0 or len(lbls) == 0:
                        continue
                    logits, n_fit = model(sents, s_max, s_min, mu, sigma)
                    if logits is None or n_fit == 0:
                        continue
                    target = torch.tensor(lbls[:n_fit], dtype=torch.float,
                                          device=device).unsqueeze(0)
                    raw = loss_fn(logits, target)
                    if args.paracnn_strict:
                        # Notebook: optimizer.step() PER DOCUMENT (effective batch_size=1).
                        optimizer.zero_grad()
                        raw.backward()
                        torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                        optimizer.step()
                    else:
                        (raw / max(len(batch_order), 1)).backward()
                    batch_loss_total += raw.item()
                    n_in_batch += 1

                if n_in_batch > 0:
                    if not args.paracnn_strict:
                        torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                        optimizer.step()
                    train_losses.append(batch_loss_total / n_in_batch)
                    pbar.set_postfix(loss=f'{train_losses[-1]:.4f}')

            scheduler.step()

            train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
            val_r1, val_r2, val_rl = evaluate(
                model, val_sents, val_labs, val_refs, device,
                s_max, s_min, mu, sigma,
            )
            print(f'  train_loss={train_loss:.4f}  '
                  f'val: R1={val_r1:.4f}  R2={val_r2:.4f}  RL={val_rl:.4f}')

            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch, train_loss, val_r1, val_r2, val_rl,
                    s_max, s_min, mu, sigma,
                ])

            if val_rl > best_rl:
                best_rl = val_rl
                torch.save(model.state_dict(), ckpt_path)
                print(f'  saved best -> {ckpt_path}  (val_RL={val_rl:.4f})')

    # ---- Test ----
    print(f'\nLoading best checkpoint for test eval: {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    test_r1, test_r2, test_rl, docs_r1, docs_r2, docs_rl, all_preds, all_refs = evaluate(
        model, test_sents, test_labs, test_refs, device,
        s_max, s_min, mu, sigma, collect=True,
    )

    print(f'\n=== Test results ({args.variant}, paracnn, {args.dataset}) ===')
    print(f'  ROUGE-1: {test_r1:.4f}')
    print(f'  ROUGE-2: {test_r2:.4f}')
    print(f'  ROUGE-L: {test_rl:.4f}')

    test_bs = ''
    try:
        from bert_score import score as bert_score_fn
        _, _, bf = bert_score_fn(all_preds, all_refs, lang='en',
                                  model_type='distilbert-base-uncased',
                                  verbose=False, device=str(device))
        test_bs = round(bf.mean().item(), 4)
        print(f'  BERTScore-F1: {test_bs:.4f}')
    except Exception as e:
        print(f'  BERTScore skipped: {e}')

    scores_path = os.path.join(log_dir, f'{tag}_test_scores.json')
    with open(scores_path, 'w') as f:
        json.dump({'r1': docs_r1, 'r2': docs_r2, 'rl': docs_rl}, f)

    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(['test', '', test_r1, test_r2, test_rl, '', '', '', test_bs])
    print(f'\nLogs:        {csv_path}')
    print(f'Scores:      {scores_path}')
    print(f'Checkpoint:  {ckpt_path}')


if __name__ == '__main__':
    main()
