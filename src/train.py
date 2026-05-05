"""
Day 2 — Train BERTSum extractive summarizer (G0 baseline) on CNN/DM.

Loads the cache produced by data.py, tokenizes per-batch with a HuggingFace
tokenizer, trains for N epochs with AdamW + linear warmup-decay, evaluates
ROUGE on val each epoch, saves the best checkpoint, and reports test ROUGE.

Run (Colab):
    !python src/train.py

Run (local):
    python src/train.py --epochs 1 --batch_size 4

Output:
    {SAVE_DIR}/../logs/{variant}_{backbone}.csv
    {SAVE_DIR}/../checkpoints/{variant}_{backbone}_best.pt
"""

import os
import csv
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm

from data import format_bertsum, setup_save_dir
from model import BERTSumExt


SEED = 42


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class BERTSumDataset(Dataset):
    """
    Tokenizes on the fly so the same cache works with any backbone.
    Each item returns lists (not tensors) — collate() does the padding.
    """
    def __init__(self, articles_sents, summaries_sents, oracle_labels,
                 indices, tokenizer, max_len=512):
        self.articles  = [articles_sents[i]  for i in indices]
        self.summaries = [summaries_sents[i] for i in indices]
        self.labels    = [oracle_labels[i]   for i in indices]
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, i):
        sents = self.articles[i]
        enc = format_bertsum(sents, self.tokenizer, max_len=self.max_len)
        n_kept = len(enc['cls_positions'])
        labels = self.labels[i][:n_kept]
        return {
            'input_ids':      enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'cls_positions':  enc['cls_positions'],
            'labels':         labels,
            'sentences':      sents[:n_kept],
            'reference':      ' '.join(self.summaries[i]),
        }


def pad(seqs, pad_value, target_len):
    return [s + [pad_value] * (target_len - len(s)) for s in seqs]


def collate(batch, pad_token_id=1):
    max_l = max(len(b['input_ids']) for b in batch)
    max_n = max(len(b['cls_positions']) for b in batch)
    return {
        'input_ids':      torch.tensor(pad([b['input_ids']      for b in batch], pad_token_id, max_l)),
        'attention_mask': torch.tensor(pad([b['attention_mask'] for b in batch], 0, max_l)),
        'cls_positions':  torch.tensor(pad([b['cls_positions']  for b in batch], 0, max_n)),
        'cls_mask':       torch.tensor(pad([[1] * len(b['cls_positions']) for b in batch], 0, max_n), dtype=torch.float),
        'labels':         torch.tensor(pad([b['labels']         for b in batch], 0, max_n), dtype=torch.float),
        'sentences':      [b['sentences']  for b in batch],
        'references':     [b['reference']  for b in batch],
    }


def linear_warmup_decay(optimizer, total_steps, warmup_ratio=0.1):
    warmup = int(total_steps * warmup_ratio)
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    return LambdaLR(optimizer, lr_lambda)


def select_summary(logits, sentences, cls_mask, k=3):
    """Top-k sentences by logit, returned in document order."""
    summaries = []
    B = logits.size(0)
    for b in range(B):
        n_real = int(cls_mask[b].sum().item())
        if n_real == 0:
            summaries.append('')
            continue
        scores = logits[b, :n_real]
        kk = min(k, n_real)
        top_idx = sorted(torch.topk(scores, kk).indices.tolist())
        summaries.append(' '.join(sentences[b][i] for i in top_idx if i < len(sentences[b])))
    return summaries


def evaluate(model, loader, device):
    """Mean ROUGE-1/2/L F-measure over a loader."""
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    r1, r2, rl = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cls_positions  = batch['cls_positions'].to(device)
            cls_mask       = batch['cls_mask'].to(device)

            logits = model(input_ids, attention_mask, cls_positions, cls_mask)
            preds = select_summary(logits.cpu(), batch['sentences'], cls_mask.cpu())
            for pred, ref in zip(preds, batch['references']):
                s = scorer.score(ref, pred)
                r1.append(s['rouge1'].fmeasure)
                r2.append(s['rouge2'].fmeasure)
                rl.append(s['rougeL'].fmeasure)
    return float(np.mean(r1)), float(np.mean(r2)), float(np.mean(rl))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',   default='roberta-base')
    parser.add_argument('--variant',    default='G0', help='Tag for log/ckpt filenames')
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--epochs',     type=int,   default=3)
    parser.add_argument('--lr',         type=float, default=2e-5)
    parser.add_argument('--max_len',       type=int,   default=512)
    parser.add_argument('--max_sents',     type=int,   default=50)
    parser.add_argument('--gauss_lr_mult', type=float, default=100.0,
                        help='LR multiplier for Gaussian parameters vs backbone')
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    save_dir = setup_save_dir()
    cache_path = os.path.join(save_dir, 'cnndm_1000.pt')
    print(f'Loading cache: {cache_path}')
    data = torch.load(cache_path, weights_only=False)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    pad_token_id = tokenizer.pad_token_id

    def make_loader(idx, shuffle):
        ds = BERTSumDataset(
            data['articles_sents'], data['summaries_sents'], data['oracle_labels'],
            idx, tokenizer, max_len=args.max_len,
        )
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            collate_fn=lambda b: collate(b, pad_token_id=pad_token_id),
            num_workers=0,
        )

    train_loader = make_loader(data['train_idx'], shuffle=True)
    val_loader   = make_loader(data['val_idx'],   shuffle=False)
    test_loader  = make_loader(data['test_idx'],  shuffle=False)

    model = BERTSumExt(backbone_name=args.backbone, variant=args.variant,
                       max_sents=args.max_sents).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {args.backbone}  trainable params: {n_params/1e6:.1f}M')

    total_steps = len(train_loader) * args.epochs

    # Gaussian parameters need a much higher lr than the pretrained backbone.
    # Using the same lr=2e-5 causes them to barely move in 3 epochs.
    gauss_param_ids = {id(p) for p in model.gaussian.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in gauss_param_ids]
    gauss_params    = list(model.gaussian.parameters())
    param_groups = [{'params': backbone_params, 'lr': args.lr}]
    if gauss_params:
        param_groups.append({'params': gauss_params, 'lr': args.lr * args.gauss_lr_mult})

    optimizer = AdamW(param_groups, betas=(0.9, 0.999), weight_decay=1e-2)
    scheduler = linear_warmup_decay(optimizer, total_steps, warmup_ratio=0.1)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    project_root = os.path.dirname(save_dir)
    log_dir  = os.path.join(project_root, 'logs')
    ckpt_dir = os.path.join(project_root, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    tag = f'{args.variant}_{args.backbone.replace("/", "_")}'
    csv_path  = os.path.join(log_dir, f'{tag}.csv')
    ckpt_path = os.path.join(ckpt_dir, f'{tag}_best.pt')

    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_r1', 'val_r2', 'val_rl',
                                 'gauss_mu', 'gauss_sigma', 'gauss_alpha_or_wnorm'])

    best_rl = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch in pbar:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cls_positions  = batch['cls_positions'].to(device)
            cls_mask       = batch['cls_mask'].to(device)
            labels         = batch['labels'].to(device)

            logits = model(input_ids, attention_mask, cls_positions, cls_mask)
            per_sent = loss_fn(logits, labels)
            loss = (per_sent * cls_mask).sum() / cls_mask.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        train_loss = float(np.mean(train_losses))
        val_r1, val_r2, val_rl = evaluate(model, val_loader, device)
        print(f'  train_loss={train_loss:.4f}  '
              f'val: R1={val_r1:.4f}  R2={val_r2:.4f}  RL={val_rl:.4f}')
        gauss_params = model.gaussian.log_params()
        if gauss_params:
            print('  gaussian: ' + '  '.join(f'{k}={v:.4f}' for k, v in gauss_params.items()))

        gp = model.gaussian.log_params()
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_r1, val_r2, val_rl,
                gp.get('mu', ''), gp.get('sigma', ''),
                gp.get('alpha', gp.get('w_norm', '')),
            ])

        if val_rl > best_rl:
            best_rl = val_rl
            torch.save(model.state_dict(), ckpt_path)
            print(f'  saved best -> {ckpt_path}  (val_RL={val_rl:.4f})')

    print(f'\nLoading best checkpoint for test eval: {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    test_r1, test_r2, test_rl = evaluate(model, test_loader, device)
    print(f'\n=== Test results ({args.variant}, {args.backbone}) ===')
    print(f'  ROUGE-1: {test_r1:.4f}')
    print(f'  ROUGE-2: {test_r2:.4f}')
    print(f'  ROUGE-L: {test_rl:.4f}')

    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(['test', '', test_r1, test_r2, test_rl])
    print(f'\nLogs:        {csv_path}')
    print(f'Checkpoint:  {ckpt_path}')


if __name__ == '__main__':
    main()
