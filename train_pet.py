"""Train and score the desk-pet instruction model.

Reuses the upstream TinyLM architecture unchanged (src/model.py from
manjunathshiva/esp32-tinyllm) and replaces only the data path and the metric.

Perplexity is not reported. The deliverable is whether a ~2M-parameter core maps
an instruction to the right action sequence, so scoring is exact-sequence
accuracy under greedy decoding constrained to the plan span -- the same
constraint the firmware would apply. Broken down by level, because one overall
number hides the only interesting failure: composition.

Loss is masked to the plan region. Training the model to predict the operator's
own words wastes capacity on a distribution it never has to generate.
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tinyllm", "src"))
from model import Config, TinyLM  # noqa: E402

import vocab as V  # noqa: E402

SEQ_LEN = 24  # measured max encoded length is 20


def load_rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def encode_split(rows, stoi, seq_len):
    ids, prompt_lens, totals, levels = [], [], [], []
    for row in rows:
        got = V.encode(row, stoi, seq_len)
        if got is None:
            continue
        seq, plen, total = got
        ids.append(seq)
        prompt_lens.append(plen)
        totals.append(total)
        levels.append(row["level"])
    return (torch.tensor(ids, dtype=torch.long),
            torch.tensor(prompt_lens), torch.tensor(totals), levels)


def masked_loss(logits, ids, prompt_lens, totals):
    """Cross-entropy over plan positions only.

    Targets are ids shifted left by one. A position is supervised when it is the
    last prompt token (predicting the first plan token) or inside the plan.
    """
    B, T, Vsz = logits.shape
    targets = torch.full((B, T), -100, dtype=torch.long, device=ids.device)
    pos = torch.arange(T, device=ids.device)[None, :]
    supervise = (pos >= (prompt_lens[:, None] - 1)) & (pos < (totals[:, None] - 1))
    shifted = torch.roll(ids, -1, dims=1)
    targets[supervise] = shifted[supervise]
    return F.cross_entropy(logits.reshape(-1, Vsz), targets.reshape(-1),
                           ignore_index=-100)


@torch.no_grad()
def score(model, stoi, itos, rows, device, seq_len, max_new=8, batch=512):
    """Greedy decode with logits masked to the plan span; exact-match by level."""
    model.eval()
    lo, hi = V.plan_span(stoi)
    allowed = torch.full((len(itos),), float("-inf"), device=device)
    allowed[lo:hi + 1] = 0.0

    per_level, action_hits, count_total, count_hits = {}, [0, 0], 0, 0
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        prompts, golds, levels = [], [], []
        for row in chunk:
            words = V.words_of(row["text"])
            seq = [V.BOS] + [stoi.get(w, stoi[V.UNK]) for w in words]
            if len(seq) + max_new > seq_len:
                seq = seq[:seq_len - max_new]
            prompts.append(seq)
            golds.append([stoi[t] for t in row["plan"]])
            levels.append(row["level"])

        # One fixed-width buffer; each row's new token is written at ITS OWN
        # cursor. Appending to a shared column instead would leave PAD between a
        # short prompt and its generated tokens, so the model would score a
        # sequence it was never trained on.
        B = len(prompts)
        idx = torch.full((B, seq_len), V.PAD, dtype=torch.long, device=device)
        lengths = torch.tensor([len(p) for p in prompts], device=device)
        for i, p in enumerate(prompts):
            idx[i, :len(p)] = torch.tensor(p, device=device)

        out = [[] for _ in prompts]
        done = torch.zeros(B, dtype=torch.bool, device=device)
        cursor = lengths.clone()
        rows_ix = torch.arange(B, device=device)
        for _ in range(max_new):
            logits, _ = model(idx)
            step = logits[rows_ix, cursor - 1]
            nxt = (step + allowed).argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, V.EOS), nxt)
            for i, tok in enumerate(nxt.tolist()):
                if not done[i] and tok != V.EOS:
                    out[i].append(tok)
            done |= nxt == V.EOS
            if bool(done.all()):
                break
            write = torch.clamp(cursor, max=seq_len - 1)
            idx[rows_ix, write] = torch.where(done, idx[rows_ix, write], nxt)
            cursor = torch.clamp(cursor + 1, max=seq_len - 1)

        for pred, gold, lvl in zip(out, golds, levels):
            bucket = per_level.setdefault(lvl, [0, 0])
            bucket[1] += 1
            if pred == gold:
                bucket[0] += 1
            # action-only and count-only breakdowns
            pa = [t for t in pred if itos[t] in V.ACTION_TOKENS]
            ga = [t for t in gold if itos[t] in V.ACTION_TOKENS]
            action_hits[1] += 1
            if pa == ga:
                action_hits[0] += 1
            pc = [t for t in pred if itos[t] in V.COUNT_TOKENS]
            gc = [t for t in gold if itos[t] in V.COUNT_TOKENS]
            if gc:
                count_total += 1
                if pc == gc:
                    count_hits += 1
    model.train()
    return per_level, action_hits, (count_hits, count_total)


def report(tag, per_level, action_hits, counts):
    print(f"\n--- {tag} ---")
    order = ["l1", "l2", "l3"] + sorted(k for k in per_level if k.startswith("neg"))
    for lvl in order:
        if lvl not in per_level:
            continue
        hit, tot = per_level[lvl]
        print(f"  {lvl:<18} {hit:>5}/{tot:<5} = {100 * hit / tot:6.2f}%")
    print(f"  {'action-only':<18} {action_hits[0]:>5}/{action_hits[1]:<5} = "
          f"{100 * action_hits[0] / action_hits[1]:6.2f}%")
    if counts[1]:
        print(f"  {'count-only':<18} {counts[0]:>5}/{counts[1]:<5} = "
              f"{100 * counts[0] / counts[1]:6.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ple", choices=["ple", "baseline"])
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ffn", type=int, default=384)
    ap.add_argument("--ple-dim", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows = load_rows("train.jsonl")
    test_rows = load_rows("test.jsonl")
    stoi, itos = V.load("vocab.json")

    ids, plens, totals, _ = encode_split(train_rows, stoi, SEQ_LEN)
    ids, plens, totals = ids.to(device), plens.to(device), totals.to(device)

    cfg = Config(arm=args.arm, vocab_size=len(itos), d_model=args.d_model,
                 n_layers=args.layers, n_heads=args.heads,
                 ffn_hidden=args.ffn, seq_len=SEQ_LEN, ple_dim=args.ple_dim)
    model = TinyLM(cfg).to(device)
    budget = model.param_budget()
    print(f"[{args.arm}] vocab={len(itos)} d_model={cfg.d_model} layers={cfg.n_layers} "
          f"ffn={cfg.ffn_hidden} ple_dim={cfg.ple_dim}")
    print(f"  core={budget['core']:,} stream={budget['stream']:,} "
          f"table={budget['table']:,} total={budget['total']:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    n = ids.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        frac = step / args.steps
        lr = (args.lr * step / args.warmup if step < args.warmup
              else 0.1 * args.lr + 0.9 * args.lr * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        pick = torch.randint(0, n, (args.batch_size,), generator=gen).to(device)
        logits, _ = model(ids[pick])
        loss = masked_loss(logits, ids[pick], plens[pick], totals[pick])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == 1:
            print(f"  step {step:>5}/{args.steps}  loss {loss.item():.4f}  "
                  f"lr {lr:.2e}  {time.time() - t0:.0f}s")
        if step % args.eval_every == 0 and step < args.steps:
            report(f"step {step} (held out)",
                   *score(model, stoi, itos, test_rows, device, SEQ_LEN))

    print(f"\ntrained in {time.time() - t0:.0f}s")
    report("FINAL held out", *score(model, stoi, itos, test_rows, device, SEQ_LEN))
    seen = [r for r in train_rows[:2000]]
    report("train subset (memorisation check)",
           *score(model, stoi, itos, seen, device, SEQ_LEN))

    if args.out:
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                    "itos": itos}, args.out)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
