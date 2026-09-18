"""Inspect a trained checkpoint: what L3 gets wrong, and how.

Exact-match accuracy says how often the plan is right. It does not say whether a
wrong plan is a swapped order, a dropped second action, or a wrong action -- and
those have very different implications. Order errors mean the model understands
both actions but not the sequence; drops mean it stops early; substitutions mean
it misread a word.
"""

import argparse
import collections
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tinyllm", "src"))
from model import Config, TinyLM  # noqa: E402

import vocab as V  # noqa: E402
from train_pet import SEQ_LEN, score  # noqa: E402


def decode_all(model, stoi, itos, rows, device, max_new=8, batch=512):
    """Greedy constrained decode, returning predicted token strings per row."""
    model.eval()
    lo, hi = V.plan_span(stoi)
    allowed = torch.full((len(itos),), float("-inf"), device=device)
    allowed[lo:hi + 1] = 0.0
    preds = []
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            prompts = []
            for row in chunk:
                seq = [V.BOS] + [stoi.get(w, stoi[V.UNK])
                                 for w in V.words_of(row["text"])]
                prompts.append(seq[:SEQ_LEN - max_new])
            B = len(prompts)
            idx = torch.full((B, SEQ_LEN), V.PAD, dtype=torch.long, device=device)
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
                        out[i].append(itos[tok])
                done |= nxt == V.EOS
                if bool(done.all()):
                    break
                write = torch.clamp(cursor, max=SEQ_LEN - 1)
                idx[rows_ix, write] = torch.where(done, idx[rows_ix, write], nxt)
                cursor = torch.clamp(cursor + 1, max=SEQ_LEN - 1)
            preds.extend(out)
    return preds


def classify(pred, gold):
    """Bucket a wrong prediction by failure mode."""
    if pred == gold:
        return "correct"
    pa = [t for t in pred if not t.startswith("<") or t in V.ACTION_TOKENS]
    ga = [t for t in gold if t in V.ACTION_TOKENS]
    pa = [t for t in pred if t in V.ACTION_TOKENS]
    if pa == ga:
        return "count_wrong"          # right actions, wrong repeat count
    if sorted(pa) == sorted(ga) and pa != ga:
        return "order_swapped"        # both actions, reversed
    if len(pa) < len(ga):
        return "action_dropped"
    if len(pa) > len(ga):
        return "action_extra"
    return "action_substituted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--test", default="test.jsonl")
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = Config(**blob["cfg"])
    itos = blob["itos"]
    stoi = {t: i for i, t in enumerate(itos)}
    model = TinyLM(cfg).to(device)
    model.load_state_dict(blob["model"])

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    preds = decode_all(model, stoi, itos, rows, device)

    buckets = collections.Counter()
    examples = collections.defaultdict(list)
    for row, pred in zip(rows, preds):
        if row["level"] != "l3":
            continue
        kind = classify(pred, row["plan"])
        buckets[kind] += 1
        if kind != "correct":
            examples[kind].append((row["text"], row["plan"], pred))

    total = sum(buckets.values())
    print(f"L3 breakdown over {total} rows")
    for kind, n in buckets.most_common():
        print(f"  {kind:<20} {n:>5}  {100 * n / total:5.1f}%")

    for kind in ("order_swapped", "action_dropped", "action_substituted",
                 "action_extra", "count_wrong"):
        if not examples[kind]:
            continue
        print(f"\n{kind}:")
        for text, gold, pred in examples[kind][:args.show]:
            print(f"  {text:<52} want {' '.join(gold):<22} got {' '.join(pred)}")

    # joiner-level view: is one connective much worse than the others?
    import templates as T
    print("\nby held-out joiner:")
    for joiner in T.HELDOUT_JOINERS:
        marker = joiner.replace("{a}", "").replace("{b}", "").strip()
        hit = tot = 0
        for row, pred in zip(rows, preds):
            if row["level"] != "l3" or marker not in row["text"]:
                continue
            tot += 1
            hit += pred == row["plan"]
        if tot:
            print(f"  {marker:<22} {hit:>5}/{tot:<5} = {100 * hit / tot:5.1f}%")


if __name__ == "__main__":
    main()
