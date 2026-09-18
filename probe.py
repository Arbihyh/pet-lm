"""Probe a checkpoint interactively: what does it emit for a given sentence?

Reports the greedy plan and the probability the model assigned to stopping
immediately (emitting <eos> as the first plan token, i.e. refusing). That number
is what a confidence gate would threshold on, so this doubles as a check on
whether such a gate is even viable.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tinyllm", "src"))
from model import Config, TinyLM  # noqa: E402

import vocab as V  # noqa: E402
from train_pet import SEQ_LEN  # noqa: E402

PROBES = [
    # real commands
    ("lie down", "command"),
    ("walk forward two steps", "command"),
    ("walk forward two steps and afterwards lie down", "command"),
    # chit-chat -> should refuse
    ("nice weather isn't it", "should refuse"),
    ("what is your name", "should refuse"),
    ("how are you today", "should refuse"),
    # unsupported -> should refuse
    ("make me a coffee", "should refuse"),
    ("turn on the light", "should refuse"),
    # near-miss: action words present, not an instruction
    ("my dog can jump", "should refuse"),
    ("do you know how to sit", "should refuse"),
    ("walking is good exercise", "should refuse"),
    ("he told me to sit down", "should refuse"),
    # nonsense
    ("um", "should refuse"),
    ("banana telephone", "should refuse"),
]


def load(ckpt, device):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = Config(**blob["cfg"])
    itos = blob["itos"]
    model = TinyLM(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, {t: i for i, t in enumerate(itos)}, itos


@torch.no_grad()
def plan_for(model, stoi, itos, text, device, max_new=8):
    lo, hi = V.plan_span(stoi)
    allowed = torch.full((len(itos),), float("-inf"), device=device)
    allowed[lo:hi + 1] = 0.0

    words = V.words_of(text)
    seq = [V.BOS] + [stoi.get(w, stoi[V.UNK]) for w in words]
    unk = sum(1 for w in words if w not in stoi)
    idx = torch.full((1, SEQ_LEN), V.PAD, dtype=torch.long, device=device)
    idx[0, :len(seq)] = torch.tensor(seq, device=device)
    cursor = len(seq)

    out, stop_prob = [], None
    for _ in range(max_new):
        logits, _ = model(idx)
        step = logits[0, cursor - 1] + allowed
        probs = F.softmax(step, dim=-1)
        if stop_prob is None:
            # P(refuse) = probability mass on stopping before any action.
            stop_prob = probs[V.EOS].item()
        nxt = int(step.argmax())
        if nxt == V.EOS:
            break
        out.append(itos[nxt])
        if cursor >= SEQ_LEN:
            break
        idx[0, cursor] = nxt
        cursor += 1
    return out, stop_prob, unk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--text", nargs="*", help="probe these instead of the builtins")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stoi, itos = load(args.ckpt, device)

    probes = [(t, "") for t in args.text] if args.text else PROBES
    print(f"{'input':<48}{'expected':<15}{'emitted':<26}{'P(stop)':>8}")
    print("-" * 97)
    for text, expected in probes:
        plan, stop, unk = plan_for(model, stoi, itos, text, device)
        emitted = " ".join(plan) if plan else "(nothing)"
        flag = ""
        if expected == "should refuse" and plan:
            flag = "  <-- WRONG, acted"
        if expected == "command" and not plan:
            flag = "  <-- WRONG, ignored"
        if unk:
            flag += f"  [{unk} unknown word(s)]"
        print(f"{text:<48}{expected:<15}{emitted:<26}{stop:>8.3f}{flag}")


if __name__ == "__main__":
    main()
