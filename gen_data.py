"""Generate the desk-pet instruction dataset.

The split is by SURFACE FORM, not by row. A random row split would put
"walk forward two steps" in train and "walk forward three steps" in test, and
the resulting accuracy would measure numeral substitution rather than
generalisation. Instead:

  train  - PHRASINGS (bare/polite/question/casual) x COUNT_PATTERNS x JOINERS
  test   - HELDOUT_PHRASINGS and HELDOUT_JOINERS only

so every test row contains at least one wording the model has never seen.

Emitted rows are dicts: {"text", "plan", "level"}. `plan` is the token sequence
the model must produce, already including <eos>. `level` labels the row for
per-level scoring; a single overall accuracy hides exactly the failure that
matters (composition).
"""

import argparse
import json
import random

import templates as T


def _numeral_unit(n, rng):
    """A numeral and a number-agreeing unit. 'one steps' would be a bad row."""
    num = rng.choice(T.NUMERALS[n])
    unit = rng.choice(T.UNITS_SINGULAR if n == 1 else T.UNITS_PLURAL)
    return num, unit


def l1_rows(heldout, rng, decorate_n=0, light=False):
    """Single action, no count.

    `decorate_n` extra decorated variants per phrasing. Raising surface diversity
    is the only thing that helps here: 226 unique L1 texts oversampled to 13,600
    rows is ~60 repeats of each template per pass, and the model starts fitting
    surface forms rather than the mapping by ~800 steps. More ROWS at the same
    diversity only raises the repeat factor.
    """
    rows = []
    for action, spec in T.ACTIONS.items():
        if heldout:
            phrasings = T.HELDOUT_PHRASINGS.get(action, [])
        else:
            phrasings = [p for reg in T.PHRASINGS[action].values() for p in reg]
        for text in phrasings:
            variants = {text}
            for _ in range(decorate_n):
                variants.add(T.decorate(text, rng, light=light))
            for surface in variants:
                rows.append({"text": surface, "plan": [spec["token"]],
                             "level": "l1"})
    return rows


def l2_rows(heldout, rng, decorate_n=0, light=False):
    """Action plus repeat count.

    Held-out side varies the numeral surface only, so l2 is reported but is not
    the load-bearing generalisation test.

    Patterns ending in "{n} times" need number agreement too: "spin right one
    times" is not English, and training on ungrammatical rows teaches the model
    that agreement carries no information.
    """
    rows = []
    for action, patterns in T.COUNT_PATTERNS.items():
        token = T.ACTIONS[action]["token"]
        for pattern in patterns:
            for n in T.NUMERALS:
                num, unit = _numeral_unit(n, rng)
                text = pattern.format(n=num, unit=unit)
                if n == 1:
                    text = text.replace(f"{num} times", "once")
                    text = text.replace(f"{num} jumps", "one jump")
                variants = {text}
                for _ in range(decorate_n):
                    variants.add(T.decorate(text, rng, light=light))
                for surface in variants:
                    rows.append({"text": surface, "plan": [token, f"<{n}>"],
                                 "level": "l2"})
    return rows


def _atoms(heldout, rng, decorate_n=0):
    """Single-action fragments usable as a composition operand, with their plans.

    Operands are decorated with `allow_wrapper=False` inside l3_rows' own call,
    because a wrapper ("can you X") around one half of a two-action instruction
    reads as a question about that half rather than a command.
    """
    atoms = []
    for row in l1_rows(heldout, rng, decorate_n, light=True):
        atoms.append((row["text"], row["plan"]))
    if not heldout:
        for row in l2_rows(heldout, rng, decorate_n, light=True):
            atoms.append((row["text"], row["plan"]))
    return atoms


def l3_rows(n_rows, heldout, rng, atom_heldout=None, decorate_n=0):
    """Two actions in a stated order.

    Two independent axes:
      atom_heldout - are the operand wordings from HELDOUT_PHRASINGS?
      heldout      - is the connective from HELDOUT_JOINERS?

    They are separated because the first run conflated them and produced a
    diagnosis-proof failure: 26.5% of L3 rows got the FIRST action right and the
    second wrong, while order was never once reversed. The model had learned
    composition; what it lacked was a confident reading of words it had only
    ever seen as a bare one-word command. Training composition over held-out
    atoms with TRAINING joiners fixes that without leaking the test, because the
    test's connective is still unseen.
    """
    if atom_heldout is None:
        atom_heldout = heldout
    # Operands are decorated only lightly: an already-decorated fragment joined
    # to another produces "how about you you should forward buddy and then ...",
    # which no operator says. The whole-instruction wrapper below carries the
    # variation instead.
    atoms = _atoms(atom_heldout, rng, 1 if decorate_n else 0)
    joiners = T.HELDOUT_JOINERS if heldout else T.JOINERS
    rows, seen = [], set()
    attempts = 0
    while len(rows) < n_rows and attempts < n_rows * 40:
        attempts += 1
        (ta, pa), (tb, pb) = rng.choice(atoms), rng.choice(atoms)
        if pa == pb:
            continue  # "sit then sit" is not a composition
        text = rng.choice(joiners).format(a=ta, b=tb)
        # A wrapper belongs around the whole instruction, never one operand.
        if decorate_n and rng.random() < 0.3:
            text = T.decorate(text, rng, allow_wrapper=True)
        if text in seen:
            continue
        seen.add(text)
        rows.append({"text": text, "plan": pa + pb, "level": "l3"})
    return rows


def negative_rows(heldout=None, rng=None, holdout_frac=0.35):
    """Empty plan.

    Split WITHIN each kind, not by kind. The first run held out `unsupported`
    and `nearmiss` whole and refusal collapsed to 7%, while the trained kinds
    scored 100%: the model had learned to reject those particular sentences, not
    to reject what it cannot act on. Refusal does not transfer across kinds --
    "make me a coffee" and "how are you" share no surface feature -- so every
    kind must be represented in training, and the held-out rows are unseen
    sentences of a kind the model has learned to refuse.

    Never decorated. Decorating negatives was measured and lost outright: it put
    "please", "can you", "now" into 86% of negative rows alongside 89% of
    positive ones, erasing the only cue separating an instruction from a remark.
    Refusal fell from 83% to 27% while L3 barely moved.
    """
    rows = []
    for kind, texts in T.NEGATIVES.items():
        items = sorted(texts)
        if heldout is None:
            chosen = items
        else:
            n_hold = max(1, int(len(items) * holdout_frac))
            # Deterministic slice so train and test never overlap.
            hold = set(items[:n_hold])
            chosen = [t for t in items if (t in hold) == heldout]
        for text in chosen:
            rows.append({"text": text, "plan": [], "level": f"neg_{kind}"})
    return rows


def oversample(rows, target, rng):
    """Repeat rows to `target`. Templates are few and exact; the model needs
    repetition to fit them, and duplicates are harmless when the test set shares
    no surface form with train."""
    if not rows:
        return []
    out = list(rows)
    while len(out) < target:
        out.append(dict(rng.choice(rows)))
    rng.shuffle(out)
    return out[:target]


def build(seed=0, n_train=40000, n_test_l3=1500, ground_heldout=True,
          decorate_n=0):
    """Train/test rows.

    `ground_heldout`: include a SMALL number of L1 rows built from the held-out
    phrasings. Without them a held-out word has an id but an untrained
    embedding, and no word-level model can infer meaning from a vector it never
    received a gradient for -- the first run scored 48% on L1 and got WORSE with
    training for exactly this reason.

    With grounding, each held-out word is seen in its bare single-action form
    only. L3 test rows then combine those words with held-out JOINERS in pairs
    the model never saw, so the test still measures composition rather than
    lexical recall. This mirrors deployment: the recogniser's word list is
    fixed and known, what varies is how words are put together.
    """
    rng = random.Random(seed)

    # ---- train ----
    # decorate_n multiplies surface diversity; n_train only multiplies repeats.
    tr_l1 = l1_rows(False, rng, decorate_n)
    tr_l2 = l2_rows(False, rng, decorate_n)
    tr_l3 = l3_rows(int(n_train * 0.22), False, rng, decorate_n=decorate_n)
    # Composition over held-out ATOM wordings but TRAINING joiners. This is the
    # fix for the 26.5% "second action wrong" bucket: those words previously
    # appeared only as bare one-word commands, so the model never had to read
    # them in a trailing clause. The test's connective stays unseen.
    tr_l3_ground = (l3_rows(int(n_train * 0.10), False, rng, atom_heldout=True,
                            decorate_n=decorate_n)
                    if ground_heldout else [])
    tr_neg = negative_rows(heldout=False)
    ground = l1_rows(True, rng, decorate_n) if ground_heldout else []

    train = (oversample(tr_l1, int(n_train * 0.26), rng)
             + oversample(tr_l2, int(n_train * 0.20), rng)
             + tr_l3
             + tr_l3_ground
             + oversample(tr_neg, int(n_train * 0.14), rng)
             + oversample(ground, int(n_train * 0.08), rng))
    rng.shuffle(train)

    # ---- test ----
    # Undecorated on purpose. Decoration is a training-side intervention, and
    # scoring against a decorated test set would move the target at the same time
    # as the treatment, making the two runs incomparable.
    te_l1 = l1_rows(True, rng)
    te_l2 = l2_rows(True, rng)
    te_l3 = l3_rows(n_test_l3, True, rng)
    te_neg = negative_rows(heldout=True)
    test = te_l1 + te_l2 + te_l3 + te_neg

    # L3 test rows must not appear verbatim in training.
    train_texts = {r["text"] for r in train}
    leaked = [r for r in test if r["level"] == "l3" and r["text"] in train_texts]
    test = [r for r in test
            if r["level"] != "l3" or r["text"] not in train_texts]
    return train, test, leaked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", type=int, default=40000)
    ap.add_argument("--test-l3", type=int, default=1500)
    ap.add_argument("--out-train", default="train.jsonl")
    ap.add_argument("--out-test", default="test.jsonl")
    ap.add_argument("--decorate", type=int, default=6,
                    help="extra decorated variants per phrasing (0 = off). "
                         "This, not --train, is what adds information.")
    args = ap.parse_args()

    train, test, leaked = build(args.seed, args.train, args.test_l3,
                                decorate_n=args.decorate)

    for path, rows in ((args.out_train, train), (args.out_test, test)):
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def tally(rows):
        counts = {}
        for r in rows:
            counts[r["level"]] = counts.get(r["level"], 0) + 1
        return dict(sorted(counts.items()))

    print(f"train {len(train):>6}  {tally(train)}")
    print(f"test  {len(test):>6}  {tally(test)}")
    print(f"dropped {len(leaked)} test rows that collided with train")

    # Repeat factor is the number that predicts memorisation, so report it.
    print("\nsurface diversity (repeat factor = rows / unique texts):")
    for level in sorted({r["level"] for r in train}):
        rows = [r for r in train if r["level"] == level]
        uniq = len({r["text"] for r in rows})
        print(f"  {level:<18} {len(rows):>6} rows  {uniq:>6} unique  "
              f"{len(rows) / uniq:>5.1f}x")
    uniq_all = len({r["text"] for r in train})
    print(f"  {'TOTAL':<18} {len(train):>6} rows  {uniq_all:>6} unique  "
          f"{len(train) / uniq_all:>5.1f}x")

    print("\nsamples:")
    for r in train[:5] + [r for r in test if r["level"] == "l3"][:2]:
        print(f"  [{r['level']:>9}] {r['text']!r:62} -> "
              f"{' '.join(r['plan']) or '(empty)'}")


if __name__ == "__main__":
    main()
