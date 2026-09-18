"""Word-level vocabulary and encoding for the desk-pet model.

Word-level, not BPE. English commands draw on a tiny closed word set (216 forms
across every template here), so BPE buys nothing and costs a 197KB merge table
plus a class of bug the upstream project documents: a prompt tokenised
differently from training is silently out of distribution. Splitting on
whitespace is exact by construction, and the device tokeniser becomes a hash
lookup instead of a merge loop.

Layout is fixed so an exported model's ids stay meaningful:
    0        <pad>
    1        <bos>
    2        <eos>
    3..14    the 12 action tokens
    15..19   count tokens <1>..<5>
    20..     input words, sorted for determinism

The plan tokens occupy a contiguous block on purpose: constrained decoding at
inference is then a slice, and the firmware can mask everything else with a
single range check instead of a token set.
"""

import json
import re

import templates as T

PAD, BOS, EOS = 0, 1, 2
SPECIALS = ["<pad>", "<bos>", "<eos>"]
ACTION_TOKENS = [spec["token"] for spec in T.ACTIONS.values()]
COUNT_TOKENS = [f"<{n}>" for n in sorted(T.NUMERALS)]
UNK = "<unk>"

_WORD = re.compile(r"[a-z0-9']+")


def words_of(text):
    """Tokenise input text. Lowercase, keep apostrophes ("isn't" -> "isn't")."""
    return _WORD.findall(text.lower())


def template_words():
    """Every word any template can emit, held-out ones included.

    Deriving the vocabulary from training rows alone was wrong: the held-out
    phrasings introduce words ("advance", "retreat", "rotate") that then encode
    as <unk>, so those rows carry no signal at all and the model cannot do
    better than guess. 46% of held-out tokens were <unk>, and accuracy DROPPED
    with training as the model sharpened a mapping for a token it never saw
    supervised.

    A word-level model cannot generalise to an unseen word -- there is no
    subword structure to fall back on. So the vocabulary covers the closed word
    set of the domain (a real deployment would do the same: the recogniser's
    grammar is fixed), and generalisation is tested where it is actually
    testable: unseen COMBINATIONS of known words.
    """
    seen = set()

    def add(text):
        seen.update(words_of(text))

    for action in T.PHRASINGS:
        for reg in T.PHRASINGS[action].values():
            for p in reg:
                add(p)
    for lst in T.HELDOUT_PHRASINGS.values():
        for p in lst:
            add(p)
    for lst in T.COUNT_PATTERNS.values():
        for p in lst:
            add(p.replace("{n}", " ").replace("{unit}", " "))
    for lst in T.NEGATIVES.values():
        for p in lst:
            add(p)
    for j in T.JOINERS + T.HELDOUT_JOINERS:
        add(j.replace("{a}", " ").replace("{b}", " "))
    for forms in T.NUMERALS.values():
        for f in forms:
            add(f)
    for u in T.UNITS_SINGULAR + T.UNITS_PLURAL:
        add(u)
    return sorted(seen)


def build_vocab(rows=None):
    """Vocabulary over the domain's closed word set.

    `rows` is accepted and ignored for call compatibility; see template_words().
    """
    words = template_words()
    itos = SPECIALS + ACTION_TOKENS + COUNT_TOKENS + [UNK] + words
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, itos


def plan_span(stoi):
    """(lo, hi) covering every emittable plan token plus <eos>.

    Decoding is restricted to this span, which makes a malformed plan
    unrepresentable instead of merely unlikely.
    """
    ids = [stoi[t] for t in ACTION_TOKENS + COUNT_TOKENS] + [EOS]
    return min(ids), max(ids)


def encode(row, stoi, seq_len):
    """One training example: <bos> words... <eos-of-input> plan... <eos>.

    The input/plan boundary is the first plan-span token, so no extra separator
    id is needed. Loss is masked to the plan region by `prompt_len`.
    """
    unk = stoi[UNK]
    ids = [BOS] + [stoi.get(w, unk) for w in words_of(row["text"])]
    prompt_len = len(ids)
    ids += [stoi[t] for t in row["plan"]] + [EOS]
    if len(ids) > seq_len:
        return None
    pad = [PAD] * (seq_len - len(ids))
    return ids + pad, prompt_len, len(ids)


def save(stoi, itos, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"itos": itos}, fh, indent=1)


def load(path):
    with open(path, encoding="utf-8") as fh:
        itos = json.load(fh)["itos"]
    return {t: i for i, t in enumerate(itos)}, itos


if __name__ == "__main__":
    import sys

    rows = [json.loads(l) for l in open("train.jsonl", encoding="utf-8")]
    stoi, itos = build_vocab(rows)
    save(stoi, itos, "vocab.json")

    lens = []
    dropped = 0
    for row in rows:
        got = encode(row, stoi, 64)
        if got is None:
            dropped += 1
        else:
            lens.append(got[2])
    print(f"vocab {len(itos)}  (specials {len(SPECIALS)}, actions {len(ACTION_TOKENS)}, "
          f"counts {len(COUNT_TOKENS)}, words {len(itos) - len(SPECIALS) - len(ACTION_TOKENS) - len(COUNT_TOKENS) - 1})")
    print(f"plan span {plan_span(stoi)}")
    print(f"encoded len: max {max(lens)}  mean {sum(lens)/len(lens):.1f}  dropped {dropped}")
    print(f"\nexample: {rows[0]['text']!r}")
    ids, plen, total = encode(rows[0], stoi, 64)
    print(f"  ids[:{total}] = {ids[:total]}")
    print(f"  prompt_len = {plen}  -> plan tokens {[itos[i] for i in ids[plen:total]]}")
