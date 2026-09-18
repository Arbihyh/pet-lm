# pet-lm

A ~2.2M-parameter instruction model for a desk-pet robot: English command in,
action sequence out, small enough to target an ESP32-S3.

The robot is an STM32F103C8T6 desk pet driving five servos (four legs, one tail)
with twelve behaviours. Its existing control path is fixed voice command words.
The question this repo answers is whether a model small enough to sit beside it
can accept free-form instructions instead, including composed ones
("walk forward two steps and afterwards lie down").

**Status: feasibility confirmed on host. Not yet on hardware.**

## Result

Trained in 95 s on an RTX 5060. Held-out accuracy, exact match on the full
action sequence under greedy decoding constrained to the plan-token span:

| level | what it tests | baseline | ple |
|---|---|---:|---:|
| L1 | single action, unseen wording | 100.0% | 100.0% |
| L2 | action + repeat count | 99.4% | 99.4% |
| **L3** | **two actions, unseen connective** | **88.9%** | **90.6%** |
| — | count value only | 99.4% | 99.4% |
| — | action sequence only | 92.0% | 92.1% |
| neg | refuse what it cannot act on | 22–100% | 44–100% |

Core parameters: 2,214,336 (baseline) / 2,436,736 (ple).

**Order is never reversed.** Across 1500 composed held-out instructions the
model did not once emit the two actions in the wrong order. Composition
semantics are learned; the residual ~9% is second-action identification, which
is a data-density problem rather than a capacity one.

## What this is built on

The architecture is [manjunathshiva/esp32-tinyllm](https://github.com/manjunathshiva/esp32-tinyllm)
(MIT), itself building on [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)
(MIT), which established Per-Layer-Embeddings on a microcontroller. `src/model.py`
from that repo is imported unchanged; this repo replaces only the data path and
the metric. Clone it as a sibling directory:

```
<parent>/
  tinyllm/     # git clone https://github.com/manjunathshiva/esp32-tinyllm
  pet-lm/      # this repo
```

## Why the design is shaped this way

**Word-level vocabulary, not BPE.** English commands in this domain draw on a
closed set of 291 word forms. BPE would buy nothing and cost a 197 KB merge
table plus the failure mode upstream documents: a prompt tokenised differently
from training is silently out of distribution with nothing in the logs. Splitting
on whitespace is exact by construction and the device tokeniser becomes a hash
lookup.

**Action tokens are the output vocabulary.** Decoding is masked to a contiguous
token span, so a malformed plan is unrepresentable rather than merely unlikely.
This replaces a constrained-decoding grammar for free.

**PLE is probably not worth keeping.** Its published advantage comes from a large
vocabulary (upstream ablation: +0.025 nats at vocab 4096 versus +0.098 at
32768). At vocab 312 the two arms are within noise of each other, and `ple` costs
10% more parameters plus four tensor groups in the firmware. Deciding this
properly needs multiple seeds; it has not been done.

**Loss is masked to the plan region.** Training the model to predict the
operator's own words would spend capacity on a distribution it never generates.

## Files

| path | what |
|---|---|
| `templates.py` | 12 actions x 4 registers, count patterns, connectives, negatives |
| `gen_data.py` | three difficulty levels, held-out splits, negative sampling |
| `vocab.py` | word-level vocabulary, encoding, plan-token span |
| `train_pet.py` | training with per-level exact-match scoring |
| `inspect_errors.py` | failure-mode attribution (order / drop / substitution / count) |

## Run

```sh
python gen_data.py            # -> train.jsonl test.jsonl
python vocab.py               # -> vocab.json
python train_pet.py --arm baseline --steps 3000 --out pet.pt
python inspect_errors.py pet.pt
```

## Evaluation discipline

Two mistakes were made and fixed while building this, both of which produced
plausible-looking numbers:

**Held-out phrasings introduced unseen words.** A word-level model cannot
generalise to a word it has no embedding for — there is no subword structure to
fall back on. 46% of held-out tokens encoded as `<unk>`, so those rows carried no
signal, and L1 accuracy *fell* from 61% to 49% as training sharpened a mapping
for a token that was never supervised. The vocabulary now covers the domain's
closed word set and generalisation is tested where it is testable: unseen
*combinations* of known words. This mirrors deployment, where the recogniser's
word list is fixed and known.

**A connective collided with a numeral.** Fixing `"jump one times"` to
`"jump once"` made `once` a count word, while a held-out connective was
`"once done"` — so `"X once done Y"` read as `"X one time"`. That joiner dropped
to 26% while the other rose to 94%, and overall L3 fell from 74% to 56%, which
looks like the other fixes failing. `done` also appeared in zero training rows,
making the connective untestable by construction. Held-out connectives are now
checked programmatically against every other word role.

The lesson both times: a single aggregate number hid the cause. Per-level and
per-connective breakdowns are why the second bug was found rather than
misattributed.

## Known limits

- **Accuracy is an upper bound.** Templates are narrower than real speech.
  Training accuracy is 100% against 90.6% held out; a real evaluation needs
  human-written instructions, which do not exist yet.
- **Refusal is unreliable**, worst on near-misses ("my dog can jump", 44%),
  and the held-out negative set is only 34 rows — too few to be conclusive.
- **English only.** The pet's current voice module is Chinese. A character-level
  Chinese vocabulary is larger, ambiguous at the single-character level
  ("走" spans multiple senses), and needs UTF-8 handling in the device
  tokeniser. English validates the architecture with one variable changed at a
  time.
- **Single seed.** No variance estimates, so the baseline/ple gap is not
  meaningful.
- **Nothing has run on hardware.** The device-side numbers in `NOTES.md` are
  calibrated projections, not measurements.

## Next

1. Human-written test set — decides whether the rest is worth building.
2. Multiple seeds; drop `ple` if it stays within noise.
3. Port the core matvec to the int8-staged path used for the output head
   (upstream stages only the head, leaving the core at a measured 18.1 MB/s
   against the head's 56.9 MB/s).
4. Export to `.cact`, pass the host golden gate, flash, measure.

## Licence

MIT, matching the upstream projects this derives from.
